#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : rpc.py
# Author: anyongjin
# Date  : 2023/4/1
import asyncio
import time

import psutil
from starlette.concurrency import run_in_threadpool
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from banbot.storage.common import *
from banbot.config.consts import *
from banbot.storage import (dba, sa, InOutOrder, InOutStatus, get_db_orders, BotTask, get_order_filters,
                            ExitTags, EnterTags)
from banbot.data.metrics import *
from banbot.util import btime
from banbot.main.addons import MarketPrice
from banbot.compute import get_context
from banbot.util.common import bufferHandler
from banbot.config import AppConfig
from banbot.rpc.api.schemas import *


def run_in_loop(func):
    async def wrapper(*args, **kwargs):
        coort = func(*args, **kwargs)
        fut = asyncio.run_coroutine_threadsafe(coort, BotGlobal.bot_loop)
        return await run_in_threadpool(fut.result)
    return wrapper


class RPCException(Exception):

    def __init__(self, message: str) -> None:
        super().__init__(self)
        self.message = message

    def __str__(self):
        return self.message

    def __json__(self):
        return {
            'msg': self.message
        }


class RPC:

    def __init__(self, bot):
        # 这里不导入LiveTrader类型提示，会导致循环引用
        self.bot = bot
        self._config = bot.config
        self._wallet = bot.wallets

    def _rpc_start(self) -> Dict[str, str]:
        """ Handler for start """
        if BotGlobal.state == BotState.RUNNING:
            return {'status': 'already running'}

        BotGlobal.state = BotState.RUNNING
        return {'status': 'starting trader ...'}

    def _rpc_stop(self) -> Dict[str, str]:
        """ Handler for stop """
        if BotGlobal.state == BotState.RUNNING:
            BotGlobal.state = BotState.STOPPED
            return {'status': 'stopping trader ...'}

        return {'status': 'already stopped'}

    def balance(self) -> Dict:
        """ Returns current account balance per crypto """
        items: List[Dict] = []
        total = 0.0

        for coin, item in self.bot.wallets.data.items():
            cur_total = item.total()
            if not cur_total:
                continue
            cur_total_fiat = cur_total * MarketPrice.get(coin)
            total += cur_total_fiat
            items.append({
                'symbol': coin,
                'free': item.available,
                'used': cur_total - item.available,
                'total': cur_total,
                'total_fiat': cur_total_fiat,
            })
        items = sorted(items, key=lambda x: x['total_fiat'], reverse=True)

        return {
            'items': items,
            'total': total,
        }

    async def open_num(self):
        open_ods = await get_db_orders(status='open')
        return dict(
            open_num=len(open_ods),
            max_num=self.bot.order_mgr.max_open_orders,
            total_cost=sum(od.enter_cost for od in open_ods)
        )

    async def pair_performance(self):
        return await InOutOrder.get_overall_performance()

    async def dash_statistics(self):
        orders = await get_db_orders()

        profit_all, profit_pct_all = [], []
        day_ts = []
        profit_closed, profit_pct_closed = [], []
        durations = []
        winning_trades = 0
        losing_trades = 0
        winning_profit = 0.0
        losing_profit = 0.0
        best_pair, best_rate = None, 0
        total_cost_closed = 0

        for od in orders:
            cur_ms = od.exit_at or btime.utcstamp()
            durations.append((cur_ms - od.enter_at) / 1000)
            profit_val = od.profit or 0.0
            profit_pct = od.profit_rate or 0.0
            profit_pct_all.append(profit_pct)
            profit_all.append(profit_val)
            if od.status == InOutStatus.FullExit:
                if profit_val > best_rate:
                    best_pair, best_rate = od.symbol, profit_pct
                if profit_val >= 0:
                    winning_trades += 1
                    winning_profit += profit_val
                else:
                    losing_trades += 1
                    losing_profit -= profit_val
                profit_closed.append(profit_val)
                profit_pct_closed.append(profit_pct)
                total_cost_closed += od.enter_cost
                done_ts = od.exit_at or od.enter_at
                day_ts.append(done_ts // 1000 // secs_day)

        closed_num = len(profit_closed)
        profit_closed_sum = sum(profit_closed) if closed_num else 0.
        profit_closed_mean = float(profit_closed_sum / closed_num) if closed_num else 0.

        all_num = len(profit_all)
        profit_all_sum = sum(profit_all) if all_num else 0.
        profit_all_mean = float(profit_all_sum / all_num) if all_num else 0.

        # 计算初始余额
        init_balance = self._wallet.fiat_value() - profit_all_sum
        total_cost = sum(od.enter_cost for od in orders)

        profit_factor = winning_profit / abs(losing_profit) if losing_profit else float('inf')
        winrate = (winning_trades / closed_num) if closed_num > 0 else 0

        # 按天分组得到每日利润
        zip_profits = list(zip(day_ts, profit_all))
        from banbot.util.misc import groupby
        profit_gps = groupby(zip_profits, lambda x: x[0])
        day_profits = [(key, sum(v[1] for v in items)) for key, items in profit_gps.items()]
        if not day_profits:
            day_ts, day_profit_vals = [], []
        else:
            day_ts, day_profit_vals = list(zip(*day_profits))

        # 计算每日利润的期望
        expectancy, expectancy_ratio = calc_expectancy(day_profit_vals)
        # 按天计算最大回撤
        abs_max_drawdown, _, _, _, _, max_drawdown = calc_max_drawdown(day_profit_vals, init_balance)

        first_ms = orders[0].enter_at if orders else None
        last_ms = orders[-1].enter_at if orders else None
        num = float(len(durations) or 1)
        profit_pct_closed_avg = sum(profit_pct_closed) / len(profit_pct_closed) if profit_pct_closed else 0
        profit_pct_all_avg = sum(profit_pct_all) / len(profit_pct_all) if profit_pct_all else 0
        return {
            'profit_closed_percent_mean': profit_pct_closed_avg * 100,
            'profit_closed_mean': profit_closed_mean,
            'profit_closed_percent_sum': profit_closed_sum / total_cost_closed * 100 if total_cost_closed else 0,
            'profit_closed_sum': profit_closed_sum,
            'profit_all_percent_mean': profit_pct_all_avg * 100,
            'profit_all_mean': profit_all_mean,
            'profit_all_percent_sum': profit_all_sum / total_cost * 100 if total_cost else 0,
            'profit_all_sum': profit_all_sum,
            'trade_count': len(orders),
            'closed_trade_count': closed_num,
            'first_trade_timestamp': first_ms,
            'latest_trade_timestamp': last_ms,
            'avg_duration': sum(durations) / num,
            'best_pair': best_pair,
            'best_pair_profit_pct': best_rate,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'profit_factor': profit_factor,
            'winrate': winrate,
            'expectancy': expectancy,
            'expectancy_ratio': expectancy_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_abs': abs_max_drawdown,
            'total_cost': total_cost,
            'bot_start_timestamp': BotGlobal.start_at,
            'run_tfs': [p[0] for p in BotGlobal.run_tf_secs],
            'exchange': BotGlobal.exg_name,
            'market': BotGlobal.market_type,
            'pairs': BotGlobal.pairs
        }

    async def stats(self) -> Dict[str, Any]:
        orders = await get_db_orders(status='his')
        # Duration
        dur: Dict[str, List[float]] = {'wins': [], 'draws': [], 'losses': []}
        # Exit reason
        exit_reasons = {}
        for od in orders:
            exit_tag = od.exit_tag or 'empty'
            if exit_tag not in exit_reasons:
                exit_reasons[exit_tag] = {'wins': 0, 'losses': 0, 'draws': 0}
            od_sta = 'wins' if od.profit > 0 else ('losses' if od.profit < 0 else 'draws')
            exit_reasons[exit_tag][od_sta] += 1

            dur[od_sta].append((od.exit_at - od.enter_at) / 1000)

        wins_dur = sum(dur['wins']) / len(dur['wins']) if len(dur['wins']) > 0 else None
        draws_dur = sum(dur['draws']) / len(dur['draws']) if len(dur['draws']) > 0 else None
        losses_dur = sum(dur['losses']) / len(dur['losses']) if len(dur['losses']) > 0 else None

        durations = {'wins': wins_dur, 'draws': draws_dur, 'losses': losses_dur}
        tag_list = []
        for tag, item in exit_reasons.items():
            item['tag'] = tag
            tag_list.append(item)
        tag_list = sorted(tag_list, key=lambda x: x['tag'])
        return {'exit_reasons': tag_list, 'durations': durations}

    async def timeunit_profit(self, timescale: int, timeunit: str = 'days') -> List[Dict]:
        init_date = datetime.now(timezone.utc).date()
        if timeunit == 'weeks':
            # weekly
            init_date = init_date - timedelta(days=init_date.weekday())  # Monday
        if timeunit == 'months':
            init_date = init_date.replace(day=1)

        def time_offset(step: int):
            if timeunit == 'months':
                return relativedelta(months=step)
            return timedelta(**{timeunit: step})

        if not (isinstance(timescale, int) and timescale > 0):
            raise RPCException('timescale must be an integer greater than 0')

        his_ods = await get_db_orders(status='his')

        if not his_ods:
            return []

        result = []
        total_value = self._wallet.fiat_value()
        min_start = min(od.enter_at for od in his_ods)
        for num in range(0, timescale):
            profitday = init_date - time_offset(num)
            stop_date = profitday + time_offset(1)
            start_ms = btime.to_utcstamp(profitday, ms=True)
            stop_ms = btime.to_utcstamp(stop_date, ms=True)
            if start_ms < min_start:
                break
            cur_profits = [od.profit for od in his_ods if start_ms <= od.enter_at <= stop_ms]

            profit_sum = sum(cur_profits)
            # Calculate this periods starting balance
            total_value -= profit_sum
            result.append({
                'date_ms': btime.to_utcstamp(profitday, ms=True, cut_int=True),
                'start_balance': total_value,
                'profit_sum': profit_sum,
                'profit_pct': round(profit_sum / total_value, 8) if total_value > 0 else 0,
                'order_num': len(cur_profits),
            })
        return result

    async def get_orders(self, source: str = 'bot', status: str = None, symbol: str = None,
                         start_time: int = 0, stop_time: int = 0, limit: int = 0, offset: int = 0,
                         with_total: bool = False) -> Dict[str, Any]:
        if source == 'bot':
            return await self.get_ban_orders(status, symbol, start_time, stop_time,
                                             limit, offset, with_total)
        elif source == 'exchange':
            return await self.get_exg_orders(symbol, start_time, limit or 10)
        elif source == 'position':
            return await self.get_exg_positions()
        else:
            return dict(code=400, msg='invalid source!')

    async def get_ban_orders(self, status: str = None, symbol: str = None,
                             start_time: int = 0, stop_time: int = 0, limit: int = 0, offset: int = 0,
                             with_total: bool = False) -> Dict[str, Any]:
        '''
        筛选符合条件的订单列表。查未平仓订单和已平仓订单都经过此接口
        '''
        sess = dba.session
        order_by = InOutOrder.enter_at.desc()
        filters = []
        if start_time:
            filters.append(InOutOrder.enter_at >= start_time)
        if stop_time:
            filters.append(InOutOrder.exit_at >= stop_time)
        orders = await get_db_orders(status=status, pairs=symbol, filters=filters,
                                     limit=limit, offset=offset, order_by=order_by)

        total_num = 0
        if with_total:
            fts = get_order_filters(task_id=BotTask.cur_id, status=status, pairs=symbol, filters=filters)
            total_num = await sess.scalar(sa.select(sa.func.count(InOutOrder.id)).filter(*fts))

        if not orders:
            return dict(data=[], total_num=total_num, offset=offset)
        results = []
        for od in orders:
            if od.exit_tag and od.exit and od.exit.price:
                cur_price = od.exit.price
            else:
                cur_price = MarketPrice.get(od.symbol)
                od.update_profits(cur_price)
            od_dict = od.dict(flat_sub=True)
            od_dict.update(dict(
                close_profit=od.profit_rate if od.exit_at else None,
                cur_price=cur_price,
            ))
            results.append(od_dict)
        return dict(data=results, total_num=total_num, offset=offset)

    @run_in_loop
    async def get_exg_orders(self, symbol: str, start_time: int, limit: int):
        od_list = await self.bot.exchange.fetch_orders(symbol, start_time, limit)
        return dict(data=od_list)

    @run_in_loop
    async def get_exg_positions(self):
        pos_list = await self.bot.exchange.fetch_account_positions()
        positions = [p for p in pos_list if p['notional']]
        return dict(data=positions)

    def _force_entry_validations(self, pair: str, order_side: str):
        if order_side not in {'long', 'short'}:
            raise RPCException("order_side must be long/short")

        if order_side == 'short' and self._config['market_type'] == 'spot':
            raise RPCException("Can't go short on Spot markets.")

        if pair not in self.bot.exchange.get_markets(tradable_only=True):
            raise RPCException('Symbol does not exist or market is not active.')
        # Check if pair quote currency equals to the stake currency.
        cur_quote = self.bot.exchange.get_pair_quote_currency(pair)
        if cur_quote not in self.bot.exchange.quote_symbols:
            raise RPCException(
                f'Wrong pair selected. Only pairs with stake-currency {cur_quote} allowed.')

    async def force_entry(self, pair: str, price: Optional[float],
                          order_type: Optional[str] = None,
                          side: Optional[str] = 'long',
                          enter_cost: Optional[float] = None,
                          enter_tag: Optional[str] = 'force_entry',
                          leverage: Optional[float] = None,
                          strategy: Optional[str] = None,
                          stoploss_price: Optional[float] = None) -> Optional[dict]:
        """
        Handler for forcebuy <asset> <price>
        Buys a pair trade at the given or current price
        """
        from banbot.strategy import BaseStrategy  # 放到顶层会导致循环引用
        from banbot.compute import TempContext
        self._force_entry_validations(pair, side)

        open_ods = await get_db_orders(pairs=pair, status='open')
        if len(open_ods) >= self.bot.order_mgr.max_open_orders:
            raise RPCException("Maximum number of trades is reached.")

        enter_dic = dict(
            tag=enter_tag or EnterTags.user_open,
            short=side == 'short',
            leverage=leverage or self._config.get('leverage'),
            enter_price=price,
            enter_order_type=order_type,
            stoploss_price=stoploss_price
        )

        if not enter_cost:
            enter_cost = BaseStrategy.custom_cost(enter_dic)
        enter_dic['legal_cost'] = enter_cost

        # execute buy
        job = next((p for p in BotGlobal.stg_symbol_tfs
                    if p[1] == pair and (not strategy or p[0] == strategy)), None)
        watch_pairs = {p[1] for p in BotGlobal.stg_symbol_tfs}
        if not job:
            raise RPCException(f'{pair}/{strategy} is not managed by bot, valid: {watch_pairs}')
        timeframe = job[2]
        pair_tf = f'{self.bot.exchange.name}_{self.bot.exchange.market_type}_{pair}_{timeframe}'
        with TempContext(pair_tf):
            ctx = get_context(pair_tf)
            od = self.bot.order_mgr.enter_order(ctx, job[0], enter_dic, do_check=False)
            start = time.time()
            while time.time() - start < 10:
                od = await InOutOrder.get(od.id)
                if not od.enter.order_id and not od.exit_tag and not od.exit:
                    await asyncio.sleep(1)
                    continue
                break
            return od.dict(flat_sub=True)

    async def force_exit(self, order_id: str) -> dict:
        """
        Handler for forceexit <id>.
        Sells the given trade at current price
        """
        tag = ExitTags.user_exit
        tag_msg = '用户通过管理面板取消'
        if order_id == 'all':
            args = dict(status='open')
        else:
            args = dict(filters=[InOutOrder.id == int(order_id)])
        open_ods = await get_db_orders(**args)
        if not open_ods:
            raise RPCException('invalid argument')
        for od in open_ods:
            await od.force_exit(tag, tag_msg)
        return dict(close_num=len(open_ods))

    @run_in_loop
    async def close_pos(self, data: ClosePosPayload):
        tasks = []
        if data.symbol == 'all':
            pos_list = await self.bot.exchange.fetch_account_positions()
            positions = [p for p in pos_list if p['notional']]
            for p in positions:
                tasks.append(dict(
                    symbol=p['symbol'],
                    amount=p['contracts'],
                    side=p['side'],
                    order_type='market',
                    price=None
                ))
        else:
            tasks.append(data.model_dump())
        close_count, done_count = 0, 0
        for task in tasks:
            params = dict()
            if BotGlobal.market_type == 'future':
                params['positionSide'] = 'SHORT' if task['side'] == 'short' else 'LONG'
            side = 'buy' if task['side'] == 'short' else 'sell'
            symbol, od_type, amount, price = task['symbol'], task['order_type'], task['amount'], task.get('price')
            od_res = await self.bot.exchange.create_order(symbol, od_type, side, amount, price, params)
            if od_res.get('id'):
                close_count += 1
                if od_res.get('filled') == od_res.get('amount'):
                    done_count += 1
        return dict(close_num=close_count, done_num=done_count)

    async def calc_profits(self, status: str):
        od_list = await get_db_orders(status=status)
        for od in od_list:
            od.update_profits(MarketPrice.get(od.symbol))
        return dict(num=len(od_list))

    def pairlist(self) -> Dict:
        """ Returns the currently active blacklist"""

        res = {'method': self.bot.pair_mgr.name_list,
               'blacklist': self.bot.pair_mgr.blacklist,
               'whitelist': self.bot.pair_mgr.whitelist,
               }
        return res

    async def set_pairs(self, for_white: bool, adds: List[str], deletes: List[str]) -> Dict:
        '''
        手动设置交易对。增加或减少。
        '''
        errors = {}
        target = self.bot.pair_mgr.whitelist if for_white else self.bot.pair_mgr.blacklist
        init_list = set(target)
        if adds:
            valid_keys = set(self.bot.exchange.get_cur_markets().keys())
            old_list = set(target)
            for pair in adds:
                if pair not in valid_keys:
                    errors[pair] = f'Pair {pair} is not a valid symbol.'
                    continue
                if pair not in old_list:
                    target.append(pair)
                else:
                    errors[pair] = f'Pair {pair} already in pairlist.'
        if deletes:
            old_list = set(target)
            for pair in deletes:
                if pair in old_list:
                    target.remove(pair)
                else:
                    errors[pair] = {
                        'error_msg': f"Pair {pair} is not in the current blacklist."
                    }
        if for_white:
            asyncio.run_coroutine_threadsafe(self.bot.add_del_pairs(init_list), self.bot.loop)
        resp = self.pairlist()
        resp['errors'] = errors
        return resp

    @staticmethod
    def get_logs(limit: Optional[int]) -> Dict[str, Any]:
        """Returns the last X logs"""
        if limit:
            buffer = bufferHandler.buffer[-limit:]
        else:
            buffer = bufferHandler.buffer
        records = [[int(r.created * 1000), r.name, r.levelname,
                   r.message + ('\n' + r.exc_text if r.exc_text else '')]
                   for r in buffer]

        # Log format:
        # [logtime-formatted, logepoch, logger-name, loglevel, message \n + exception]
        # e.g. [1598520901097, "freqtrade.worker", "INFO", "Starting worker develop"]

        return {'logs': records}

    def set_allow_trade_after(self, cost_secs: int) -> Dict[str, int]:
        """
        在给定的时间到达前，禁止交易。
        :param cost_secs: 正数表示禁用一段时间开单；0表示立刻允许入场
        """
        start_from = btime.utcstamp() + cost_secs * 1000
        self.bot.order_mgr.disable_until = start_from
        return dict(allow_trade_at=start_from)

    def reload_config(self) -> Dict[str, str]:
        """ Handler for reload_config. """
        if AppConfig.obj:
            AppConfig.obj.config = None
        config = AppConfig.get()
        self._config = config
        self.bot.config = config
        # TODO: 这里应该给这些组件实现reload_config方法
        self.bot.order_mgr.config = config
        self.bot.wallets.config = config
        self.bot.pair_mgr.config = config
        self.bot.exchange.config = config
        self.bot.data_mgr.config = config
        return {'status': 'Reloading config ...'}

    def bot_info(self) -> Dict[str, Any]:
        return dict(
            cpu_pct=psutil.cpu_percent(interval=1),
            ram_pct=psutil.virtual_memory().percent,
            last_process=self.bot.last_process,
            allow_trade_at=self.bot.order_mgr.disable_until
        )

    @run_in_loop
    async def incomes(self, intype: str, symbol: str, start_time: int, limit: int):
        """
        获取账户损益资金流水
        """
        res_list = await self.bot.exchange.get_incomes(intype, symbol, start_time, limit)
        exg_name = self.bot.exchange.name
        if exg_name == 'binance':
            for item in res_list:
                info = item.get('info')
                if info:
                    item['account'] = info.get('info')
                    item['tradeId'] = info.get('tradeId')
                    del item['info']
        return dict(data=res_list)


