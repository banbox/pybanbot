#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : rpc.py
# Author: anyongjin
# Date  : 2023/4/1
import asyncio
import time

import psutil
from datetime import datetime, timedelta, timezone, date
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzlocal
from banbot.storage.common import *
from banbot.config.consts import *
from banbot.storage import db, sa, InOutOrder, InOutStatus, get_db_orders, BotTask, get_order_filters
from banbot.data.metrics import *
from banbot.util import btime
from banbot.main.addons import MarketPrice
from banbot.compute import get_context
from banbot.util.common import bufferHandler
from banbot.config import AppConfig


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

    def balance(self, stake_currency: str, fiat_display_currency: str) -> Dict:
        """ Returns current account balance per crypto """
        currencies: List[Dict] = []
        total = 0.0

        for coin, balance in self.bot.wallets.data.items():
            cur_total = balance.total
            if not cur_total:
                continue
            total += cur_total
            currencies.append({
                'currency': coin,
                'free': balance.available,
                'balance': cur_total,
                'used': cur_total - balance.available,
                'stake': stake_currency,
                'side': 'long',
                'leverage': 1,
                'position': 0,
                'is_position': False,
            })

        return {
            'currencies': currencies,
            'total': total,
            'symbol': fiat_display_currency,
            'stake': stake_currency,
        }

    def open_num(self):
        open_ods = InOutOrder.open_orders()
        return dict(
            current=len(open_ods),
            max=self.bot.order_mgr.max_open_orders,
            total_stake=sum(od.enter_cost for od in open_ods)
        )

    def performance(self):
        return InOutOrder.get_overall_performance()

    def trade_statistics(self, stake_currency: str):
        orders = InOutOrder.get_orders()

        profit_all = []
        day_ts = []
        profit_closed = []
        durations = []
        winning_trades = 0
        losing_trades = 0
        winning_profit = 0.0
        losing_profit = 0.0
        best_pair, best_rate = None, 0
        total_cost = 0

        for od in orders:
            if od.exit_at:
                durations.append((od.exit_at - od.enter_at) / 1000)
            profit_val = od.profit or 0.0
            profit_pct = od.profit_rate or 0.0
            if profit_val > best_rate:
                best_pair, best_rate = od.symbol, profit_pct
            if od.status < InOutStatus.FullExit:
                if profit_val >= 0:
                    winning_trades += 1
                    winning_profit += profit_val
                else:
                    losing_trades += 1
                    losing_profit -= profit_val
            else:
                profit_closed.append(profit_val)
            profit_all.append(profit_val)
            day_ts.append(od.exit_at // 1000 // secs_day)
            total_cost += od.enter_cost_real

        closed_num = len(profit_closed)
        profit_closed_sum = sum(profit_closed) if closed_num else 0.
        profit_closed_mean = float(profit_closed_sum / closed_num) if closed_num else 0.

        all_num = len(profit_all)
        profit_all_sum = sum(profit_all) if all_num else 0.
        profit_all_mean = float(profit_all_sum / all_num) if all_num else 0.

        # 计算初始余额
        init_balance = self._wallet.fiat_value() - profit_all_sum
        trading_volume = sum(od.enter_cost for od in orders)

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
        return {
            'profit_closed_percent_mean': round(profit_closed_mean * 100, 2),
            'profit_closed_ratio_mean': profit_closed_mean,
            'profit_closed_percent_sum': round(profit_closed_sum * 100, 2),
            'profit_closed_ratio_sum': profit_closed_sum,
            'profit_all_percent_mean': round(profit_all_mean * 100, 2),
            'profit_all_ratio_mean': profit_all_mean,
            'profit_all_percent_sum': round(profit_all_sum * 100, 2),
            'profit_all_ratio_sum': profit_all_sum,
            'trade_count': len(orders),
            'closed_trade_count': closed_num,
            'first_trade_date': btime.to_datestr(first_ms),
            'first_trade_timestamp': first_ms,
            'latest_trade_date': btime.to_datestr(last_ms),
            'latest_trade_timestamp': last_ms,
            'avg_duration': str(timedelta(seconds=sum(durations) / num)).split('.')[0],
            'best_pair': best_pair,
            'best_pair_profit_ratio': best_rate,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'profit_factor': profit_factor,
            'winrate': winrate,
            'expectancy': expectancy,
            'expectancy_ratio': expectancy_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_abs': abs_max_drawdown,
            'trading_volume': trading_volume,
            'bot_start_timestamp': BotGlobal.start_at,
            'bot_start_date': btime.to_datestr(BotGlobal.start_at)
        }

    def stats(self) -> Dict[str, Any]:
        orders = InOutOrder.his_orders()
        # Duration
        dur: Dict[str, List[float]] = {'wins': [], 'draws': [], 'losses': []}
        # Exit reason
        exit_reasons = {}
        for od in orders:
            if od.exit_tag not in exit_reasons:
                exit_reasons[od.exit_tag] = {'wins': 0, 'losses': 0, 'draws': 0}
            od_sta = 'wins' if od.profit > 0 else ('losses' if od.profit < 0 else 'draws')
            exit_reasons[od.exit_tag][od_sta] += 1

            dur[od_sta].append((od.exit_at - od.enter_at) / 1000)

        wins_dur = sum(dur['wins']) / len(dur['wins']) if len(dur['wins']) > 0 else None
        draws_dur = sum(dur['draws']) / len(dur['draws']) if len(dur['draws']) > 0 else None
        losses_dur = sum(dur['losses']) / len(dur['losses']) if len(dur['losses']) > 0 else None

        durations = {'wins': wins_dur, 'draws': draws_dur, 'losses': losses_dur}
        return {'exit_reasons': exit_reasons, 'durations': durations}

    def timeunit_profit(self, timescale: int, stake_currency: List[str], timeunit: str = 'days') -> Dict[str, Any]:
        start_date = datetime.now(timezone.utc).date()
        if timeunit == 'weeks':
            # weekly
            start_date = start_date - timedelta(days=start_date.weekday())  # Monday
        if timeunit == 'months':
            start_date = start_date.replace(day=1)

        def time_offset(step: int):
            if timeunit == 'months':
                return relativedelta(months=step)
            return timedelta(**{timeunit: step})

        if not (isinstance(timescale, int) and timescale > 0):
            raise RPCException('timescale must be an integer greater than 0')

        profit_units: Dict[date, Dict] = {}
        daily_stake = self._wallet.fiat_value()

        his_ods = InOutOrder.his_orders()
        for day in range(0, timescale):
            profitday = start_date - time_offset(day)
            stop_date = profitday + time_offset(1)
            start_ms = btime.to_utcstamp(profitday, ms=True)
            stop_ms = btime.to_utcstamp(stop_date, ms=True)
            cur_profits = [od.profit for od in his_ods if start_ms <= od.enter_at <= stop_ms]

            curdayprofit = sum(cur_profits)
            # Calculate this periods starting balance
            daily_stake -= curdayprofit
            profit_units[profitday] = {
                'amount': curdayprofit,
                'daily_stake': daily_stake,
                'rel_profit': round(curdayprofit / daily_stake, 8) if daily_stake > 0 else 0,
                'trades': len(cur_profits),
            }

        data = [
            {
                'date': key,
                'abs_profit': value["amount"],
                'starting_balance': value["daily_stake"],
                'rel_profit': value["rel_profit"],
                'trade_count': value["trades"],
            }
            for key, value in profit_units.items()
        ]
        return {
            'stake_currency': stake_currency,
            'data': data
        }

    def trade_status(self, trade_ids: List[int] = None) -> List[Dict[str, Any]]:
        """
        Below follows the RPC backend it is prefixed with rpc_ to raise awareness that it is
        a remotely exposed function
        """
        # Fetch open trades
        if trade_ids:
            orders = get_db_orders(BotTask.cur_id, filters=[InOutOrder.id.in_(trade_ids)])
        else:
            orders = InOutOrder.open_orders()

        if not orders:
            raise RPCException('no active trade')
        else:
            results = []
            for od in orders:
                if od.is_open:
                    cur_price = MarketPrice.get(od.symbol)
                    od.update_by_price(cur_price)
                else:
                    cur_price = od.close_rate
                od_dict = od.dict()
                od_dict.update(dict(
                    close_profit=od.profit_rate if od.exit_at else None,
                    current_rate=cur_price,
                ))
                results.append(od_dict)
            return results

    def trade_history(self, limit: int, offset: int = 0, order_by_id: bool = False) -> Dict:
        """ Returns the X last trades """
        order_by = InOutOrder.id if order_by_id else InOutOrder.exit_at.desc()
        orders = get_db_orders(BotTask.cur_id, limit=limit, offset=offset, order_by=order_by)
        output = [od.dict() for od in orders]

        sess = db.session
        fts = get_order_filters(task_id=BotTask.cur_id, status='his')
        total_trades = sess.scalar(sa.select(sa.func.count(InOutOrder.id)).filter(*fts))

        return {
            "trades": output,
            "trades_count": len(output),
            "offset": offset,
            "total_trades": total_trades,
        }

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

    async def force_entry(self, pair: str, price: Optional[float], *,
                         order_type: Optional[str] = None,
                         order_side: Optional[str] = 'long',
                         stake_amount: Optional[float] = None,
                         enter_tag: Optional[str] = 'force_entry',
                         leverage: Optional[float] = None) -> Optional[dict]:
        """
        Handler for forcebuy <asset> <price>
        Buys a pair trade at the given or current price
        """
        from banbot.strategy import BaseStrategy  # 放到顶层会导致循环引用
        from banbot.compute import TempContext
        self._force_entry_validations(pair, order_side)

        open_ods = InOutOrder.open_orders(pairs=pair)
        if len(open_ods) >= self.bot.order_mgr.max_open_orders:
            raise RPCException("Maximum number of trades is reached.")

        enter_dic = dict(
            tag=enter_tag,
            short=order_side == 'short',
            leverage=leverage,
            enter_price=price,
            enter_order_type=order_type,
        )

        if not stake_amount:
            stake_amount = BaseStrategy.custom_cost(enter_dic)
        enter_dic['legal_cost'] = stake_amount

        # execute buy
        job = next((p for p in BotGlobal.stg_symbol_tfs if p[1] == pair), None)
        watch_pairs = {p[1] for p in BotGlobal.stg_symbol_tfs}
        if not job:
            raise RPCException(f'{pair} is not managed by bot, valid: {watch_pairs}')
        timeframe = job[2]
        pair_tf = f'{self.bot.exchange.name}_{self.bot.exchange.market_type}_{pair}_{timeframe}'
        with TempContext(pair_tf):
            ctx = get_context(pair_tf)
            od = self.bot.order_mgr.enter_order(ctx, job[0], enter_dic)
            if od:
                start = time.time()
                sess = db.session
                while time.time() - start < 10:
                    od = InOutOrder.get(sess, od.id)
                    if not od.enter.order_id and not od.exit_tag and not od.exit:
                        await asyncio.sleep(1)
                        continue
                    break
                result = od.dict()
                result['enter'] = od.enter.dict()
                if od.exit:
                    result['exit'] = od.exit.dict()
                return result
        raise RPCException(f'Failed to enter position for {pair}.')

    def force_exit(self, trade_id: str) -> Dict[str, str]:
        """
        Handler for forceexit <id>.
        Sells the given trade at current price
        """
        if trade_id == 'all':
            open_ods = InOutOrder.open_orders()
            for od in open_ods:
                od.force_exit()
            return {'result': 'Created exit orders for all open trades.'}

        od = get_db_orders(BotTask.cur_id, filters=[InOutOrder.id == int(trade_id)])
        if not od:
            raise RPCException('invalid argument')
        od[0].force_exit()
        return {'result': f'Created exit order for trade {trade_id}.'}

    def blacklist(self, add: Optional[List[str]] = None) -> Dict:
        """ Returns the currently active blacklist"""
        errors = {}
        if add:
            valid_keys = set(self.bot.exchange.get_markets().keys())
            old_list = set(self.bot.pair_mgr.blacklist)
            for pair in add:
                if pair not in valid_keys:
                    errors[pair] = f'Pair {pair} is not a valid symbol.'
                    continue
                if pair in old_list:
                    self.bot.pair_mgr.blacklist.append(pair)
                else:
                    errors[pair] = f'Pair {pair} already in pairlist.'

        res = {'method': self.bot.pair_mgr.name_list,
               'length': len(self.bot.pair_mgr.blacklist),
               'blacklist': self.bot.pair_mgr.blacklist,
               'errors': errors}
        return res

    def blacklist_delete(self, delete: List[str]) -> Dict:
        """ Removes pairs from currently active blacklist """
        errors = {}
        old_list = set(self.bot.pair_mgr.blacklist)
        for pair in delete:
            if pair in old_list:
                self.bot.pair_mgr.blacklist.remove(pair)
            else:
                errors[pair] = {
                    'error_msg': f"Pair {pair} is not in the current blacklist."
                }
        resp = self.blacklist()
        resp['errors'] = errors
        return resp

    def whitelist(self) -> Dict:
        """ Returns the currently active whitelist"""
        res = {'method': self.bot.pair_mgr.name_list,
               'length': len(self.bot.pair_mgr.whitelist),
               'whitelist': self.bot.pair_mgr.whitelist
               }
        return res

    @staticmethod
    def get_logs(limit: Optional[int]) -> Dict[str, Any]:
        """Returns the last X logs"""
        if limit:
            buffer = bufferHandler.buffer[-limit:]
        else:
            buffer = bufferHandler.buffer
        records = [[btime.to_datestr(r.created),
                   r.created * 1000, r.name, r.levelname,
                   r.message + ('\n' + r.exc_text if r.exc_text else '')]
                   for r in buffer]

        # Log format:
        # [logtime-formatted, logepoch, logger-name, loglevel, message \n + exception]
        # e.g. ["2020-08-27 11:35:01", 1598520901097.9397,
        #       "freqtrade.worker", "INFO", "Starting worker develop"]

        return {'log_count': len(records), 'logs': records}

    def stopentry(self) -> Dict[str, str]:
        """
        Handler to stop buying, but handle open trades gracefully.
        """
        # Set 'max_open_trades' to 0
        self._config['max_open_trades'] = 0
        self.bot.order_mgr.max_open_orders = 0

        return {'status': 'No more entries will occur from now. Run /reload_config to reset.'}

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
            last_process=self.bot.last_process
        )
