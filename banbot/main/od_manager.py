#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : main.py
# Author: anyongjin
# Date  : 2023/3/30
import asyncio

from collections import OrderedDict
from asyncio import Queue

from banbot.storage.orders import *
from banbot.util.common import logger, SingletonArg
from banbot.main.wallets import CryptoWallet, WalletsLocal
from banbot.strategy.base import BaseStrategy
from banbot.data.provider import *
from banbot.util.misc import *


class OrderBook():
    def __init__(self, **kwargs):
        self.bids = kwargs.get('bids')
        self.asks = kwargs.get('asks')
        self.symbol = kwargs.get('symbol')
        self.timestamp = kwargs.get('timestamp') or btime.time()

    def limit_price(self, side: str, depth: float):
        data_arr = self.bids if side == 'buy' else self.asks
        vol_sum, last_price = 0, 0
        for price, amount in data_arr:
            vol_sum += amount
            last_price = price
            if vol_sum >= depth:
                break
        if vol_sum < depth:
            logger.warning('depth not enough, require: {0:.5f} cur: {1:.5f}, len: {2}', depth, vol_sum, len(data_arr))
        return last_price


class OrderManager(metaclass=SingletonArg):
    def __init__(self, config: dict, exchange: CryptoExchange, wallets: WalletsLocal, data_hd: DataProvider,
                 callback: Callable):
        self.config = config
        self.name = exchange.name
        self.exchange = exchange
        self.wallets = wallets
        self.data_mgr = data_hd
        self.prices = dict()  # 所有产品对法币的价格
        self.network_cost = 3.  # 模拟网络延迟
        self.callback = callback
        self.fatal_stop = dict()
        self.last_ts = btime.time()  # 记录上次订单时间戳，方便对比钱包时间戳是否正确
        self._load_fatal_stop()
        self.disabled = False
        self.forbid_pairs = set()
        self.live_mode = btime.run_mode in TRADING_MODES
        self.pair_fee_limits = AppConfig.obj.exchange_cfg.get('pair_fee_limits')

    def _load_fatal_stop(self):
        fatal_cfg = self.config.get('fatal_stop')
        if not fatal_cfg:
            return
        for k, v in fatal_cfg.items():
            self.fatal_stop[int(k)] = v

    def _fire(self, od: InOutOrder, enter: bool):
        with TempContext(f'{od.symbol}/{od.timeframe}'):
            self.callback(od, enter)

    def allow_pair(self, pair: str) -> bool:
        if self.disabled:
            # 触发系统交易熔断时，禁止入场，允许出场
            logger.warning('order enter forbid, fatal stop, %s', pair)
            return False
        return pair not in self.forbid_pairs

    async def try_dump(self):
        pass

    def process_pair_orders(self, pair_tf: str, enters: List[Tuple[str, str, float]],
                            exits: List[Tuple[str, str]], exit_keys: Dict[int, str])\
            -> Tuple[List[InOutOrder], List[InOutOrder]]:
        '''
        批量创建指定交易对的订单
        :param pair_tf: 交易对
        :param enters: [(strategy, enter_tag, cost), ...]
        :param exits: [(strategy, exit_tag), ...]
        :param exit_keys: Dict(order_key, exit_tag)
        :return:
        '''
        ctx = get_context(pair_tf)
        pair, _, _, _ = get_cur_symbol(ctx)
        allow_enter = self.allow_pair(pair)
        if not allow_enter and not (exits or exit_keys):
            logger.debug('pair enter disable: %s', [pair, enters])
            return [], []
        buy_price, sell_price = self._get_pair_prices(pair)
        enter_ods, exit_ods = [], []
        if enters and btime.run_mode not in NORDER_MODES:
            if allow_enter:
                for stg_name, enter_tag, cost in enters:
                    enter_ods.append(self.enter_order(ctx, stg_name, enter_tag, cost, buy_price))
            else:
                logger.debug('pair enter not allow: %s', enters)
        if exits:
            for stg_name, exit_tag in exits:
                exit_ods.extend(self.exit_open_orders(exit_tag, sell_price, stg_name, pair))
        if exit_keys:
            for od_id, ext_tag in exit_keys.items():
                od = InOutOrder.get(od_id)
                exit_ods.append(self.exit_order(ctx, od, ext_tag, sell_price))
        enter_ods = [od for od in enter_ods if od]
        exit_ods = [od for od in exit_ods if od]
        return enter_ods, exit_ods

    def _get_pair_prices(self, pair: str) -> Tuple[Union[float, Callable], Union[float, Callable]]:
        price = self.data_mgr.get_latest_ohlcv(pair)[ccol]
        return price, price

    def enter_order(self, ctx: Context, strategy: str, tag: str, cost: float, price: Union[float, Callable]
                    ) -> Optional[InOutOrder]:
        '''
        策略产生入场信号，执行入场订单。（目前仅支持做多）
        :param ctx:
        :param strategy:
        :param tag:
        :param cost: 对应的法币价值。当定价币不是USD时，会计算其对应花费数量
        :param price:
        :return:
        '''
        if btime.run_mode in NORDER_MODES:
            return
        pair, base_s, quote_s, timeframe = get_cur_symbol(ctx)
        if lock_od := InOutOrder.get_order(pair, strategy, tag):
            # 同一交易对，同一策略，同一信号，只允许一个订单
            logger.debug('order lock, enter forbid: %s', lock_od)
            if random.random() < 0.2 and lock_od.elp_num_exit > 5:
                logger.error('lock order exit timeout: %s', lock_od)
            return
        quote_cost = self.wallets.get_avaiable_by_cost(quote_s, cost, self.last_ts)
        if not quote_cost or not self.allow_pair(pair):
            logger.debug('wallet empty or pair disable: %f', quote_cost)
            return
        od = InOutOrder(
            symbol=pair,
            timeframe=timeframe,
            quote_cost=quote_cost,
            enter_price=price,
            enter_tag=tag,
            enter_at=int(ctx[bar_arr][-1][0]),
            strategy=strategy
        )
        od.save()
        if btime.run_mode in TRADING_MODES:
            logger.info('enter order {0} {1} cost: {2:.2f}', od.symbol, od.enter_tag, cost)
        self._put_order(od, True)
        return od

    def _put_order(self, od: InOutOrder, is_enter: bool):
        pass

    def _fill_pending_enter(self, candle: np.ndarray, od: InOutOrder):
        enter_price = self._sim_market_price(od.symbol, od.timeframe, candle)
        if not od.enter.amount:
            od.enter.amount = od.quote_cost / enter_price
        quote_amount = enter_price * od.enter.amount
        ctx = get_context(f'{od.symbol}/{od.timeframe}')
        _, base_s, quote_s, timeframe = get_cur_symbol(ctx)
        fees = self.exchange.calc_funding_fee(od.enter)
        if fees['rate']:
            od.enter.fee = fees['rate']
            od.enter.fee_type = fees['currency']
        base_amt = od.enter.amount * (1 - od.enter.fee)
        self.wallets.update_wallets(**{quote_s: -quote_amount, base_s: base_amt})
        self.last_ts = btime.time()
        od.status = InOutStatus.FullEnter
        od.enter_at = int(ctx[bar_arr][-1][0])
        od.enter.filled = od.enter.amount
        od.enter.average = enter_price
        od.enter.status = OrderStatus.Close
        if not od.enter.price:
            od.enter.price = enter_price
        self._fire(od, True)

    def _fill_pending_exit(self, candle: np.ndarray, od: InOutOrder):
        exit_price = self._sim_market_price(od.symbol, od.timeframe, candle)
        quote_amount = exit_price * od.exit.amount
        fees = self.exchange.calc_funding_fee(od.exit)
        if fees['rate']:
            od.exit.fee = fees['rate']
            od.exit.fee_type = fees['currency']
        ctx = get_context(f'{od.symbol}/{od.timeframe}')
        pair, base_s, quote_s, timeframe = get_cur_symbol(ctx)
        quote_amt = quote_amount * (1 - od.exit.fee)
        self.wallets.update_wallets(**{quote_s: quote_amt, base_s: -od.exit.amount})
        self.last_ts = btime.time()
        od.status = InOutStatus.FullExit
        od.exit_at = int(ctx[bar_arr][-1][0])
        od.update_exit(
            status=OrderStatus.Close,
            price=exit_price,
            filled=od.exit.amount,
            average=exit_price,
        )
        self._finish_order(od)
        self._fire(od, False)

    def _sim_market_price(self, pair: str, timeframe: str, candle: np.ndarray) -> float:
        '''
        计算从收到bar数据，到订单提交到交易所的时间延迟：对应的价格。
        阳线和阴线对应不同的模拟方法。
        阳线一般是先略微下调，再上冲到最高点，最后略微回调出现上影线。
        阴线一般是先略微上调，再下跌到最低点，最后略微回调出现下影线。
        :return:
        '''
        rate = min(1., self.network_cost / tf_to_secs(timeframe))
        if candle is None:
            candle = self.data_mgr.get_latest_ohlcv(pair)
        open_p, high_p, low_p, close_p = candle[ocol: vcol]
        if open_p <= close_p:
            # 阳线，一般是先下调走出下影线，然后上升到最高点，最后略微回撤，出现上影线
            a, b, c = open_p - low_p, high_p - low_p, high_p - close_p
            total_len = a + b + c
            if not total_len:
                return close_p
            a_end_rate, b_end_rate = a / total_len, (a + b) / total_len
            if rate <= a_end_rate:
                start, end, pos_rate = open_p, low_p, rate / a_end_rate
            elif rate <= b_end_rate:
                start, end, pos_rate = low_p, high_p, (rate - a_end_rate) / (b_end_rate - a_end_rate)
            else:
                start, end, pos_rate = high_p, close_p, (rate - b_end_rate) / (1 - b_end_rate)
        else:
            # 阴线，一般是先上升走出上影线，然后下降到最低点，最后略微回调，出现下影线
            a, b, c = high_p - open_p, high_p - low_p, close_p - low_p
            total_len = a + b + c
            if not total_len:
                return close_p
            a_end_rate, b_end_rate = a / total_len, (a + b) / total_len
            if rate <= a_end_rate:
                start, end, pos_rate = open_p, high_p, rate / a_end_rate
            elif rate <= b_end_rate:
                start, end, pos_rate = high_p, low_p, (rate - a_end_rate) / (b_end_rate - a_end_rate)
            else:
                start, end, pos_rate = low_p, close_p, (rate - b_end_rate) / (1 - b_end_rate)
        return start * (1 - pos_rate) + end * pos_rate

    def fill_pending_orders(self, symbol: str = None, timeframe: str = None, candle: Optional[np.ndarray] = None):
        '''
        填充等待交易所响应的订单。不可用于实盘；可用于回测、模拟实盘等。
        此方法内部会访问锁：ctx_lock，请勿在TempContext中调用此方法
        :param symbol:
        :param timeframe:
        :param candle:
        :return:
        '''
        if btime.run_mode == btime.RunMode.LIVE:
            raise RuntimeError('fill_pending_orders unavaiable in LIVE mode')
        op_orders = InOutOrder.open_orders()
        for od in op_orders:
            if symbol and od.symbol != symbol or timeframe and od.timeframe != timeframe:
                continue
            if od.exit_tag and od.exit and od.exit.status != OrderStatus.Close:
                self._fill_pending_exit(candle, od)
            elif od.enter.status != OrderStatus.Close:
                self._fill_pending_enter(candle, od)

    def exit_open_orders(self, exit_tag: str, price: float, strategy: str = None,
                               pair: str = None) -> List[InOutOrder]:
        order_list = InOutOrder.open_orders(strategy, pair)
        result = []
        is_force = exit_tag == 'force_exit'
        for od in order_list:
            if not od.can_close():
                # 订单正在退出、或刚入场需等到下个bar退出
                if not is_force:
                    continue
                # 正在退出的exit_order不会处理，刚入场的交给exit_order退出
            ctx = get_context(f'{od.symbol}/{od.timeframe}')
            if self.exit_order(ctx, od, exit_tag, price):
                result.append(od)
        return result

    def exit_order(self, ctx: Context, od: InOutOrder, exit_tag: str, price: Union[float, Callable]
                   ) -> Optional[InOutOrder]:
        if od.exit_tag:
            return
        od.exit_tag = exit_tag
        od.exit_at = int(ctx[bar_arr][-1][0])
        _, base_s, quote_s, _ = get_cur_symbol(ctx)
        candle = self.data_mgr.get_latest_ohlcv(od.symbol)
        hprice, lprice, cprice = candle[hcol: vcol]
        if not price:
            # 为提供价格时，以最低价卖出（即吃单方）
            price = lprice * 2 - hprice
        ava_amt, lock_amt = self.wallets.get(base_s, od.enter.create_at)
        exit_amount = od.enter.filled
        if 0 < ava_amt < exit_amount or abs(ava_amt - exit_amount) / exit_amount <= 0.03:
            exit_amount = ava_amt
        od.update_exit(price=price, amount=exit_amount)
        if btime.run_mode in TRADING_MODES:
            cost = cprice * exit_amount
            logger.info('exit order {0} {1} got ~: {2:.2f}', od.symbol, od.exit_tag, cost)
        self._put_order(od, False)
        return od

    def _finish_order(self, od: InOutOrder):
        od.enter.trades.clear()
        if od.exit:
            od.exit.trades.clear()
        fee_rate = od.enter.fee + od.exit.fee
        od.profit_rate = float(od.exit.price / od.enter.price) - 1 - fee_rate
        od.profit = float(od.profit_rate * od.enter.price * od.enter.amount)
        if self.pair_fee_limits and fee_rate and od.symbol not in self.forbid_pairs:
            limit_fee = self.pair_fee_limits.get(od.symbol)
            if limit_fee is not None and fee_rate > limit_fee * 2:
                self.forbid_pairs.add(od.symbol)
                logger.error('%s fee Over limit: %f', od.symbol, self.pair_fee_limits.get(od.symbol, 0))
        od.del_lock()
        if self.live_mode:
            db.session.commit()

    def update_by_bar(self, pair_arr: np.ndarray):
        op_orders = InOutOrder.open_orders()
        # 更新订单利润
        for od in op_orders:
            od.update_by_bar(pair_arr)
        # 更新价格
        pair, base_s, quote_s, timeframe = get_cur_symbol()
        if quote_s.find('USD') >= 0:
            self.prices[base_s] = float(pair_arr[-1, ccol])

    def calc_custom_exits(self, pair_arr: np.ndarray, strategy: BaseStrategy) -> Dict[int, str]:
        result = dict()
        op_orders = InOutOrder.open_orders()
        if not op_orders:
            return result
        pair, _, _, _ = get_cur_symbol()
        cur_strategy = strategy.name
        # 调用策略的自定义退出判断
        for od in op_orders:
            if od.strategy != cur_strategy or od.symbol != pair or not od.can_close():
                continue
            if ext_tag := strategy.custom_exit(pair_arr, od):
                result[od.id] = ext_tag
        return result

    def check_fatal_stop(self):
        for check_mins, bad_ratio in self.fatal_stop.items():
            fatal_loss = self.calc_fatal_loss(check_mins)
            if fatal_loss >= bad_ratio:
                logger.error('fatal loss {0:.2f}% in {1} mins, Disable!', fatal_loss * 100, check_mins)
                self.disabled = True
                break

    def calc_fatal_loss(self, back_mins: int) -> float:
        '''
        计算系统级别最近n分钟内，账户余额损失百分比
        :param back_mins:
        :return:
        '''
        fin_loss = 0
        min_timestamp = btime.to_utcstamp(btime.now() - btime.timedelta(minutes=back_mins))
        his_orders = InOutOrder.his_orders()
        for i in range(len(his_orders) - 1, -1, -1):
            od = his_orders[i]
            if od.enter.create_at < min_timestamp:
                break
            fin_loss += od.profit
        if fin_loss >= 0:
            return 0
        fin_loss = abs(fin_loss)
        return fin_loss / (fin_loss + self.get_legal_value())

    def _symbol_price(self, symbol: str):
        if symbol not in self.prices:
            raise RuntimeError(f'{symbol} price to USD unknown')
        return self.prices[symbol]

    def _get_legal_value(self, symbol: str):
        if symbol not in self.wallets.data:
            return 0
        amount = sum(self.wallets.data[symbol])
        if symbol.find('USD') >= 0:
            return amount
        elif not amount:
            return 0
        return amount * self._symbol_price(symbol)

    def get_legal_value(self, symbol: str = None):
        '''
        获取某个产品的法定价值。USDT直接返回。BTC等需要计算。
        :param symbol:
        :return:
        '''
        if symbol:
            return self._get_legal_value(symbol)
        else:
            result = 0
            for key in self.wallets.data:
                result += self._get_legal_value(key)
            return result


class LiveOrderManager(OrderManager):
    def __init__(self, config: dict, exchange: CryptoExchange, wallets: CryptoWallet, data_hd: LiveDataProvider,
                 callback: Callable):
        super(LiveOrderManager, self).__init__(config, exchange, wallets, data_hd, callback)
        self.exchange = exchange
        self.exg_orders: Dict[str, Order] = dict()
        self.unmatch_trades: Dict[str, dict] = dict()
        self.handled_trades: Dict[str, int] = OrderedDict()
        self.max_market_rate = config.get('max_market_rate', 0.0001)
        self.odbook_ttl: int = config.get('odbook_ttl', 500)
        self.odbooks: Dict[str, OrderBook] = dict()
        self.order_q = Queue(1000)  # 最多1000个待执行订单
        # 限价单的深度对应的秒级成交量
        self.limit_vol_secs = config.get('limit_vol_secs', 10)

    async def _get_odbook_price(self, pair: str, side: str, depth: float):
        '''
        获取订单簿指定深度价格。用于生成限价单价格
        :param pair:
        :param side:
        :param depth:
        :return:
        '''
        odbook = self.odbooks.get(pair)
        if not odbook or odbook.timestamp + self.odbook_ttl < time.monotonic() * 1000:
            od_res = await self.exchange.fetch_order_book(pair, 1000)
            self.odbooks[pair] = OrderBook(**od_res)
        return self.odbooks[pair].limit_price(side, depth)

    async def calc_price(self, pair: str, vol_secs=0):
        # 如果taker的费用为0，直接使用市价单，否则获取订单簿，使用限价单
        candle = self.data_mgr.get_latest_ohlcv(pair)
        high_price, low_price, close_price, vol_amount = candle[hcol: vcol + 1]
        od = Order(symbol=pair, order_type='limit', side='buy', amount=vol_amount, price=close_price)
        fees = self.exchange.calc_funding_fee(od)
        if fees['rate'] > self.max_market_rate and btime.run_mode in TRADING_MODES:
            # 手续费率超过指定市价单费率，使用限价单
            # 取过去300s数据计算；限价单深度=min(60*每秒平均成交量, 最后30s总成交量)
            his_ohlcvs = await self.exchange.fetch_ohlcv(pair, '1s', limit=300)
            vol_arr = np.array(his_ohlcvs)[:, vcol]
            if not vol_secs:
                vol_secs = self.limit_vol_secs
            depth = min(np.average(vol_arr) * vol_secs * 2, np.sum(vol_arr[-vol_secs:]))
            buy_price = await self._get_odbook_price(pair, 'buy', depth)
            sell_price = await self._get_odbook_price(pair, 'sell', depth)
        else:
            buy_price = high_price * 2 - low_price
            sell_price = low_price * 2 - high_price
        return buy_price, sell_price

    def _get_pair_prices(self, pair: str, vol_sec=0):
        _buy_price, _sell_price = None, None

        async def buy_price():
            nonlocal _buy_price, _sell_price
            if _buy_price is None:
                _buy_price, _sell_price = await self.calc_price(pair, vol_sec)
            return _buy_price

        async def sell_price():
            nonlocal _buy_price, _sell_price
            if _sell_price is None:
                _buy_price, _sell_price = await self.calc_price(pair, vol_sec)
            return _sell_price
        return buy_price, sell_price

    async def _consume_unmatchs(self, sub_od: Order):
        for trade in list(self.unmatch_trades.values()):
            if trade['symbol'] != sub_od.symbol or trade['order'] != sub_od.order_id:
                continue
            trade_key = f"{trade['symbol']}_{trade['id']}"
            del self.unmatch_trades[trade_key]
            if trade_key in self.handled_trades or sub_od.status == OrderStatus.Close:
                continue
            logger.info('exec unmatch trade: %s', trade)
            await self._update_order(sub_od, trade)

    def _check_new_trades(self, sub_od: Order, trades: List[dict]):
        if not trades:
            return 0, 0
        handled_cnt = 0
        for trade in trades:
            od_key = f"{trade['symbol']}_{trade['id']}"
            if od_key in self.handled_trades:
                handled_cnt += 1
                continue
            self.handled_trades[od_key] = 1
            sub_od.trades.append(trade)
        return len(trades) - handled_cnt, handled_cnt

    def _update_order_res(self, od: InOutOrder, is_enter: bool, data: dict):
        sub_od = od.enter if is_enter else od.exit
        cur_ts = data['timestamp']
        if cur_ts < sub_od.last_ts:
            return
        sub_od.last_ts = cur_ts
        order_status, fee, filled = data.get('status'), data.get('fee'), float(data.get('filled', 0))
        if filled > 0:
            filled_price = safe_value_fallback(data, 'average', 'price', sub_od.price)
            sub_od.update_props(average=filled_price, filled=filled, status=OrderStatus.PartOk)
            od.status = InOutStatus.PartEnter if is_enter else InOutStatus.PartExit
            if fee and fee.get('rate'):
                sub_od.fee = fee.get('rate')
                sub_od.fee_type = fee.get('currency')
            # 下单后立刻有成交的，认为是taker方（ccxt返回的信息中未明确）
            fee_key = f'{od.symbol}_taker'
            self.exchange.pair_fees[fee_key] = sub_od.fee_type, sub_od.fee
        if order_status in {'expired', 'rejected', 'closed', 'canceled'}:
            sub_od.status = OrderStatus.Close
            if sub_od.filled and sub_od.average:
                sub_od.price = sub_od.average
            if filled == 0:
                if is_enter:
                    # 入场订单，0成交，被关闭；整体状态为：完全退出
                    od.status = InOutStatus.FullExit
                else:
                    # 出场订单，0成交，被关闭，整体状态为：已入场
                    od.status = InOutStatus.FullEnter
                logger.warning('%s is %s by %s, no filled', od, order_status, self.name)
            else:
                od.status = InOutStatus.FullEnter if is_enter else InOutStatus.FullExit
        if od.status == InOutStatus.FullExit:
            self._finish_order(od)

    async def _update_subod_by_ccxtres(self, od: InOutOrder, is_enter: bool, order: dict):
        sub_od = od.enter if is_enter else od.exit
        async with sub_od.lock():
            if sub_od.order_id:
                # 如修改订单价格，order_id会变化
                del self.exg_orders[f'{od.symbol}_{sub_od.order_id}']
            sub_od.order_id = order["id"]
            exg_key = f'{od.symbol}_{sub_od.order_id}'
            self.exg_orders[exg_key] = sub_od
            logger.debug('create order: %s %s %s', od.symbol, sub_od.order_id, order)
            new_num, old_num = self._check_new_trades(sub_od, order['trades'])
            if new_num:
                self._update_order_res(od, is_enter, order)
        await self._consume_unmatchs(sub_od)

    def _finish_order(self, od: InOutOrder):
        super(LiveOrderManager, self)._finish_order(od)
        exg_inkey = f'{od.symbol}_{od.enter.order_id}'
        if exg_inkey in self.exg_orders:
            self.exg_orders.pop(exg_inkey)
        if od.exit:
            exg_outkey = f'{od.symbol}_{od.exit.order_id}'
            if exg_outkey in self.exg_orders:
                self.exg_orders.pop(exg_outkey)

    async def _create_exg_order(self, od: InOutOrder, is_enter: bool):
        sub_od = od.enter if is_enter else od.exit
        side, amount, price = sub_od.side, sub_od.amount, sub_od.price
        order = await self.exchange.create_limit_order(od.symbol, side, amount, price)
        self.last_ts = btime.time()
        # 创建订单返回的结果，可能早于listen_orders_forever，也可能晚于listen_orders_forever
        await self._update_subod_by_ccxtres(od, is_enter, order)
        self._fire(od, is_enter)

    def _put_order(self, od: InOutOrder, is_enter: bool):
        if btime.run_mode != btime.RunMode.LIVE:
            return
        self.order_q.put_nowait(OrderJob(od.id, is_enter))

    async def _update_bnb_order(self, od: Order, data: dict):
        info = data['info']
        state = info['X']
        if state == 'NEW':
            return
        cur_ts = info['E']
        if cur_ts < od.last_ts:
            # 收到的订单更新不一定按服务器端顺序。故早于已处理的时间戳的跳过
            return
        od.last_ts = cur_ts  # 记录上次更新的时间戳，避免旧数据覆盖新数据
        if state in {'CANCELED', 'REJECTED', 'EXPIRED', 'EXPIRED_IN_MATCH'}:
            od.update_props(status=OrderStatus.Close)
        sess = db.session
        inout_od: InOutOrder = sess.query(InOutOrder).get(od.inout_id)
        if state in {'FILLED', 'PARTIALLY_FILLED'}:
            od_status = OrderStatus.Close if state == 'FILLED' else OrderStatus.PartOk
            filled, total_cost = float(info['z']), float(info['Z'])
            fee_val, last_amt = float(info['n']), float(info['l'])
            kwargs = dict(status=od_status, order_type=info['o'], filled=filled, average=total_cost/filled)
            if od_status == OrderStatus.Close:
                kwargs['price'] = kwargs['average']
            if fee_val:
                kwargs['fee_type'] = info['N']
                kwargs['fee'] = fee_val / last_amt
            od.update_props(**kwargs)
            mtaker = 'maker' if info['m'] else 'taker'
            fee_key = f'{od.symbol}_{mtaker}'
            self.exchange.pair_fees[fee_key] = od.fee_type, od.fee
            if od_status == OrderStatus.Close:
                if od.enter:
                    inout_od.status = InOutStatus.FullEnter
                else:
                    inout_od.status = InOutStatus.FullExit
        else:
            logger.error('unknown bnb order status: %s, %s', state, data)
            return
        if inout_od.status == InOutStatus.FullExit:
            self._finish_order(inout_od)
        self._fire(inout_od, od.enter)

    async def _update_order(self, od: Order, data: dict):
        async with od.lock():
            od.trades.append(data)
            if self.name.find('binance') >= 0:
                await self._update_bnb_order(od, data)
            else:
                raise ValueError(f'unsupport exchange to update order: {self.name}')

    @loop_forever
    async def listen_orders_forever(self):
        trades = await self.exchange.watch_my_trades()
        logger.debug('get my trades: %s', trades)
        related_ods = set()
        for data in trades:
            trade_key = f"{data['symbol']}_{data['id']}"
            if trade_key in self.handled_trades:
                continue
            od_key = f"{data['symbol']}_{data['order']}"
            if od_key not in self.exg_orders:
                self.unmatch_trades[trade_key] = data
                continue
            sub_od = self.exg_orders[od_key]
            await self._update_order(sub_od, data)
            related_ods.add(sub_od)
        for sub_od in related_ods:
            await self._consume_unmatchs(sub_od)
        if len(self.handled_trades) > 500:
            cut_keys = list(self.handled_trades.keys())[-300:]
            self.handled_trades = OrderedDict.fromkeys(cut_keys, value=1)
        exp_unmatchs = []
        for trade_key, trade in list(self.unmatch_trades.items()):
            if btime.time() - trade['timestamp'] / 1000 >= 10:
                exp_unmatchs.append(trade)
                del self.unmatch_trades[trade_key]
        if exp_unmatchs:
            logger.warning('expired unmatch orders: %s', exp_unmatchs)

    async def _exec_order_enter(self, od: InOutOrder):
        if od.exit_tag:
            # 订单已被取消，不再提交到交易所
            return
        if isinstance(od.enter.price, Callable):
            od.enter.price = await od.enter.price()
        if not od.enter.amount:
            od.enter.amount = od.quote_cost / od.enter.price
        await self._create_exg_order(od, True)

    async def _exec_order_exit(self, od: InOutOrder):
        if (not od.enter.amount or od.enter.filled < od.enter.amount) and od.enter.status < OrderStatus.Close:
            # 可能也尚未入场。或者尚未完全入场
            if od.enter.order_id:
                try:
                    res = await self.exchange.cancel_order(od.enter.order_id, od.symbol)
                    await self._update_subod_by_ccxtres(od, True, res)
                except ccxt.OrderNotFound:
                    pass
            if not od.enter.filled:
                od.update_exit(price=od.enter.price)
                self._finish_order(od)
                # 这里未入场直接退出的，不应该fire
                return
            self._fire(od, True)
        if isinstance(od.exit.price, Callable):
            od.exit.price = await od.exit.price()
        # 检查入场订单是否已成交，如未成交则直接取消
        await self._create_exg_order(od, False)

    async def exec_order(self, job: OrderJob):
        od: InOutOrder = db.session.query(InOutOrder).get(job.od_id)
        if job.is_enter:
            await self._exec_order_enter(od)
        else:
            await self._exec_order_exit(od)

    async def consume_queue(self):
        while True:
            job = await self.order_q.get()
            try:
                await self.exec_order(job)
            except Exception:
                logger.exception('consume order exception: %s', job)
            self.order_q.task_done()

    @loop_forever
    async def trail_open_orders_forever(self):
        timeouts = self.config.get('limit_vol_secs', 5) * 2
        if btime.run_mode == RunMode.LIVE:
            await self._trail_open_orders(timeouts)
        await asyncio.sleep(timeouts)

    async def _trail_open_orders(self, timeouts: int):
        '''
        跟踪未关闭的订单，根据市场价格及时调整，避免长时间无法成交
        :return:
        '''
        op_orders = InOutOrder.open_orders()
        if not op_orders:
            return
        exp_orders = [od for od in op_orders if od.pending_type(timeouts)]
        if not exp_orders:
            return
        logger.info(f'pending open orders: {exp_orders}')
        from itertools import groupby
        for pair, od_list in groupby(exp_orders, lambda x: x.symbol):
            buy_price, sell_price = self._get_pair_prices(pair, round(self.limit_vol_secs * 0.5))
            buy_price = await buy_price()
            sell_price = await sell_price()
            for od in od_list:
                if od.exit and od.exit_tag:
                    if sell_price >= od.exit.price:
                        continue
                    od.exit.create_at = btime.time()
                    logger.info('change price exit %s price: %f -> %f', od.key, od.exit.price, sell_price)
                    od.exit.price = sell_price
                    if not od.exit.order_id:
                        await self._exec_order_exit(od)
                    else:
                        left_amount = od.exit.amount - od.exit.filled
                        res = await self.exchange.edit_limit_order(od.exit.order_id, od.symbol, od.exit.side,
                                                                   left_amount, sell_price)
                        await self._update_subod_by_ccxtres(od, False, res)
                else:
                    if buy_price <= od.enter.price:
                        continue
                    od.enter.create_at = btime.time()
                    logger.info('change price enter %s price: %f -> %f', od.key, od.enter.price, buy_price)
                    od.enter.price = buy_price
                    if not od.enter.order_id:
                        await self._exec_order_enter(od)
                    else:
                        left_amount = od.enter.amount - od.enter.filled
                        res = await self.exchange.edit_limit_order(od.enter.order_id, od.symbol, od.enter.side,
                                                                   left_amount, buy_price)
                        await self._update_subod_by_ccxtres(od, True, res)
