#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : main.py
# Author: anyongjin
# Date  : 2023/3/30
import copy
from asyncio import Queue
from collections import OrderedDict

from banbot.data.provider import *
from banbot.main.wallets import CryptoWallet, WalletsLocal
from banbot.storage.orders import *
from banbot.strategy.base import BaseStrategy
from banbot.util.common import SingletonArg
from banbot.util.misc import *
from banbot.data.tools import auto_fetch_ohlcv
from banbot.util.num_utils import to_pytypes


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
    def __init__(self, config: dict, wallets: WalletsLocal, data_hd: DataProvider, callback: Callable):
        self.config = config
        self.market_type = config.get('market_type')
        self.leverage = config.get('leverage', 3)  # 杠杆倍数
        self.name = data_hd.exg_name
        self.wallets = wallets
        self.data_mgr = data_hd
        self.callback = callback
        self.fatal_stop = dict()
        self.last_ts = btime.time()  # 记录上次订单时间戳，方便对比钱包时间戳是否正确
        self._load_fatal_stop()
        self.disabled = False
        self.forbid_pairs = set()
        self.pair_fee_limits = AppConfig.obj.exchange_cfg.get('pair_fee_limits')

    def _load_fatal_stop(self):
        fatal_cfg = self.config.get('fatal_stop')
        if not fatal_cfg:
            return
        for k, v in fatal_cfg.items():
            self.fatal_stop[int(k)] = v

    def _fire(self, od: InOutOrder, enter: bool):
        pair_tf = f'{self.name}_{self.data_mgr.market}_{od.symbol}_{od.timeframe}'
        with TempContext(pair_tf):
            try:
                self.callback(od, enter)
            except Exception:
                logger.exception(f'fire od callback fail {od.id}, enter: {enter}')

    def get_context(self, od: InOutOrder):
        pair_tf = f'{self.name}_{self.market_type}_{od.symbol}_{od.timeframe}'
        return get_context(pair_tf)

    def allow_pair(self, pair: str) -> bool:
        if self.disabled:
            # 触发系统交易熔断时，禁止入场，允许出场
            logger.warning('order enter forbid, fatal stop, %s', pair)
            return False
        return pair not in self.forbid_pairs

    async def try_dump(self):
        pass

    def process_orders(self, pair_tf: str, enters: List[Tuple[str, dict]],
                       exits: List[Tuple[str, dict]], exit_keys: Dict[int, dict])\
            -> Tuple[List[InOutOrder], List[InOutOrder]]:
        '''
        批量创建指定交易对的订单
        :param pair_tf: 交易对
        :param enters: [(strategy, enter_tag, cost), ...]
        :param exits: [(strategy, exit_tag), ...]
        :param exit_keys: Dict(order_key, exit_tag)
        :return:
        '''
        if enters or exits or exit_keys:
            logger.debug('bar signals: %s %s %s', enters, exits, exit_keys)
        else:
            return [], []
        ctx = get_context(pair_tf)
        exs, _ = get_cur_symbol(ctx)
        enter_ods, exit_ods = [], []
        if enters:
            if btime.allow_order_enter(ctx) and self.allow_pair(exs.symbol):
                for stg_name, sigin in enters:
                    enter_ods.append(self.enter_order(ctx, stg_name, sigin, do_check=False))
                enter_ods = [od for od in enter_ods if od]
            else:
                logger.debug('pair %s enter not allow: %s', exs.symbol, enters)
        if exits:
            for stg_name, sigout in exits:
                exit_ods.extend(self.exit_open_orders(sigout, None, stg_name, exs.symbol))
        if exit_keys:
            for od_id, sigout in exit_keys.items():
                od = InOutOrder.get(od_id)
                exit_ods.append(self.exit_order(od, sigout))
        exit_ods = [od for od in exit_ods if od]
        return enter_ods, exit_ods

    def enter_order(self, ctx: Context, strategy: str, sigin: dict, price: Optional[float] = None,
                    do_check=True) -> Optional[InOutOrder]:
        '''
        策略产生入场信号，执行入场订单。（目前仅支持做多）
        :param ctx:
        :param strategy:
        :param sigin:
        :param price:
        :param do_check: 是否执行入场检查
        :return:
        '''
        if 'short' not in sigin:
            if self.market_type == 'spot':
                sigin['short'] = False
            else:
                raise ValueError(f'`short` is required from `on_entry` in market: {self.market_type}')
        elif self.market_type == 'spot' and sigin.get('short'):
            # 现货市场，忽略做空单
            return
        exs, timeframe = get_cur_symbol(ctx)
        if do_check and (not btime.allow_order_enter(ctx) or not self.allow_pair(exs.symbol)):
            logger.debug('pair %s enter not allowed', exs.symbol)
            return
        tag = sigin.pop('tag')
        if 'leverage' not in sigin and self.market_type == 'future':
            sigin['leverage'] = self.leverage
        ent_side = 'sell' if sigin.get('short') else 'buy'
        od_key = f'{exs.symbol}|{strategy}|{ent_side}|{tag}|{btime.time_ms()}'
        legal_cost = self.wallets.enter_od(exs, sigin, od_key, self.last_ts)
        if not legal_cost:
            # 余额不足
            return
        od = InOutOrder(
            **sigin,
            sid=exs.id,
            symbol=exs.symbol,
            timeframe=timeframe,
            enter_price=price,
            enter_tag=tag,
            enter_at=btime.time_ms(),
            init_price=to_pytypes(ctx[bar_arr][-1][ccol]),
            strategy=strategy
        )
        if btime.run_mode in LIVE_MODES:
            logger.info('enter order {0} {1} cost: {2:.2f}', od.symbol, od.enter_tag, legal_cost)
        self._put_order(od, True)
        return od

    def _put_order(self, od: InOutOrder, is_enter: bool):
        pass

    def exit_open_orders(self, sigout: dict, price: Optional[float] = None, strategy: str = None,
                         pairs: Union[str, List[str]] = None, is_force=False, od_dir: str = None) -> List[InOutOrder]:
        order_list = InOutOrder.open_orders(strategy, pairs)
        result = []
        if od_dir == 'both':
            pass
        elif od_dir == 'long' or sigout.get('short') == False:
            order_list = [od for od in order_list if not od.short]
        elif od_dir == 'short' or sigout.get('short'):
            order_list = [od for od in order_list if od.short]
        elif self.market_type != 'spot':
            raise ValueError(f'`od_dir` is required in market: {self.market_type}')
        for od in order_list:
            if not od.can_close():
                # 订单正在退出、或刚入场需等到下个bar退出
                if not is_force:
                    continue
                # 正在退出的exit_order不会处理，刚入场的交给exit_order退出
            if self.exit_order(od, copy.copy(sigout), price):
                result.append(od)
        return result

    def exit_order(self, od: InOutOrder, sigout: dict, price: Optional[float] = None) -> Optional[InOutOrder]:
        if od.exit_tag:
            return
        od.exit_tag = sigout.pop('tag')
        od.exit_at = btime.time_ms()
        get_amount = od.enter.filled * (1 - od.enter.fee)  # 扣除手续费后才是实际得到的
        self.wallets.exit_od(od, get_amount, self.last_ts)
        od.update_exit(**sigout, price=price, amount=get_amount)
        od.save()
        if btime.run_mode in LIVE_MODES:
            logger.info('exit order {0} {1}', od, od.exit_tag)
        self._put_order(od, False)
        return od

    def _finish_order(self, od: InOutOrder):
        fee_rate = od.enter.fee + od.exit.fee
        if od.exit.price and od.enter.price:
            od.update_by_price(od.exit.price)
        if self.pair_fee_limits and fee_rate and od.symbol not in self.forbid_pairs:
            limit_fee = self.pair_fee_limits.get(od.symbol)
            if limit_fee is not None and fee_rate > limit_fee * 2:
                self.forbid_pairs.add(od.symbol)
                logger.error('%s fee Over limit: %f', od.symbol, self.pair_fee_limits.get(od.symbol, 0))
        od.save()

    def update_by_bar(self, row):
        if btime.run_mode not in LIVE_MODES:
            self.wallets.update_at = btime.time()
        op_orders = InOutOrder.open_orders()
        # 更新订单利润
        close_price = float(row[ccol])
        for od in op_orders:
            od.update_by_price(close_price)
        if self.market_type == 'future' and not btime.prod_mode():
            # 期货合约需要计算更新保证金
            bomb_ods = self.wallets.update_ods(op_orders)
            for od in bomb_ods:
                self.exit_order(od, dict(tag='bomb'), price=close_price)

    def calc_custom_exits(self, pair_arr: np.ndarray, strategy: BaseStrategy) -> Dict[int, dict]:
        result = dict()
        exs, _ = get_cur_symbol()
        op_orders = InOutOrder.open_orders(strategy.name, exs.symbol)
        if not op_orders:
            return result
        # 调用策略的自定义退出判断
        for od in op_orders:
            if not od.can_close():
                continue
            sigout = strategy.custom_exit(pair_arr, od)
            if sigout:
                result[od.id] = sigout
        return result

    def check_fatal_stop(self):
        with db():
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

    def _get_legal_value(self, symbol: str):
        if symbol not in self.wallets.data:
            return 0
        amount = self.wallets.data[symbol].total
        if symbol.find('USD') >= 0:
            return amount
        elif not amount:
            return 0
        return amount * MarketPrice.get(symbol)

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

    def cleanup(self):
        pass


class LocalOrderManager(OrderManager):
    obj: Optional['LocalOrderManager'] = None

    def __init__(self, config: dict, exchange: CryptoExchange, wallets: WalletsLocal, data_hd: DataProvider,
                 callback: Callable):
        super(LocalOrderManager, self).__init__(config, wallets, data_hd, callback)
        LocalOrderManager.obj = self
        self.exchange = exchange
        self.network_cost = 3.  # 模拟网络延迟

    def update_by_bar(self, row):
        super(LocalOrderManager, self).update_by_bar(row)
        if not btime.prod_mode():
            exs, timeframe = get_cur_symbol()
            affect_num = self.fill_pending_orders(exs.symbol, timeframe, row)
            # if affect_num:
            #     logger.info(f"wallets: {self.wallets}")

    def force_exit(self, od: InOutOrder, tag: Optional[str] = None, price: float = None):
        if not tag:
            tag = 'force_exit'
        self.exit_order(od, dict(tag=tag), price)
        if not price:
            candle = self.data_mgr.get_latest_ohlcv(od.symbol)
            price = self._sim_market_price(od.symbol, od.timeframe, candle)
        self._fill_pending_exit(od, price)

    def _fill_pending_enter(self, od: InOutOrder, price: float):
        enter_price = self.exchange.pres_price(od.symbol, price)
        sub_od = od.enter
        if not sub_od.amount:
            if od.short and od.leverage == 1:
                # 现货空单，必须给定数量
                raise ValueError('`enter_amount` is require for short order')
            sub_od.amount = self.exchange.pres_amount(od.symbol, od.quote_cost / enter_price)
        ctx = self.get_context(od)
        fees = self.exchange.calc_fee(sub_od.symbol, sub_od.order_type, sub_od.side, sub_od.amount, sub_od.price)
        if fees['rate']:
            sub_od.fee = fees['rate']
            sub_od.fee_type = fees['currency']
        self.wallets.confirm_od_enter(od, enter_price)
        update_time = ctx[bar_arr][-1][0] + self.network_cost * 1000
        self.last_ts = update_time / 1000
        od.status = InOutStatus.FullEnter
        sub_od.update_at = update_time
        sub_od.filled = sub_od.amount
        sub_od.average = enter_price
        sub_od.status = OrderStatus.Close
        if not sub_od.price:
            sub_od.price = enter_price
        self._fire(od, True)

    def _fill_pending_exit(self, od: InOutOrder, exit_price: float):
        sub_od = od.exit
        fees = self.exchange.calc_fee(sub_od.symbol, sub_od.order_type, sub_od.side, sub_od.amount, sub_od.price)
        if fees['rate']:
            sub_od.fee = fees['rate']
            sub_od.fee_type = fees['currency']
        ctx = self.get_context(od)
        update_time = ctx[bar_arr][-1][0] + self.network_cost * 1000
        self.last_ts = update_time / 1000
        od.status = InOutStatus.FullExit
        sub_od.update_at = update_time
        od.update_exit(
            status=OrderStatus.Close,
            price=exit_price,
            filled=sub_od.amount,
            average=exit_price,
        )
        self._finish_order(od)
        # 用计算的利润更新钱包
        self.wallets.confirm_od_exit(od, exit_price)
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
        if btime.prod_mode():
            raise RuntimeError('fill_pending_orders unavaiable in PROD mode')
        op_orders = InOutOrder.open_orders()
        affect_num = 0
        for od in op_orders:
            if symbol and od.symbol != symbol or timeframe and od.timeframe != timeframe:
                continue
            price = self._sim_market_price(od.symbol, od.timeframe, candle)
            if od.exit_tag and od.exit and od.exit.status != OrderStatus.Close:
                self._fill_pending_exit(od, price)
                affect_num += 1
            elif od.enter.status != OrderStatus.Close:
                self._fill_pending_enter(od, price)
                affect_num += 1
        return affect_num

    def cleanup(self):
        self.exit_open_orders(dict(tag='bot_stop'), 0, od_dir='both')
        self.fill_pending_orders()
        if not self.config.get('no_db'):
            InOutOrder.dump_to_db()


class LiveOrderManager(OrderManager):
    obj: Optional['LiveOrderManager'] = None

    def __init__(self, config: dict, exchange: CryptoExchange, wallets: CryptoWallet, data_hd: LiveDataProvider,
                 callback: Callable):
        super(LiveOrderManager, self).__init__(config, wallets, data_hd, callback)
        LiveOrderManager.obj = self
        self.exchange = exchange
        self.wallets: CryptoWallet = self.wallets
        self.exg_orders: Dict[Tuple[str, str], int] = dict()
        self.unmatch_trades: Dict[str, dict] = dict()
        self.handled_trades: Dict[str, int] = OrderedDict()
        self.od_type = config.get('order_type', 'limit')
        self.max_market_rate = config.get('max_market_rate', 0.0001)
        self.odbook_ttl: int = config.get('odbook_ttl', 500)
        self.odbooks: Dict[str, OrderBook] = dict()
        self.order_q = Queue(1000)  # 最多1000个待执行订单
        # 限价单的深度对应的秒级成交量
        self.limit_vol_secs = config.get('limit_vol_secs', 10)
        # 交易对的价格缓存，键：pair+vol，值：买入价格，卖出价格，过期时间s
        self._pair_prices: Dict[str, Tuple[float, float, float]] = dict()
        if self.od_type not in {'limit', 'market'}:
            raise ValueError(f'invalid order type: {self.od_type}, `limit` or `market` is accepted')
        if self.market_type == 'future' and self.od_type == 'limit':
            raise ValueError('only market order type is supported for future (as watch trades is not avaiable on bnb)')

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
        if not candle:
            # 机器人刚启动，没有最新bar时，如果有之前的未完成订单，这里需要给默认值
            high_price, low_price, close_price, vol_amount = 99999, 0.0000001, 1, 10
        else:
            high_price, low_price, close_price, vol_amount = candle[hcol: vcol + 1]
        od = Order(symbol=pair, order_type='limit', side='buy', amount=vol_amount, price=close_price)
        fees = self.exchange.calc_fee(od.symbol, od.order_type, od.side, od.amount, od.price)
        if fees['rate'] > self.max_market_rate and btime.run_mode in LIVE_MODES:
            # 手续费率超过指定市价单费率，使用限价单
            # 取过去5m数据计算；限价单深度=min(60*每秒平均成交量, 最后30s总成交量)
            exs = ExSymbol.get(self.exchange.name, self.exchange.market_type, pair)
            his_ohlcvs = await auto_fetch_ohlcv(self.exchange, exs, '1m', limit=5)
            vol_arr = np.array(his_ohlcvs)[:, vcol]
            if not vol_secs:
                vol_secs = self.limit_vol_secs
            avg_vol_sec = np.sum(vol_arr) / 5 / 60
            last_vol = vol_arr[-1]
            depth = min(avg_vol_sec * vol_secs * 2, last_vol * vol_secs / 60)
            buy_price = await self._get_odbook_price(pair, 'buy', depth)
            sell_price = await self._get_odbook_price(pair, 'sell', depth)
        else:
            buy_price = high_price * 2
            sell_price = low_price / 2
        return buy_price, sell_price

    async def _get_pair_prices(self, pair: str, vol_sec=0):
        key = f'{pair}_{round(vol_sec * 1000)}'
        cache_val = self._pair_prices.get(key)
        if cache_val and cache_val[-1] > btime.utctime():
            return cache_val[:2]

        # 计算后缓存3s有效
        buy_price, sell_price = await self.calc_price(pair, vol_sec)
        self._pair_prices[key] = (buy_price, sell_price, btime.utctime() + 3)

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

    def _check_new_trades(self, trades: List[dict]):
        if not trades:
            return 0, 0
        handled_cnt = 0
        for trade in trades:
            od_key = f"{trade['symbol']}_{trade['id']}"
            if od_key in self.handled_trades:
                handled_cnt += 1
                continue
            self.handled_trades[od_key] = 1
        return len(trades) - handled_cnt, handled_cnt

    def _update_order_res(self, od: InOutOrder, is_enter: bool, data: dict):
        sub_od = od.enter if is_enter else od.exit
        data_info = data.get('info') or dict()
        cur_ts = data['timestamp']
        if not cur_ts and self.name == 'binance':
            # 币安期货返回时间戳需要从info.updateTime取
            cur_ts = int(data_info.get('updateTime', '0'))
        if cur_ts < sub_od.update_at:
            logger.info(f'trade is out of date, skip: {data} {od} {is_enter}')
            return False
        sub_od.update_at = cur_ts
        sub_od.order_id = data['id']
        sub_od.amount = data.get('amount')
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
                logger.warning('%s[%s] is %s by %s, no filled', od, is_enter, order_status, self.name)
            else:
                od.status = InOutStatus.FullEnter if is_enter else InOutStatus.FullExit
        if od.status == InOutStatus.FullExit:
            self._finish_order(od)
        return True

    async def _update_subod_by_ccxtres(self, od: InOutOrder, is_enter: bool, order: dict):
        sub_od = od.enter if is_enter else od.exit
        async with sub_od.lock():
            if sub_od.order_id:
                # 如修改订单价格，order_id会变化
                del self.exg_orders[(od.symbol, sub_od.order_id)]
            sub_od.order_id = order["id"]
            exg_key = od.symbol, sub_od.order_id
            self.exg_orders[exg_key] = sub_od.id
            logger.debug('create order: %s %s %s', od.symbol, sub_od.order_id, order)
            new_num, old_num = self._check_new_trades(order['trades'])
            if new_num or self.market_type != 'spot':
                # 期货市场未返回trades
                new_change = self._update_order_res(od, is_enter, order)
                if new_change and self.market_type != 'spot':
                    await self.wallets.update_balance()
        await self._consume_unmatchs(sub_od)
        db.session.commit()

    def _finish_order(self, od: InOutOrder):
        super(LiveOrderManager, self)._finish_order(od)
        exg_inkey = od.symbol, od.enter.order_id
        if exg_inkey in self.exg_orders:
            self.exg_orders.pop(exg_inkey)
        if od.exit:
            exg_outkey = od.symbol, od.exit.order_id
            if exg_outkey in self.exg_orders:
                self.exg_orders.pop(exg_outkey)

    async def _create_exg_order(self, od: InOutOrder, is_enter: bool):
        sub_od = od.enter if is_enter else od.exit
        side, amount, price = sub_od.side, sub_od.amount, sub_od.price
        if od.leverage and self.exchange.leverages.get(od.symbol) != od.leverage:
            await self.exchange.set_leverage(od.leverage, od.symbol)
        params = dict()
        if self.market_type == 'future':
            params['positionSide'] = 'LONG' if od.enter.side == 'buy' else 'SHORT'
            # params.update(closePosition=True, triggerPrice=price)  # 止损单
            # params.update(closePosition=True, takeProfitPrice=price)  # 止盈单
        order = await self.exchange.create_order(od.symbol, self.od_type, side, amount, price, params)
        # 创建订单返回的结果，可能早于listen_orders_forever，也可能晚于listen_orders_forever
        try:
            await self._update_subod_by_ccxtres(od, is_enter, order)
            self._fire(od, is_enter)
        except Exception:
            logger.exception(f'error after put exchange order: {od}')

    def _put_order(self, od: InOutOrder, is_enter: bool):
        if not btime.prod_mode():
            return
        if is_enter:
            od.quote_cost = self.exchange.pres_cost(od.symbol, od.quote_cost)
            od.save()
        self.order_q.put_nowait(OrderJob(od.id, is_enter))

    async def _update_bnb_order(self, od: Order, data: dict):
        info = data['info']
        state = info['X']
        if state == 'NEW':
            return
        cur_ts = info['E']
        if cur_ts < od.update_at:
            # 收到的订单更新不一定按服务器端顺序。故早于已处理的时间戳的跳过
            return
        od.update_at = cur_ts  # 记录上次更新的时间戳，避免旧数据覆盖新数据
        od.amount = float(info['q'])
        if state in {'CANCELED', 'REJECTED', 'EXPIRED', 'EXPIRED_IN_MATCH'}:
            od.update_props(status=OrderStatus.Close)
        inout_status = None
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
                    inout_status = InOutStatus.FullEnter
                else:
                    inout_status = InOutStatus.FullExit
            self.last_ts = btime.time()
        else:
            logger.error('unknown bnb order status: %s, %s', state, data)
            return
        # 将sub_od的修改保存
        db.session.commit()
        inout_od: InOutOrder = InOutOrder.get(od.inout_id)
        if inout_status:
            inout_od.status = inout_status
        if inout_status == InOutStatus.FullExit:
            self._finish_order(inout_od)
        self._fire(inout_od, od.enter)

    async def _update_order(self, od: Order, data: dict):
        async with od.lock():
            if self.name.find('binance') >= 0:
                await self._update_bnb_order(od, data)
            else:
                raise ValueError(f'unsupport exchange to update order: {self.name}')

    @loop_forever
    async def listen_orders_forever(self):
        if self.market_type != 'spot':
            # 币安只有现货市场支持订单更新
            return 'exit'
        try:
            trades = await self.exchange.watch_my_trades()
        except ccxt.NetworkError as e:
            logger.error(f'watch_my_trades net error: {e}')
            return
        logger.debug('get my trades: %s', trades)
        with db():
            sess = db.session
            related_ods = set()
            for data in trades:
                trade_key = f"{data['symbol']}_{data['id']}"
                if trade_key in self.handled_trades:
                    continue
                od_key = data['symbol'], data['order']
                if od_key not in self.exg_orders:
                    self.unmatch_trades[trade_key] = data
                    continue
                sub_od = sess.query(Order).get(self.exg_orders[od_key])
                await self._update_order(sub_od, data)
                related_ods.add(sub_od)
                sess.commit()
            for sub_od in related_ods:
                await self._consume_unmatchs(sub_od)
            sess.commit()
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
        if not od.enter.price:
            enter_price = (await self._get_pair_prices(od.symbol, self.limit_vol_secs))[0]
            od.enter.price = self.exchange.pres_price(od.symbol, enter_price)
        if not od.enter.amount:
            if not od.quote_cost:
                raise ValueError(f'quote_cost is required to calc enter_amount')
            od.enter.amount = self.exchange.pres_amount(od.symbol, od.quote_cost / od.enter.price)
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
        if not od.exit.price:
            od.exit.price = (await self._get_pair_prices(od.symbol, self.limit_vol_secs))[1]
        # 检查入场订单是否已成交，如未成交则直接取消
        await self._create_exg_order(od, False)

    async def consume_queue(self):
        while True:
            job = await self.order_q.get()
            od: Optional[InOutOrder] = None
            try:
                with db():
                    od = InOutOrder.get(job.od_id)
                    if job.is_enter:
                        await self._exec_order_enter(od)
                    else:
                        await self._exec_order_exit(od)
                    db.session.commit()
            except Exception:
                if od:
                    await od.force_exit()
                    logger.exception('consume order exception: %s, force exit', job)
                else:
                    logger.exception('consume order exception: %s', job)
            self.order_q.task_done()

    async def edit_pending_order(self, od: InOutOrder, is_enter: bool, price: float):
        sub_od = od.enter if is_enter else od.exit
        sub_od.price = price
        if not sub_od.order_id:
            await self._exec_order_enter(od)
        else:
            left_amount = sub_od.amount - sub_od.filled
            try:
                if self.market_type == 'future':
                    await self.exchange.cancel_order(sub_od.order_id, od.symbol)
                    res = await self.exchange.create_order(od.symbol, self.od_type, sub_od.side, left_amount, price)
                else:
                    res = await self.exchange.edit_limit_order(sub_od.order_id, od.symbol, sub_od.side,
                                                               left_amount, price)
                await self._update_subod_by_ccxtres(od, is_enter, res)
            except ccxt.InvalidOrder as e:
                logger.error('edit invalid order: %s', e)

    @loop_forever
    async def trail_open_orders_forever(self):
        timeouts = self.config.get('limit_vol_secs', 5) * 2
        if btime.prod_mode():
            try:
                with db():
                    await self._trail_open_orders(timeouts)
            except Exception:
                logger.exception('_trail_open_orders error')
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
        # logger.info(f'pending open orders: {exp_orders}')
        sess = db.session
        from itertools import groupby
        exp_orders = sorted(exp_orders, key=lambda x: x.symbol)
        for pair, od_list in groupby(exp_orders, lambda x: x.symbol):
            buy_price, sell_price = await self._get_pair_prices(pair, round(self.limit_vol_secs * 0.5))
            od_list: List[InOutOrder] = list(od_list)
            for od in od_list:
                if od.exit and od.exit_tag:
                    sub_od, is_enter, new_price = od.exit, False, sell_price
                else:
                    sub_od, is_enter, new_price = od.enter, True, buy_price
                if not sub_od.price:
                    continue
                price_chg = new_price - sub_od.price
                price_chg = price_chg if is_enter else -price_chg
                if price_chg <= 0.:
                    # 新价格更不容易成交，跳过
                    continue
                sub_od.create_at = btime.time()
                logger.info('change %s price %s: %f -> %f', sub_od.side, od.key, sub_od.price, new_price)
                await self.edit_pending_order(od, is_enter, new_price)
                sess.commit()

    async def cleanup(self):
        with db():
            exit_ods = self.exit_open_orders(dict(tag='bot_stop'), 0, is_force=True, od_dir='both')
            if exit_ods:
                logger.info('exit %d open trades', len(exit_ods))
        await self.order_q.join()
