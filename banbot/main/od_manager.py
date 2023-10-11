#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : main.py
# Author: anyongjin
# Date  : 2023/3/30
import traceback
from asyncio import Queue
from collections import OrderedDict

from banbot.data.provider import *
from banbot.main.wallets import CryptoWallet, WalletsLocal
from banbot.storage.orders import *
from banbot.util.common import SingletonArg
from banbot.util.misc import *
from banbot.data.tools import auto_fetch_ohlcv
from banbot.util.num_utils import to_pytypes

min_dust = 0.00000001


class OrderBook:
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
    def __init__(self, config: dict, exchange: CryptoExchange, wallets: WalletsLocal,
                 data_hd: DataProvider, callback: Callable):
        self.config = config
        self.exchange = exchange
        self.market_type = config.get('market_type')
        self.leverage = config.get('leverage', 3)  # 杠杆倍数
        self.name = data_hd.exg_name
        self.wallets = wallets
        self.data_mgr = data_hd
        self.callback = callback
        self.max_open_orders = config.get('max_open_orders', 100)
        self.fatal_stop = dict()
        self.last_ts = 0  # 记录上次订单时间戳，方便对比钱包时间戳是否正确
        self._load_fatal_stop()
        self.disable_until = 0
        '禁用交易，到指定时间再允许，13位毫秒时间戳'
        self.fatal_stop_hours: float = config.get('fatal_stop_hours', 8)
        '全局止损时禁止时间，默认8小时'
        self.forbid_pairs = set()
        self.pair_fee_limits = AppConfig.obj.exchange_cfg.get('pair_fee_limits')
        self.unready_ids = set()  # 向order_q添加任务后，尚未commit保存，所以也暂存到un_ready_ids

    def _load_fatal_stop(self):
        fatal_cfg = self.config.get('fatal_stop')
        if not fatal_cfg:
            return
        for k, v in fatal_cfg.items():
            self.fatal_stop[int(k)] = v

    async def _fire(self, od: InOutOrder, enter: bool):
        from banbot.util.misc import run_async
        pair_tf = f'{self.name}_{self.data_mgr.market}_{od.symbol}_{od.timeframe}'
        with TempContext(pair_tf):
            try:
                await run_async(self.callback, od, enter)
            except Exception:
                logger.exception(f'fire od callback fail {od.id} {od}, enter: {enter} {traceback.format_stack()}')

    def get_context(self, od: InOutOrder):
        pair_tf = f'{self.name}_{self.market_type}_{od.symbol}_{od.timeframe}'
        return get_context(pair_tf)

    def allow_pair(self, pair: str) -> bool:
        if self.disable_until > btime.utcstamp():
            # 触发系统交易熔断时，禁止入场，允许出场
            logger.warning('order enter forbid, fatal stop, %s', pair)
            return False
        return pair not in self.forbid_pairs

    async def try_dump(self):
        pass

    async def process_orders(self, pair_tf: str, enters: List[Tuple[str, dict]],
                       exits: List[Tuple[str, dict]], edit_triggers: List[Tuple[InOutOrder, str]])\
            -> Tuple[List[InOutOrder], List[InOutOrder]]:
        '''
        批量创建指定交易对的订单
        :param pair_tf: 交易对
        :param enters: [(strategy, enter_tag, cost), ...]
        :param exits: [(strategy, exit_tag), ...]
        :param edit_triggers: 编辑止损止盈订单
        :return:
        '''
        sess = dba.session
        if enters or exits:
            logger.debug('bar signals: %s %s', enters, exits)
            ctx = get_context(pair_tf)
            exs, _ = get_cur_symbol(ctx)
            enter_ods, exit_ods = [], []
            if enters:
                if btime.allow_order_enter(ctx) and self.allow_pair(exs.symbol):
                    open_ods = await InOutOrder.open_orders()
                    if len(open_ods) < self.max_open_orders:
                        for stg_name, sigin in enters:
                            try:
                                ent_od = await self.enter_order(ctx, stg_name, sigin, do_check=False)
                                enter_ods.append(ent_od)
                            except LackOfCash as e:
                                logger.warning(f'enter fail, lack of cash: {e} {stg_name} {sigin}')
                                await self.on_lack_of_cash()
                            except Exception as e:
                                logger.warning(f'enter fail: {e} {stg_name} {sigin}')
                else:
                    logger.debug('pair %s enter not allow: %s', exs.symbol, enters)
            if exits:
                for stg_name, sigout in exits:
                    exit_ods.extend(await self.exit_open_orders(sigout, None, stg_name, exs.symbol))
        else:
            enter_ods, exit_ods = [], []
        if btime.run_mode in btime.LIVE_MODES:
            await sess.flush()
            enter_ods = [od.detach(sess) for od in enter_ods if od]
            exit_ods = [od.detach(sess) for od in exit_ods if od]
        for od, prefix in edit_triggers:
            if od in exit_ods or od.enter.status < OrderStatus.PartOk:
                # 订单未成交时，不应该挂止损
                continue
            self._put_order(od, OrderJob.ACT_EDITTG, prefix)
        return enter_ods, exit_ods

    async def enter_order(self, ctx: Context, strategy: str, sigin: dict, do_check=True) -> Optional[InOutOrder]:
        '''
        策略产生入场信号，执行入场订单。（目前仅支持做多）
        :param ctx:
        :param strategy:
        :param sigin: tag,short,legal_cost,cost_rate, stoploss_price, takeprofit_price
        :param do_check: 是否执行入场检查
        :return:
        '''
        if 'short' not in sigin:
            if self.market_type == 'spot':
                sigin['short'] = False
            else:
                raise ValueError(f'`short` is required in market: {self.market_type}')
        elif self.market_type == 'spot' and sigin.get('short'):
            # 现货市场，忽略做空单
            raise ValueError('short order unavaiable in spot market')
        exs, timeframe = get_cur_symbol(ctx)
        if do_check and (not btime.allow_order_enter(ctx) or not self.allow_pair(exs.symbol)):
            raise RuntimeError('pair %s enter not allowed' % exs.symbol)
        tag = sigin.pop('tag')
        if 'leverage' not in sigin and self.market_type == 'future':
            sigin['leverage'] = self.leverage
        ent_side = 'short' if sigin.get('short') else 'long'
        od_key = f'{exs.symbol}|{strategy}|{ent_side}|{tag}|{btime.time_ms()}'
        # 如果余额不足会发出异常
        legal_cost = self.wallets.enter_od(exs, sigin, od_key, self.last_ts)
        od = InOutOrder(
            **sigin,
            sid=exs.id,
            symbol=exs.symbol,
            timeframe=timeframe,
            enter_tag=tag,
            enter_at=btime.time_ms(),
            init_price=to_pytypes(ctx[bar_arr][-1][ccol]),
            strategy=strategy
        )
        od.quote_cost = self.exchange.pres_cost(od.symbol, od.quote_cost)
        await od.save()
        if btime.run_mode in LIVE_MODES:
            logger.info('enter order {0} {1} cost: {2:.2f}', od.symbol, od.enter_tag, legal_cost)
        self._put_order(od, OrderJob.ACT_ENTER)
        return od

    def _put_order(self, od: InOutOrder, action: str, data: str = None):
        pass

    async def exit_open_orders(self, sigout: dict, price: Optional[float] = None, strategy: str = None,
                         pairs: Union[str, List[str]] = None, is_force=False, od_dir: str = None) -> List[InOutOrder]:
        is_exact = False
        if 'order_id' in sigout:
            is_exact = True
            order_list = [await InOutOrder.get(sigout.pop('order_id'))]
        else:
            order_list = await InOutOrder.open_orders(strategy, pairs)
        result = []
        if not is_exact:
            # 精确指定退出订单ID时，忽略方向过滤
            if od_dir == 'both':
                pass
            elif od_dir == 'long' or sigout.get('short') is False:
                order_list = [od for od in order_list if not od.short]
            elif od_dir == 'short' or sigout.get('short'):
                order_list = [od for od in order_list if od.short]
            elif self.market_type != 'spot':
                raise ValueError(f'`od_dir` is required in market: {self.market_type}')
        if 'enter_tag' in sigout:
            enter_tag = sigout.pop('enter_tag')
            order_list = [od for od in order_list if od.enter_tag == enter_tag]
        if not is_force:
            # 非强制退出，筛选可退出订单
            order_list = [od for od in order_list if od.can_close()]
        if not order_list:
            return result
        # 计算要退出的数量
        all_amount = sum(od.enter_amount for od in order_list)
        exit_amount = all_amount
        if 'amount' in sigout:
            exit_amount = sigout.pop('amount')
        elif 'exit_rate' in sigout:
            exit_amount = all_amount * sigout.pop('exit_rate')
        for od in order_list:
            if not od.enter_amount:
                continue
            cur_ext_rate = exit_amount / od.enter_amount
            if cur_ext_rate < 0.01:
                break
            try:
                exit_amount -= od.enter_amount
                cur_out = copy.copy(sigout)
                if cur_ext_rate < 0.99:
                    cur_out['amount'] = od.enter_amount * cur_ext_rate
                res_od = await self.exit_order(od, cur_out, price)
                if res_od:
                    result.append(res_od)
            except Exception as e:
                logger.error(f'exit order fail: {e} {od}')
        return result

    async def exit_order(self, od: InOutOrder, sigout: dict, price: Optional[float] = None) -> Optional[InOutOrder]:
        if od.exit_tag:
            return
        exit_amt = sigout.get('amount')
        if exit_amt:
            exit_rate = exit_amt / (od.enter.amount * (1 - od.enter.fee))
            if exit_rate < 0.99:
                # 要退出的部分不足99%，分割出一个小订单，用于退出。
                part = od.cut_part(exit_amt)
                # 这里part的key和原始的一样，所以part作为src_key
                tgt_key, src_key = od.key, part.key
                exs = ExSymbol.get_by_id(od.sid)
                self.wallets.cut_part(src_key, tgt_key, exs.base_code, (1 - exit_rate))
                self.wallets.cut_part(src_key, tgt_key, exs.quote_code, (1 - exit_rate))
                return await self.exit_order(part, sigout, price)
        od.exit_tag = sigout.pop('tag')
        od.exit_at = btime.time_ms()
        if price:
            sigout['price'] = price
        od.update_exit(**sigout)
        self.wallets.exit_od(od, od.exit.amount, self.last_ts)
        await od.save()
        if btime.run_mode in LIVE_MODES:
            logger.info('exit order {0} {1}', od, od.exit_tag)
        self._put_order(od, OrderJob.ACT_EXIT)
        return od

    async def _finish_order(self, od: InOutOrder):
        fee_rate = od.enter.fee + od.exit.fee
        if od.exit.price and od.enter.price:
            od.update_by_price(od.exit.price)
        if self.pair_fee_limits and fee_rate and od.symbol not in self.forbid_pairs:
            limit_fee = self.pair_fee_limits.get(od.symbol)
            if limit_fee is not None and fee_rate > limit_fee * 2:
                self.forbid_pairs.add(od.symbol)
                logger.error('%s fee Over limit: %f', od.symbol, self.pair_fee_limits.get(od.symbol, 0))
        await od.save()

    async def update_by_bar(self, row):
        if btime.run_mode not in LIVE_MODES:
            self.wallets.update_at = btime.time()
        exs, _ = get_cur_symbol()
        op_orders = await InOutOrder.open_orders()
        cur_orders = [od for od in op_orders if od.symbol == exs.symbol]
        # 更新订单利润
        close_price = float(row[ccol])
        for od in cur_orders:
            od.update_by_price(close_price)
        if op_orders and self.market_type == 'future' and not btime.prod_mode():
            # 为合约更新此定价币的所有订单保证金和钱包情况
            quote_suffix = exs.quote_suffix()
            quote_orders = [od for od in op_orders if od.symbol.endswith(quote_suffix)]
            try:
                self.wallets.update_ods(quote_orders)
            except AccountBomb:
                # 保存订单状态
                for od in quote_orders:
                    await od.save()
                raise

    async def on_lack_of_cash(self):
        pass

    async def check_fatal_stop(self):
        if self.disable_until >= btime.utcstamp():
            return
        async with dba():
            for check_mins, bad_ratio in self.fatal_stop.items():
                fatal_loss = await self.calc_fatal_loss(check_mins)
                if fatal_loss >= bad_ratio:
                    logger.error(f'{check_mins}分钟内损失{(fatal_loss * 100):.2f}%，禁止下单{self.fatal_stop_hours}小时!')
                    self.disable_until = btime.utcstamp() + self.fatal_stop_hours * 60 * 60000
                    break

    async def calc_fatal_loss(self, back_mins: int) -> float:
        '''
        计算系统级别最近n分钟内，账户余额损失百分比
        :param back_mins:
        :return:
        '''
        fin_loss = 0
        min_time_ms = btime.to_utcstamp(btime.now() - btime.timedelta(minutes=back_mins), ms=True)
        min_time_ms = max(min_time_ms, BotGlobal.start_at)
        his_orders = await InOutOrder.his_orders()
        for i in range(len(his_orders) - 1, -1, -1):
            od = his_orders[i]
            if od.enter.create_at < min_time_ms:
                break
            fin_loss += od.profit
        if fin_loss >= 0:
            return 0
        fin_loss = abs(fin_loss)
        total_legal = self.wallets.total_legal()
        return fin_loss / (fin_loss + total_legal)

    def cleanup(self):
        pass


class LocalOrderManager(OrderManager):
    obj: Optional['LocalOrderManager'] = None

    def __init__(self, config: dict, exchange: CryptoExchange, wallets: WalletsLocal, data_hd: DataProvider,
                 callback: Callable):
        super(LocalOrderManager, self).__init__(config, exchange, wallets, data_hd, callback)
        LocalOrderManager.obj = self
        self.stake_currency = config.get('stake_currency') or []
        self.exchange = exchange
        self.network_cost = 3.  # 模拟网络延迟

    async def update_by_bar(self, row):
        await super(LocalOrderManager, self).update_by_bar(row)
        if not btime.prod_mode():
            exs, timeframe = get_cur_symbol()
            affect_num = await self.fill_pending_orders(exs.symbol, timeframe, row)
            if affect_num:
                logger.debug("wallets: %s", self.wallets)

    async def on_lack_of_cash(self):
        lack_num = 0
        for currency in self.stake_currency:
            fiat_val = self.wallets.fiat_value(currency)
            if fiat_val < MIN_STAKE_AMOUNT:
                lack_num += 1
        if lack_num < len(self.stake_currency):
            logger.debug('%s/%s sybol lack %s', lack_num, len(self.stake_currency), self.wallets.data)
            return
        open_num = len(await InOutOrder.open_orders())
        if open_num == 0:
            # 如果余额不足，且没有打开的订单，则终止回测
            BotGlobal.state = BotState.STOPPED
        else:
            logger.info('%s open orders , dont stop', open_num)

    def force_exit(self, od: InOutOrder, tag: Optional[str] = None, price: float = None):
        if not tag:
            tag = 'force_exit'
        self.exit_order(od, dict(tag=tag), price)
        if not price:
            candle = self.data_mgr.get_latest_ohlcv(od.symbol)
            price = self._sim_market_price(od.symbol, od.timeframe, candle)
        self._fill_pending_exit(od, price)

    async def _fill_pending_enter(self, od: InOutOrder, price: float):
        enter_price = self.exchange.pres_price(od.symbol, price)
        sub_od = od.enter
        if not sub_od.amount:
            if od.short and self.market_type == 'spot' and od.leverage == 1:
                # 现货空单，必须给定数量
                raise ValueError('`enter_amount` is require for short order')
            ent_amount = od.quote_cost / enter_price
            try:
                sub_od.amount = self.exchange.pres_amount(od.symbol, ent_amount)
            except ccxt.InvalidOrder as e:
                err_msg = f'{od} pres enter amount fail: {od.quote_cost} {ent_amount} : {e}'
                logger.warning(err_msg)
                od.local_exit(ExitTags.fatal_err, status_msg=err_msg)
                await od.save()
                return
        ctx = self.get_context(od)
        fees = self.exchange.calc_fee(sub_od.symbol, sub_od.order_type, sub_od.side, sub_od.amount, sub_od.price)
        if fees['rate']:
            sub_od.fee = fees['rate']
            sub_od.fee_type = fees['currency']
        self.wallets.confirm_od_enter(od, enter_price)
        update_time = ctx[bar_time][0] + self.network_cost * 1000
        self.last_ts = update_time / 1000
        od.status = InOutStatus.FullEnter
        sub_od.update_at = update_time
        sub_od.filled = sub_od.amount
        sub_od.average = enter_price
        sub_od.status = OrderStatus.Close
        if not sub_od.price:
            sub_od.price = enter_price
        await self._fire(od, True)

    async def _fill_pending_exit(self, od: InOutOrder, exit_price: float):
        sub_od = od.exit
        fees = self.exchange.calc_fee(sub_od.symbol, sub_od.order_type, sub_od.side, sub_od.amount, sub_od.price)
        if fees['rate']:
            sub_od.fee = fees['rate']
            sub_od.fee_type = fees['currency']
        ctx = self.get_context(od)
        update_time = ctx[bar_time][0] + self.network_cost * 1000
        self.last_ts = update_time / 1000
        od.status = InOutStatus.FullExit
        sub_od.update_at = update_time
        od.update_exit(
            status=OrderStatus.Close,
            price=exit_price,
            filled=sub_od.amount,
            average=exit_price,
        )
        await self._finish_order(od)
        # 用计算的利润更新钱包
        self.wallets.confirm_od_exit(od, exit_price)
        await self._fire(od, False)

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

    async def fill_pending_orders(self, symbol: str = None, timeframe: str = None, candle: Optional[np.ndarray] = None):
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
        op_orders = await InOutOrder.open_orders(pairs=symbol)
        affect_num = 0
        for od in op_orders:
            if timeframe and od.timeframe != timeframe:
                continue
            price = self._sim_market_price(od.symbol, od.timeframe, candle)
            if od.exit_tag and od.exit and od.exit.status != OrderStatus.Close:
                await self._fill_pending_exit(od, price)
                affect_num += 1
            elif od.enter.status != OrderStatus.Close:
                await self._fill_pending_enter(od, price)
                affect_num += 1
        return affect_num

    async def cleanup(self):
        await self.exit_open_orders(dict(tag='bot_stop'), 0, od_dir='both')
        await self.fill_pending_orders()
        if not self.config.get('no_db'):
            await InOutOrder.dump_to_db()


class LiveOrderManager(OrderManager):
    obj: Optional['LiveOrderManager'] = None

    def __init__(self, config: dict, exchange: CryptoExchange, wallets: CryptoWallet, data_hd: LiveDataProvider,
                 callback: Callable):
        super(LiveOrderManager, self).__init__(config, exchange, wallets, data_hd, callback)
        LiveOrderManager.obj = self
        self.exchange = exchange
        self.wallets: CryptoWallet = wallets
        self.exg_orders: Dict[Tuple[str, str], Tuple[int, int]] = dict()
        self.unmatch_trades: Dict[str, dict] = dict()
        '未匹配交易，key: symbol, order_id；每个订单只保留一个未匹配交易，也只应该有一个'
        self.handled_trades: Dict[str, int] = OrderedDict()  # 有序集合，使用OrderedDict实现
        self.od_type = config.get('order_type', 'market')
        self.max_market_rate = config.get('max_market_rate', 0.0001)
        self.odbook_ttl: int = config.get('odbook_ttl', 500)
        self.odbooks: Dict[str, OrderBook] = dict()
        self.order_q = Queue(1000)  # 最多1000个待执行订单
        self.take_over_stgy: str = config.get('take_over_stgy')
        self.old_unsubmits = set()
        self.allow_take_over = self.take_over_stgy and self.name.find('binance') >= 0 and self.market_type == 'future'
        '目前仅币安期货支持接管用户订单'
        self.limit_vol_secs = config.get('limit_vol_secs', 10)
        '限价单的深度对应的秒级成交量'
        self._pair_prices: Dict[str, Tuple[float, float, float]] = dict()
        '交易对的价格缓存，键：pair+vol，值：买入价格，卖出价格，过期时间s'
        if self.od_type not in {'limit', 'market'}:
            raise ValueError(f'invalid order type: {self.od_type}, `limit` or `market` is accepted')
        if self.market_type == 'future' and self.od_type == 'limit':
            raise ValueError('only market order type is supported for future (as watch trades is not avaiable on bnb)')
        self.last_lack_ts = 0
        '上次通知余额不足的时间戳，10位秒级'

    async def sync_orders_with_exg(self):
        '''
        将交易所最新状态本地订单和进行
        先通过fetch_account_positions抓取交易所所有币的仓位情况。
        如果本地没有未平仓订单：
            如果交易所没有持仓：忽略
            如果交易所有持仓：视为用户开的新订单，创建新订单跟踪
        如果本地有未平仓订单：
             获取本地订单的最后时间作为起始时间，通过fetch_orders接口查询此后所有订单。
             从交易所订单记录来确定未平仓订单的当前状态：已平仓、部分平仓、未平仓
             对于冗余的仓位，视为用户开的新订单，创建新订单跟踪。
        '''
        positions = await self.exchange.fetch_account_positions()
        positions = [p for p in positions if p['notional']]
        cur_symbols = {p['symbol'] for p in positions}
        op_ods = await InOutOrder.open_orders()
        if not op_ods:
            since_ms = 0
        else:
            since_ms = max([od.enter_at for od in op_ods])
            cur_symbols.update({od.symbol for od in op_ods})
        res_odlist = set()
        for pair in cur_symbols:
            cur_ods = [od for od in op_ods if od.symbol == pair]
            cur_pos = [p for p in positions if p['symbol'] == pair]
            if not cur_pos and not cur_ods:
                continue
            long_pos = next((p for p in cur_pos if p['side'] == 'long'), None)
            short_pos = next((p for p in cur_pos if p['side'] == 'short'), None)
            await self._sync_pair_orders(pair, long_pos, short_pos, cur_ods, since_ms)
            res_odlist.update(cur_ods)
        new_ods = res_odlist - set(op_ods)
        old_ods = res_odlist.intersection(op_ods)
        del_ods = set(op_ods) - res_odlist
        if old_ods:
            logger.info(f'恢复{len(old_ods)}个未平仓订单：{old_ods}')
        if new_ods:
            logger.info(f'开始跟踪{len(new_ods)}个用户下单：{new_ods}')
        return len(old_ods), len(new_ods), len(del_ods), list(res_odlist)

    async def _sync_pair_orders(self, pair: str, long_pos: dict, short_pos: dict,
                                op_ods: List[InOutOrder], since_ms: int):
        '''
        对指定币种，将交易所订单状态同步到本地。机器人刚启动时执行。
        '''
        if op_ods:
            # 本地有未平仓订单，从交易所获取订单记录，尝试恢复订单状态。
            sess = dba.session
            ex_orders = await self.exchange.fetch_orders(pair, since_ms)
            for exod in ex_orders:
                if exod['status'] != 'closed':
                    # 跳过未完成的订单
                    continue
                client_id: str = exod['clientOrderId']
                if client_id.startswith(BotGlobal.bot_name):
                    # 这里不应该有当前机器人的订单，除非是两个同名的机器人交易同一个账户
                    logger.error(f'unexpect order for bot: {BotGlobal.bot_name}: {exod}')
                await self._apply_history_order(op_ods, exod)
            long_pos_amt = long_pos['contracts'] if long_pos else 0.
            short_pos_amt = short_pos['contracts'] if short_pos else 0.
            # 检查剩余的打开订单是否和仓位匹配，如不匹配强制关闭对应的订单
            del_ods = []
            for iod in op_ods:
                od_amt = iod.enter.filled - (iod.exit.filled if iod.exit else 0)
                if od_amt * iod.init_price < 1:
                    # TODO: 这里计算的quote价值，后续需要改为法币价值
                    if iod.status < InOutStatus.FullExit:
                        msg = '订单没有入场仓位'
                        iod.local_exit(ExitTags.fatal_err, iod.init_price, status_msg=msg)
                    del_ods.append(iod)
                    continue
                pos_amt = short_pos_amt if iod.short else long_pos_amt
                pos_amt -= od_amt
                if iod.short:
                    short_pos_amt = pos_amt
                else:
                    long_pos_amt = pos_amt
                if pos_amt > od_amt * -0.01:
                    await iod.save()
                else:
                    msg = f'订单在交易所没有对应仓位，交易所：{(pos_amt + od_amt):.5f}'
                    iod.local_exit(ExitTags.fatal_err, iod.init_price, status_msg=msg)
                    del_ods.append(iod)
            [op_ods.remove(od) for od in del_ods]
            if long_pos:
                long_pos['contracts'] = long_pos_amt
            if short_pos:
                short_pos['contracts'] = short_pos_amt
            await sess.flush()
        if not self.take_over_stgy:
            return
        if long_pos and long_pos['contracts'] > min_dust:
            long_od = self._create_order_from_position(long_pos)
            if long_od:
                await long_od.save()
                op_ods.append(long_od)
        if short_pos and short_pos['contracts'] > min_dust:
            short_od = self._create_order_from_position(short_pos)
            if short_od:
                await short_od.save()
                op_ods.append(short_od)

    def _create_order_from_position(self, pos: dict):
        '''
        从ccxt返回的持仓情况创建跟踪订单。
        '''
        exs = ExSymbol.get(self.name, self.market_type, pos['symbol'])
        average = pos['entryPrice']
        filled = pos['contracts']
        ent_od_type = self.od_type
        market = self.exchange.markets[exs.symbol]
        is_short = pos['side'] == 'short'
        # 持仓信息没有手续费，直接从当前机器人订单类型推断手续费，可能和实际的手续费不同
        fee_rate = market['taker'] if ent_od_type == 'market' else market['maker']
        fee_name = exs.quote_code if self.market_type == 'future' else exs.base_code
        tag_ = '开空' if is_short else '开多'
        logger.info(f'[仓]{tag_}：price:{average}, amount: {filled}, fee: {fee_rate}')
        return self._create_inout_od(exs, is_short, average, filled, ent_od_type, fee_rate,
                                     fee_name, btime.time_ms(), OrderStatus.Close)

    def _create_inout_od(self, exs: ExSymbol, short: bool, average: float, filled: float,
                         ent_od_type: str, fee_rate: float, fee_name: str, enter_at: int,
                         ent_status: int, ent_odid: str = None):
        job = next((p for p in BotGlobal.stg_symbol_tfs if p[0] == self.take_over_stgy and p[1] == exs.symbol), None)
        if not job:
            logger.warning(f'take over job not found: {exs.symbol} {self.take_over_stgy}')
            return
        leverage = self.exchange.get_leverage(exs.symbol)
        quote_cost = filled * average / leverage
        io_status = InOutStatus.FullEnter if ent_status == OrderStatus.Close else InOutStatus.PartEnter
        return InOutOrder(
            sid=exs.id,
            symbol=exs.symbol,
            timeframe=job[2],
            short=short,
            status=io_status,
            enter_price=average,
            enter_average=average,
            enter_amount=filled,
            enter_filled=filled,
            enter_status=ent_status,
            enter_order_type=ent_od_type,
            enter_tag=EnterTags.third,
            enter_at=enter_at,
            enter_update_at=enter_at,
            enter_fee=fee_rate,
            enter_fee_type=fee_name,
            init_price=average,
            strategy=self.take_over_stgy,
            enter_order_id=ent_odid,
            leverage=leverage,
            quote_cost=quote_cost,
        )

    async def _apply_history_order(self, od_list: List[InOutOrder], od: dict):
        info: dict = od['info']
        is_short = info['positionSide'] == 'SHORT'
        is_sell = od['side'] == 'sell'
        is_reduce_only = od['reduceOnly']
        exs = ExSymbol.get(self.name, self.market_type, od['symbol'])
        market = self.exchange.markets[exs.symbol]
        # 订单信息没有手续费，直接从当前机器人订单类型推断手续费，可能和实际的手续费不同
        fee_rate = market['taker'] if od['type'] == 'market' else market['maker']
        fee_name = exs.quote_code if self.market_type == 'future' else exs.base_code
        od_price, od_amount, od_time = od['average'], od['filled'], od['timestamp']

        async def _apply_close_od():
            nonlocal od_amount
            for iod in list(od_list):
                if iod.short != is_short:
                    continue
                before_amt = od_amount
                od_amount = (await self._try_fill_exit(iod, od_amount, od_price, od_time, od['id'], od['type'],
                                                       fee_name, fee_rate))[0]
                fill_amt = before_amt - od_amount
                tag_ = '平空' if is_short else '平多'
                logger.info(
                    f'{tag_}：price:{od_price}, amount: {fill_amt}, {od["type"]}, fee: {fee_rate} {od_time} id: {od["id"]}')
                if iod.status == InOutStatus.FullExit:
                    od_list.remove(iod)
                if od_amount <= min_dust:
                    break
            if not is_reduce_only and od_amount > min_dust:
                # 剩余数量，创建相反订单
                tag_ = '开空' if is_short else '开多'
                logger.info(f'{tag_}：price:{od_price}, amount: {od_amount}, {od["type"]}, fee: {fee_rate} {od_time} id: {od["id"]}')
                iod = self._create_inout_od(exs, is_short, od_price, od_amount, od['type'], fee_rate, fee_name,
                                            od_time, OrderStatus.Close, od['id'])
                if iod:
                    od_list.append(iod)

        if is_short == is_sell:
            # 开多，或开空
            tag = '开空' if is_short else '开多'
            logger.info(f'{tag}：price:{od_price}, amount: {od_amount}, {od["type"]}, fee: {fee_rate} {od_time} id: {od["id"]}')
            od = self._create_inout_od(exs, is_short, od_price, od_amount, od['type'], fee_rate, fee_name,
                                       od_time, OrderStatus.Close, od['id'])
            if od:
                od_list.append(od)
        else:
            # 平多，或平空
            await _apply_close_od()

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
        od = Order(symbol=pair, order_type=self.od_type, side='buy', amount=vol_amount, price=close_price)
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

    def _consume_unmatchs(self, sub_od: Order):
        for trade in list(self.unmatch_trades.values()):
            if trade['symbol'] != sub_od.symbol or trade['order'] != sub_od.order_id:
                continue
            trade_key = f"{trade['symbol']}_{trade['id']}"
            del self.unmatch_trades[trade_key]
            if trade_key in self.handled_trades or sub_od.status == OrderStatus.Close:
                continue
            logger.info('exec unmatch trade: %s', trade)
            self._update_order(sub_od, trade)

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

    async def _update_order_res(self, od: InOutOrder, is_enter: bool, data: dict):
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
            await self._finish_order(od)
        return True

    async def _update_subod_by_ccxtres(self, od: InOutOrder, is_enter: bool, order: dict):
        sub_od = od.enter if is_enter else od.exit
        if sub_od.order_id and (od.symbol, sub_od.order_id) in self.exg_orders:
            # 如修改订单价格，order_id会变化
            del self.exg_orders[(od.symbol, sub_od.order_id)]
        sub_od.order_id = order["id"]
        exg_key = od.symbol, sub_od.order_id
        self.exg_orders[exg_key] = od.id, sub_od.id
        logger.debug('create order: %s %s %s', od.symbol, sub_od.order_id, order)
        new_num, old_num = self._check_new_trades(order['trades'])
        if new_num or self.market_type != 'spot':
            # 期货市场未返回trades
            await self._update_order_res(od, is_enter, order)
        self._consume_unmatchs(sub_od)

    async def _finish_order(self, od: InOutOrder):
        await super(LiveOrderManager, self)._finish_order(od)
        exg_inkey = od.symbol, od.enter.order_id
        if exg_inkey in self.exg_orders:
            self.exg_orders.pop(exg_inkey)
        if od.exit:
            exg_outkey = od.symbol, od.exit.order_id
            if exg_outkey in self.exg_orders:
                self.exg_orders.pop(exg_outkey)

    async def _edit_trigger_od(self, od: InOutOrder, prefix: str, try_kill=True):
        trigger_oid = od.get_info(f'{prefix}oid')
        params = dict()
        params['positionSide'] = 'SHORT' if od.short else 'LONG'
        trig_price = od.get_info(f'{prefix}price')
        if trig_price:
            params.update(closePosition=True, triggerPrice=trig_price)  # 止损单
        side = 'buy' if od.short else 'sell'
        amount = od.enter_amount
        try:
            order = await self.exchange.create_order(od.symbol, self.od_type, side, amount, trig_price, params)
        except ccxt.OrderImmediatelyFillable:
            logger.error(f'[{od.id}] stop order, {side} {od.symbol}, price: {trig_price:.4f} invalid, skip')
            return
        except ccxt.ExchangeError as e:
            err_msg = str(e)
            if err_msg.find('-4045,') > 0:
                if try_kill:
                    # binanceusdm {"code":-4045,"msg":"Reach max stop order limit."}
                    logger.error('max stop orders reach, try cancel invalid orders...')
                    await self.exchange.cancel_invalid_orders()
                    await self._edit_trigger_od(od, prefix, False)
                    return
                else:
                    logger.warning('max stop orders reach, put trigger order skip...')
                    # 这里不返回，取消已有的触发订单
            else:
                raise e
        od.set_info(**{f'{prefix}oid': order['id']})
        if trigger_oid and order['status'] == 'open':
            try:
                await self.exchange.cancel_order(trigger_oid, od.symbol)
            except ccxt.OrderNotFound:
                logger.error(f'[{od.id}] cancel old stop order fail, not found: {od.symbol}, {trigger_oid}')

    async def _cancel_trigger_ods(self, od: InOutOrder):
        '''
        取消订单的关联订单。订单在平仓时，关联的止损单止盈单不会自动退出，需要调用此方法退出
        '''
        prefixs = ['stoploss_', 'takeprofit_']
        cancel_res = []
        for prefix in prefixs:
            trigger_oid = od.get_info(f'{prefix}oid')
            if not trigger_oid:
                continue
            try:
                res = await self.exchange.cancel_order(trigger_oid, od.symbol)
                cancel_res.append(res)
            except ccxt.OrderNotFound:
                logger.error(f'cancel stop order fail, not found: {od.symbol}, {trigger_oid}')

    async def _create_exg_order(self, od: InOutOrder, is_enter: bool):
        sub_od = od.enter if is_enter else od.exit
        old_lev = self.exchange.get_leverage(od.symbol, False)
        if is_enter and od.leverage and old_lev != od.leverage:
            item = await self.exchange.set_leverage(od.leverage, od.symbol)
            if item.leverage < od.leverage:
                # 此币种杠杆比较小，对应缩小金额
                rate = item.leverage / od.leverage
                od.leverage = item.leverage
                sub_od.amount *= rate
                od.quote_cost *= rate
        od_type = sub_od.order_type or self.od_type
        if not sub_od.price and od_type.find('market') < 0:
            # 非市价单时，计算价格
            buy_price, sell_price = (await self._get_pair_prices(od.symbol, self.limit_vol_secs))
            cur_price = buy_price if sub_od.side == 'buy' else sell_price
            sub_od.price = self.exchange.pres_price(od.symbol, cur_price)
        side, amount, price = sub_od.side, sub_od.amount, sub_od.price
        params = dict()
        if self.market_type == 'future':
            params['positionSide'] = 'SHORT' if od.short else 'LONG'
        order = await self.exchange.create_order(od.symbol, od_type, side, amount, price, params)
        print_args = [is_enter, od.symbol, od_type, side, amount, price, params, order]
        logger.debug('create exg order res: %s, %s, %s, %s, %s, %s, %s, %s', *print_args)
        # 创建订单返回的结果，可能早于listen_orders_forever，也可能晚于listen_orders_forever
        try:
            await self._update_subod_by_ccxtres(od, is_enter, order)
            if is_enter:
                if od.get_info('stoploss_price'):
                    await self._edit_trigger_od(od, 'stoploss_')
                if od.get_info('takeprofit_price'):
                    await self._edit_trigger_od(od, 'takeprofit_')
            else:
                # 平仓，取消关联订单
                await self._cancel_trigger_ods(od)
            if sub_od.status == OrderStatus.Close:
                logger.debug('fire od: %s %s %s %s', is_enter, sub_od.status, sub_od.filled, sub_od.amount)
                await self._fire(od, is_enter)
        except Exception:
            logger.exception(f'error after put exchange order: {od}')

    def _put_order(self, od: InOutOrder, action: str, data: str = None):
        if not btime.prod_mode():
            return
        if action == OrderJob.ACT_EDITTG:
            tg_price = od.get_info(data + 'price')
            logger.debug('edit push: %s %s', od, tg_price)
        self.unready_ids.add(od.id)
        self.order_q.put_nowait(OrderJob(od.id, action, data))

    async def _update_bnb_order(self, od: Order, data: dict):
        info: dict = data['info']
        state = info['X']
        if state == 'NEW':
            return
        cur_ts = info.get('E') or info.get('T')  # 合约订单没有E
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
            filled, fee_val = float(info['z']), float(info['n'])
            if self.market_type == 'future':
                average = float(info['ap'])
            else:
                average = float(info['Z']) / filled
            kwargs = dict(status=od_status, order_type=info['o'], filled=filled, average=average)
            if od_status == OrderStatus.Close:
                kwargs['price'] = kwargs['average']
            if fee_val:
                fee_name = info['N'] or ''
                kwargs['fee_type'] = fee_name
                if od.symbol.endswith(fee_name):
                    # 期货市场手续费以USD计算
                    fee_val /= average
                kwargs['fee'] = fee_val / filled
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
        inout_od: InOutOrder = await InOutOrder.get(od.inout_id)
        if inout_status:
            inout_od.status = inout_status
        if inout_status == InOutStatus.FullExit:
            await self._finish_order(inout_od)
            logger.debug('fire exg od: %s', inout_od)
            await self._fire(inout_od, od.enter)

    async def _update_order(self, od: Order, data: dict):
        if od.status == OrderStatus.Close:
            logger.warning(f'order: {od.inout_id} enter: {od.enter} complete: {od}, ignore trade: {data}')
            return
        if self.name.find('binance') >= 0:
            await self._update_bnb_order(od, data)
        else:
            raise ValueError(f'unsupport exchange to update order: {self.name}')

    @loop_forever
    async def listen_orders_forever(self):
        try:
            trades = await self.exchange.watch_my_trades()
        except ccxt.NetworkError as e:
            logger.error(f'watch_my_trades net error: {e}')
            return
        logger.debug('get my trades: %s', trades)
        async with dba():
            sess = dba.session
            related_ods = set()
            for data in trades:
                symbol = data['symbol']
                if symbol not in BotGlobal.pairs:
                    # 忽略不处理的交易对
                    continue
                trade_key = f"{symbol}_{data['id']}"
                if trade_key in self.handled_trades:
                    continue
                od_key = symbol, data['order']
                if od_key not in self.exg_orders:
                    self.unmatch_trades[trade_key] = data
                    continue
                iod_id, sub_id = self.exg_orders[od_key]
                async with BanLock(f'iod_{iod_id}', 5, force_on_fail=True):
                    sub_od: Order = await sess.get(Order, sub_id)
                    await self._update_order(sub_od, data)
                    related_ods.add(sub_od)
            for sub_od in related_ods:
                async with BanLock(f'iod_{sub_od.inout_id}', 5, force_on_fail=True):
                    self._consume_unmatchs(sub_od)
            if len(self.handled_trades) > 500:
                cut_keys = list(self.handled_trades.keys())[-300:]
                self.handled_trades = OrderedDict.fromkeys(cut_keys, value=1)

    async def _try_fill_exit(self, iod: InOutOrder, filled: float, od_price: float, od_time: int, order_id: str,
                             order_type: str, fee_name: str, fee_rate: float):
        '''
        尝试平仓，用于从第三方交易中更新机器人订单的平仓状态
        '''
        if not iod.enter.filled:
            return filled, iod
        if iod.exit and iod.exit.amount:
            ava_amount = iod.exit.amount - iod.exit.filled
        else:
            ava_amount = iod.enter_amount
        if filled >= ava_amount * 0.99:
            fill_amt = ava_amount
            filled -= ava_amount
            part = iod.cut_part(fill_amt)
        else:
            fill_amt = filled
            filled = 0
            part = iod
        part.update_exit(amount=iod.enter.amount, filled=fill_amt, order_type=order_type,
                         order_id=order_id, price=od_price, average=od_price,
                         status=OrderStatus.Close, fee=fee_rate, fee_type=fee_name,
                         create_at=od_time, update_at=od_time)
        part.exit_tag = ExitTags.third
        part.exit_at = od_time
        part.status = InOutStatus.FullExit
        await iod.save()
        if not part.id:
            # 没有id说明是分离出来的订单，需要保存
            await part.save()
        return filled, part

    async def _exit_any_bnb_order(self, trade: dict) -> bool:
        info: dict = trade['info']
        pair, filled = trade['symbol'], float(info['z'])
        if not filled:
            # 忽略未成交的交易
            return False
        is_short = info.get('ps') == 'SHORT'
        od_side = trade['side']
        open_ods = await InOutOrder.open_orders(pairs=pair)
        open_ods = [od for od in open_ods if od.short == is_short and od.enter.side != od_side]
        if not open_ods:
            # 没有同方向，相反操作的订单
            return False
        is_reduce_only = info.get('R', False)
        od_price, od_time = float(info['ap']), trade['timestamp']
        order_id, od_type = trade['order'], trade['type']
        fee_name = info['N'] or ''
        fee_val = float(info['n'])
        if fee_val:
            fee_name = info['N'] or ''
            if pair.endswith(fee_name):
                # 期货市场手续费以USD计算
                fee_val /= od_price
            fee_val /= filled
        for iod in open_ods:
            # 尝试平仓
            filled, part = await self._try_fill_exit(iod, filled, od_price, od_time, order_id,
                                                     od_type, fee_name, fee_val)
            if part.status == InOutStatus.FullExit:
                await self._finish_order(part)
                logger.debug('exit : %s by third %s', part, trade)
                await self._fire(part, False)
            if filled <= min_dust:
                break
        if not is_reduce_only and filled > min_dust and self.allow_take_over:
            # 有剩余数量，创建相反订单
            exs = ExSymbol.get(self.name, self.market_type, pair)
            iod = self._create_inout_od(exs, is_short, od_price, filled, od_type, fee_val, fee_name, od_time,
                                        OrderStatus.Close, order_id)
            if iod:
                await iod.save()
                logger.debug('enter for left: %s', iod)
                await self._fire(iod, True)
        return True

    async def _exit_any_order(self, trade: dict) -> bool:
        '''
        检查交易流是否是平仓操作
        '''
        if self.name.find('binance') >= 0:
            return await self._exit_any_bnb_order(trade)
        else:
            raise ValueError(f'unsupport exchange to _exit_any_order: {self.name}')

    async def _handle_unmatches(self):
        '''
        处理超时未匹配的订单，1s执行一次
        '''
        exp_after = btime.time_ms() - 1000  # 未匹配交易，1s过期
        cur_unmatchs = [(trade_key, trade) for trade_key, trade in self.unmatch_trades.items()
                        if trade['timestamp'] < exp_after]
        left_unmats = []
        for trade_key, trade in cur_unmatchs:
            matched = False
            if await self._exit_any_order(trade):
                # 检测是否是平仓订单
                matched = True
            elif self.allow_take_over and (await self._create_order_from_bnb(trade)):
                # 检测是否应该接管第三方下单
                matched = True
            if not matched:
                left_unmats.append(trade)
            del self.unmatch_trades[trade_key]
        if left_unmats:
            logger.warning('expired unmatch orders: %s', left_unmats)

    async def _create_order_from_bnb(self, trade: dict):
        info: dict = trade['info']
        state: str = info['X']
        if state != 'FILLED':
            # 只对完全入场的尝试跟踪
            return
        side: str = info['S']
        position: str = info.get('ps')
        if self.market_type == 'future':
            if position == 'LONG' and side == 'SELL' or position == 'SHORT' and side == 'BUY':
                # 忽略平仓的订单
                return
        else:
            if side == 'SELL':
                # 现货市场卖出即平仓，忽略平仓
                return
        exs = ExSymbol.get(self.name, self.market_type, trade['symbol'])
        od_status = OrderStatus.Close if state == 'FILLED' else OrderStatus.PartOk
        filled, fee_val = float(info['z']), float(info['n'])
        if self.market_type == 'future':
            average = float(info['ap'])
        else:
            average = float(info['Z']) / filled
        fee_name = info['N'] or ''
        if fee_name and exs.symbol.endswith(fee_name):
            # 期货市场手续费以USD计算
            fee_val /= average
        fee_rate = fee_val / filled
        is_short = position == 'SHORT'
        od = self._create_inout_od(exs, is_short, average, filled, info['o'], fee_rate, fee_name,
                                   btime.time_ms(), od_status, trade['id'])
        if od:
            await od.save()
            logger.debug('enter od: %s', od)
            await self._fire(od, True)
            stg_list = BotGlobal.pairtf_stgs.get(f'{exs.symbol}_{od.timeframe}')
            stg = next((stg for stg in stg_list if stg.name == self.take_over_stgy), None)
            if stg:
                stg.init_third_od(od)
        return od

    async def _exec_order_enter(self, od: InOutOrder):
        if od.exit_tag:
            # 订单已被取消，不再提交到交易所
            return
        if not od.enter.amount:
            if not od.quote_cost:
                raise ValueError(f'quote_cost is required to calc enter_amount')
            try:
                real_price = MarketPrice.get(od.symbol)
                # 这里应使用市价计算数量，因传入价格可能和市价相差很大
                od.enter.amount = self.exchange.pres_amount(od.symbol, od.quote_cost / real_price)
            except Exception:
                logger.error(f'pres_amount for order fail: {od.dict()}')

        async def force_exit_od():
            od.local_exit(ExitTags.force_exit, status_msg='InsufficientFunds')
            sess = dba.session
            await sess.delete(od)
            if od.enter:
                await sess.delete(od.enter)
            await sess.flush()
        try:
            await self._create_exg_order(od, True)
        except ccxt.InsufficientFunds:
            await force_exit_od()
            err_msg = f'InsufficientFunds open cancel: {od}'
            if btime.time() - self.last_lack_ts > 3600:
                logger.error(err_msg)
                self.last_lack_ts = btime.time()
            else:
                logger.warning(err_msg)
        except ccxt.ExchangeError as e:
            err_msg = str(e)
            if err_msg.find(':-4061') > 0:
                side = 'SHORT' if od.short else 'LONG'
                logger.error(f'enter fail, SideNotMatch: order: {side}, {e}')
                await force_exit_od()
            else:
                raise

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
                await self._finish_order(od)
                await self._cancel_trigger_ods(od)
                # 这里未入场直接退出的，不应该fire
                return
            logger.debug('exit uncomple od: %s', od)
            await self._fire(od, True)
        # 检查入场订单是否已成交，如未成交则直接取消
        await self._create_exg_order(od, False)

    async def consume_queue(self):
        while True:
            job: OrderJob = await self.order_q.get()
            while job.od_id in self.unready_ids:
                await asyncio.sleep(0.005)
            await self.exec_od_job(job)
            self.order_q.task_done()
            if BotGlobal.state == BotState.STOPPED and not self.order_q.qsize():
                break

    async def exec_od_job(self, job: OrderJob):
        try:
            od: Optional[InOutOrder] = None
            try:
                async with BanLock(f'iod_{job.od_id}', 5, force_on_fail=True):
                    async with dba():
                        od = await InOutOrder.get(job.od_id)
                        if job.action == OrderJob.ACT_ENTER:
                            await self._exec_order_enter(od)
                        elif job.action == OrderJob.ACT_EXIT:
                            await self._exec_order_exit(od)
                        elif job.action == OrderJob.ACT_EDITTG:
                            await self._edit_trigger_od(od, job.data)
                        else:
                            logger.error(f'unsupport order job type: {job.action}')
            except Exception as e:
                if od and job.action in {OrderJob.ACT_ENTER, OrderJob.ACT_EXIT}:
                    err_msg = str(e)
                    async with BanLock(f'iod_{job.od_id}', 5, force_on_fail=True):
                        async with dba():
                            od = await InOutOrder.get(job.od_id)
                            # 平仓时报订单无效，说明此订单在交易所已退出-2022 ReduceOnly Order is rejected
                            od.local_exit(ExitTags.fatal_err, status_msg=err_msg)
                    logger.exception('consume order %s: %s, force exit: %s', type(e), e, job)
                else:
                    logger.exception('consume order exception: %s', job)
        except Exception:
            logger.exception("consume order_q error")

    async def edit_pending_order(self, od: InOutOrder, is_enter: bool, price: float):
        sub_od = od.enter if is_enter else od.exit
        sub_od.price = price
        if not sub_od.order_id:
            await self._exec_order_enter(od)
            return
        left_amount = sub_od.amount - sub_od.filled
        try:
            if self.market_type == 'future':
                await self.exchange.cancel_order(sub_od.order_id, od.symbol)
                od_type = sub_od.order_type or self.od_type
                params = dict()
                if self.market_type == 'future':
                    params['positionSide'] = 'SHORT' if od.short else 'LONG'
                res = await self.exchange.create_order(od.symbol, od_type, sub_od.side, left_amount, price, params)
            else:
                res = await self.exchange.edit_limit_order(sub_od.order_id, od.symbol, sub_od.side,
                                                           left_amount, price)
            await self._update_subod_by_ccxtres(od, is_enter, res)
        except ccxt.InvalidOrder as e:
            logger.exception('edit invalid order: %s, %s', e, od)

    @loop_forever
    async def trail_unmatches_forever(self):
        '''
        1s一次轮训处理未匹配的订单。尝试跟踪用户下单。
        '''
        if not btime.prod_mode():
            return
        try:
            async with dba():
                await self._handle_unmatches()
        except Exception:
            logger.exception('trail_unmatches_forever error')
        await asyncio.sleep(1)

    @loop_forever
    async def trail_open_orders_forever(self):
        timeouts = self.config.get('limit_vol_secs', 5) * 2
        if not btime.prod_mode():
            return
        try:
            async with dba():
                await self._trail_open_orders(timeouts)
        except Exception:
            logger.exception('_trail_open_orders error')
        await asyncio.sleep(timeouts)

    async def _trail_open_orders(self, timeouts: int):
        '''
        跟踪未关闭的订单，根据市场价格及时调整，避免长时间无法成交
        :return:
        '''
        op_orders = await InOutOrder.open_orders()
        if not op_orders:
            return
        exp_orders = [od for od in op_orders if od.pending_type(timeouts)]
        if not exp_orders:
            return
        logger.debug('pending open orders: %s', exp_orders)
        sess = dba.session
        from itertools import groupby
        exp_orders = sorted(exp_orders, key=lambda x: x.symbol)
        unsubmits = []  # 记录长期未提交到交易所的订单
        for pair, od_list in groupby(exp_orders, lambda x: x.symbol):
            buy_price, sell_price = await self._get_pair_prices(pair, round(self.limit_vol_secs * 0.5))
            od_list: List[InOutOrder] = list(od_list)
            for od in od_list:
                if od.exit and od.exit_tag:
                    sub_od, is_enter, new_price = od.exit, False, sell_price
                else:
                    sub_od, is_enter, new_price = od.enter, True, buy_price
                if not sub_od.order_id:
                    od_msg = str(od)
                    if od_msg not in self.old_unsubmits:
                        unsubmits.append(od_msg)
                        self.old_unsubmits.add(od_msg)
                    continue
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
                await sess.flush()
        from banbot.rpc import Notify, NotifyType
        if unsubmits and Notify.instance:
            Notify.send(type=NotifyType.EXCEPTION, status=f'超时未提交订单：{unsubmits}')

    @loop_forever
    async def watch_leverage_forever(self):
        if self.market_type != 'future':
            return 'exit'
        try:
            msg = await self.exchange.watch_account_update()
        except ccxt.NetworkError as e:
            logger.error(f'watch_account_update net error: {e}')
            return
        leverage = msg.get('leverage')
        if not leverage:
            # 忽略保证金状态更新
            return
        symbol = msg['symbol']
        if symbol not in self.exchange.leverages:
            await self.exchange.update_symbol_leverages([symbol], leverage)
        else:
            self.exchange.leverages[symbol].leverage = leverage

    @loop_forever
    async def watch_price_forever(self):
        if self.market_type != 'future':
            return 'exit'
        try:
            price_list = await self.exchange.watch_mark_prices()
        except ccxt.NetworkError as e:
            logger.error(f'watch_price_forever net error: {e}')
            return
        for item in price_list:
            MarketPrice.set_new_price(item['symbol'], item['markPrice'])

    async def cleanup(self):
        if btime.run_mode not in btime.LIVE_MODES:
            # 实盘模式和实时模拟时，停止机器人不退出订单
            async with dba():
                exit_ods = await self.exit_open_orders(dict(tag='bot_stop'), 0, is_force=True, od_dir='both')
                if exit_ods:
                    logger.info('exit %d open trades', len(exit_ods))
        await self.order_q.join()
