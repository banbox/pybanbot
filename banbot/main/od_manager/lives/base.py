#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : main.py
# Author: anyongjin
# Date  : 2023/3/30
import json

from banbot.main.od_manager.base import *
from asyncio import Queue
from collections import OrderedDict
from banbot.main.wallets import CryptoWallet
from banbot.data.tools import auto_fetch_ohlcv
from sqlalchemy.orm import object_session


class LiveOrderMgr(OrderManager):
    obj: Optional['LiveOrderMgr'] = None

    def __init__(self, config: dict, exchange: CryptoExchange, wallets: CryptoWallet, data_hd: LiveDataProvider,
                 callback: Callable):
        super(LiveOrderMgr, self).__init__(config, exchange, wallets, data_hd, callback)
        LiveOrderMgr.obj = self
        self.exchange = exchange
        self.wallets: CryptoWallet = wallets
        self._done_keys: Dict[Tuple[str, str], Any] = OrderedDict()  # 有序集合
        '已完成的exorder的id主键列表'
        self.exg_orders: Dict[Tuple[str, str], Tuple[int, int]] = dict()
        '交易所币+订单ID到数据库订单ID的映射。用于从推送的订单流更新订单状态。'
        self.unmatch_trades: Dict[str, dict] = dict()
        '未匹配交易，key: symbol, order_id；每个订单只保留一个未匹配交易，也只应该有一个'
        self.handled_trades: Dict[str, int] = OrderedDict()  # 有序集合，使用OrderedDict实现
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
        his_ods = await get_db_orders(pairs=list(cur_symbols), status='his')
        his_ods = sorted(his_ods, key=lambda o: o.enter_at, reverse=True)
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
            his_od = next((o for o in his_ods if o.symbol == pair), None)
            prev_tf = his_od.timeframe if his_od else None
            await self._sync_pair_orders(pair, long_pos, short_pos, since_ms, cur_ods, prev_tf)
            res_odlist.update(cur_ods)
        new_ods = res_odlist - set(op_ods)
        old_ods = res_odlist.intersection(op_ods)
        del_ods = set(op_ods) - res_odlist
        if old_ods:
            logger.info(f'恢复{len(old_ods)}个未平仓订单：{old_ods}')
        if new_ods:
            logger.info(f'新开始跟踪{len(new_ods)}个用户下单：{new_ods}')
        return len(old_ods), len(new_ods), len(del_ods), list(res_odlist)

    async def _sync_pair_orders(self, pair: str, long_pos: dict, short_pos: dict, since_ms: int,
                                op_ods: List[InOutOrder], prev_tf: str = None):
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
                await self._apply_history_order(op_ods, exod, prev_tf)
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
                        await iod.save()
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
                    await iod.save()
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
            long_od = self._create_order_from_position(long_pos, prev_tf)
            if long_od:
                await long_od.save()
                op_ods.append(long_od)
        if short_pos and short_pos['contracts'] > min_dust:
            short_od = self._create_order_from_position(short_pos, prev_tf)
            if short_od:
                await short_od.save()
                op_ods.append(short_od)

    def _create_order_from_position(self, pos: dict, prev_tf: str):
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
        fee_rate = market['taker'] if ent_od_type == OrderType.Market.value else market['maker']
        fee_name = exs.quote_code if self.market_type == 'future' else exs.base_code
        tag_ = '开空' if is_short else '开多'
        logger.info(f'[仓]{tag_}：price:{average}, amount: {filled}, fee: {fee_rate}')
        return self._create_inout_od(exs, is_short, average, filled, ent_od_type, fee_rate,
                                     fee_name, btime.time_ms(), OrderStatus.Close, prev_tf=prev_tf)

    def _create_inout_od(self, exs: ExSymbol, short: bool, average: float, filled: float,
                         ent_od_type: str, fee_rate: float, fee_name: str, enter_at: int,
                         ent_status: int, ent_odid: str = None, prev_tf: str = None):
        job = next((p for p in BotGlobal.stg_symbol_tfs if p[0] == self.take_over_stgy and p[1] == exs.symbol), None)
        timeframe = job[2] if job else prev_tf
        if not timeframe and BotGlobal.state == BotState.RUNNING:
            # 启动后，跳过未跟踪的。初始化阶段，允许未跟踪的币。
            logger.warning(f'take over job not found: {exs.symbol} {self.take_over_stgy}')
            return
        leverage = self.exchange.get_leverage(exs.symbol)
        quote_cost = filled * average / leverage
        io_status = InOutStatus.FullEnter if ent_status == OrderStatus.Close else InOutStatus.PartEnter
        return InOutOrder(
            sid=exs.id,
            symbol=exs.symbol,
            timeframe=timeframe,
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

    def _is_short(self, trade: dict):
        info: dict = trade['info']
        if self.name == 'binance':
            return info['positionSide'] == 'SHORT'
        else:
            raise ValueError(f'unsupport exchange: {self.name}')

    async def _apply_history_order(self, od_list: List[InOutOrder], od: dict, prev_tf: str):
        is_short = self._is_short(od)
        is_sell = od['side'] == 'sell'
        is_reduce_only = od['reduceOnly']
        exs = ExSymbol.get(self.name, self.market_type, od['symbol'])
        market = self.exchange.markets[exs.symbol]
        # 订单信息没有手续费，直接从当前机器人订单类型推断手续费，可能和实际的手续费不同
        fee_rate = market['taker'] if od['type'] == OrderType.Market.value else market['maker']
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
                logger.info(
                    f'{tag_}：price:{od_price}, amount: {od_amount}, {od["type"]}, fee: {fee_rate} {od_time} id: {od["id"]}')
                iod = self._create_inout_od(exs, is_short, od_price, od_amount, od['type'], fee_rate, fee_name,
                                            od_time, OrderStatus.Close, od['id'], prev_tf=prev_tf)
                if iod:
                    od_list.append(iod)

        if is_short == is_sell:
            # 开多，或开空
            tag = '开空' if is_short else '开多'
            logger.info(
                f'{tag}：price:{od_price}, amount: {od_amount}, {od["type"]}, fee: {fee_rate} {od_time} id: {od["id"]}')
            od = self._create_inout_od(exs, is_short, od_price, od_amount, od['type'], fee_rate, fee_name,
                                       od_time, OrderStatus.Close, od['id'], prev_tf=prev_tf)
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
        # 这里只使用手续费率，所以提供假的amount和price即可
        od = Order(symbol=pair, order_type=self.od_type, side='buy', amount=10, price=1.)
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
            buy_price = sell_price = MarketPrice.get(pair)
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

    async def _consume_unmatchs(self, od: InOutOrder, sub_od: Order):
        has_match = False
        for trade in list(self.unmatch_trades.values()):
            if trade['symbol'] != sub_od.symbol or trade['order'] != sub_od.order_id:
                continue
            trade_key = f"{trade['symbol']}_{trade['id']}"
            del self.unmatch_trades[trade_key]
            if trade_key in self.handled_trades or sub_od.status == OrderStatus.Close:
                continue
            logger.info('exec unmatch trade: %s', trade)
            cur_mat = await self._update_order(od, sub_od, trade)
            has_match = has_match or cur_mat
        return has_match

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

    def _get_trade_ts(self, trade: dict):
        data_info = trade.get('info') or dict()
        cur_ts = trade['timestamp']
        if not cur_ts and self.name == 'binance':
            # 币安期货返回时间戳需要从info.updateTime取
            cur_ts = int(data_info.get('updateTime', '0'))
        return cur_ts

    def _update_order_res(self, od: InOutOrder, is_enter: bool, data: dict):
        sub_od = od.enter if is_enter else od.exit
        cur_ts = self._get_trade_ts(data)
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
            if fee and fee.get('cost'):
                sub_od.fee = fee.get('cost')
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
        if sub_od.order_id:
            # 如修改订单价格，order_id会变化
            self._done_keys[(od.symbol, sub_od.order_id)] = 1
        sub_od.order_id = order["id"]
        exg_key = od.symbol, sub_od.order_id
        self.exg_orders[exg_key] = od.id, sub_od.id
        logger.debug('create order: %s %s %s', od.symbol, sub_od.order_id, order)
        new_num, old_num = self._check_new_trades(order['trades'])
        if new_num or self.market_type != 'spot':
            # 期货市场未返回trades
            self._update_order_res(od, is_enter, order)
        else:
            logger.debug('no new trades: %s %s %s', od.symbol, sub_od.order_id, order)
        await self._consume_unmatchs(od, sub_od)
        logger.debug('apply ccxtres to order: %s %s %s %s', od, is_enter, order, sub_od.dict())

    def _finish_order(self, od: InOutOrder):
        if od.enter.order_id:
            self._done_keys[(od.symbol, od.enter.order_id)] = 1
        if od.exit and od.exit.order_id:
            self._done_keys[(od.symbol, od.exit.order_id)] = 1
        super(LiveOrderMgr, self)._finish_order(od)
        if len(self._done_keys) > 1500:
            done_keys = list(self._done_keys.keys())
            self._done_keys = OrderedDict.fromkeys(done_keys[800:], value=1)
            del_keys = done_keys[:800]
            for k in del_keys:
                if k in self.exg_orders:
                    self.exg_orders.pop(k)

    async def _edit_trigger_od(self, od: InOutOrder, prefix: str, try_kill=True):
        trigger_oid = od.get_info(f'{prefix}oid')
        params = dict()
        params['positionSide'] = 'SHORT' if od.short else 'LONG'
        trig_price = od.get_info(f'{prefix}price')
        if trig_price:
            params.update(closePosition=True)
            if prefix == 'takeprofit_':
                params.update(takeProfitPrice=trig_price)
            elif prefix == 'stoploss_':
                params.update(stopLossPrice=trig_price)
            else:
                raise ValueError(f'invalid prefix: {prefix}')
        side = 'buy' if od.short else 'sell'
        amount = od.enter.amount
        order = None
        try:
            logger.debug('create trigger %s %s %s %s %s %s %s',
                         prefix, od.symbol, self.od_type, side, amount, trig_price, params)
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
        if order:
            od.set_info(**{f'{prefix}oid': order['id']})
            logger.debug('save trigger oid: %s %s %s %s', prefix, order, od.info, trigger_oid)
        if trigger_oid and (not order or order['status'] == 'open'):
            try:
                await self.exchange.cancel_order(trigger_oid, od.symbol)
            except ccxt.OrderNotFound:
                logger.warning(f'[{od.id}] cancel old stop order fail, not found: {od.symbol}, {trigger_oid}')

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
        if not sub_od.price and od_type.find(OrderType.Market.value) < 0:
            # 非市价单时，计算价格
            buy_price, sell_price = (await self._get_pair_prices(od.symbol, self.limit_vol_secs))
            cur_price = buy_price if sub_od.side == 'buy' else sell_price
            sub_od.price = self.exchange.pres_price(od.symbol, cur_price)
        if not sub_od.amount:
            if is_enter:
                raise ValueError(f'amount is required for enter: {od}, {sub_od.amount}')
            sub_od.amount = od.enter.filled
            if not sub_od.amount:
                # 没有入场，直接本地退出。
                od.status = InOutStatus.FullExit
                od.update_exit(price=od.enter.price)
                await od.save()
                self._finish_order(od)
                await self._cancel_trigger_ods(od)
                return
        side, amount, price = sub_od.side, sub_od.amount, sub_od.price
        params = dict()
        if self.market_type == 'future':
            params['positionSide'] = 'SHORT' if od.short else 'LONG'
        print_args = [is_enter, od.symbol, od_type, side, amount, price, params]
        od_res = None
        try:
            od_res = await self.exchange.create_order(od.symbol, od_type, side, amount, price, params)
            print_args.append(od_res)
            logger.debug('create exg order res: %s, %s, %s, %s, %s, %s, %s, %s', *print_args)
        except Exception as e:
            catched = False
            if isinstance(e, ccxt.errors.InvalidOrder):
                if str(e).find('-2022') > 0:
                    catched = True
                    logger.warning(f"{od.key} id:{od.id} {print_args} reduce only is rejected")
                    tag = (od.enter_tag if is_enter else od.exit_tag) or ExitTags.fatal_err
                    od.local_exit(tag, status_msg=str(e))
            if not catched:
                raise ValueError(f'create exg order fail: {print_args}')
        # 创建订单返回的结果，可能早于listen_orders_forever，也可能晚于listen_orders_forever
        try:
            if od_res:
                await self._update_subod_by_ccxtres(od, is_enter, od_res)
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
                self._fire(od, is_enter)
        except Exception:
            logger.exception(f'error after put exchange order: {od}')

    def _put_order(self, od: InOutOrder, action: str, data: str = None):
        if not btime.prod_mode():
            return
        if action == OrderJob.ACT_EDITTG:
            tg_price = od.get_info(data + 'price')
            logger.debug('edit push: %s %s', od, tg_price)
        od_id = od.id

        def do_put(success: bool):
            if success:
                self.order_q.put_nowait(OrderJob(od_id, action, data))

        dba.add_callback(dba.session, do_put)

    async def _apply_exg_order(self, od: InOutOrder, sub_od: Order, data: dict):
        raise ValueError(f'unsupport exchange to update order: {self.name}')

    async def _update_order(self, od: InOutOrder, sub_od: Order, data: dict):
        if sub_od.status == OrderStatus.Close:
            tag = 'enter' if sub_od.enter else 'exit'
            logger.debug('order: %s %s complete, ignore trade: %s', sub_od, tag, data)
            return
        await self._apply_exg_order(od, sub_od, data)
        return True

    @loop_forever
    async def listen_orders_forever(self):
        try:
            trades = await self.exchange.watch_my_trades()
        except ccxt.NetworkError as e:
            logger.error(f'watch_my_trades net error: {e}')
            return
        logger.debug('get my trades: %s', trades)
        valid_items = []
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
            if od_key in self._done_keys:
                continue
            valid_items.append((od_key, data))
        if not valid_items:
            return
        for od_key, data in valid_items:
            iod_id, sub_id = self.exg_orders[od_key]
            async with LocalLock(f'iod_{iod_id}', 5, force_on_fail=True):
                async with dba():
                    od = await InOutOrder.get(iod_id)
                    tracer = InOutTracer([od])
                    sub_od: Order = od.enter if od.enter.id == sub_id else od.exit
                    await self._update_order(od, sub_od, data)
                    logger.debug('update order by push %s %s', data, sub_od.dict())
                    has_match = await self._consume_unmatchs(od, sub_od)
                    if has_match:
                        logger.debug('update order by unmatch %s', sub_od.dict())
                    await tracer.save()
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
            ava_amount = iod.enter.filled
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
        iod.save_mem()
        if not part.id:
            # 没有id说明是分离出来的订单，需要保存
            await part.save()
        return filled, part

    async def _exec_order_enter(self, od: InOutOrder):
        if od.exit_tag:
            # 订单已被取消，不再提交到交易所
            return
        if not od.enter.amount:
            if not od.quote_cost:
                legal_cost = od.get_info('legal_cost')
                if legal_cost:
                    exs = ExSymbol.get_by_id(od.sid)
                    od.quote_cost = self.wallets.get_amount_by_legal(exs.quote_code, legal_cost)
                else:
                    raise ValueError(f'quote_cost is required to calc enter_amount')
            amount, real_price = None, None
            try:
                real_price = MarketPrice.get(od.symbol)
                amount = od.quote_cost / real_price
                # 这里应使用市价计算数量，因传入价格可能和市价相差很大
                od.enter.amount = self.exchange.pres_amount(od.symbol, amount)
            except Exception as e:
                logger.error(f'pres_amount for order fail: {e} {od.dict()}, price: {real_price}, amt: {amount}')

        async def force_del_od():
            od.local_exit(ExitTags.force_exit, status_msg='del')
            sess = dba.session
            await sess.delete(od)
            if od.enter:
                await sess.delete(od.enter)
            await sess.flush()

        try:
            await self._create_exg_order(od, True)
        except ccxt.InsufficientFunds:
            await force_del_od()
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
                await force_del_od()
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
                od.status = InOutStatus.FullExit
                od.update_exit(price=od.enter.price)
                await od.save()
                self._finish_order(od)
                await self._cancel_trigger_ods(od)
                # 这里未入场直接退出的，不应该fire
                return
            logger.debug('exit uncomple od: %s', od)
            self._fire(od, True)
        # 检查入场订单是否已成交，如未成交则直接取消
        await self._create_exg_order(od, False)

    async def consume_queue(self):
        reset_ctx()
        while True:
            job: OrderJob = await self.order_q.get()
            await self.exec_od_job(job)
            self.order_q.task_done()
            if BotGlobal.state == BotState.STOPPED and not self.order_q.qsize():
                break

    async def exec_od_job(self, job: OrderJob):
        try:
            async with LocalLock(f'iod_{job.od_id}', 5, force_on_fail=True):
                async with dba():
                    tracer = InOutTracer()
                    try:
                        od = await InOutOrder.get(job.od_id)
                        tracer.trace([od])
                        self._check_od_sess(od)
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
                            # 平仓时报订单无效，说明此订单在交易所已退出-2022 ReduceOnly Order is rejected
                            od.local_exit(ExitTags.fatal_err, status_msg=err_msg)
                            logger.exception('consume order %s: %s, force exit: %s', type(e), e, job)
                        else:
                            logger.exception('consume order exception: %s', job)
                    await tracer.save()
        except Exception:
            logger.exception("consume order_q error")

    def _check_od_sess(self, od: InOutOrder):
        sess: SqlSession = dba.session
        sess1 = object_session(od)
        same1 = sess1 is sess
        same1_1 = od in sess.identity_map.values()
        sess2 = object_session(od.enter)
        same2 = sess2 is sess
        same2_1 = od.enter in sess.identity_map.values()
        logger.info(f"check order in sess: {same1} {same1_1} {same2} {same2_1} {od} {sess} {sess1} {sess2}")

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
            return 'exit'
        try:
            unmatches = self._get_expire_unmatches()
            if unmatches:
                async with dba():
                    await self._handle_unmatches(unmatches)
        except Exception:
            logger.exception('trail_unmatches_forever error')
        await asyncio.sleep(3)

    async def _exit_by_exg_order(self, trade: dict) -> bool:
        """检查交易流是否是平仓操作"""
        raise ValueError(f'unsupport exchange to _exit_by_exg_order: {self.name}')

    def _get_expire_unmatches(self) -> List[Tuple[str, dict]]:
        exp_after = btime.time_ms() - 1000  # 未匹配交易，1s过期
        return [(trade_key, trade) for trade_key, trade in self.unmatch_trades.items()
                if trade['timestamp'] < exp_after]

    async def _handle_unmatches(self, cur_unmatchs: List[Tuple[str, dict]]):
        '''
        处理超时未匹配的订单，1s执行一次
        '''
        left_unmats = []
        for trade_key, trade in cur_unmatchs:
            matched = False
            if await self._exit_by_exg_order(trade):
                # 检测是否是平仓订单
                matched = True
            elif self.allow_take_over and (await self._trace_exg_order(trade)):
                # 检测是否应该接管第三方下单
                matched = True
            if not matched:
                left_unmats.append(trade)
            del self.unmatch_trades[trade_key]
        if left_unmats:
            logger.warning('expired unmatch orders: %s', left_unmats)

    async def _trace_exg_order(self, trade: dict):
        raise ValueError(f'unsupport exchange to _trace_exg_order: {self.name}')

    @loop_forever
    async def trail_unfill_orders_forever(self):
        if not self.config.get('auto_edit_limit') or not btime.prod_mode():
            # 未启用，退出
            return 'exit'
        timeouts = self.config.get('limit_vol_secs', 5) * 2
        try:
            exp_orders = [od for k, od in BotCache.open_ods if od.pending_type(timeouts)]
            if exp_orders:
                # 当缓存有符合条件的未成交订单时，才尝试执行，避免荣誉的数据库访问
                async with dba():
                    await self._trail_unfill_orders(timeouts)
        except Exception:
            logger.exception('_trail_open_orders error')
        await asyncio.sleep(timeouts)

    async def _trail_unfill_orders(self, timeouts: int):
        '''
        跟踪未成交的订单，根据市场价格及时调整，避免长时间无法成交
        :return:
        '''
        op_orders = await InOutOrder.open_orders()
        if not op_orders:
            return
        exp_orders = [od for od in op_orders if od.pending_type(timeouts)]
        if not exp_orders:
            return
        logger.debug('pending open orders: %s', exp_orders)
        tracer = InOutTracer(exp_orders)
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
        await tracer.save()
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

    async def cleanup(self):
        if btime.run_mode not in btime.LIVE_MODES:
            # 实盘模式和实时模拟时，停止机器人不退出订单
            async with dba():
                ext_dic = dict(tag='bot_stop')
                exit_ods = await self.exit_open_orders(ext_dic, 0, is_force=True, od_dir='both',
                                                       with_unopen=True)
                if exit_ods:
                    logger.info('exit %d open trades', len(exit_ods))
        await self.order_q.join()

    @classmethod
    def init(cls, config: dict, exchange: CryptoExchange, wallets: CryptoWallet, data_hd: LiveDataProvider,
             callback: Callable):
        if exchange.name == 'binance':
            from banbot.main.od_manager.lives import BinanceOrderMgr
            return BinanceOrderMgr(config, exchange, wallets, data_hd, callback)
        raise ValueError(f'unsupport order mgr for {exchange.name}')
