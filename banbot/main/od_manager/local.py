#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : local.py
# Author: anyongjin
# Date  : 2023/11/4
from banbot.main.od_manager.base import *


class LocalOrderManager(OrderManager):
    obj: Optional['LocalOrderManager'] = None

    def __init__(self, config: dict, exchange: CryptoExchange, wallets: WalletsLocal, data_hd: DataProvider,
                 callback: Callable):
        super(LocalOrderManager, self).__init__(config, exchange, wallets, data_hd, callback)
        LocalOrderManager.obj = self
        self.stake_currency = config.get('stake_currency') or []
        self.exchange = exchange
        self.network_cost = 0  # 模拟网络延迟

    def update_by_bar(self, all_opens: List[InOutOrder], pair: str, timeframe: str, row):
        if not self.network_cost:
            self.network_cost = 0.3 if timeframe == 'ws' else 3.
        super(LocalOrderManager, self).update_by_bar(all_opens, pair, timeframe, row)
        if all_opens and not btime.prod_mode():
            cur_ods = [od for od in all_opens if od.symbol == pair]
            affect_num = self.fill_pending_orders(cur_ods, timeframe, row)
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

    async def force_exit(self, od: InOutOrder, tag: Optional[str] = None, price: float = None):
        if not tag:
            tag = 'force_exit'
        await self.exit_order(od, dict(tag=tag), price)
        if not price:
            price = MarketPrice.get(od.symbol)
        self._fill_pending_exit(od, price)

    def _fill_pending_enter(self, od: InOutOrder, price: float):
        try:
            self.wallets.enter_od(od, self.last_ts)
        except LackOfCash as e:
            # 余额不足
            od.local_exit(ExitTags.force_exit, status_msg=str(e))
            return
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
                exs = ExSymbol.get_by_id(od.sid)
                self.wallets.cancel(od.key, exs.quote_code)
                return
        fees = self.exchange.calc_fee(sub_od.symbol, sub_od.order_type, sub_od.side, sub_od.amount, enter_price)
        if fees['rate']:
            sub_od.fee = fees['cost']
            sub_od.fee_type = fees['currency']
        if not sub_od.price:
            sub_od.price = enter_price
        ctx = self.get_context(od)
        self.wallets.confirm_od_enter(od, enter_price)
        update_time = ctx[bar_time][0] + self.network_cost * 1000
        self.last_ts = update_time / 1000
        od.status = InOutStatus.FullEnter
        sub_od.create_at = update_time
        sub_od.update_at = update_time
        sub_od.filled = sub_od.amount
        sub_od.average = enter_price
        sub_od.status = OrderStatus.Close
        self._fire(od, True)

    def _fill_pending_exit(self, od: InOutOrder, exit_price: float):
        sub_od = od.exit
        self.wallets.exit_od(od, sub_od.amount, self.last_ts)
        fees = self.exchange.calc_fee(sub_od.symbol, sub_od.order_type, sub_od.side, sub_od.amount, exit_price)
        if fees['cost']:
            sub_od.fee = fees['cost'] or 0
            sub_od.fee_type = fees['currency']
        ctx = self.get_context(od)
        update_time = ctx[bar_time][0] + self.network_cost * 1000
        self.last_ts = update_time / 1000
        od.status = InOutStatus.FullExit
        sub_od.create_at = update_time
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

    def _sim_market_price(self, timeframe: str, candle: np.ndarray) -> float:
        '''
        根据收到的下一个bar数据，计算到订单提交到交易所的时间延迟：对应的价格。
        阳线和阴线对应不同的模拟方法。
        阳线一般是先略微下调，再上冲到最高点，最后略微回调出现上影线。
        阴线一般是先略微上调，再下跌到最低点，最后略微回调出现下影线。
        :return:
        '''
        rate = min(1., self.network_cost / tf_to_secs(timeframe))
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

    def fill_pending_orders(self, open_ods: List[InOutOrder], timeframe: str = None, candle: Optional[tuple] = None):
        '''
        填充等待交易所响应的订单。不可用于实盘；可用于回测、模拟实盘等。
        此方法内部会访问锁：ctx_lock，请勿在TempContext中调用此方法
        :param open_ods:
        :param timeframe:
        :param candle:
        :return:
        '''
        if btime.prod_mode():
            raise RuntimeError('fill_pending_orders unavaiable in PROD mode')
        affect_num = 0
        for od in open_ods:
            if timeframe and od.timeframe != timeframe:
                continue
            cur_candle = candle or self.data_mgr.get_latest_ohlcv(od.symbol)
            if od.exit_tag and od.exit and od.exit.status != OrderStatus.Close:
                sub_od = od.exit
            elif od.enter.status != OrderStatus.Close:
                sub_od = od.enter
            else:
                # 已入场完成，尚未出现出场信号，检查是否触发止损
                if not od.exit_tag:
                    self._check_od_trigger(od, cur_candle)
                continue
            # 更新待执行的订单
            od_type = sub_od.order_type or self.od_type
            if od_type == OrderType.Limit.value and sub_od.price:
                price = sub_od.price
                if sub_od.side == 'buy':
                    if price < cur_candle[lcol]:
                        continue
                    if price > cur_candle[ocol]:
                        # 买价高于市价，以市价成交
                        price = cur_candle[ocol]
                elif sub_od.side == 'sell':
                    if price > cur_candle[hcol]:
                        continue
                    elif price < cur_candle[ocol]:
                        # 卖价低于市价，以市价成交
                        price = cur_candle[ocol]
            else:
                # 按网络延迟，模拟成交价格，和开盘价接近
                price = self._sim_market_price(od.timeframe, cur_candle)
            if sub_od.enter:
                self._fill_pending_enter(od, price)
            else:
                self._fill_pending_exit(od, price)
            affect_num += 1
        return affect_num

    def _check_od_trigger(self, od: InOutOrder, candle: tuple):
        """检查是否触发止盈止损"""
        sl_price = od.get_info('stoploss_price')
        tp_price = od.get_info('takeprofit_price')
        if not (sl_price or tp_price):
            return False
        high_price, low_price = candle[hcol], candle[lcol]
        if sl_price and (od.short and high_price >= sl_price or
                         not od.short and low_price <= sl_price):
            od.local_exit(ExitTags.stoploss, sl_price)
        elif tp_price and (od.short and low_price <= tp_price or
                           not od.short and high_price >= tp_price):
            od.local_exit(ExitTags.takeprofit, tp_price)
        else:
            return False
        self.wallets.exit_od(od, od.exit.amount, self.last_ts)
        # 用计算的利润更新钱包
        self.wallets.confirm_od_exit(od, od.exit.price)
        return True

    async def cleanup(self):
        await self.exit_open_orders(dict(tag='bot_stop'), 0, od_dir='both', with_unopen=True)
        open_ods = await InOutOrder.open_orders()

        tracer = InOutTracer(open_ods)
        self.fill_pending_orders(open_ods)
        await tracer.save()

        if not self.config.get('no_db'):
            await InOutOrder.dump_to_db()
