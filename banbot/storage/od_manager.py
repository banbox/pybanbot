#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : main.py
# Author: anyongjin
# Date  : 2023/3/30
import inspect

import orjson

from banbot.bar_driven.tainds import get_cur_symbol, timeframe_secs, bar_arr
from banbot.exchange.crypto_exchange import loop_forever, CryptoExchange
from banbot.storage.orders import *
from banbot.util.common import logger, SingletonArg
from banbot.util import btime
from banbot.main.wallets import CryptoWallet, WalletsLocal
from banbot.strategy.base import BaseStrategy
from banbot.util.misc import *


class OrderManager(metaclass=SingletonArg):
    def __init__(self, exg_name: str, wallets: WalletsLocal, dump_path: str = None):
        self.name = exg_name
        self.wallets = wallets
        self.exg_orders: Dict[str, Order] = dict()
        self.open_orders: Dict[str, InOutOrder] = dict()  # 尚未退出的订单
        self.his_list: List[InOutOrder] = []  # 历史已完成或取消的订单
        self.network_cost = 0.6  # 模拟网络延迟
        self.callbacks: List[Callable] = []
        self.dump_path = dump_path

    def _fire(self, key: str, enter: bool):
        if key not in self.callbacks:
            return
        od = self.open_orders[key]
        for func in self.callbacks:
            if inspect.iscoroutinefunction(func):
                call_async(func(od, enter))
            else:
                func(od, enter)

    def try_dump(self):
        if not self.dump_path:
            return
        result = dict(
            open_ods=[od.to_dict() for key, od in self.open_orders.items()],
            his_ods=[od.to_dict() for od in self.his_list],
        )
        with open(self.dump_path, 'wb') as fout:
            fout.write(orjson.dumps(result))

    def enter_order(self, strategy: str, pair: str, tag: str, cost: float, price: float):
        lock_key = f'{pair}_{tag}_{strategy}'
        if lock_key in self.open_orders:
            # 同一交易对，同一策略，同一信号，只允许一个订单
            return
        quote_cost = self.wallets.get_avaiable_by_cost(pair.split('/')[1], cost)
        if not quote_cost:
            return
        if TradeLock.get_pair_locks(lock_key):
            logger.warning(f'fail to create order: {lock_key} is locked')
            return
        amount = quote_cost / price
        od = InOutOrder(
            symbol=pair,
            enter_price=price,
            enter_amount=amount,
            enter_tag=tag,
            enter_at=bar_num.get(),
            strategy=strategy
        )
        self.open_orders[lock_key] = od
        TradeLock.lock(lock_key, btime.now() + btime.timedelta(minutes=1))
        if btime.run_mode == btime.RunMode.LIVE:
            cost = od.enter.price * od.enter.amount
            logger.info(f'order {od.symbol} {od.enter_tag} {od.enter.price} cost: {cost:.2f}')
        self._enter_order(od)

    def _enter_order(self, od: InOutOrder):
        pass

    def _fill_pending_enter(self, arr: np.ndarray, od: InOutOrder):
        enter_price = self._sim_market_price(arr)
        quote_amount = enter_price * od.enter.amount
        _, base_s, quote_s, timeframe = get_cur_symbol()
        self.wallets.update_wallets(**{quote_s: -quote_amount, base_s: od.enter.amount})
        od.status = InOutStatus.FullEnter
        od.enter_at = bar_num.get()
        od.enter.filled = od.enter.amount
        od.enter.average = enter_price
        od.enter.status = OrderStatus.Close
        if not od.enter.price:
            od.enter.price = enter_price
        # TODO: 根据交易对情况设置手续费
        TradeLock.unlock(od.key)
        self._fire(od.key, True)
        self.try_dump()

    def _fill_pending_exit(self, arr: np.ndarray, od: InOutOrder):
        exit_price = self._sim_market_price(arr)
        quote_amount = exit_price * od.enter.amount
        pair, base_s, quote_s, timeframe = get_cur_symbol()
        self.wallets.update_wallets(**{quote_s: quote_amount, base_s: -od.enter.amount})
        od.status = InOutStatus.FullExit
        od.exit_at = bar_num.get()
        od.update_exit(
            status=OrderStatus.Close,
            price=exit_price,
            filled=od.enter.amount,
            average=exit_price,
        )
        # TODO: 根据交易对情况设置手续费
        self._finish_order(od.strategy, pair, od.enter_tag)
        self._fire(od.key, False)
        self.try_dump()

    def _sim_market_price(self, arr: np.ndarray) -> float:
        '''
        计算从收到bar数据，到订单提交到交易所的时间延迟：对应的价格。
        :return:
        '''
        rate = min(1, self.network_cost / timeframe_secs.get())
        return arr[-1, 0] * (1 - rate) + arr[-1, 3] * rate

    def fill_pending_orders(self, arr: np.ndarray):
        '''
        填充等待交易所响应的订单。不可用于实盘；可用于回测、模拟实盘等。
        :param arr:
        :return:
        '''
        if btime.run_mode == btime.RunMode.LIVE:
            raise RuntimeError('fill_pending_orders unavaiable in LIVE mode')
        for od in list(self.open_orders.values()):
            if od.status == InOutStatus.Init:
                self._fill_pending_enter(arr, od)
            elif od.status == InOutStatus.FullEnter and od.exit_tag:
                self._fill_pending_exit(arr, od)

    def exit_open_orders(self, strategy: str, exit_tag: str, pair: str = None):
        if not self.open_orders:
            return
        for key, od in self.open_orders.items():
            if pair and not key.startswith(pair):
                continue
            if strategy and od.strategy != strategy:
                continue
            self.exit_order(od, exit_tag)

    def exit_order(self, od: InOutOrder, exit_tag: str):
        if od.exit_tag:
            return
        od.exit_tag = exit_tag
        # 默认以最低价卖出（即吃单方）
        od.update_exit(price=bar_arr.get()[-1, 2])
        if btime.run_mode == btime.RunMode.LIVE:
            cost = od.exit.price * od.exit.amount
            logger.info(f'exit {od.symbol} {od.exit_tag} {od.exit.price} got: {cost:.2f}')
        self._exit_order(od)

    def _exit_order(self, od: InOutOrder):
        '''
        调用此方法前，已将exit_tag写入到订单中
        :param od:
        :return:
        '''
        pass

    def _finish_order(self, strategy: str, pair: str, tag: str):
        od: InOutOrder = self.open_orders.pop(f'{pair}_{tag}_{strategy}')
        od.profit_rate = od.exit.price / od.enter.price - 1
        od.profit = (od.exit.price - od.enter.price) * od.enter.amount
        self.his_list.append(od)

    def update_by_bar(self, pair_arr: np.ndarray):
        if not self.open_orders:
            return
        # 更新订单利润
        for tag in self.open_orders:
            self.open_orders[tag].update_by_bar(pair_arr)

    def check_custom_exits(self, pair_arr: np.ndarray, strategy: BaseStrategy):
        if not self.open_orders:
            return
        pair, _, _, _ = get_cur_symbol()
        cur_strategy = strategy.__class__.__name__
        # 调用策略的自定义退出判断
        for od in list(self.open_orders.values()):
            if not od.can_close() or od.strategy != cur_strategy:
                continue
            if ext_tag := strategy.custom_exit(pair_arr, od):
                self.exit_order(od, ext_tag)


class LiveOrderManager(OrderManager):
    def __init__(self, exchange: CryptoExchange, wallets: CryptoWallet, dump_path: str = None):
        super(LiveOrderManager, self).__init__(exchange.name, wallets, dump_path)
        self.exchange = exchange

    async def _create_exg_order(self, od: InOutOrder, is_enter: bool):
        sub_od = od.enter if is_enter else od.exit
        side, amount, price = sub_od.side, sub_od.amount, sub_od.price
        order = await self.exchange.create_limit_order(od.symbol, side, amount, price)
        order_status, fee, filled = order.get('status'), order.get('fee'), float(order.get('filled', 0))
        if filled > 0:
            filled_price = safe_value_fallback(order, 'average', 'price', price)
            sub_od.update(average=filled_price, filled=filled, status=OrderStatus.PartOk)
            od.status = InOutStatus.PartEnter if is_enter else InOutStatus.PartExit
            if fee and fee.get('cost'):
                sub_od.fee = fee.get('cost')
                sub_od.fee_type = fee.get('currency')
        if order_status in {'expired', 'rejected', 'closed'}:
            sub_od.status = OrderStatus.Close
            od.status = InOutStatus.FullEnter if is_enter else InOutStatus.FullExit
            if filled == 0:
                logger.warning(f'{od} is {order_status} by {self.name}, no filled')
        if od.status > InOutStatus.Init:
            self._fire(od.key, is_enter)
            self.try_dump()

    async def _enter_order(self, od: InOutOrder):
        if btime.run_mode == btime.RunMode.LIVE:
            await self._create_exg_order(od, True)

    async def _exit_order(self, od: InOutOrder):
        if btime.run_mode == btime.RunMode.LIVE:
            await self._create_exg_order(od, False)

    def _update_bnb_order(self, od: Order, data: dict):
        info = data['info']
        state = info['X']
        if state == 'NEW':
            return
        if state in {'CANCELED', 'REJECTED', 'EXPIRED', 'EXPIRED_IN_MATCH'}:
            od.update(status=OrderStatus.Close)
        if state in {'FILLED', 'PARTIALLY_FILLED'}:
            od_status = OrderStatus.Close if state == 'FILLED' else OrderStatus.PartOk
            filled, total_cost, fee_val = float(data['z']), float(data['Z']), float(data['n'])
            kwargs = dict(status=od_status, order_type=data['o'], filled=filled, average=total_cost/filled)
            if not od.price and od_status == OrderStatus.Close:
                kwargs['price'] = kwargs['average']
            if data['N']:
                kwargs['fee_type'] = data['N']
            if fee_val:
                kwargs['fee'] = od.fee + fee_val
            od.update(**kwargs)
            if od_status == OrderStatus.Close:
                inout_od = self.open_orders[od.inout_key]
                if od.enter:
                    inout_od.status = InOutStatus.FullEnter
                else:
                    inout_od.status = InOutStatus.FullExit
        else:
            logger.error(f'unknown bnb order status: {state}, {data}')
            return
        self._fire(od.inout_key, od.enter)
        self.try_dump()

    def _update_order(self, od: Order, data: dict):
        if self.name.find('binance') >= 0:
            self._update_bnb_order(od, data)
        else:
            raise ValueError(f'unsupport exchange to update order: {self.name}')

    @loop_forever
    async def update_forever(self):
        orders = await self.exchange.watch_my_trades()
        for data in orders:
            key = f"{data['symbol']}_{data['order']}"
            if key not in self.exg_orders:
                logger.warning(f'update order {key} not found in {self.name}')
                continue
            od = self.exg_orders[key]
            self._update_order(od, data)
