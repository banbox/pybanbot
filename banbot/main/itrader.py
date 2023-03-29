#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : itrader.py
# Author: anyongjin
# Date  : 2023/3/17
from __future__ import annotations

from banbot.strategy.base import *
from banbot.config import *
from banbot.config import consts
from banbot.util.common import logger


class Trader:
    def __init__(self):
        self.strategy: Optional[BaseStrategy] = None
        self.pairlist: List[Tuple[str, str]] = cfg.get('pairlist')
        self.his_orders: List[Order] = []  # 历史订单
        self.open_orders: Dict[str, Order] = dict()  # 键是tag，值是字典
        self.stake_amount: float = cfg.get('stake_amount', 1000)
        self.wallets = dict()  # 当前钱包
        self._bar_listeners: List[Tuple[int, Callable]] = []
        self.cur_time = datetime.now(timezone.utc)
        # 每次切换交易对时更新
        self.pair = 'BTC/USDT'
        self.timeframe = ''
        self.base_symbol, self.stake_symbol = self.pair.split('/')

    def on_data_feed(self, row: np.ndarray):
        arr = np.expand_dims(row, 0)
        state = pair_state.get()
        if not bar_num.get():
            pair_arr = self.strategy.on_bar(arr)
            state['pad_len'] = pair_arr.shape[1] - arr.shape[1]
            bar_arr.set(pair_arr)
        else:
            pair_arr = np.append(bar_arr.get(), append_nan_cols(arr, state['pad_len']), axis=0)
            self.strategy.state = dict()
            # 计算指标
            pair_arr = self.strategy.on_bar(pair_arr)
            bar_arr.set(pair_arr)
            # 调用监听器
            self._fire_listeners()
            if self.open_orders:
                # 更新订单利润
                for tag in self.open_orders:
                    self.open_orders[tag].update_by_bar(pair_arr)
            # 调用策略生成入场和出场信号
            entry_tag = self.strategy.on_entry(pair_arr)
            exit_tag = self.strategy.on_exit(pair_arr)
            if entry_tag and not exit_tag:
                state['last_enter'] = bar_num.get()
                self.on_new_order(entry_tag)
            elif exit_tag:
                self.on_close_orders(exit_tag)
            if self.open_orders:
                # 调用策略的自定义退出判断
                for od in list(self.open_orders.values()):
                    if not od.can_close():
                        continue
                    if ext_tag := self.strategy.custom_exit(pair_arr, od):
                        self.close_order(od.enter_tag, ext_tag)

    def _fire_listeners(self):
        '''
        触发监听当前bar的回调函数
        :return:
        '''
        res_listeners = []
        for od_num, func in self._bar_listeners:
            if od_num == bar_num.get():
                func(bar_arr.get())
            elif od_num > bar_num.get():
                res_listeners.append((od_num, func))
        self._bar_listeners = res_listeners

    def _update_wallet(self, symbol: str, amount: float, is_frz=True):
        ava_val, frz_val = self.wallets.get(symbol)
        if amount > 0:
            # 增加钱包金额，不影响冻结值，直接更新
            # TODO: 取消订单时，可能需要增加可用余额，减少冻结金额
            ava_val += amount
        elif abs(frz_val / abs(amount) - 1) <= 0.02:
            ava_val = ava_val + frz_val + amount
            frz_val = 0
        else:
            ava_val += amount
            if is_frz:
                frz_val -= amount
        self.wallets[symbol] = (max(0, ava_val), max(0, frz_val))

    def update_wallets(self, **kwargs):
        '''
        更新钱包，可同时更新两个钱包，或只更新一个钱包：
        只更新一个钱包时，变化值记录为冻结。
        :param kwargs:
        :return:
        '''
        items = list(kwargs.items())
        assert 0 < len(items) <= 2, 'update wallets should be 2 keys'
        if len(items) == 2:
            # 同时更新2个钱包时，必须是一增一减
            (keya, vala), (keyb, valb) = items
            assert vala * valb < 0, 'two amount should different signs'
            self._update_wallet(keya, vala, False)
            self._update_wallet(keyb, valb, False)
        else:
            self._update_wallet(*items[0])

    def on_new_order(self, tag: str, stoploss: float = None, ):
        if tag in self.open_orders:
            # 同一交易对，同一信号，只允许一个订单
            return
        lock_key = f'{self.pair}_{tag}'
        if TradeLock.get_pair_locks(lock_key, self.cur_time):
            logger.warning(f'fail to create order: {lock_key} is locked')
            return
        # 记录到open_orders
        state = self.strategy.state
        score = state.get('entry_score', 1)
        stake_amount = round(self.stake_amount * score, 2)
        if stake_amount < consts.MIN_STAKE_AMOUNT:
            return
        ava_val, frz_val = self.wallets.get(self.stake_symbol)
        if ava_val < consts.MIN_STAKE_AMOUNT:
            return
        stake_amount = min(stake_amount, ava_val)
        pair_arr = bar_arr.get()
        self.open_orders[tag] = Order(
            symbol=self.pair,
            order_type=state.get('order_type', 'market'),
            amount=stake_amount / pair_arr[-1, 3],
            enter_tag=tag,
            enter_at=bar_num.get(),
            stoploss=stoploss
        )
        TradeLock.lock(lock_key, self.cur_time + timedelta(seconds=1))
        self._new_order(tag)

    def _new_order(self, tag: str):
        raise NotImplementedError('`_new_order` need to be implemented')

    def on_close_orders(self, exit_tag: str):
        if not self.open_orders:
            return
        for tag in self.open_orders:
            self.close_order(tag, exit_tag)

    def close_order(self, entry_tag: str, exit_tag: str):
        od: Order = self.open_orders.get(entry_tag)
        if not od or od.exit_tag:
            return
        od.exit_tag = exit_tag
        self._close_order(entry_tag)

    def _close_order(self, entry_tag: str):
        '''
        调用此方法前，已将exit_tag写入到订单中
        :param entry_tag:
        :return:
        '''
        raise NotImplementedError('`close_order` need to be implemented')

    def _finish_order(self, tag: str):
        od: Order = self.open_orders.pop(tag)
        od.profit_rate = od.stop_price / od.price - 1
        od.profit = (od.stop_price - od.price) * od.amount
        self.his_orders.append(od)

    def force_exit_all(self):
        exit_tag = 'force_exit'
        for tag in self.open_orders:
            self.close_order(tag, exit_tag)
        bar_num.set(bar_num.get() + 1)
        self._fire_listeners()

    def run(self, make_strategy):
        raise NotImplementedError('`run` is not implemented')

