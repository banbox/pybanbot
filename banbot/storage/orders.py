#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : trades.py
# Author: anyongjin
# Date  : 2023/3/21

from banbot.util import btime
from datetime import datetime
from banbot.bar_driven.tainds import *
from banbot.util.misc import del_dict_prefix, add_dict_prefix
from banbot.util.num_utils import to_pytypes
from typing import *

import numpy as np


class OrderStatus:
    Init = 0
    PartOk = 1
    Close = 2  # cancel, expired, close; 部分成交也可能是这个状态


class InOutStatus:
    Init = 0
    PartEnter = 1
    FullEnter = 2
    PartExit = 3
    FullExit = 4


class TradeLock:
    _locks: List['TradeLock'] = []

    def __init__(self, **kwargs):
        self.key: str = kwargs['key']
        self.side: str = kwargs.get('side')
        self.reason: str = kwargs.get('reason')
        self.unlock_time: Optional[datetime] = kwargs.get('unlock_time')

    @staticmethod
    def reset_locks():
        TradeLock._locks = []

    @staticmethod
    def lock(pair: str, until: datetime, side: str = '*', reason: Optional[str] = None):
        obj = TradeLock(key=pair, side=side, unlock_time=until, reason=reason)
        TradeLock._locks.append(obj)
        return obj

    @staticmethod
    def get_pair_locks(key: Optional[str], side: str = '*'):
        now = btime.now()
        return [lock for lock in TradeLock._locks if (
                lock.unlock_time >= now
                and (key is None or lock.key == key)
                and (lock.side == '*' or lock.side == side)
        )]

    @staticmethod
    def unlock(key: str, side: str = '*'):
        locks = TradeLock.get_pair_locks(key, side)
        for l in locks:
            TradeLock._locks.remove(l)


class Order:
    '''
    交易所订单；一次买入（卖出）就会产生一个订单
    同一交易所下，symbol+order_id可唯一确定一个订单。
    '''
    def __init__(self, **kwargs):
        self.symbol: str = kwargs['symbol']
        od_type = kwargs.get('order_type')
        price = kwargs.get('price')
        if not od_type:
            od_type = 'limit' if price else 'market'
        elif od_type == 'limit' and not price:
            raise ValueError('`price` is required for limit order')
        self.order_type: str = od_type
        self.enter: bool = kwargs.get('enter', False)  # 是否是入场侧订单
        self.order_id: str = kwargs.get('order_id')
        self.inout_key: str = kwargs.get('inout_key')
        self.side: str = kwargs.get('side', 'buy')
        self.amount: float = kwargs['amount']  # 交易量，等同于volume；
        # 下面属性可能需要更新
        self.price: float = price  # 入场价格，市价单此项为空
        self.status: int = kwargs.get('status', OrderStatus.Init)
        self.filled: float = kwargs.get('filled')  # 已成交数量
        self.average: float = kwargs.get('average')  # 平均成交价格
        self.fee: float = kwargs.get('fee', 0)
        self.fee_type: str = kwargs.get('fee_type')

    def update(self, **kwargs):
        self.status: int = kwargs.get('status', self.status)
        self.order_type: str = kwargs.get('order_type', self.order_type)
        self.price: int = kwargs.get('price', self.price)
        self.filled: float = kwargs.get('filled', self.filled)  # 已成交数量
        self.average: float = kwargs.get('average', self.average)  # 平均成交价格
        self.fee: float = kwargs.get('fee', self.fee)
        self.fee_type: str = kwargs.get('fee_type')

    def to_dict(self) -> dict:
        return dict(
            symbol=self.symbol,
            order_id=self.order_id,
            order_type=self.order_type,
            inout_key=self.inout_key,
            side=self.side,
            price=to_pytypes(self.price),
            amount=to_pytypes(self.amount),
            status=self.status,
            filled=to_pytypes(self.filled),
            average=to_pytypes(self.average),
            fee=to_pytypes(self.fee),
            fee_type=self.fee_type,
            cost=to_pytypes(self.price * self.amount)
        )

    def __str__(self):
        return f'{self.side} {self.amount:.5f} with {self.price}'


class InOutOrder:
    '''
    策略逻辑订单（包含入场、出场两个Order）
    为避免过度复杂，不支持市价单按定价金额买入（需按基准产品数量买入）
    一个交易所的所有订单维护在一个OrderManager中
    '''

    def __init__(self, **kwargs):
        self.symbol: str = kwargs['symbol']
        self.status: int = kwargs.get('status', InOutStatus.Init)
        self.enter_tag: str = kwargs.get('enter_tag')
        self.enter_at: int = kwargs.get('enter_at')
        self.exit_tag: str = kwargs.get('exit_tag')
        self.exit_at: int = kwargs.get('exit_at')
        self.timestamp = btime.time()
        self.strategy: str = kwargs.get('strategy')
        self.key = f'{self.symbol}_{self.enter_tag}_{self.strategy}'
        enter_kwargs = del_dict_prefix(kwargs, 'enter_')
        self.enter: Order = Order(**enter_kwargs, inout_key=self.key, enter=True)
        # exit_kwargs = del_dict_prefix(kwargs, 'exit_')
        self.exit: Optional[Order] = None

        self.stoploss: float = kwargs.get('stoploss')
        self.profit_rate: float = kwargs.get('profit_rate', 0.0)
        self.profit: float = kwargs.get('profit', 0.0)

    def can_close(self):
        return self.status > InOutStatus.Init and not self.exit_tag and bar_num.get() > self.enter_at

    def to_dict(self) -> dict:
        result = dict(
            symbol=self.symbol,
            status=self.status,
            key=self.key,
            enter_tag=self.enter_tag,
            enter_at=self.enter_at,
            exit_tag=self.exit_tag,
            exit_at=self.exit_at,
            timestamp=self.timestamp,
            stoploss=self.stoploss,
            profit_rate=self.profit_rate,
            profit=self.profit,
            duration=self.exit_at - self.enter_at,
        )
        result.update(add_dict_prefix(self.enter.to_dict(), 'enter_'))
        if self.exit:
            result.update(add_dict_prefix(self.exit.to_dict(), 'exit_'))
        return result

    def update_enter(self, **kwargs):
        self.enter.update(**kwargs)

    def update_exit(self, **kwargs):
        if not self.exit:
            kwargs.update(dict(
                symbol=self.enter.symbol,
                inout_key=self.enter.inout_key,
                side='sell' if self.enter.side == 'buy' else 'buy',
            ))
            if not kwargs.get('amount'):
                # 未提供时，默认全部卖出
                kwargs['amount'] = self.enter.amount
            self.exit = Order(**kwargs)
        else:
            self.exit.update(**kwargs)

    def update(self, **kwargs):
        self.status: int = kwargs.get('status', self.status)
        self.exit_tag: str = kwargs.get('exit_tag', self.exit_tag)
        self.exit_at: int = kwargs.get('exit_at', self.exit_at)

        self.stoploss: float = kwargs.get('stoploss', self.stoploss)
        self.profit_rate: float = kwargs.get('profit_rate', self.profit_rate)
        self.profit: float = kwargs.get('profit', self.profit)

    def update_by_bar(self, arr: np.ndarray):
        '''
        此方法由接口调用，策略中不应该调用此方法。
        :param arr:
        :return:
        '''
        if self.status in {InOutStatus.Init, InOutStatus.FullExit}:
            return
        # TODO: 当定价货币不是USD时，这里需要计算对应USD的利润
        self.profit = (arr[-1, ccol] - self.enter.price) * self.enter.amount
        self.profit_rate = arr[-1, ccol] / self.enter.price - 1

    def __str__(self):
        return f'[{self.key}] {self.enter} || {self.exit}'
