#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : trades.py
# Author: anyongjin
# Date  : 2023/3/21

from banbot.compute.utils import logger
from datetime import datetime, timezone, timedelta
from typing import *


class OrderStatus:
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
        self.lock_time = datetime.now(timezone.utc)
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
    def get_pair_locks(key: Optional[str], now: Optional[datetime] = None, side: str = '*'):
        if not now:
            now = datetime.now(timezone.utc)
        return [lock for lock in TradeLock._locks if (
                lock.unlock_time >= now
                and (key is None or lock.key == key)
                and (lock.side == '*' or lock.side == side)
        )]

    @staticmethod
    def unlock(key: str, now: Optional[datetime] = None, side: str = '*'):
        if not now:
            now = datetime.now(timezone.utc)
        locks = TradeLock.get_pair_locks(key, now, side)
        for l in locks:
            TradeLock._locks.remove(l)


class Order:
    '''
    策略逻辑订单（包含入场、出场等子订单）
    为避免过度复杂，不支持市价单按定价金额买入（需按基准产品数量买入）
    '''
    def __init__(self, **kwargs):
        self.symbol: str = kwargs['symbol']
        od_type = kwargs.get('order_type')
        price = kwargs.get('price')
        if not od_type:
            od_type = 'limit' if price else 'market'
        elif od_type == 'market' and price:
            logger.warning('`price` ignored for market order!')
        elif od_type == 'limit' and not price:
            raise ValueError('`price` is required for limit order')
        self.order_type: str = od_type
        self.side: str = kwargs.get('side', 'buy')
        self.price: float = price  # 入场价格，市价单此项为空
        self.amount: float = kwargs['amount']  # 交易量，等同于volume；
        self.enter_tag: str = kwargs.get('enter_tag')
        self.enter_at: int = kwargs.get('enter_at')
        # 下面属性可能需要更新
        self.status: int = kwargs.get('status', OrderStatus.Init)
        self.filled: float = kwargs.get('filled')  # 已成交数量
        self.average: float = kwargs.get('average')  # 平均成交价格
        self.exit_tag: str = kwargs.get('exit_tag')
        self.exit_at: int = kwargs.get('exit_at')
        self.stop_price: float = kwargs.get('stop_price')
        self.funding_fee: float = kwargs.get('funding_fee', 0)

        self.stoploss: float = kwargs.get('stoploss')
        self.profit_rate: float = kwargs.get('profit_rate', 0.0)
        self.profit: float = kwargs.get('profit', 0.0)

    def can_close(self):
        return self.status > OrderStatus.Init and not self.exit_tag

    def to_dict(self) -> dict:
        return dict(
            symbol=self.symbol,
            order_type=self.order_type,
            side=self.side,
            price=self.price,
            amount=self.amount,
            enter_tag=self.enter_tag,
            enter_at=self.enter_at,
            status=self.status,
            filled=self.filled,
            average=self.average,
            exit_tag=self.exit_tag,
            exit_at=self.exit_at,
            stop_price=self.stop_price,
            funding_fee=self.funding_fee,
            profit_rate=self.profit_rate,
            profit=self.profit,
            duration=self.exit_at - self.enter_at
        )

    def update(self, **kwargs):
        self.status: int = kwargs.get('status', self.status)
        self.filled: float = kwargs.get('filled', self.filled)  # 已成交数量
        self.average: float = kwargs.get('average', self.average)  # 平均成交价格
        self.exit_tag: str = kwargs.get('exit_tag', self.exit_tag)
        self.exit_at: int = kwargs.get('exit_at', self.exit_at)
        self.stop_price: float = kwargs.get('stop_price', self.stop_price)
        self.funding_fee: float = kwargs.get('funding_fee', self.funding_fee)

        self.stoploss: float = kwargs.get('stoploss', self.stoploss)
        self.profit_rate: float = kwargs.get('profit_rate', self.profit_rate)
        self.profit: float = kwargs.get('profit', self.profit)
