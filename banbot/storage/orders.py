#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : trades.py
# Author: anyongjin
# Date  : 2023/3/21

from datetime import datetime
from banbot.compute.tainds import *
from banbot.util.misc import del_dict_prefix
from banbot.storage.base import *
from banbot.util.redis_helper import AsyncRedis, SyncRedis
from banbot.exchange.exchange_utils import tf_to_secs
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


class Order(BaseDbModel):
    '''
    交易所订单；一次买入（卖出）就会产生一个订单
    同一交易所下，symbol+order_id可唯一确定一个订单。
    '''

    __tablename__ = 'order'

    __table_args__ = (
        sa.Index('idx_od_inout_id', 'inout_id'),
        sa.Index('idx_od_status', 'status'),
    )

    id = Column(sa.Integer, primary_key=True)
    inout_id = Column(sa.Integer)
    symbol = Column(sa.String(50))
    enter = Column(sa.Boolean, default=False)
    order_type = Column(sa.String(50), default='limit')
    order_id = Column(sa.String(164))  # 交易所订单ID，如修改订单会变化，记录最新的值
    side = Column(sa.String(10))
    create_at = Column(sa.BIGINT)  # 创建时间，13位整型时间戳
    price = Column(sa.Float)  # 入场价格，市价单此项为空
    average = Column(sa.Float)  # 平均成交价格
    amount = Column(sa.Float)  # 交易量；这里无需扣除手续费，这里和实际钱包到账金额不同
    filled = Column(sa.Float)  # 已成交数量，这里不用扣除手续费，完全成交时和amount相等
    status = Column(sa.SMALLINT, default=OrderStatus.Init)
    fee = Column(sa.Float)
    fee_type = Column(sa.String(10))
    update_at = Column(sa.BIGINT)  # 上次更新的交易所时间戳，如果trade小于此值，则是旧的数据不更新

    async def lock(self):
        redis = AsyncRedis()
        return redis.lock(f'order_{self.id}', with_conn=True)

    def __str__(self):
        return f'{self.side} {self.amount:.5f} with {self.price}'


class InOutOrder(BaseDbModel):
    '''
    策略逻辑订单（包含入场、出场两个Order）
    为避免过度复杂，不支持市价单按定价金额买入（需按基准产品数量买入）
    一个交易所的所有订单维护在一个OrderManager中
    '''
    __tablename__ = 'inout_order'

    __table_args__ = (
        sa.Index('idx_io_task_id', 'task_id'),
        sa.Index('idx_io_status', 'status'),
    )

    id = Column(sa.Integer, primary_key=True)
    task_id = Column(sa.Integer)
    symbol = Column(sa.String(50))
    timeframe = Column(sa.String(5))
    status = Column(sa.SMALLINT, default=InOutStatus.Init)
    enter_tag = Column(sa.String(30))
    exit_tag = Column(sa.String(30))
    enter_at = Column(sa.BIGINT)  # 13位时间戳，和bar的时间戳保持一致
    exit_at = Column(sa.BIGINT)  # 13位时间戳，和bar的时间戳保持一致
    strategy = Column(sa.String(20))
    profit_rate = Column(sa.Float, default=0)
    profit = Column(sa.Float, default=0)

    @orm.reconstructor
    def __init__(self, **kwargs):
        super(InOutOrder, self).__init__(**kwargs)
        # 花费定价币数量，当价格不确定，amount可先不设置，后续通过此字段/价格计算amount
        self.quote_cost = kwargs.get('quote_cost')
        if self.id:
            sess = db.session
            rows = sess.query(Order).filter(Order.inout_id == self.id).all()
            self.enter = next((r for r in rows if r.enter), None)
            sess.exit = next((r for r in rows if not r.enter), None)
        else:
            enter_kwargs = del_dict_prefix(kwargs, 'enter_')
            self.enter: Order = Order(**enter_kwargs, enter=True)
            self.exit: Optional[Order] = None
            if 'exit_amount' in kwargs:
                exit_kwargs = del_dict_prefix(kwargs, 'exit_')
                self.exit = Order(**exit_kwargs)

    @property
    def key(self):
        return f'{self.symbol}_{self.enter_tag}_{self.strategy}'

    def _elp_num_offset(self, time_ms: int):
        ctx = get_context(f'{self.symbol}/{self.timeframe}')
        tf_secs = tf_to_secs(self.timeframe)
        return round((ctx[bar_arr][-1][0] - time_ms) / tf_secs / 1000)

    @property
    def elp_num_enter(self):
        return self._elp_num_offset(self.enter_at)

    @property
    def elp_num_exit(self):
        if not self.exit_at:
            return -1
        return self._elp_num_offset(self.exit_at)

    def can_close(self):
        if self.exit_tag:
            return False
        return self.elp_num_enter > 0

    def pending_type(self, timeouts: int):
        if self.exit and self.exit_tag:
            if btime.time() - self.exit.create_at < timeouts or self.exit.status == OrderStatus.Close:
                # 尚未超时，或者订单已关闭，本次不处理
                return
            return 'exit'
        elif self.enter.status != OrderStatus.Close:
            if btime.time() - self.enter.create_at < timeouts:
                # 尚未超时，本次不处理
                return
            return 'enter'

    def update_exit(self, **kwargs):
        if not self.exit:
            kwargs.update(dict(
                symbol=self.enter.symbol,
                inout_key=self.enter.inout_key,
                side='sell' if self.enter.side == 'buy' else 'buy',
            ))
            if not kwargs.get('amount'):
                # 未提供时，默认全部卖出。（这里模拟手续费扣除）
                kwargs['amount'] = self.enter.filled * (1 - self.enter.fee)
            self.exit = Order(**kwargs)
        else:
            self.exit.update_props(**kwargs)

    def update_by_bar(self, arr: np.ndarray):
        '''
        此方法由接口调用，策略中不应该调用此方法。
        :param arr:
        :return:
        '''
        if not self.status or self.status == InOutStatus.FullExit:
            return
        # TODO: 当定价货币不是USD时，这里需要计算对应USD的利润
        self.profit = (arr[-1, ccol] - self.enter.price) * self.enter.amount
        self.profit_rate = arr[-1, ccol] / self.enter.price - 1

    def save(self):
        sess = db.session
        if not self.id:
            sess.add(self)
            sess.flush()
        if self.enter and not self.enter.id:
            if not self.enter.inout_id:
                self.enter.inout_id = self.id
            sess.add(self.enter)
        if self.exit and not self.exit.id:
            if not self.exit.inout_id:
                self.exit.inout_id = self.id
            sess.add(self.exit)
        sess.commit()
        with SyncRedis() as redis:
            redis.set(self.key, self.id)

    @classmethod
    def get_orders(cls, strategy: str = None, pair: str = None, filters: List = None) -> List['InOutOrder']:
        from banbot.storage.bot_task import BotTask
        sess = db.session
        where_list = [InOutOrder.task_id == BotTask.cur_id]
        if filters:
            where_list.extend(filters)
        if strategy:
            where_list.append(InOutOrder.strategy == strategy)
        if pair:
            where_list.append(InOutOrder.symbol == pair)
        return sess.query(InOutOrder).filter(*where_list).all()

    @classmethod
    def open_orders(cls, strategy: str = None, pair: str = None) -> List['InOutOrder']:
        filters = [InOutOrder.status < InOutStatus.FullExit]
        return cls.get_orders(strategy, pair, filters)

    @classmethod
    def his_orders(cls) -> List['InOutOrder']:
        return cls.get_orders(filters=[InOutOrder.status == InOutStatus.FullExit])

    @classmethod
    def is_lock(cls, symbol: str, strategy: str, enter_tag: str) -> int:
        key = f'{symbol}_{enter_tag}_{strategy}'
        with SyncRedis() as redis:
            return redis.get(key) or 0

    def del_lock(self):
        with SyncRedis() as redis:
            return redis.delete(self.key)

    def __str__(self):
        return f'[{self.key}] {self.enter} || {self.exit}'


class OrderJob:
    def __init__(self, od_id: int, is_enter: bool):
        self.od_id = od_id
        self.is_enter = is_enter

