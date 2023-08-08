#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : wallets.py
# Author: anyongjin
# Date  : 2023/3/29

from numbers import Number
from typing import *
from dataclasses import dataclass, field

import ccxt

from banbot.config.consts import MIN_STAKE_AMOUNT
from banbot.exchange.crypto_exchange import CryptoExchange, loop_forever
from banbot.util import btime
from banbot.util.common import logger


@dataclass
class ItemWallet:
    available: float = 0
    '可用余额'
    pendings: Dict[str, float] = field(default_factory=dict)
    '买入卖出时锁定金额，键可以是订单id'
    frozens: Dict[str, float] = field(default_factory=dict)
    '空单等长期冻结金额，键可以是订单id'

    @property
    def total(self):
        sum_val = self.available
        for k, v in self.pendings.items():
            sum_val += v
        for k, v in self.frozens.items():
            sum_val += v
        return sum_val


class WalletsLocal:
    def __init__(self):
        self.data: Dict[str, ItemWallet] = dict()
        self.update_at = btime.time()
        self.prices = dict()  # 保存各个币相对USD的价格，仅用于回测，从od_manager更新

    def set_wallets(self, **kwargs):
        for key, val in kwargs.items():
            if isinstance(val, ItemWallet):
                self.data[key] = val
            elif isinstance(val, (int, float)):
                self.data[key] = ItemWallet(available=val)
            else:
                raise ValueError(f'unsupport val type: {key} {type(val)}')

    def cost_ava(self, lock_key: str, symbol: str, amount: float, negative: bool = False, after_ts: float = 0,
                 min_rate: float = 0.1) -> float:
        '''
        从某个币的可用余额中扣除，仅用于回测
        :param lock_key: 锁定的键
        :param symbol: 币代码
        :param amount: 金额
        :param negative: 是否允许负数余额（空单用到）
        :param after_ts: 是否要求更新时间戳
        :param min_rate: 最低扣除比率
        :return 实际扣除数量
        '''
        assert self.update_at - after_ts >= -1, f'wallet expired, expect > {after_ts}, current: {self.update_at}'
        if symbol not in self.data:
            self.data[symbol] = ItemWallet()
        wallet = self.data[symbol]
        src_amount = wallet.available
        if src_amount >= amount or negative:
            # 余额充足，或允许负数，直接扣除
            real_cost = amount
        elif src_amount / amount > min_rate:
            # 差额在近似允许范围内，扣除实际值
            real_cost = src_amount
        else:
            return 0
        if symbol.find('USD') >= 0 and real_cost < MIN_STAKE_AMOUNT:
            return 0
        wallet.available -= real_cost
        wallet.pendings[lock_key] = real_cost
        self.update_at = btime.time()
        return real_cost

    def cost_frozen(self, lock_key: str, symbol: str, amount: float, after_ts: float = 0):
        '''
        从frozen中扣除，如果不够，从available扣除剩余部分
        扣除后，添加到pending中
        '''
        assert self.update_at - after_ts >= -1, f'wallet expired, expect > {after_ts}, current: {self.update_at}'
        if symbol not in self.data:
            return 0
        wallet = self.data[symbol]
        frozen_amt = wallet.frozens.get(lock_key, 0)
        if frozen_amt:
            del wallet.frozens[lock_key]
        # 将冻结的剩余部分归还到available，正负都有可能
        wallet.available += frozen_amt - amount
        real_cost = amount
        if wallet.available < 0:
            real_cost += wallet.available
            wallet.available = 0
        wallet.pendings[lock_key] = real_cost
        self.update_at = btime.time()
        return real_cost

    def confirm_pending(self, lock_key: str, src_key: str, src_amount: float, tgt_key: str, tgt_amount: float,
                        to_frozen: bool = False):
        '''
        从src中确认扣除，添加到tgt的余额中
        '''
        self.update_at = btime.time()
        src, tgt = self.data.get(src_key), self.data.get(tgt_key)
        if not src or not tgt:
            return False
        pending_amt = src.pendings.get(lock_key, 0)
        if not pending_amt:
            return False
        left_pending = pending_amt - src_amount
        del src.pendings[lock_key]
        src.available += left_pending  # 剩余pending归还到available，（正负都可能）
        if to_frozen:
            tgt.frozens[lock_key] = tgt_amount
        else:
            tgt.available += tgt_amount
        return True

    def cancel(self, lock_key: str, symbol: str, from_pending: bool = True):
        '''
        取消对币种的数量锁定(frozens/pendings)，重新加到available上
        '''
        self.update_at = btime.time()
        wallet = self.data.get(symbol)
        if not wallet:
            return
        src_dic = wallet.pendings if from_pending else wallet.frozens
        src_amount = src_dic.get(lock_key)
        if not src_amount:
            return
        del src_dic[lock_key]
        wallet.available += src_amount

    def get(self, symbol: str, after_ts: float = 0):
        assert self.update_at - after_ts >= -1, f'wallet ts expired: {self.update_at} > {after_ts}'
        if symbol not in self.data:
            self.data[symbol] = ItemWallet()
        return self.data[symbol]

    def _get_symbol_price(self, symbol: str):
        if symbol in self.prices:
            return self.prices[symbol]
        if symbol.find('USD') >= 0:
            return 1
        raise ValueError(f'unsupport quote symbol: {symbol}')

    def get_amount_by_legal(self, symbol: str, legal_cost: float):
        '''
        根据花费的USDT计算需要的数量，并返回可用数量
        :param symbol: 产品，不是交易对。如：USDT
        :param legal_cost: 花费法币金额（一般是USDT）
        '''
        price = self._get_symbol_price(symbol)
        return legal_cost / price

    def total_legal(self, symbols: Iterable[str] = None):
        legal_sum = 0
        if symbols:
            data = {k: self.data[k] for k in symbols if k in self.data}
        else:
            data = self.data
        for key, item in data.items():
            legal_sum += item.total * self._get_symbol_price(key)
        return legal_sum

    def __str__(self):
        from io import StringIO
        builder = StringIO()
        for key, item in self.data.items():
            pend_sum = sum([v for k, v in item.pendings.items()])
            frozen_sum = sum([v for k, v in item.frozens.items()])
            builder.write(f"{key}: {item.available:.4f}|{pend_sum:.4f}|{frozen_sum:.4f} ")
        return builder.getvalue()


class CryptoWallet(WalletsLocal):
    def __init__(self, config: dict, exchange: CryptoExchange):
        super(CryptoWallet, self).__init__()
        self.exchange = exchange
        self.config = config
        self._symbols = set()

    def _get_symbol_price(self, symbol: str):
        if symbol not in self.exchange.quote_prices:
            raise ValueError(f'unsupport quote symbol: {symbol}')
        return self.exchange.quote_prices[symbol]

    def _update_local(self, balances: dict):
        message = []
        for symbol in self._symbols:
            state = balances[symbol]
            free, used = state['free'], state['used']
            self.data[symbol] = ItemWallet(available=free, pendings={'*': used})
            if free + used < 0.00001:
                continue
            message.append(f'{symbol}: {free}/{used}')
        return '  '.join(message)

    async def init(self, pairs: List[str]):
        for p in pairs:
            self._symbols.update(p.split('/'))
        balances = await self.exchange.fetch_balance()
        self.update_at = btime.time()
        logger.info('load balances: %s', self._update_local(balances))

    @loop_forever
    async def update_forever(self):
        try:
            balances = await self.exchange.watch_balance()
        except ccxt.NetworkError as e:
            logger.error(f'watch balance net error: {e}')
            return
        self.update_at = btime.time()
        logger.info('update balances: %s', self._update_local(balances))

