#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : wallets.py
# Author: anyongjin
# Date  : 2023/3/29

from numbers import Number
from typing import *
from dataclasses import dataclass

import ccxt

from banbot.config.consts import MIN_STAKE_AMOUNT
from banbot.exchange.crypto_exchange import CryptoExchange, loop_forever
from banbot.util import btime
from banbot.util.common import logger


@dataclass
class ItemWallet:
    available: float = 0
    '可用余额'
    pending: float = 0
    '买入卖出时锁定金额'
    frozen: float = 0
    '空单等长期冻结金额'

    @property
    def total(self):
        return self.available + self.pending + self.frozen


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

    def cost_ava(self, key: str, amount: float, negative: bool = False, after_ts: float = 0,
                 min_rate: float = 0.1) -> float:
        '''
        从某个币的可用余额中扣除，仅用于回测
        :param key: 币代码
        :param amount: 金额
        :param negative: 是否允许负数余额（空单用到）
        :param after_ts: 是否要求更新时间戳
        :param min_rate: 最低扣除比率
        :return 实际扣除数量
        '''
        assert self.update_at - after_ts >= -1, f'wallet expired, expect > {after_ts}, current: {self.update_at}'
        if key not in self.data:
            self.data[key] = ItemWallet()
        wallet = self.data[key]
        src_amount = wallet.available
        if src_amount >= amount or negative:
            # 余额充足，或允许负数，直接扣除
            real_cost = amount
        elif src_amount / amount > min_rate:
            # 差额在近似允许范围内，扣除实际值
            real_cost = src_amount
        else:
            return 0
        if key.find('USD') >= 0 and real_cost < MIN_STAKE_AMOUNT:
            return 0
        wallet.available -= real_cost
        wallet.pending += real_cost
        self.update_at = btime.time()
        return real_cost

    def cost_frozen(self, key: str, amount: float, after_ts: float = 0):
        '''
        从frozen中扣除，如果不够，从available扣除剩余部分
        扣除后，添加到pending中
        '''
        assert self.update_at - after_ts >= -1, f'wallet expired, expect > {after_ts}, current: {self.update_at}'
        if key not in self.data:
            return 0
        wallet = self.data[key]
        real_cost = amount
        wallet.frozen -= real_cost
        if wallet.frozen < 0:
            wallet.available += wallet.frozen
            wallet.frozen = 0
        if wallet.available < 0:
            real_cost += wallet.available
            wallet.available = 0
        wallet.pending += real_cost
        self.update_at = btime.time()
        return real_cost

    def confirm_pending(self, src_key: str, src_amount: float, tgt_key: str, tgt_amount: float,
                        to_frozen: bool = False):
        '''
        从src中确认扣除，添加到tgt的余额中
        '''
        self.update_at = btime.time()
        src, tgt = self.data.get(src_key), self.data.get(tgt_key)
        if not src or not tgt:
            return False
        if src.pending >= src_amount:
            src.pending -= src_amount
        elif src.pending / src_amount > 0.999:
            src.pending = 0
        else:
            return False
        if to_frozen:
            tgt.frozen += tgt_amount
        else:
            tgt.available += tgt_amount
        return True

    def cancel(self, symbol: str, amount: float, from_pending: bool = True):
        '''
        取消对币种的数量锁定(frozen/pending)，重新加到available上
        '''
        self.update_at = btime.time()
        wallet = self.data.get(symbol)
        if not wallet:
            return
        src_amount = wallet.pending if from_pending else wallet.frozen
        if src_amount >= amount:
            real_amount = amount
        elif src_amount:
            real_amount = src_amount
        else:
            return
        wallet.available += real_amount
        if from_pending:
            wallet.pending -= real_amount
        else:
            wallet.frozen -= real_amount

    def get(self, symbol: str, after_ts: float = 0):
        assert self.update_at - after_ts >= -1, f'wallet ts expired: {self.update_at} > {after_ts}'
        if symbol not in self.data:
            self.data[symbol] = ItemWallet()
        return self.data[symbol]

    def _get_symbol_price(self, symbol: str):
        if symbol in self.prices:
            return self.prices[symbol]
        raise ValueError(f'unsupport quote symbol: {symbol}')

    def get_amount_by_legal(self, symbol: str, legal_cost: float):
        '''
        根据花费的USDT计算需要的数量，并返回可用数量
        :param symbol: 产品，不是交易对。如：USDT
        :param legal_cost: 花费法币金额（一般是USDT）
        '''
        if symbol.find('USD') >= 0:
            return legal_cost
        else:
            price = self._get_symbol_price(symbol)
            return legal_cost / price


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
            self.data[symbol] = ItemWallet(available=free, pending=used)
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

