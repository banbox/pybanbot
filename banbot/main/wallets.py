#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : wallets.py
# Author: anyongjin
# Date  : 2023/3/29

from banbot.exchange.crypto_exchange import CryptoExchange, loop_forever
from banbot.util.common import logger
from banbot.util import btime
from banbot.config.consts import MIN_STAKE_AMOUNT
from numbers import Number
from typing import List


class WalletsLocal:
    def __init__(self):
        self.data = dict()
        self.update_at = btime.time()

    def set_wallets(self, **kwargs):
        for key, val in kwargs.items():
            if isinstance(val, (tuple, list)):
                assert len(val) == 2
                self.data[key] = val
            elif isinstance(val, Number):
                self.data[key] = val, 0
            else:
                raise ValueError(f'unsupport val type: {key} {type(val)}')

    def _update_wallet(self, symbol: str, amount: float, is_frz=True):
        ava_val, frz_val = self.data.get(symbol)
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
        self.data[symbol] = (max(0, ava_val), max(0, frz_val))

    def update_wallets(self, **kwargs):
        '''
        【目前仅在回测、模拟实盘等模式下使用】
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
            assert vala * valb < 0, f'two amount should different signs {vala} - {valb}'
            self._update_wallet(keya, vala, False)
            self._update_wallet(keyb, valb, False)
        else:
            self._update_wallet(*items[0])
        self.update_at = btime.time()

    def get(self, symbol: str, after_ts: int = 0):
        assert self.update_at - after_ts >= -1, f'wallet ts expired: {self.update_at} > {after_ts}'
        if symbol not in self.data:
            return 0, 0
        return self.data[symbol]

    def _get_symbol_price(self, symbol: str):
        raise ValueError(f'unsupport quote symbol: {symbol}')

    def get_avaiable_by_cost(self, symbol: str, cost: float, after_ts: int = 0):
        '''
        根据花费的USDT计算需要的数量，并返回可用数量
        :param symbol:
        :param cost:
        :param after_ts:
        :return:
        '''
        assert self.update_at - after_ts >= -1, f'wallet ava ts expired: {self.update_at} > {after_ts}'
        if symbol.find('USD') >= 0:
            price = 1
        else:
            price = self._get_symbol_price(symbol)
        req_amount = (cost * 0.99) / price
        ava_val, frz_val = self.get(symbol)
        fin_amount = min(req_amount, ava_val)
        if fin_amount < MIN_STAKE_AMOUNT:
            return 0
        return fin_amount


class CryptoWallet(WalletsLocal):
    def __init__(self, config: dict, exchange: CryptoExchange):
        super(CryptoWallet, self).__init__()
        self.exchange = exchange
        self.config = config
        self._symbols = set()

    def _get_symbol_price(self, symbol: str):
        if symbol not in self.exchange.quote_symbols:
            raise ValueError(f'unsupport quote symbol: {symbol}')
        return self.exchange.quote_symbols[symbol]

    def _update_local(self, balances: dict):
        message = []
        for symbol in self._symbols:
            state = balances[symbol]
            free, used = state['free'], state['used']
            self.data[symbol] = free, used
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
        balances = await self.exchange.watch_balance()
        self.update_at = btime.time()
        logger.info('update balances: %s', self._update_local(balances))

