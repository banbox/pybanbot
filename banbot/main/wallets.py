#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : wallets.py
# Author: anyongjin
# Date  : 2023/3/29
from banbot.exchange.crypto_exchange import CryptoExchange
from banbot.util.common import logger


class WalletsLocal:
    def __init__(self):
        self.data = dict()

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

    def get(self, symbol: str):
        if symbol not in self.data:
            return 0, 0
        return self.data[symbol]


class CryptoWallet(WalletsLocal):
    def __init__(self, config: dict, exchange: CryptoExchange):
        super(CryptoWallet, self).__init__()
        self.exchange = exchange
        self.config = config
        pairlist = config.get('pairlist')
        self._symbols = set()
        for pair, _ in pairlist:
            a, b = pair.split('/')
            self._symbols.add(a)
            self._symbols.add(b)

    async def init(self):
        balances = await self.exchange.fetch_balance()
        message = []
        for symbol in self._symbols:
            state = balances[symbol]
            self.data[symbol] = state['free'], state['used']
            message.append(f'{symbol}: {self.data[symbol][0]}/{self.data[symbol][1]}')
        logger.info(f'load balances: {"  ".join(message)}')
