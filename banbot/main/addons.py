#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : addons.py
# Author: anyongjin
# Date  : 2023/8/10


class MarketPrice:
    '''
    市场所有币对的最新价格
    '''
    bar_prices = dict()  # 来自bar的每个币的最新价格，仅用于回测等。键可以是交易对，也可以是币的code
    prices = dict()  # 交易对的最新订单簿价格，仅用于实时模拟或实盘。键可以是交易对，也可以是币的code

    @classmethod
    def get(cls, symbol: str) -> float:
        '''
        获取指定币种的价格
        '''
        from banbot.storage import BotGlobal
        is_pair = len(symbol.split('/')) > 1
        if BotGlobal.live_mode:
            if symbol in cls.prices:
                return cls.prices[symbol]
            if is_pair:
                cls.prices[symbol] = None  # 记录下，实盘时会定期更新最新价格
        if symbol in cls.bar_prices:
            return cls.bar_prices[symbol]
        elif not is_pair and symbol.find('USD') >= 0:
            return 1
        raise ValueError(f'unsupport quote symbol: {symbol}')

    @classmethod
    def pairs(cls):
        return [k for k in cls.prices if k.find('/') > 0]

    @classmethod
    def set_bar_price(cls, pair: str, price: float):
        cls.bar_prices[pair] = price
        if pair.find('USD') > 0:
            cls.bar_prices[pair.split('/')[0]] = price

    @classmethod
    def set_new_price(cls, pair: str, price: float):
        cls.prices[pair] = price
        if pair.find('USD') > 0:
            cls.prices[pair.split('/')[0]] = price
