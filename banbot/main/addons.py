#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : addons.py
# Author: anyongjin
# Date  : 2023/8/10
from banbot.storage import BotGlobal


class MarketPrice:
    '''
    市场所有币对的最新价格
    '''
    bar_prices = dict()  # 来自bar的每个币的最新价格，仅用于回测等
    prices = dict()  # 交易对的最新订单簿价格，仅用于实时模拟或实盘

    @classmethod
    def get(cls, symbol: str) -> float:
        '''
        获取指定币种的价格
        '''
        if BotGlobal.live_mode:
            if symbol in cls.prices:
                return cls.prices[symbol]
            cls.prices[symbol] = None  # 记录下，实盘时会定期更新最新价格
        if symbol in cls.bar_prices:
            return cls.bar_prices[symbol]
        raise ValueError(f'unsupport quote symbol: {symbol}')
