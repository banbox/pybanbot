#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : whack_mole.py
# Author: anyongjin
# Date  : 2023/2/28
from banbot.exchange.bnbscan import *


class LiveTrader(BnbScan):
    '''
    超短线策略。针对币安BTC交易无手续费。
    均值回归：
      成交量剧增&价格猛烈变化
      接下来5~10周期价格逐渐回归

    下降周期&突然巨量向上，是买入信号

    5周期均值向上突破，震荡较小，前期属于熊市，是买入信号
    '''

    def __init__(self):
        super(LiveTrader, self).__init__()

    def on_data_feed(self, last_secs, tags: List):
        if 'kline' not in tags:
            return
        if self.is_first:
            self.arr, ptn = self.strategy.on_bar(self.klines[-1:])
            self.pad_len = self.arr.shape[1] - self.klines.shape[1]
            self.is_first = False
        else:
            self.arr = np.append(self.arr, append_nan_cols(self.klines[-1:], self.pad_len), axis=0)
            self.arr, ptn = self.strategy.on_bar(self.arr)
            tag = self.strategy.on_entry(self.arr)


