#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : oversold_up.py
# Author: anyongjin
# Date  : 2023/4/11

from banbot.strategy.base import *


class OverSoldUp(BaseStrategy):
    '''
    超跌反弹策略。当短期内连续下跌，且反弹超过一定比例时，入场。
    '''
    warmup_num = 600

    def __init__(self, config: dict):
        super(OverSoldUp, self).__init__(config)
        self.rsi = StaRSI(14)
        self.macd = StaMACD(7, 20, 14)
        self.atr = StaATR()

    def on_bar(self, arr: np.ndarray):
        ccolse = arr[-1, ccol]
        self.rsi(ccolse)
        self.macd(ccolse)
        self.atr(arr)

    def on_entry(self, arr: np.ndarray) -> Optional[str]:
        if len(self.rsi) < 10 or np.isnan(self.rsi[-10]):
            return
        macd_up = self.macd.macd_arr[-1] > self.macd.singal_arr[-1]
        rsi_btm = min(self.rsi[-5:]) <= 40 or min(self.rsi[-10:]) <= 33
        rsi_up = self.rsi[-1] >= self.rsi[-2]
        if rsi_btm and rsi_up and macd_up:
            return 'up'

    def custom_exit(self, arr: np.ndarray, od: InOutOrder) -> Optional[str]:
        elp_num = od.elp_num_enter
        atr_val = self.atr[- elp_num - 1]
        loss_thres = [-atr_val, -0.5 * atr_val, atr_val, 2 * atr_val, 3 * atr_val]
        return trail_stop_loss(arr, od.init_price, elp_num, odlens=[4, 8, 15, 30],
                               loss_thres=loss_thres, back_rates=[0.44, 0.28, 0.13])

