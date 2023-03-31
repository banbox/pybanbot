#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : macd_cross.py
# Author: anyongjin
# Date  : 2023/3/27
from banbot.strategy.base import *


class MACDCross(BaseStrategy):
    '''
    基于MACD+MA5策略。
    当MA5处于上升通道，当前bar阳线，且MACD也上升时，入场。
    使用跟踪止损出场
    '''
    def __init__(self, config: dict):
        super(MACDCross, self).__init__(config)
        self.macd = StaMACD(11, 38, 14)
        self.ma5 = StaSMA(5)

    def on_bar(self, arr: np.ndarray):
        self.ma5(arr[-1, 3])
        self.macd(arr[-1, 3])

    def on_entry(self, arr: np.ndarray) -> Optional[str]:
        if np.isnan(self.ma5.arr[-1]) or np.isnan(LongVar.get(LongVar.bar_len).val):
            return
        max_chg, real, solid_rate, hline_rate, lline_rate = self.bar_rates(arr, -1)
        len_ok = real > LongVar.get(LongVar.bar_len).val * 0.4 or solid_rate > 0.1
        macd_up = self.macd.macd_arr[-1] > self.macd.singal_arr[-1]
        ma5_up = self.ma5.arr[-1] > self.ma5.arr[-3]
        price_up = arr[-1, 3] > min(arr[-1, 0], arr[-2, 3])
        if price_up and len_ok and ma5_up and macd_up:
            return 'ent'

    def custom_exit(self, arr: np.ndarray, od: InOutOrder) -> Optional[str]:
        return trail_stop_loss(arr, od)

