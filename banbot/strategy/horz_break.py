#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : horz_break.py
# Author: anyongjin
# Date  : 2023/4/9
import numpy as np

from banbot.strategy.base import *


class HorzBreak(BaseStrategy):
    '''
    TrendModelSys  T07
    检测一定周期内MACD频繁交叉，认为是震荡周期，查找震荡周期内K线的最高点，当价格突破此最高点时视为入场信号
    '''
    warmup_num = 600
    back_dist = 50
    cross_num = 4

    def __init__(self, config: dict):
        super(HorzBreak, self).__init__(config)
        self.macd = StaMACD(9, 26, 7)
        self.ma5 = StaSMA(5)
        self.cross_loc = []
        self.macd_dir = 0
        self.max_high = 9999999
        self.min_low = -1

    def log_macd_cross(self, arr: np.ndarray):
        macd, signl = self.macd(arr[-1, ccol])
        cmacd_dir = macd - signl
        if cmacd_dir:
            if self.macd_dir and cmacd_dir * self.macd_dir < 0:
                bar_len = LongVar.get(LongVar.bar_len).val
                prex = self.cross_loc[-1] - 1 if self.cross_loc else 0
                self.max_high = np.max(arr[prex:, hcol]) + bar_len
                self.min_low = np.min(arr[prex:, lcol]) - bar_len
                self.cross_loc.append(bar_num.get())
                self.cross_loc = self.cross_loc[-10:]
            self.macd_dir = cmacd_dir
        return self.cross_loc

    def on_bar(self, arr: np.ndarray):
        ccolse = arr[-1, ccol]
        self.ma5(ccolse)
        self.log_macd_cross(arr)

    def on_entry(self, arr: np.ndarray) -> Optional[str]:
        ccolse = arr[-1, ccol]
        x_locs = self.cross_loc
        if len(x_locs) >= self.cross_num and (bar_num.get() - x_locs[-self.cross_num]) <= self.back_dist \
                and (ccolse > self.max_high or ccolse < self.min_low):
            # 检查最近4个MACD交叉点，要求第四个不超过50周期，表示前面处于震荡行情。
            start_loc = len(x_locs) - self.cross_num
            while start_loc > 0 and bar_num.get() - x_locs[start_loc - 1] <= self.back_dist:
                start_loc -= 1
            start = max(0, x_locs[start_loc] - 5)
            horz_arr = arr[start: x_locs[-1]]
            max_high, min_low = np.max(horz_arr[:, hcol]), np.min(horz_arr[:, lcol])
            bar_len = LongVar.get(LongVar.bar_len).val
            if ccolse > max_high + bar_len and self.ma5.arr[-1] > self.ma5.arr[-3]:
                return 'upx'
            # elif ccolse < min_low - bar_len and self.ma5.arr[-1] < self.ma5.arr[-3]:
            #     return 'downx'

    def custom_exit(self, arr: np.ndarray, od: InOutOrder) -> Optional[str]:
        elp_num = od.elp_num_enter
        max_loss, max_up, back_rate = trail_info(arr, elp_num, od.enter.price)
        return trail_stop_loss_core(elp_num, max_up, max_loss, back_rate, odlens=[2, 4, 8, 16],
                                    loss_thres=[-0.7, 0., 1.5, 2., 3.6], back_rates=[0.44, 0.28, 0.13])
