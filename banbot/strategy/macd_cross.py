#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : macd_cross.py
# Author: anyongjin
# Date  : 2023/3/27
from banbot.strategy.base import *
from banbot.bar_driven.addons import *


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
        ccolse = arr[-1, ccol]
        self.ma5(ccolse)
        self.macd(ccolse)

    def on_entry(self, arr: np.ndarray) -> Optional[str]:
        long_bar_len = LongVar.get(LongVar.bar_len).val
        if np.isnan(self.ma5.arr[-1]) or np.isnan(long_bar_len):
            return
        fea_start = fea_col_start.get()
        max_chg, real, solid_rate, hline_rate, lline_rate = arr[-1, fea_start: fea_start + 5]
        len_ok = real > long_bar_len * 0.2 and solid_rate > 0.3
        macd_up = self.macd.macd_arr[-1] > self.macd.singal_arr[-1]
        ma5_up = self.ma5.arr[-1] > self.ma5.arr[-3]
        price_up = arr[-1, ccol] > max(arr[-1, ocol], arr[-2, ccol])
        if price_up and len_ok and ma5_up and macd_up:
            return 'ent'

    def custom_exit(self, arr: np.ndarray, od: InOutOrder) -> Optional[str]:
        elp_num = bar_num.get() - od.enter_at
        max_loss, max_up, back_rate = trail_info(arr, elp_num, od.enter.price)
        return trail_stop_loss_core(elp_num, max_up, max_loss, back_rate, odlens=[1, 2, 4, 8],
                                    loss_thres=[-0.1, 0.2, 1.5, 2., 3.6], back_rates=[0.44, 0.28, 0.13])

