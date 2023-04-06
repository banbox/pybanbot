#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : macd_cross.py
# Author: anyongjin
# Date  : 2023/3/27
from banbot.strategy.base import *
from banbot.util.common import logger


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
        self.ma5(arr[-1, ccol])
        self.macd(arr[-1, ccol])

    def on_entry(self, arr: np.ndarray) -> Optional[str]:
        long_bar_len = LongVar.get(LongVar.bar_len).val
        # logger.info(f'entry, long bar len: {long_bar_len}')
        if np.isnan(self.ma5.arr[-1]) or np.isnan(long_bar_len):
            return
        fea_start = fea_col_start.get()
        max_chg, real, solid_rate, hline_rate, lline_rate = arr[-1, fea_start: fea_start + 5]
        len_ok = real > long_bar_len * 0.4 or solid_rate > 0.1
        macd_up = self.macd.macd_arr[-1] > self.macd.singal_arr[-1]
        ma5_up = self.ma5.arr[-1] > self.ma5.arr[-3]
        price_up = arr[-1, ccol] > min(arr[-1, ocol], arr[-2, ccol])
        if price_up and len_ok and ma5_up and macd_up:
            return 'ent'

    def custom_exit(self, arr: np.ndarray, od: InOutOrder) -> Optional[str]:
        return trail_stop_loss(arr, od)

