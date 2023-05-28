#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : macd_cross.py
# Author: anyongjin
# Date  : 2023/3/27
from banbot.strategy.base import *


class MACDCross(BaseStrategy):
    '''
    注意：这是高频策略，由于滑点和延迟问题，实际收益可能只有回测的一半。
    基于MACD+MA5策略。
    当MA5处于上升通道，当前bar阳线，且MACD也上升时，入场。
    使用跟踪止损出场。
    测试效果不佳的：
    最近3个bar的2个是星线，或者大阳线，且处于下降趋势，则退场

    此策略适用于0手续费且频繁震荡的行情。
    BTC/TUSD
    20230322-20230412  趋势上升    获利22%
    20230412-20230528  震荡或下跌  获利1% (请使用NoFee)
    '''
    warmup_num = 600
    version = 4

    def __init__(self, config: dict):
        super(MACDCross, self).__init__(config)
        self.macd = StaMACD(11, 38, 14)
        self.ma5 = StaSMA(5)
        self.atr = StaATR(5)
        self.kdj = StaKDJ()

    def on_bar(self, arr: np.ndarray):
        self._update_inds(arr, self.ma5, self.macd, self.atr, self.kdj)
        kdj = self.kdj[-1]
        self._log_cross('kdj', kdj[0] - kdj[1])

    def on_entry(self, arr: np.ndarray) -> Optional[str]:
        long_bar_len = LongVar.get(LongVar.bar_len).val
        if np.isnan(self.ma5[-1]) or np.isnan(long_bar_len):
            return
        if self._cross_dist('kdj', 1) > self.kdj.period:
            return
        cur_close = arr[-1, ccol]
        fea_start = fea_col_start.get()
        max_chg, real, solid_rate, hline_rate, lline_rate = arr[-1, fea_start: fea_start + 5]
        len_ok = real > long_bar_len * 0.2 and solid_rate > 0.3
        macd_val, macd_sig = self.macd[-1]
        macd_up = macd_val > macd_sig
        ma5_up = self.ma5[-1] > self.ma5[-3]
        price_up = cur_close > max(arr[-1, ocol], arr[-2, ccol])
        if price_up and len_ok and ma5_up and macd_up:
            return 'ent'

    def custom_exit(self, arr: np.ndarray, od: InOutOrder) -> Optional[str]:
        elp_num = od.elp_num_enter
        atr_val = self.atr[- elp_num - 1]
        loss_thres = [-1.2 * atr_val, -0.85 * atr_val, atr_val, 2.7 * atr_val, 4 * atr_val]
        return trail_stop_loss(arr, od.init_price, elp_num, loss_thres, odlens=[10, 10, 16, 25],
                               back_rates=[0.44, 0.28, 0.13])
        # 如下参数组，对高波动性市场获利更好，但震荡时亏损也略多，整体收益不如上面
        # return trail_stop_loss_core(elp_num, max_up, max_loss, back_rate, odlens=[7, 20, 40, 70],
        #                             loss_thres=[-1.5, -1., 2., 4., 7.], back_rates=[0.44, 0.28, 0.13])

