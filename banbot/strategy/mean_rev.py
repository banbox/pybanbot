#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : mean_rev.py
# Author: anyongjin
# Date  : 2023/3/4
import numpy as np

from banbot.strategy.base import *
from banbot.compute.addons import *
from typing import Tuple, Optional


class MeanRev(BaseStrategy):
    '''
    均值回归策略。主要基于：MA5，MA20，MA120
    MA5和MA20始终趋向于回归到MA100

      震荡/上升时突然剧烈下跌 & 成交量剧增 & MA100在上方  会逐渐上升（等待反转信号）
      震荡/下降时突然剧烈上涨 & 成交量剧增 & MA100在下方  会逐渐下跌（等待反转信号）

      震荡时(小幅度下跌 | 巨跌) & 见顶信号 & 成交量剧增 & MA100在下方  会继续下跌

      下降周期 & MA100在上方 & 量价巨增  买入信号（等待确认信号）

      下降周期 & 阳线带长下影  见底买入信号（可能只有两三个周期上涨，应尽快止损）

      NTRRoll较低，波动较小，小阳+小阴交替，20均线上涨，上涨信号

      MA5震荡 + MA20下跌 + 低于MA100 + 一系列中阳线，上涨信号

      MA20下降+量价巨跌+低于MA100+连续两次超过MA20均线向上  上涨信号


      MA5向上，3窗口NATR较低，前期属于熊市，位于MA100下方，是买入信号
    '''

    def __init__(self, config: dict, debug_ids: Optional[set] = None):
        super().__init__(config)
        # 原始列：open, high, low, close, volume, count, long_vol
        # max_chg, real, solid_rate, hline_rate, lline_rate
        self.ma120 = StaSMA(120)
        self.ma20 = StaSMA(20)
        self.ma5 = StaSMA(5)
        self.tr = StaTR()
        self.natr = StaNATR()
        self.ntr_rol = StaNTRRoll()
        self.nvol = StaNVol()
        self.macd = StaMACD()
        self.debug_ids = debug_ids or set()
        self._is_debug = False

    def _init_state_fn(self):
        self._state_fn = dict(
            big_vol_prc=make_big_vol_prc(self.nvol, self.ntr_rol),
            calc_shorts=make_calc_shorts(self.ma5, self.ma20, self.ma120)
        )

    def on_bar(self, arr: np.ndarray):
        '''
        针对每个蜡烛计算后续用到的指标。
        :param arr: 二维数组，每个元素是一个蜡烛：[open, high, low, close, volume, count, long_vol]
        :return:
        '''
        ccolse = arr[-1, ccol]
        self.ma5(ccolse)
        self.ma20(ccolse)
        self.ma120(ccolse)
        self.macd(ccolse)
        self.tr(arr)
        self.natr(arr)
        self.ntr_rol(arr)
        self.nvol(arr)
        if not self._state_fn:
            self._init_state_fn()
        self._is_debug = self.debug_ids and bar_num.get() - 1 in self.debug_ids
        self.patterns.append(detect_pattern(arr))
        # 记录均线极值点
        log_ma_extrems([
            (self.extrems_ma5, self.ma5, 2),
            (self.extrems_ma20, self.ma20, 1),
            (self.extrems_ma120, self.ma120, 0.5),
        ])
        # 记录MA5和MA20的交叉点
        self._log_ma_cross(self.ma5, self.ma20)

    def _sudden_huge_rev(self, arr: np.ndarray) -> Optional[Tuple[str, float, dict]]:
        if np.isnan(self.ma120.arr[-1]):
            return
        huge_score = self._calc_state('big_vol_prc', arr)
        if huge_score < 0.1:
            return
        
        cur_bar_num = bar_num.get()
        close_chg = arr[-1, ccol] - arr[-2, ccol]
        bar_avg_len = LongVar.get(LongVar.bar_len).val
        ma5_chg = self.ma5.arr[-1] - self.ma5.arr[-10]
        ma20_down = self.ma20.arr[-1] - self.ma20.arr[-10] < bar_avg_len * -0.5
        ma120 = self.ma120.arr[-1]

        cur_singals = dict()
        if close_chg > 0:
            # 巨量向上
            if ma5_chg < bar_avg_len * -1.5 and ma20_down and arr[-1, ccol] + bar_avg_len < ma120:
                # 下降周期 & MA100在上方 & 量价巨增  买入信号（等待确认信号）
                cur_singals['huge_up_rev'] = cur_bar_num, huge_score
            elif ma5_chg > bar_avg_len * 1.5:
                # 前面也上涨：继续上涨的信号。
                return 'up_ensure', huge_score, cur_singals
            # 背离：价格成交量巨幅向上。可能是买入信号，也可能会迅速回归。
            elif close_chg > bar_avg_len * 3:
                cur_singals['huge_up_rev'] = cur_bar_num, huge_score
        else:
            # 巨量向下
            cur_singals['huge_down_rev'] = cur_bar_num, huge_score
        return '', 0, cur_singals

    def sudden_huge_rev(self, arr: np.ndarray) -> Tuple[Optional[str], float]:
        '''
        突发价格巨大背离，巨量，背离MA120，会逐渐回归（需等待反转）。
          震荡/上升时突然剧烈下跌 & 成交量剧增 & MA100在上方  会逐渐上升（等待反转信号）
          震荡/下降时突然剧烈上涨 & 成交量剧增 & MA100在下方  会逐渐下跌（等待反转信号）
        :return:
        '''
        phuge_up = self.long_sigs.get('huge_up_rev')
        phuge_down = self.long_sigs.get('huge_down_rev')
        bar_res = self._sudden_huge_rev(arr)
        tag, score, sign_updates = None, 0, dict()
        if bar_res:
            sign_updates = bar_res[2]
            if bar_res[1] > 0.1:
                return bar_res[:2]
        copen, chigh, clow, close = arr[-1, ocol:vcol]
        bar_avg_len = LongVar.get(LongVar.bar_len).val
        fea_start = fea_col_start.get()
        max_chg, real, solid_rate, hline_rate, lline_rate = arr[-1, fea_start: fea_start + 5]
        cur_bar_num = bar_num.get()
        back_len = 7
        if not phuge_up or cur_bar_num - phuge_up[0] > back_len:
            phuge_up = phuge_down
        if phuge_up and cur_bar_num - phuge_up[0] <= back_len:
            cur_ma5, sign_close = self.ma5.arr[-1], arr[phuge_up[0] - cur_bar_num, ccol]
            if solid_rate >= 0.3 and max_chg >= bar_avg_len and cur_ma5 - sign_close >= bar_avg_len:
                # 3周期内信号，当前实体至少70%，有足够的实体长度
                tag = 'huge_up_rev' if copen < close else 'huge_down_rev'
                score = phuge_up[1]
                sign_updates = dict(huge_up_rev=None, huge_down_rev=None)
        self.long_sigs.update(sign_updates)
        return tag, score

    def _up_score(self, arr: np.ndarray, up_val) -> float:
        fea_start = fea_col_start.get()
        max_chg, real, solid_rate, hline_rate, lline_rate = arr[-1, fea_start: fea_start + 5]
        if arr[-1, ocol] >= arr[-1, ccol] or solid_rate < 0.5:
            # 最后一个必须阳线，实体50%
            return 0
        if real >= up_val and solid_rate >= 0.7:
            # 单个上涨较多，实体70%
            return solid_rate / 0.8
        if arr[-1, ccol] - arr[-2, ocol] >= up_val and arr[-2, fea_start + 2] >= 0.5:
            # 两个上涨较多，实体均不小于50%
            return np.average(arr[-2:, fea_start + 2]) / 0.66
        if arr[-1, ccol] - arr[-3, ocol] >= up_val and np.min(arr[-3:, fea_start + 2]) >= 0.5:
            # 三个上涨较多，实体不小于50%
            return np.average(arr[-3:, fea_start + 2]) / 0.66
        return 0

    def ma_cross_entry(self, arr: np.ndarray):
        ma5, ma20 = self.ma5.arr[-1], self.ma20.arr[-1]
        if ma5 < ma20:  # or not self.ma_cross or bar_num.get() - self.ma_cross[-1] > 7
            return None, 0
        sign_key, back_len = 'xma', 5
        old_sta = self.long_sigs.get(sign_key)
        if old_sta and bar_num.get() - old_sta[0] <= back_len or arr.shape[0] < back_len:
            return None, 0
        score = self._up_score(arr, LongVar.get(LongVar.bar_len).val)
        if score < 0.5:
            return None, 0
        # 如果是背离MA120巨量+巨价；则不入场
        prc_chg, close_s120 = arr[-1, ccol] - arr[-1, ocol], arr[-1, ccol] - self.ma120.arr[-1]
        huge_score = self._calc_state('big_vol_prc', arr)
        if huge_score > 0.1 and prc_chg * close_s120 > 0:
            return None, 0
        # 如果邻近历史极值点，则不入场；测试没有这里利润高些
        # start_id = len(self.extrems_ma5) - 1
        # end_id = max(start_id - 10, 0)
        # exm_min, exm_max = -long_bar_avg.val, long_bar_avg.val * 3
        # exm_near_len = 4
        # for i in range(start_id, end_id, -1):
        #     exm = self.extrems_ma5[i]
        #     prow_id = exm[0] - bar_num.get() - 1
        #     if prow_id + exm_near_len >= 0:
        #         exm_range_arr = arr[prow_id - exm_near_len:, 3]
        #     else:
        #         exm_range_arr = arr[prow_id - exm_near_len: prow_id + exm_near_len, 3]
        #     if exm[-1] == 1:
        #         prow_id += np.argmax(exm_range_arr) - exm_near_len
        #     else:
        #         prow_id += np.argmin(exm_range_arr) - exm_near_len
        #     prc_dst = arr[prow_id, hcol] - arr[-1, ccol]
        #     if exm_min < prc_dst < exm_max:
        #         return None, 0

        self.long_sigs[sign_key] = bar_num.get(), score
        return sign_key, score

    def on_entry(self, arr: np.ndarray) -> str:
        '''
        检测入场信号。调用此方法前必须已经调用on_bar
        :param arr:
        :return:
        '''
        # self.ma_cross_entry(arr)
        enter_tags = [self.sudden_huge_rev(arr)]
        enter_tags = sorted(enter_tags, key=lambda x: x[1], reverse=True)
        final_tag = enter_tags[0]
        if final_tag[1] > 0.1:
            return final_tag[0]

    # def on_exit(self, arr: np.ndarray) -> str:
    #     huge_score = self._calc_state('big_vol_prc', arr)
    #     short_sigs = self._calc_state('calc_shorts', arr, self.patterns, huge_score)
    #     if short_sigs:
    #         for key, score in short_sigs:
    #             self.short_sigs[key] = bar_num.get(), score
    #         if short_sigs[0][1] > 0.8:
    #             return short_sigs[0][0]

    def custom_exit(self, arr: np.ndarray, od: InOutOrder) -> Optional[str]:
        '''
        判断订单是否应该平仓。
        如果是刚开仓，稍有趋势不对（连续两个见顶信号）应该立刻平仓。
        如果已经有些许盈利，应该使用止损线
        :param arr:
        :param od:
        :return:
        '''
        elp_num = bar_num.get() - od.enter_at
        return trail_stop_loss(arr, od.enter.price, elp_num)
        # profit = arr[-1, ccol] - od.price
        # stable_score = profit / LongVar.get(LongVar.sub_malong).val
        # if stable_score < 1:
        #     short_sigs = self._get_sigs('short')
        #     if not short_sigs:
        #         return None
        #     short_score = short_sigs[0][1]
        #     bar_len = LongVar.get(LongVar.bar_len).val
        #     if profit < bar_len and short_score >= 0.4:
        #         return short_sigs[0][0]
        #     elif profit > bar_len and short_score >= 0.7:
        #         return short_sigs[0][0]
            