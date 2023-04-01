#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : TrendModelSys.py
# Author: anyongjin
# Date  : 2023/2/21
from banbot.strategy.base import *


class TrendModelSys(BaseStrategy):
    '''
    Futures Truth Magazine杂志策略排行第9
    https://mp.weixin.qq.com/s?__biz=MzkyODI5ODcyMA==&mid=2247484039&idx=1&sn=defcd9c0c03653ed1078ba98392af315&scene=21#wechat_redirect
    【核心思想】
    以震荡区间的最高点/最低点作为突破上下轨阈值。突破时入场。

    【背景】
    趋势行情出来之前，行情往往是持续震荡。震荡判断方法：MACD短期内经常交叉。

    这个策略是一个趋势突破型的交易系统，利用MACD快(DIF)慢(DEA)线的金叉死叉，
    当证券价格突破过去N次金/死叉记录的“最高/低价±0.5倍ATR”时，开仓；
    若持有多头仓位，当证券价格回落至M根K线最低点平仓；
    若持有空头仓位，当证券价格上升至过去M根K线最高点平仓。

    个人感觉，这个策略值得借鉴最大的亮点就是“关键点位思想”，
    以前突破通道的确定往往是根据过去整一段时间的指标值/因子值，而这里只考虑关键点位的指标值/因子值。
    例如唐奇安通道上轨是过去N个交易日的最大值，那关键点位就是可以只考虑过去N个金叉对应的最高价的最大值，
    这里只是只是打个比方，类推就好了，就如同以下评论，点评得非常到位。
    '''
    def __init__(self):
        super(TrendModelSys, self).__init__()
        self.macd = StaMACD(9, 26, 4)
        self.atr = StaATR(4)
        self.cross_dsct = 50  # 要求最远交叉点的最大距离
        self.back_num = 4  # 往前查看的交叉点的数量
        state: dict = pair_state.get()
        state['cross_exm'] = []

    def log_macd_cross(self, arr: np.ndarray) -> List[Tuple[int, float, float]]:
        '''
        发生交叉时，记录位置和最高值，最低值
        :param arr:
        :return:
        '''
        state: dict = pair_state.get()
        macd, signl = self.macd(arr[-1, ccol])
        cmacd_dir = macd - signl
        if cmacd_dir:
            macd_dir = state.get('macd_dir', 0)
            if macd_dir and cmacd_dir * macd_dir < 0:
                state['cross_exm'].append((bar_num.get(), arr[-1, hcol], arr[-1, lcol]))
                state['cross_exm'] = state['cross_exm'][-10:]
            state['macd_dir'] = cmacd_dir
        return state['cross_exm']

    def on_bar(self, arr: np.ndarray):
        half_atr = self.atr(arr) * 0.5
        cross_exms = self.log_macd_cross(arr)
        if len(cross_exms) >= self.back_num:
            # 检查最近4个MACD交叉点，要求第四个不超过50周期，表示前面处于震荡行情。
            max_high, min_low, xid = -1, 9999999, len(cross_exms)
            while xid > 0:
                xid -= 1
                xitem = cross_exms[xid]
                if bar_num.get() - xitem[0] > self.cross_dsct:
                    break
                max_high = max(max_high, xitem[1])
                min_low = min(min_low, xitem[2])
            if len(cross_exms) - xid >= self.back_num:
                if arr[-1, ccol] > max_high + half_atr:
                    # 当前价格突破最高值+ATR，入场
                    pair_state.get()['up_cross'] = bar_num.get()
                elif arr[-1, ccol] < min_low - half_atr:
                    pair_state.get()['down_cross'] = bar_num.get()

    def on_entry(self, arr: np.ndarray) -> str:
        state: dict = pair_state.get()
        if state.get('up_cross') == bar_num.get():
            return 'upx'

    def on_exit(self, arr: np.ndarray) -> str:
        state: dict = pair_state.get()
        if state.get('down_cross') == bar_num.get():
            return 'downx'
