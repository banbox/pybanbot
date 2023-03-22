#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : big_bear_rev.py
# Author: anyongjin
# Date  : 2023/3/1
from banbot.bar_driven.base import *


class BigBearRev(BaseStrategy):
    '''
    行情：震荡/上升时。5日均线大部分位于100日均线下方。
    突然剧烈下跌，成交量剧增，主动买入占比较少。
    原理：市价单大单看跌。后续如果不继续砸单，则多半立刻恢复。否则会继续下跌。
    '''

    fea_cols = ['div5', 'div20', 'div120']

    def __init__(self):
        pass

    def on_entry(self, arr: np.ndarray) -> str:
        pass

