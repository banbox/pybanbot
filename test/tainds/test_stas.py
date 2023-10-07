#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_stas.py
# Author: anyongjin
# Date  : 2023/8/30
from test.common import *


def test_series_state():
    sma3 = SMA(Bar.close, 3)
    sma_32 = SMA(sma3, 2)
    sma_33 = SMA(sma3, 2)
    print(sma_32[0:])
    print(sma_33[0:])
    print(f'complete: {len(sma3)} {len(sma_32)} {len(sma_33)}')
    return sma_33


def test_run():
    calc_state_func(test_series_state)

