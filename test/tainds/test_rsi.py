#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_rsi.py
# Author: anyongjin
# Date  : 2023/4/12
import numpy as np

from banbot.bar_driven.tainds import *
import talib as ta
from test.common import *

input_vals = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28,
              46.00, 46.03, 46.41, 46.22, 45.64, 46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57,
              43.42, 42.66, 43.13]
input_arr = np.array(input_vals)
period = 14


def test_main():
    result = ta.RSI(input_arr, timeperiod=period)
    res_rsi = RSI(input_arr, period)
    with TempContext('BTC/TUSD/1m'):
        bar_num.set(1)
        rsi = StaRSI(period)
        res_rsi_s = []
        for v in input_arr:
            bar_num.set(bar_num.get() + 1)
            res_rsi_s.append(rsi(v))
        res_rsi_s = np.array(res_rsi_s)
    assert_arr_equal(result, res_rsi)
    assert_arr_equal(result, res_rsi_s)
