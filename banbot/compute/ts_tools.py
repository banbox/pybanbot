#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : ts_tools.py
# Author: anyongjin
# Date  : 2023/3/16
import numpy as np


def resample_candlesticks(arr: np.ndarray, period: int, max_num: int = 0) -> np.ndarray:
    out_arr = np.zeros((0, arr.shape[1]), arr.dtype)
    for end in range(arr.shape[0], -1, -period):
        start = end - period
        if start < 0:
            return out_arr
        out_arr = np.concatenate([arr[end - 1: end, :], out_arr], axis=0)
        row = out_arr[0, :]
        row[0] = arr[start, 0]
        row[1] = np.max(arr[start: end, 1])
        row[2] = np.min(arr[start: end, 2])
        row[4] = np.sum(arr[start: end, 4])
        row[5] = np.sum(arr[start: end, 5])
        row[6] = np.sum(arr[start: end, 6])
        copen, chigh, clow, close = row[:4]
        dust = min(0.00001, close * 0.0001)
        max_chg = dust + chigh - clow
        real = abs(close - copen)
        solid_rate = real / max_chg
        hline_rate = (chigh - max(close, copen)) / max_chg
        lline_rate = (min(close, copen) - clow) / max_chg
        row[list(range(7, 12))] = max_chg, real, solid_rate, hline_rate, lline_rate
        if max_num and out_arr.shape[0] >= max_num:
            return out_arr
    return out_arr
