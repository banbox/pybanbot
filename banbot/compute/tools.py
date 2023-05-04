#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : ts_tools.py
# Author: anyongjin
# Date  : 2023/3/16
from banbot.compute.tainds import *
from banbot.util.common import logger
from banbot.util import btime


def append_new_bar(row: np.ndarray, tf_secs: int) -> np.ndarray:
    result = bar_arr.get()
    copen, chigh, clow, close = row[ocol:vcol]
    dust = min(0.00001, max(close, 0.001) * 0.0001)
    max_chg = dust + chigh - clow
    real = abs(close - copen)
    solid_rate = real / max_chg
    hline_rate = (chigh - max(close, copen)) / max_chg
    lline_rate = (min(close, copen) - clow) / max_chg
    bar_num.set(bar_num.get() + 1)
    crow = np.concatenate([row, [max_chg, real, solid_rate, hline_rate, lline_rate]], axis=0)
    exp_crow = np.expand_dims(crow, axis=0)
    if not len(result):
        fea_col_start.set(len(row))
        result = exp_crow
    else:
        result = np.append(result, exp_crow, axis=0)[-1000:]
    if len(result) >= 2:
        sub_diff = round((result[-1][0] - result[-2][0]) / tf_secs / 1000)
        if sub_diff > 1:
            stf, bar_date = symbol_tf.get(), btime.to_datestr(result[-2][0])
            logger.error(f'{stf} {sub_diff - 1} bar lost after {bar_date}')
    bar_arr.set(result)
    LongVar.update(result)
    return result


def resample_candlesticks(arr: np.ndarray, period: int, max_num: int = 0) -> np.ndarray:
    out_arr = np.zeros((0, arr.shape[1]), arr.dtype)
    for end in range(arr.shape[0], -1, -period):
        start = end - period
        if start < 0:
            return out_arr
        out_arr = np.concatenate([arr[end - 1: end, :], out_arr], axis=0)
        row = out_arr[0, :]
        row[0] = arr[start, ocol]
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
