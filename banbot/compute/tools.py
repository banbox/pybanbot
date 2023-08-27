#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : ts_tools.py
# Author: anyongjin
# Date  : 2023/3/16
from banbot.compute.sta_inds import *
from banbot.util import btime
from banbot.util.common import logger


def append_new_bar(row: list, tf_secs: int, check_err=True) -> np.ndarray:
    bar_start_time = int(row[0])
    new_ts_range = (bar_start_time, bar_start_time + tf_secs * 1000)
    old_rg = bar_time.get()
    if check_err and old_rg[1] and old_rg[1] > bar_start_time:
        exp_date = f'{old_rg[1]}({btime.to_datestr(old_rg[1])})'
        get_date = f'{bar_start_time}({btime.to_datestr(bar_start_time)})'
        raise ValueError(f'{symbol_tf.get()}, expect: {exp_date} get invalid bar {get_date}')
    result = bar_arr.get()
    prefix = symbol_tf.get() + '_'
    copen, chigh, clow, close, vol = row[ocol:vcol + 1]
    Bar.open = SeriesVar(prefix + 'o', float(copen))
    Bar.high = SeriesVar(prefix + 'h', float(chigh))
    Bar.low = SeriesVar(prefix + 'l', float(clow))
    Bar.close = SeriesVar(prefix + 'c', float(close))
    Bar.vol = SeriesVar(prefix + 'v', float(vol))
    bar_num.set(bar_num.get() + 1)
    bar_time.set(new_ts_range)
    if not len(result):
        result = np.array(row).reshape((1, -1))
        LongStat.init()
    else:
        osp = result.shape
        result = np.resize(result, (osp[0] + 1, osp[1]))
        result[-1, :] = row
        if osp[0] > 1500:
            result = result[-1000:]
    if check_err and len(result) >= 2:
        sub_diff = round((result[-1][0] - result[-2][0]) / tf_secs / 1000)
        if sub_diff > 1:
            stf, bar_date = symbol_tf.get(), btime.to_datestr(result[-2][0])
            cur_date = btime.to_datestr(result[-1][0])
            logger.error(f'{stf} {sub_diff - 1} bar lost after {bar_date}, get: {cur_date}')
    bar_arr.set(result)
    LongStat.update()
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
