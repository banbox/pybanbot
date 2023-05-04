#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : common.py
# Author: anyongjin
# Date  : 2023/5/4
from banbot.bar_driven.tainds import *
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
