#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : exchange_utils.py
# Author: anyongjin
# Date  : 2023/3/25
import ccxt
from typing import *
from banbot.util import btime
_tfsecs_map = dict()


def max_sub_timeframe(timeframes: List[str], current: str, force_sub=False) -> Tuple[str, int]:
    '''
    返回交易所支持的最大子时间帧
    :param timeframes: 交易所支持的所有时间帧  exchange.timeframes.keys()
    :param current: 当前要求的时间帧
    :param force_sub: 是否强制使用更细粒度的时间帧，即使当前时间帧支持
    :return:
    '''
    tf_secs = tf_to_secs(current)
    pairs = [(tf, tf_to_secs(tf)) for tf in timeframes]
    pairs = sorted(pairs, key=lambda x: x[1])
    all_tf, all_tf_secs = list(zip(*pairs))
    rev_tf_secs = all_tf_secs[::-1]
    for i in range(len(rev_tf_secs)):
        if force_sub and tf_secs == rev_tf_secs[i]:
            continue
        if tf_secs % rev_tf_secs[i] == 0:
            return all_tf[len(all_tf_secs) - i - 1], rev_tf_secs[i]


def tf_to_secs(timeframe: str) -> int:
    """
    Translates the timeframe interval value written in the human readable
    form ('1m', '5m', '1h', '1d', '1w', etc.) to the number
    of seconds for one timeframe interval.
    """
    if timeframe not in _tfsecs_map:
        _tfsecs_map[timeframe] = ccxt.Exchange.parse_timeframe(timeframe)
    return _tfsecs_map[timeframe]


def get_back_ts(tf_secs: int, back_period: int, in_ms: bool = True) -> Tuple[int, int]:
    cur_time_int = int(btime.time())
    to_ms = (cur_time_int - cur_time_int % tf_secs)
    since_ms = to_ms - back_period * tf_secs
    if in_ms:
        since_ms *= 1000
        to_ms *= 1000
    return since_ms, to_ms
