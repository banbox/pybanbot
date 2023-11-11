#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : tf_utils.py
# Author: anyongjin
# Date  : 2023/10/22
from typing import *

import six

from banbot.config.consts import *


_tfsecs_map = dict(ws=5)
_secstf_map = {5: 'ws'}
_tfsecs_origins = [
    (604800, 345600, '1970-01-05'),  # 周级别，从1970-01-05星期一开始
]


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
        import ccxt
        secs = ccxt.Exchange.parse_timeframe(timeframe)
        _tfsecs_map[timeframe] = secs
        _secstf_map[secs] = timeframe
    return _tfsecs_map[timeframe]


def get_tfalign_origin(tf: Union[int, str]) -> Tuple[str, int]:
    if isinstance(tf, six.string_types):
        tf = tf_to_secs(tf)
    tf_secs = tf
    for secs, origin, org_date in _tfsecs_origins:
        if tf_secs < secs:
            break
        if tf_secs % secs == 0:
            return org_date, origin
    return '1970-01-01', 0


def align_tfsecs(time_secs: Union[int, float], tf_secs: int):
    """
    将给定的10位秒级时间戳，转为指定时间周期下，的头部开始时间戳
    """
    if time_secs > 1000000000000:
        raise ValueError('10 digit timestamp is require for align_tfsecs')
    origin_off = 0
    for secs, origin, _ in _tfsecs_origins:
        if tf_secs < secs:
            break
        if tf_secs % secs == 0:
            origin_off = origin
            break
    if not origin_off:
        return time_secs // tf_secs * tf_secs
    return (time_secs - origin_off) // tf_secs * tf_secs + origin_off


def align_tfmsecs(time_msecs: int, tf_msecs: int):
    """
    将给定的13位毫秒级时间戳，转为指定时间周期下，的头部开始时间戳
    """
    if time_msecs < 1000000000000:
        raise ValueError('13 digit timestamp is require for align_tfmsecs')
    if tf_msecs < 1000:
        raise ValueError('milliseconds tf_msecs is require for align_tfmsecs')
    return align_tfsecs(time_msecs // 1000, tf_msecs // 1000) * 1000


def secs_to_tf(tfsecs: int) -> str:
    if tfsecs not in _secstf_map:
        if tfsecs >= secs_year:
            _secstf_map[tfsecs] = str(tfsecs // secs_year) + 'y'
        elif tfsecs >= secs_mon:
            _secstf_map[tfsecs] = str(tfsecs // secs_mon) + 'M'
        elif tfsecs >= secs_week:
            _secstf_map[tfsecs] = str(tfsecs // secs_week) + 'w'
        elif tfsecs >= secs_day:
            _secstf_map[tfsecs] = str(tfsecs // secs_day) + 'd'
        elif tfsecs >= secs_hour:
            _secstf_map[tfsecs] = str(tfsecs // secs_hour) + 'h'
        elif tfsecs >= secs_min:
            _secstf_map[tfsecs] = str(tfsecs // secs_min) + 'm'
        elif tfsecs >= 1:
            _secstf_map[tfsecs] = str(tfsecs) + 's'
        else:
            raise ValueError(f'unsupport tfsecs: {tfsecs}')
    return _secstf_map[tfsecs]


def tfsecs(num: int, timeframe: str):
    return num * tf_to_secs(timeframe)

