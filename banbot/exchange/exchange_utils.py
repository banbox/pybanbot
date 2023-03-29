#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : exchange_utils.py
# Author: anyongjin
# Date  : 2023/3/25
import math
import time

import ccxt
import numpy as np
from banbot.config.consts import *
from typing import *


def max_sub_timeframe(timeframes: List[str], current: str, force_sub=False) -> Tuple[str, int]:
    '''
    返回交易所支持的最大子时间帧
    :param timeframes: 交易所支持的所有时间帧  exchange.timeframes.keys()
    :param current: 当前要求的时间帧
    :param force_sub: 是否强制使用更细粒度的时间帧，即使当前时间帧支持
    :return:
    '''
    tf_secs = timeframe_to_seconds(current)
    pairs = [(tf, timeframe_to_seconds(tf)) for tf in timeframes]
    pairs = sorted(pairs, key=lambda x: x[1])
    all_tf, all_tf_secs = list(zip(*pairs))
    rev_tf_secs = all_tf_secs[::-1]
    for i in range(len(rev_tf_secs)):
        if force_sub and tf_secs == rev_tf_secs[i]:
            continue
        if tf_secs % rev_tf_secs[i] == 0:
            return all_tf[len(all_tf_secs) - i - 1], rev_tf_secs[i]


def timeframe_to_seconds(timeframe: str) -> int:
    """
    Translates the timeframe interval value written in the human readable
    form ('1m', '5m', '1h', '1d', '1w', etc.) to the number
    of seconds for one timeframe interval.
    """
    return ccxt.Exchange.parse_timeframe(timeframe)


def get_check_interval(timeframe_secs: int) -> float:
    '''
    根据监听的交易对和时间帧。计算最小检查间隔。
    <60s的通过WebSocket获取数据，检查更新间隔可以比较小。
    1m及以上的通过API的秒级接口获取数据，3s更新一次
    :param timeframe_secs:
    :return:
    '''
    if timeframe_secs <= 3:
        check_interval = 0.2
    elif timeframe_secs <= 10:
        check_interval = 0.5
    elif timeframe_secs < 60:
        check_interval = 1
    else:
        return 3
    return check_interval


def trade2ohlc(trade: dict) -> Tuple[int, float, float, float, float, float, int]:
    price = trade['price']
    return trade['timestamp'], price, price, price, price, trade['amount'], 1


def build_ohlcvc(details, tf_secs: int, prefire: float = 0., since=None, ohlcvs=None):
    '''
    从交易或子OHLC数组中，构建或更新更粗粒度OHLC数组。
    :param details: 可以是交易列表或子OHLC列表。[dict] or [[t,o,h,l,c,v,cnt]]
    :param tf_secs: 指定要构建的时间粒度，单位：秒
    :param prefire: 是否提前触发构建完成；用于在特定信号时早于其他交易者提早发出信号
    :param since:
    :param ohlcvs: 已有的待更新数组
    :return:
    '''
    ms = tf_secs * 1000
    off_ms = round(ms * prefire)
    ohlcvs = ohlcvs or []
    (timestamp, copen, high, low, close, volume, count) = (0, 1, 2, 3, 4, 5, 6)
    for detail in details:
        row = list(trade2ohlc(detail)) if isinstance(detail, dict) else list(detail)
        # 按给定粒度重新格式化时间戳
        row[timestamp] = int(math.floor((row[timestamp] + off_ms) / ms) * ms)
        if since and row[timestamp] < since:
            continue
        if not ohlcvs or (row[timestamp] >= ohlcvs[-1][timestamp] + ms):
            # moved to a new timeframe -> create a new candle from opening trade
            ohlcvs.append(row)
        else:
            prow = ohlcvs[-1]
            # still processing the same timeframe -> update opening trade
            prow[high] = max(prow[high], row[high])
            prow[low] = min(prow[low], row[low])
            prow[close] = row[close]
            prow[volume] += row[volume]
            if len(row) > count:
                prow[count] += row[count]
    return ohlcvs
