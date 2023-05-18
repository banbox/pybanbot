#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : exchange_utils.py
# Author: anyongjin
# Date  : 2023/3/25
from typing import *

import ccxt

from banbot.util import btime

_tfsecs_map = dict()
_secstf_map = dict()


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
        tfsecs = ccxt.Exchange.parse_timeframe(timeframe)
        _tfsecs_map[timeframe] = tfsecs
        _secstf_map[tfsecs] = timeframe
    return _tfsecs_map[timeframe]


def secs_to_tf(tfsecs: int) -> str:
    return _secstf_map.get(tfsecs)


def tfsecs(num: int, timeframe: str):
    return num * tf_to_secs(timeframe)


def get_back_ts(tf_secs: int, back_period: int, in_ms: bool = True) -> Tuple[int, int]:
    cur_time_int = int(btime.time())
    to_ms = (cur_time_int - cur_time_int % tf_secs)
    since_ms = to_ms - back_period * tf_secs
    if in_ms:
        since_ms *= 1000
        to_ms *= 1000
    return since_ms, to_ms


def text_markets(market_map: Dict[str, Any], min_num: int = 10):
    from tabulate import tabulate
    from itertools import groupby
    headers = ['Quote', 'Count', 'Active', 'Spot', 'Future', 'Margin', 'TakerFee', 'MakerFee']
    records = []
    markets = list(market_map.values())
    markets = sorted(markets, key=lambda x: x['quote'])
    for key, group in groupby(markets, key=lambda x: x['quote']):
        glist = list(group)
        if len(glist) < min_num:
            continue
        active = len([m for m in glist if m.get('active', True)])
        spot = len([m for m in glist if m.get('spot')])
        future = len([m for m in glist if m.get('future')])
        margin = len([m for m in glist if m.get('margin')])
        taker_gps = [f"{tk}/{len(list(tg))}" for tk, tg in groupby(glist, key=lambda x: x['taker'])]
        taker_text = '  '.join(taker_gps)
        maker_gps = [f"{tk}/{len(list(tg))}" for tk, tg in groupby(glist, key=lambda x: x['maker'])]
        maker_text = '  '.join(maker_gps)
        records.append((
            key, len(glist), active, spot, future, margin, taker_text, maker_text
        ))
    records = sorted(records, key=lambda x: x[1], reverse=True)
    return tabulate(records, headers, 'orgtbl')
