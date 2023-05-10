#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : btime.py
# Author: anyongjin
# Date  : 2023/3/30
'''
此模块包含所有与时间相关的函数和方法。
实盘和回测都从这里调用。
请避免直接调用datatime和time类库
'''
import datetime
from datetime import timedelta
from banbot.config.consts import *
from typing import Union

global cur_timestamp, run_mode

run_mode = RunMode.DRY_RUN
cur_timestamp = 0


def time():
    global cur_timestamp
    if run_mode in TRADING_MODES:
        import time
        return time.time()
    elif not cur_timestamp:
        import time
        cur_timestamp = time.time()
    return cur_timestamp


def time_ms() -> int:
    return int(time() * 1000)


def now():
    if run_mode in TRADING_MODES:
        return datetime.datetime.now()
    return datetime.datetime.utcfromtimestamp(cur_timestamp)


def to_datetime(timestamp: float = None):
    if not timestamp:
        timestamp = time()
    else:
        if timestamp >= 1000000000000:
            timestamp /= 1000
    return datetime.datetime.utcfromtimestamp(timestamp)


def to_utcstamp(dt, ms=False, round_int=False) -> Union[int, float]:
    stamp = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    if ms:
        stamp *= 1000
    if round_int:
        stamp = round(stamp)
    return stamp


def to_datestr(timestamp: int = None, fmt: str = '%Y-%m-%d %H:%M:%S'):
    dt = to_datetime(timestamp)
    return dt.strftime(fmt)


def allow_order_enter(ctx=None) -> bool:
    if run_mode in NORDER_MODES:
        return False
    if run_mode not in TRADING_MODES:
        return True
    from banbot.compute.ctx import bar_time
    bar_start, bar_end = bar_time.get() if not ctx else ctx[bar_time]
    import time as stime
    delay_fac = (stime.time() * 1000 - bar_end) / (bar_end - bar_start)
    return delay_fac <= 0.8


def prod_mode():
    return run_mode == RunMode.LIVE
