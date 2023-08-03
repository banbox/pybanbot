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
from datetime import timedelta  # noqa
from typing import Union

from banbot.config.consts import *

global cur_timestamp, run_mode

run_mode = RunMode.DRY_RUN
cur_timestamp = 0


def utctime() -> float:
    '''
    获取秒级的UTC时间戳，浮点类型。
    time.time()和datetime.datetime.now(datetime.timezone.utc).timestamp()都可以获取UTC的毫秒时间戳。
    但后者精度略高些
    '''
    return datetime.datetime.now(datetime.timezone.utc).timestamp()


def utcstamp() -> int:
    '''
    获取毫秒级UTC时间戳，整数类型
    '''
    stamp = datetime.datetime.now(datetime.timezone.utc).timestamp()
    return int(stamp * 1000)


def time() -> float:
    global cur_timestamp
    if run_mode in LIVE_MODES:
        return utctime()
    elif not cur_timestamp:
        cur_timestamp = utctime()
    return cur_timestamp


def time_ms() -> int:
    return int(time() * 1000)


def now():
    '''
    实盘模式下返回真实时间戳。
    回测模式返回bar对应时间戳
    '''
    if run_mode in LIVE_MODES:
        return datetime.datetime.now(datetime.timezone.utc)
    return datetime.datetime.utcfromtimestamp(cur_timestamp)


def to_datetime(timestamp: float = None):
    if not timestamp:
        timestamp = time()
    else:
        if timestamp >= 1000000000000:
            timestamp /= 1000
    return datetime.datetime.utcfromtimestamp(timestamp)


def to_utcstamp(dt, ms=False, cut_int=False) -> Union[int, float]:
    if isinstance(dt, datetime.datetime):
        if dt.tzinfo != datetime.timezone.utc:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        stamp = dt.timestamp()
    elif isinstance(dt, (int, float)):
        stamp = dt
    else:
        raise TypeError(f'unsupport type: {type(dt)} {dt}')
    if ms:
        stamp *= 1000
    if cut_int:
        stamp = int(stamp)
    return stamp


def to_datestr(ts_or_dt: Union[float, datetime.datetime] = None, fmt: str = '%Y-%m-%d %H:%M:%S'):
    if not ts_or_dt:
        return ''
    if ts_or_dt and not isinstance(ts_or_dt, datetime.datetime):
        dt = to_datetime(ts_or_dt)
    else:
        dt = ts_or_dt
    return dt.strftime(fmt)


def sys_timezone() -> str:
    '''
    获取当前系统时区代码：Asia/Shanghai
    '''
    import tzlocal
    tz = tzlocal.get_localzone()
    return str(tz)


def allow_order_enter(ctx=None) -> bool:
    if run_mode in NORDER_MODES:
        return False
    if run_mode not in LIVE_MODES:
        return True
    from banbot.compute.ctx import bar_time
    bar_start, bar_end = bar_time.get() if not ctx else ctx[bar_time]
    import time as stime
    delay_fac = (stime.time() * 1000 - bar_end) / (bar_end - bar_start)
    return delay_fac <= 0.8


def prod_mode():
    return run_mode == RunMode.PROD
