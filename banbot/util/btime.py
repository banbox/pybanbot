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
import calendar
from datetime import timedelta
from banbot.config.consts import *

global cur_timestamp, run_mode

run_mode = RunMode.DRY_RUN
cur_timestamp = 1679883802


def time():
    if run_mode in TRADING_MODES:
        import time
        return time.time()
    return cur_timestamp


def now():
    if run_mode in TRADING_MODES:
        return datetime.datetime.now()
    return datetime.datetime.utcfromtimestamp(cur_timestamp)


def to_datetime(timestamp: int = None):
    if not timestamp:
        timestamp = time()
    else:
        timestamp = int(timestamp)
        if timestamp >= 1000000000000:
            timestamp = int(timestamp / 1000)
    return datetime.datetime.utcfromtimestamp(timestamp)


def to_utcstamp(dt):
    return calendar.timegm(dt.timetuple())


def to_datestr(timestamp: int = None, fmt: str = '%Y-%m-%d %H:%M:%S'):
    dt = to_datetime(timestamp)
    return dt.strftime(fmt)
