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


class TempRunMode:
    def __init__(self, mode: RunMode):
        self.tmp_mode = mode
        self.bak_mode = run_mode

    def __enter__(self):
        global run_mode
        run_mode = self.tmp_mode

    def __exit__(self, exc_type, exc_val, exc_tb):
        global run_mode
        run_mode = self.bak_mode

