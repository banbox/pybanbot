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
import asyncio
import datetime
from datetime import timedelta
from banbot.config.consts import *

global cur_timestamp, run_mode

run_mode = RunMode.DRY_RUN
cur_timestamp = 1679883802


def add_secs(secs: float):
    global cur_timestamp
    if run_mode in TRADING_MODES:
        print('`add_secs` not avaiable in trading mode!')
        return
    cur_timestamp += secs


async def sleep(secs: float):
    if run_mode in TRADING_MODES:
        await asyncio.sleep(secs)
    else:
        add_secs(secs)


def time():
    if run_mode in TRADING_MODES:
        import time
        return time.time()
    return cur_timestamp


def now():
    if run_mode in TRADING_MODES:
        return datetime.datetime.now()
    return datetime.datetime.utcfromtimestamp(cur_timestamp)
