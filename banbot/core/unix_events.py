#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : unix_events.py
# Author: anyongjin
# Date  : 2023/4/5
import sys
if sys.platform == 'win32':  # pragma: no cover
    raise ImportError('Signals are not really supported on Windows')
import asyncio
from selectors import *
from asyncio.unix_events import *
from banbot.util import btime


class BanSelector(DefaultSelector):
    def __init__(self):
        super(BanSelector, self).__init__()

    def select(self, timeout=None):
        if btime.run_mode not in btime.TRADING_MODES:
            btime.cur_timestamp += timeout
            timeout = 0
        return super(BanSelector, self).select(timeout)


class BanSelectorEventLoop(_UnixSelectorEventLoop):

    def __init__(self, selector=None):
        if selector is None:
            selector = BanSelector()
        super(BanSelectorEventLoop, self).__init__(selector)

    def time(self) -> float:
        return btime.time()


class BanSelectorEventLoopPolicy(_UnixDefaultEventLoopPolicy):
    _loop_factory = BanSelectorEventLoop


asyncio.set_event_loop_policy(BanSelectorEventLoopPolicy())
