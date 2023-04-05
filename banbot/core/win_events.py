#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : ban_proactor.py
# Author: anyongjin
# Date  : 2023/4/5
'''
此模块模拟asyncio的异步时间流逝。
用于回测时异步代码的预期按顺序执行和获得可靠的回测时间。
在asyncio中，协程是通过EventLoop.call_at加入等待队列。
然后通过_run_once定时扫描等待队列，等待预期的时间，然后执行最近的等待函数。
时间的等待是在_run_once的下面两行中实现的：
>>>        event_list = self._selector.select(timeout)
>>>        self._process_events(event_list)
在Windows中，通过IocpProactor的_poll来睡眠指定时间。
在unix中，_selector是selectors.DefaultSelector的实例。
'''
import sys

if sys.platform != 'win32':  # pragma: no cover
    raise ImportError('win32 only')
import asyncio
from asyncio.windows_events import *
from asyncio.events import BaseDefaultEventLoopPolicy
from banbot.util import btime

INFINITE = 0xffffffff


class BanIocpProactor(IocpProactor):
    def __init__(self, concurrency=0xffffffff):
        super(BanIocpProactor, self).__init__(concurrency)

    def select(self, timeout=None):
        if not self._results:
            if btime.run_mode not in btime.TRADING_MODES:
                btime.cur_timestamp += timeout
                timeout = 0
            self._poll(timeout)
        tmp = self._results
        self._results = []
        try:
            return tmp
        finally:
            # Needed to break cycles when an exception occurs.
            tmp = None


class BanProactorEventLoop(ProactorEventLoop):

    def __init__(self, proactor=None):
        if proactor is None:
            proactor = BanIocpProactor()
        super(BanProactorEventLoop, self).__init__(proactor)

    def time(self) -> float:
        return btime.time()


class BanProactorEventLoopPolicy(BaseDefaultEventLoopPolicy):
    _loop_factory = BanProactorEventLoop


asyncio.set_event_loop_policy(BanProactorEventLoopPolicy())

