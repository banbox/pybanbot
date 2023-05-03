#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : async_events.py
# Author: anyongjin
# Date  : 2023/4/5
'''
引入此模块将针对async使用自定义的EventLoop。
当btime.run_mode不是TRADING_MODES时，跳过所有sleep等待，不睡眠；
即`asyncio.sleep`实际上不会导致休眠，只会修改btime.cur_timestamp
此行为针对一些http请求可能导致超时，故对这些请求应该将btime.run_mode改为TRADING_MODES，请求完后再改回来

此模块模拟asyncio的异步时间流逝。
用于回测时异步代码的预期按顺序执行和获得可靠的回测时间。
在asyncio中，协程是通过EventLoop.call_at加入等待队列。
然后通过_run_once定时扫描等待队列，等待预期的时间，然后执行最近的等待函数。
时间的等待是在_run_once的下面行中实现的：
>>>        event_list = self._selector.select(timeout)
在Windows中，通过IocpProactor的_poll来睡眠指定时间。
在unix中，_selector是selectors.DefaultSelector的实例。

要避免执行sleep，有两种方案：
1. 任务加入时，预期回调时间改为当前时间，目标时间存储在另一个字段用于排序
2. 重构实际等待时(上方_selector.select)的方法，不执行等待

这里使用第一种方法（第二种方法涉及不同平台，实现较复杂）：
当并未处于实时模式时，call_at加入的时候，回调时间是当前时间，来确保立刻得到执行。
处于实时模式时，回调时间是当前时间+delay
'''
import sys
import asyncio
import time
import heapq
from asyncio.events import TimerHandle
from banbot.util import btime


class BanTimerHandle(TimerHandle):
    def __init__(self, when, callback, args, loop, context=None):
        self.sval = when  # 用于排序的值
        # self.start = btime.cur_timestamp  # 记录调度时间
        if btime.run_mode not in btime.TRADING_MODES:
            when = time.monotonic()
        super(BanTimerHandle, self).__init__(when, callback, args, loop, context)

    # 模拟时钟不在这里修改，这里修改会出现并发读写错乱，交由外部手动修改
    # def _run(self) -> None:
    #     if self.sval > self._when:
    #         btime.cur_timestamp = self.start + self.sval - self._when
    #     super(BanTimerHandle, self)._run()

    def _repr_info(self):
        info = super()._repr_info()
        pos = 2 if self._cancelled else 1
        info.insert(pos, f'when={self.sval}')
        return info

    def __hash__(self):
        return hash(self.sval)

    def __lt__(self, other):
        if isinstance(other, BanTimerHandle):
            return self.sval < other.sval
        elif isinstance(other, TimerHandle):
            return self.sval < other._when
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, BanTimerHandle):
            return self.sval < other.sval or self.__eq__(other)
        elif isinstance(other, TimerHandle):
            return self.sval < other._when or self.__eq__(other)
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, BanTimerHandle):
            return self.sval > other.sval
        elif isinstance(other, TimerHandle):
            return self.sval > other._when
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, BanTimerHandle):
            return self.sval > other.sval or self.__eq__(other)
        elif isinstance(other, TimerHandle):
            return self.sval > other._when or self.__eq__(other)
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, BanTimerHandle):
            return (self.sval == other.sval and
                    self._callback == other._callback and
                    self._args == other._args and
                    self._cancelled == other._cancelled)
        elif isinstance(other, TimerHandle):
            return (self.sval == other._when and
                    self._callback == other._callback and
                    self._args == other._args and
                    self._cancelled == other._cancelled)
        return NotImplemented


def call_at(self, when, callback, *args, context=None):
    """copy from asyncio.base_events
    """
    self._check_closed()
    if self._debug:
        self._check_thread()
        self._check_callback(callback, 'call_at')
    timer = BanTimerHandle(when, callback, args, self, context)
    if timer._source_traceback:
        del timer._source_traceback[-1]
    heapq.heappush(self._scheduled, timer)
    timer._scheduled = True
    return timer


if sys.platform == 'win32':  # pragma: no cover
    from asyncio.windows_events import *
    from asyncio.events import BaseDefaultEventLoopPolicy

    class BanEventLoop(ProactorEventLoop):
        # windows默认使用ProactorEventLoop，这里也可从SelectorEventLoop继承
        pass

    class BanEventLoopPolicy(BaseDefaultEventLoopPolicy):
        _loop_factory = BanEventLoop
else:
    from asyncio.unix_events import _UnixSelectorEventLoop, _UnixDefaultEventLoopPolicy

    class BanEventLoop(_UnixSelectorEventLoop):
        pass

    class BanEventLoopPolicy(_UnixDefaultEventLoopPolicy):
        _loop_factory = BanEventLoop


# BanEventLoop.call_at = call_at
# asyncio.set_event_loop_policy(BanEventLoopPolicy())
