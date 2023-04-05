#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : events_base.py
# Author: anyongjin
# Date  : 2023/4/5
import time
import heapq
from asyncio.events import TimerHandle
from banbot.util import btime


class BanTimerHandle(TimerHandle):
    def __init__(self, when, callback, args, loop, context=None):
        self.sval = when
        self.start = btime.cur_timestamp
        if btime.run_mode not in btime.TRADING_MODES:
            when = time.monotonic()
        super(BanTimerHandle, self).__init__(when, callback, args, loop, context)

    def _run(self) -> None:
        if self.sval > self._when:
            btime.cur_timestamp = self.start + self.sval - self._when
        super(BanTimerHandle, self)._run()

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

