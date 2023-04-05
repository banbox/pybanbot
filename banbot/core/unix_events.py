#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : unix_events.py
# Author: anyongjin
# Date  : 2023/4/5
import sys
if sys.platform == 'win32':  # pragma: no cover
    raise ImportError('Signals are not really supported on Windows')
import asyncio
import heapq
from banbot.core.events_base import *
from asyncio.unix_events import _UnixSelectorEventLoop, _UnixDefaultEventLoopPolicy


class BanSelectorEventLoop(_UnixSelectorEventLoop):

    def call_at(self, when, callback, *args, context=None):
        """Like call_later(), but uses an absolute time.

        Absolute time corresponds to the event loop's time() method.
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


class BanSelectorEventLoopPolicy(_UnixDefaultEventLoopPolicy):
    _loop_factory = BanSelectorEventLoop


asyncio.set_event_loop_policy(BanSelectorEventLoopPolicy())
