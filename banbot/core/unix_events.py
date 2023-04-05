#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : unix_events.py
# Author: anyongjin
# Date  : 2023/4/5
import sys
if sys.platform == 'win32':  # pragma: no cover
    raise ImportError('Signals are not really supported on Windows')
import asyncio
from banbot.core.events_base import *
from asyncio.unix_events import _UnixSelectorEventLoop, _UnixDefaultEventLoopPolicy


class BanSelectorEventLoop(_UnixSelectorEventLoop):
    pass


BanSelectorEventLoop.call_at = call_at


class BanSelectorEventLoopPolicy(_UnixDefaultEventLoopPolicy):
    _loop_factory = BanSelectorEventLoop


asyncio.set_event_loop_policy(BanSelectorEventLoopPolicy())
