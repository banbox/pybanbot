#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : ban_proactor.py
# Author: anyongjin
# Date  : 2023/4/5
import sys

if sys.platform != 'win32':  # pragma: no cover
    raise ImportError('win32 only')
import asyncio
from banbot.core.events_base import *
from asyncio.windows_events import *
from asyncio.events import BaseDefaultEventLoopPolicy


class BanProactorEventLoop(ProactorEventLoop):
    pass


BanProactorEventLoop.call_at = call_at


class BanProactorEventLoopPolicy(BaseDefaultEventLoopPolicy):
    _loop_factory = BanProactorEventLoop


asyncio.set_event_loop_policy(BanProactorEventLoopPolicy())

