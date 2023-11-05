#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : support.py
# Author: anyongjin
# Date  : 2023/11/5

import collections
from typing import Deque, OrderedDict
from asyncio import Future
from banbot.util.misc import *
from banbot.util.common import *


class BanEvent:
    _listeners: Dict[str, Deque[Future]] = dict()
    _queue = asyncio.Queue(1000)
    '事件队列'
    _handlers: Dict[str, OrderedDict[Callable, bool]] = dict()
    '事件处理函数'
    _running = False

    @classmethod
    def set(cls, name: str, data):
        if name not in cls._listeners:
            return
        for fut in cls._listeners[name]:
            if not fut.done():
                fut.set_result(data)

    @classmethod
    def get_future(cls, name: str):
        if name not in cls._listeners:
            cls._listeners[name] = collections.deque()
        queue = cls._listeners[name]
        fut = asyncio.get_running_loop().create_future()
        queue.append(fut)
        return fut

    @classmethod
    async def wait_future(cls, name: str, fut: Future, timeout: int = None):
        if name not in cls._listeners:
            cls._listeners[name] = collections.deque()
        queue = cls._listeners[name]
        try:
            if timeout:
                return await asyncio.wait_for(fut, timeout)
            else:
                return await fut
        finally:
            if fut in queue:
                queue.remove(fut)

    @classmethod
    async def wait(cls, name: str, timeout: int = None):
        """等待一个事件被触发，然后处理，完成后移除监听"""
        if name not in cls._listeners:
            cls._listeners[name] = collections.deque()
        queue = cls._listeners[name]
        fut = asyncio.get_running_loop().create_future()
        queue.append(fut)
        try:
            if timeout:
                return await asyncio.wait_for(fut, timeout)
            else:
                return await fut
        finally:
            queue.remove(fut)

    # 事件处理部分

    @classmethod
    def emit(cls, name: str, data):
        """触发一个事件"""
        if not cls._running:
            logger.error(f'event handlers not running, emit fail: {name}')
        cls._queue.put_nowait((name, data))

    @classmethod
    def on(cls, name: str, funz: Callable, with_db=False):
        """监听一个事件，重复触发执行"""
        if name not in cls._handlers:
            cls._handlers[name] = collections.OrderedDict()
        handlers = cls._handlers[name]
        handlers[funz] = with_db

    @classmethod
    def off(cls, name: str, funz: Callable):
        """取消监听一个事件"""
        if name not in cls._handlers:
            return
        handlers = cls._handlers[name]
        if funz not in handlers:
            return
        del handlers[funz]

    @classmethod
    async def run_forever(cls):
        from banbot.storage import reset_ctx
        reset_ctx()
        name, data = None, None
        cls._running = True
        while True:
            try:
                name, data = await cls._queue.get()
                await cls._run_event(name, data)
            except Exception as e:
                logger.exception(f"BanEvent.on {name} error: {e}")
            cls._queue.task_done()

    @classmethod
    async def _run_event(cls, name: str, data):
        if name not in cls._handlers:
            logger.info(f'event {name} has no listeners')
            return
        handlers = cls._handlers[name]
        need_db = any(handlers.values())
        if not need_db:
            for funz in handlers:
                await run_async(funz, data)
            return
        from banbot.storage import dba
        async with dba():
            for funz in handlers:
                await run_async(funz, data)
