#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : async_events.py
# Author: anyongjin
# Date  : 2023/4/5
'''
引入此模块将针对async使用自定义的EventLoop。
当btime.run_mode不是TRADING_MODES时，自行调整修改，不睡眠

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


if sys.platform == 'win32':  # pragma: no cover
    from .win_events import *
else:
    from .unix_events import *

