#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : async_events.py
# Author: anyongjin
# Date  : 2023/4/5
'''
引入此模块将针对async使用自定义的EventLoop。
当btime.run_mode不是TRADING_MODES时，自行调整修改，不睡眠
'''
import sys


if sys.platform == 'win32':  # pragma: no cover
    from .win_events import *
else:
    from .unix_events import *

