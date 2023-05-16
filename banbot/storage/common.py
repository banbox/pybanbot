#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : global.py
# Author: anyongjin
# Date  : 2023/4/1
from banbot.config.consts import BotState
from typing import Optional


class BotGlobal:
    state = BotState.STOPPED

    stg_hash: Optional[str] = None
    '''策略+版本号的哈希值；如果和上次相同说明策略没有变化，可使用一个任务'''

    is_wramup = False
    '当前是否处于预热状态'
