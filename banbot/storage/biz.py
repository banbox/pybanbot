#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : bix.py
# Author: anyongjin
# Date  : 2023/11/9
from typing import *
from banbot.storage.orders import InOutOrder


class BotCache:
    open_ods: Dict[int, InOutOrder] = dict()
    '全部打开的订单'

    updod_at = 0
    '上次刷新open_ods的时间戳，13位'
