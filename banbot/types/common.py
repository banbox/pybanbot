#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : common.py
# Author: anyongjin
# Date  : 2023/11/6
from dataclasses import dataclass


@dataclass
class PairInfo:
    """策略额外需要的信息"""
    pair: str
    timeframe: str
    warmup_num: int = 30
