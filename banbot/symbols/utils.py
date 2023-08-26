#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : utils.py
# Author: anyongjin
# Date  : 2023/8/26
from typing import *
from banbot.storage import ExSymbol

__all__ = ['group_symbols']


def group_symbols(symbols: Iterable[str]) -> Dict[str, List[str]]:
    '''
    将交易对按定价币分组，返回base部分列表形成的字典。用于通知时减少冗余信息
    '''
    from itertools import groupby
    items = []
    for pair in symbols:
        exs = ExSymbol(symbol=pair)
        items.append((exs.base_code, exs.quote_code))
    items = sorted(items, key=lambda x: x[1])
    gps = groupby(items, key=lambda x: x[1])
    result = dict()
    for key, gp in gps:
        result[key] = sorted([item[0] for item in gp])
    return result
