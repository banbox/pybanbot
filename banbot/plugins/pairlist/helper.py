#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : helper.py
# Author: anyongjin
# Date  : 2023/4/19
import re
from typing import List


def search_pairlist(wildcardpl: List[str], available_pairs: List[str],
                    keep_invalid: bool = False) -> List[str]:
    '''
    根据给定的可能包含通配符的列表，从交易所支持的所有交易对中，尝试搜索匹配并返回匹配的列表。
    :param wildcardpl:
    :param available_pairs:
    :param keep_invalid:
    :return:
    '''
    result = []
    for pair_wc in wildcardpl:
        try:
            comp = re.compile(pair_wc, re.IGNORECASE)
            matches = [pair for pair in available_pairs if re.fullmatch(comp, pair)]
            if not matches and keep_invalid and re.fullmatch(r'^[A-Za-z0-9/-]+$', pair_wc):
                matches.append(pair_wc)
            result += matches
        except re.error as err:
            raise ValueError(f"Wildcard error in {pair_wc}, {err}")
    return result
