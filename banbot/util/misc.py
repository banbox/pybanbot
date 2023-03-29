#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : misc.py
# Author: anyongjin
# Date  : 2023/3/22
import time
import logging
import sys


def utime(secs: int = 0, as_ms: bool = True):
    multipler = 1000 if as_ms else 1
    return round((time.time() + secs) * multipler)


def call_async(async_fn, *args, **kwargs):
    import asyncio
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(async_fn(*args, **kwargs))
    loop.close()
    return result


def nearly_group(data, max_pct=0.1, do_sort=True):
    '''
    近似分组方法，对于给定的一组数据，如果相邻之间相差在max_pct范围内，则认为是一组
    不适用于部分数据接近0的情况
    :param do_sort: 是否需要排序，如果data是无序的，需要改为升序
    :param data: [0.1, 0.11, 0.15, 1, 1.01, 5, 5.2, 9, 9.4, 9.31]
    :param max_pct: 0.1
    :return: [(0.105, 0.2), (0.15, 0.1), (1.005, 0.2), (5.1, 0.2), (9.253, 0.3)]
    返回分组后的结果，每一项代表一个组，第一个表示组内均值，第二个表示该组数量占比
    '''
    if data is None or len(data) == 0:
        return []
    if len(data) == 1:
        return [[data[0], 1]]
    import numpy as np
    from itertools import groupby
    if do_sort:
        data = sorted(data)
    data = np.array(data)
    distences = data[1:] - data[:-1]
    group_idxs = list(range(len(data)))  # 长度与data相同，初始为索引，后面合并后同一组的值相同
    dis_ids = zip(range(len(distences)), distences)
    dis_ids = list(sorted(dis_ids, key=lambda x: x[1]))  # 按间隔升序
    for p in dis_ids:
        idx, dis = p
        if dis < min(abs(data[idx + 1]), abs(data[idx])) * max_pct:
            old_val = group_idxs[idx + 1]
            group_idxs[idx + 1] = group_idxs[idx]
            cur_idx = idx + 2
            while cur_idx < len(group_idxs) and group_idxs[cur_idx] == old_val:
                group_idxs[cur_idx] = group_idxs[idx]
                cur_idx += 1
    feas = zip(range(len(group_idxs)), group_idxs)
    groups = []
    for key, glist in groupby(feas, lambda x: x[1]):
        glist = list(glist)
        avg = sum(map(lambda x: data[x[0]], glist)) / len(glist)
        groups.append([avg, len(glist)])
        for it in glist:
            group_idxs[it[0]] = key
    groups = list(sorted(groups, key=lambda x: x[1], reverse=True))
    return [[gp[0], gp[1] / len(data)] for gp in groups], group_idxs
