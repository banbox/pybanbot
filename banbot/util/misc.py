#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : misc.py
# Author: anyongjin
# Date  : 2023/3/22
import asyncio
import time
from typing import List, Tuple
import sys


def is_debug():
    gettrace = getattr(sys, 'gettrace', None)
    return gettrace and gettrace()


def utime(secs: int = 0, as_ms: bool = True):
    multipler = 1000 if as_ms else 1
    return round((time.time() + secs) * multipler)


async def run_async(func, *args, **kwargs):
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return func(*args, **kwargs)


def parallel_jobs(func, args_list: List[tuple]):
    '''
    并行执行异步任务。用于对一个函数调用多次，且顺序不重要的情况。
    :param func:
    :param args_list: 每一项是调用的参数。一个列表表示args参数，列表中有两项，且第一个是列表第二个字典，则分别作为args和kwargs
    :return: [dict(data=return, args=args, kwargs=kwargs), ...]
    '''
    async def wrap_func(*args, **kwargs):
        res = await func(*args, **kwargs)
        return dict(data=res, args=args, kwargs=kwargs)

    jobs = []
    for job in args_list:
        if len(job) == 2 and isinstance(job[0], (tuple, list)) and isinstance(job[1], dict):
            args, kwargs = job
        else:
            args, kwargs = job, {}
        jobs.append(asyncio.create_task(wrap_func(*args, **kwargs)))
    return asyncio.as_completed(jobs)


def safe_value_fallback(obj: dict, key1: str, key2: str, default_value=None):
    """
    Search a value in obj, return this if it's not None.
    Then search key2 in obj - return that if it's not none - then use default_value.
    Else falls back to None.
    """
    if key1 in obj and obj[key1] is not None:
        return obj[key1]
    else:
        if key2 in obj and obj[key2] is not None:
            return obj[key2]
    return default_value


def add_dict_prefix(data: dict, prefix: str) -> dict:
    return {f'{prefix}{k}': v for k, v in data.items()}


def del_dict_prefix(data: dict, prefix: str, with_others=True) -> dict:
    result = dict()
    pre_len = len(prefix)
    for key, val in data.items():
        if key.startswith(prefix):
            result[key[pre_len:]] = val
        elif with_others:
            result[key] = val
    return result


def deep_merge_dicts(source, destination, allow_null_overrides: bool = True):
    """
    使用source的值覆盖destination
    Values from Source override destination, destination is returned (and modified!!)
    Sample:
    >>> a = { 'first' : { 'rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == { 'first' : { 'rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            deep_merge_dicts(value, node, allow_null_overrides)
        elif value is not None or allow_null_overrides:
            destination[key] = value

    return destination


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


def build_fallback_serial(raise_error=True):
    import datetime
    import decimal

    def json_serial(obj):
        import enum
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        elif isinstance(obj, decimal.Decimal):
            return str(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, enum.Enum):
            return obj.value
        if raise_error:
            raise TypeError("Type %s not serializable" % type(obj))
        return str(obj)
    return json_serial


def json_dumps(val, **kwargs):
    import orjson
    json_serial = build_fallback_serial()
    return orjson.dumps(val, default=json_serial, **kwargs)


def safe_json_dumps(val, **kwargs):
    import orjson
    json_serial = build_fallback_serial(False)
    return orjson.dumps(val, default=json_serial, **kwargs)
