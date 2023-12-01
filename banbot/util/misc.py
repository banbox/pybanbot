#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : misc.py
# Author: anyongjin
# Date  : 2023/3/22
import asyncio
import sys
from typing import List, Optional, Callable, Dict, Any, ClassVar, Set
_run_env = None


def is_debug():
    gettrace = getattr(sys, 'gettrace', None)
    return gettrace and gettrace()


def get_run_env():
    global _run_env
    if not _run_env:
        import os
        _run_env = os.environ.get('ban_run_env') or 'dev'
    return _run_env


def is_prod_run():
    return get_run_env() == 'prod'


def hash_text(text: str, method: str = None):
    '''
    生成文本的哈希值
    :param text:
    :param method:
    :return:
    '''
    return hash_data(text.encode(), method)


def hash_data(data, method=None):
    '''
    生成文件的哈希值
    :param data:
    :param method: 计算哈希的方法, md5, sha1(默认)
    :return:
    '''
    import hashlib
    file_hash = hashlib.new(method if method else 'sha1')
    step = 4096
    if hasattr(data, 'encode'):
        data = data.encode()
    for i in range(0, len(data), step):
        file_hash.update(data[i: i + step])
    return file_hash.hexdigest()


async def run_async(func, *args, timeout=0, **kwargs):
    if asyncio.iscoroutinefunction(func):
        if not timeout:
            return await func(*args, **kwargs)
        return await asyncio.wait_for(func(*args, **kwargs), timeout)
    return func(*args, **kwargs)


def parallel_jobs(func, args_list: List[tuple]):
    '''
    并行执行异步任务。用于对一个函数调用多次，且顺序不重要的情况。
    :param func:
    :param args_list: 每一项是调用的参数。如果列表中有两项，且第一个是列表第二个字典，则分别作为args和kwargs；否则整个列表表示args参数
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
        jobs.append(wrap_func(*args, **kwargs))
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


def del_dict_prefix(data: dict, prefix: str, *skips) -> dict:
    result = dict()
    pre_len = len(prefix)
    del_keys = set()
    skip_keys = set(skips) if skips else set()
    for key, val in data.items():
        if key.startswith(prefix):
            sub_key = key[pre_len:]
            if sub_key in skip_keys:
                continue
            result[sub_key] = val
            del_keys.add(key)
    for key in del_keys:
        del data[key]
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


def groupby(data, key: Callable) -> Dict[Any, List]:
    '''
    对给定的数据进行分组。
    返回：[(key1, list1), ...]
    '''
    if data is None or not len(data):
        return dict()
    from itertools import groupby as gpby
    data = sorted(data, key=key)
    gps = gpby(data, key=key)
    return {key: list(gp) for key, gp in gps}


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


def get_module_classes(module, base_cls: type):
    import inspect
    clsmembers = inspect.getmembers(module, inspect.isclass)
    result = []
    for (name, cld_cls) in clsmembers:
        if not issubclass(cld_cls, base_cls) or cld_cls == base_cls:
            continue
        result.append(cld_cls)
    return result


def ensure_event_loop():
    '''
    检查事件循环是否存在，如不存在则设置。
    用于线程的事件循环初始化
    '''
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def new_async_thread():
    '''
    开启一个新的线程专门运行长期执行的异步任务
    '''
    import threading
    loop = asyncio.new_event_loop()

    def _worker():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return loop, thread


class LazyTqdm:
    def __init__(self, *args, **kwargs):
        from tqdm import tqdm
        from banbot.util import btime
        import sys
        self.show_bar = btime.run_mode not in btime.LIVE_MODES or sys.stdout.isatty()
        self.bar: Optional[tqdm] = None
        self.args = args
        self.kwargs = kwargs

    def update(self, n=1):
        if not self.show_bar:
            return
        if self.bar is None:
            from tqdm import tqdm
            self.bar = tqdm(*self.args, **self.kwargs)
        self.bar.update(n)

    def close(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None


class LocalLock:
    '''
    带超时的异步锁。只对同一进程的不同协程或线程有效。
    根据键维护锁对象。超时未获取锁时发出asyncio.TimeoutError异常
    慎用此锁，使用不当会导致连接被拒绝
    当和数据库对象一起使用时，应当在锁内申请数据库连接，以确保数据库对象能被正确保存
    '''
    _locks: ClassVar[Dict[str, asyncio.Lock]] = dict()
    _holds: ClassVar[Dict[str, int]] = dict()
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()  # 全局锁用于同步 _locks 和 _holds

    def __init__(self, key: str, timeout=60, force_on_fail=False):
        '''
        :param key: 锁的键
        :param timeout: 超时时间，默认60秒
        :param force_on_fail: 超时是否强制获取锁而不发出异常
        '''
        self.key = key
        self.timeout = timeout
        self.force = force_on_fail
        self._obj: Optional[asyncio.Lock] = None

    async def __aenter__(self):
        async with self._lock:
            if self.key not in self._locks:
                self._locks[self.key] = asyncio.Lock()
                self._holds[self.key] = 1
            else:
                self._holds[self.key] += 1

        lock = self._locks[self.key]

        try:
            if not self.timeout:
                await lock.acquire()
            else:
                await asyncio.wait_for(lock.acquire(), self.timeout)
            self._obj = lock
        except asyncio.TimeoutError:
            async with self._lock:
                if self.key in self._holds:
                    self._holds[self.key] -= 1
            if self.force:
                from banbot.util.common import logger
                logger.error(f'get lock timeout, force lock: {self.key}')
            else:
                raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._obj is None:
            return
        try:
            if self._obj.locked():
                self._obj.release()
        finally:
            async with self._lock:
                if self.key in self._holds:
                    self._holds[self.key] -= 1
                hold_num = self._holds.get(self.key) or 0
                if hold_num <= 0 and self.key in self._locks:
                    # 只有当此锁没有其他使用者时，才删除引用
                    del self._locks[self.key]
                    if self.key in self._holds:
                        del self._holds[self.key]


class Sleeper:
    "Group sleep calls allowing instant cancellation of all"
    tasks: ClassVar[Set[asyncio.Task]] = set()

    @classmethod
    async def sleep(cls, delay, result=None):
        coro = asyncio.sleep(delay, result=result)
        task = asyncio.ensure_future(coro)
        cls.tasks.add(task)
        try:
            return await task
        except asyncio.CancelledError:
            return result
        finally:
            cls.tasks.remove(task)

    @classmethod
    async def cancel_all(cls):
        "Coroutine cancelling tasks"
        cancelled = set()
        for task in cls.tasks:
            if task.cancel():
                cancelled.add(task)
        await asyncio.wait(cls.tasks)
        cls.tasks -= cancelled
        return len(cancelled)
