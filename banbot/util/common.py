#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : utils.py
# Author: anyongjin
# Date  : 2023/2/11
import asyncio
import collections.abc
import logging
import sys
import threading
import time


def bell():
    import winsound
    freq = 440  # Hz
    winsound.Beep(freq, 500)


def to_snake_case(name):
    import re
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonArg(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        cls_key = f'{cls.__name__}{args}{kwargs}'
        if cls_key not in cls._instances:
            cls._instances[cls_key] = super(SingletonArg, cls).__call__(*args, **kwargs)
        return cls._instances[cls_key]


class Instance(object):
    '''
    Singleton helper class
    '''
    new_lock = threading.Lock()
    lock_dict = {}
    obj_dict = {}

    @staticmethod
    def get(cls, *args, **kwargs):
        cls_name = str(cls)
        return Instance.getobj(cls_name, cls, *args, **kwargs)

    @staticmethod
    def get_lock(key):
        if key not in Instance.lock_dict:
            with Instance.new_lock:
                if key not in Instance.lock_dict:
                    Instance.lock_dict[key] = threading.Lock()
        return Instance.lock_dict[key]

    @staticmethod
    def getobj(key, create_func, *args, **kwargs):
        '''
        get a singleton object by key
        :param key:
        :param create_func: 创建函数或值
        :param args:
        :param kwargs:
        :return:
        '''
        if key not in Instance.obj_dict:
            with Instance.get_lock(key):
                if key not in Instance.obj_dict:
                    if callable(create_func):
                        Instance.obj_dict[key] = create_func(*args, **kwargs)
                    else:
                        Instance.obj_dict[key] = create_func
        return Instance.obj_dict[key]

    @staticmethod
    def setobj(key, obj):
        with Instance.get_lock(key):
            Instance.obj_dict[key] = obj

    @staticmethod
    def remove(obj):
        cls_name = str(obj) if not isinstance(obj, str) else obj
        if cls_name in Instance.obj_dict:
            with Instance.lock_dict[cls_name]:
                if cls_name in Instance.obj_dict:
                    del Instance.obj_dict[cls_name]


class MeasureTime:
    def __init__(self, prefix: str = ''):
        from typing import List, Tuple
        self.prefix = prefix
        self.history: List[Tuple[str, float]] = []
        self.disable = False

    def start_for(self, name: str):
        if self.disable:
            return
        self.history.append((self.prefix + name, time.monotonic()))

    def print_all(self, top_n: int = 0, min_cost: float = 0):
        if self.disable or not self.history:
            return False
        self.history.append((self.prefix + 'end', time.monotonic()))
        cost_list, tags = [], set()
        for i in range(len(self.history) - 1):
            item, nt = self.history[i], self.history[i + 1]
            cost_list.append((item[0], nt[1] - item[1], 1))
            tags.add(item[0])
        tag_repeat = False
        if len(cost_list) > len(tags):
            tag_repeat = True
            # 有标签重复多次，按总时间排序
            from itertools import groupby
            cost_list = sorted(cost_list, key=lambda x: x[0])
            res_group = groupby(cost_list, key=lambda x: x[0])
            cost_list = []
            for key, group in res_group:
                gp_list = list(group)
                cost_sum = sum(list(map(lambda x: x[1], gp_list)))
                cost_list.append((key, cost_sum, len(gp_list)))
        if len(cost_list) > 2:
            cost_total = self.history[-1][1] - self.history[0][1]
            cost_list.append((self.prefix + 'total', cost_total, 0))
        cost_list = sorted(cost_list, key=lambda x: x[1], reverse=True)
        max_name_len = max(len(n[0]) for n in cost_list)
        max_name_len = max(max_name_len + 3, 12)
        title, cost_t = 'action', 'cost(s)'
        time_len = 10
        head_text = f'{title:<{max_name_len}}{cost_t:<{time_len}}'
        if tag_repeat:
            head_text += 'count'
        result = [head_text]
        for i, item in enumerate(cost_list):
            if top_n and i >= top_n:
                break
            if item[1] < min_cost:
                continue
            msg = f'{item[0]:<{max_name_len}}{item[1]:<{time_len}.5f}'
            if tag_repeat:
                msg += str(item[2])
            result.append(msg)
        if len(result) > 1:
            print('\n'.join(result))
        return True

    def total_secs(self):
        if not self.history:
            return 0
        return self.history[-1][1] - self.history[0][1]


class StrFormatLogRecord(logging.LogRecord):
    def getMessage(self) -> str:
        msg = str(self.msg)
        if self.args:
            try:
                msg %= ()
            except TypeError:
                # Either or the two is expected indicating there's
                # a placeholder to interpolate:
                #
                # - not all arguments converted during string formatting
                # - format requires a mapping" expected
                #
                # If would've been easier if Python printf-style behaved
                # consistently for "'' % (1,)" and "'' % {'foo': 1}". But
                # it raises TypeError only for the former case.
                msg %= self.args
            else:
                # There's special case of first mapping argument. See
                # duner init of logging.LogRecord.
                if isinstance(self.args, collections.abc.Mapping):
                    msg = msg.format(**self.args)
                else:
                    msg = msg.format(*self.args)
        return msg


class NotifyHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super(NotifyHandler, self).__init__(level)
        self.loop = None

    def emit(self, record: logging.LogRecord) -> None:
        from banbot.rpc.rpc_manager import RPCManager, RPCMessageType
        from banbot.util import btime
        try:
            if not RPCManager.instance or btime.run_mode not in btime.LIVE_MODES:
                return
            try:
                if self.loop is None:
                    self.loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            msg = self.format(record)
            self.loop.create_task(RPCManager.instance.send_msg(dict(
                type=RPCMessageType.EXCEPTION,
                status=msg,
            )))
        except RecursionError:  # See issue 36272
            raise
        except Exception:
            self.handleError(record)


def get_logger(level=logging.INFO):
    log = logging.getLogger('banbot')
    log.setLevel(level)
    log.propagate = False
    if log.hasHandlers():
        return log
    # 定义handler的输出格式
    formatter = logging.Formatter(fmt='%(asctime)s %(process)d %(levelname)s %(message)s')
    # 使用UTC时间
    formatter.converter = time.gmtime
    low_handler = logging.StreamHandler(sys.stdout)
    low_handler.setLevel(level)
    low_handler.setFormatter(formatter)
    low_handler.addFilter(lambda r: r.levelno < logging.ERROR)
    log.addHandler(low_handler)

    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    log.addHandler(error_handler)

    notify_handler = NotifyHandler(logging.ERROR)
    notify_handler.setFormatter(logging.Formatter(fmt='%(message)s'))
    log.addHandler(notify_handler)
    return log


logging.setLogRecordFactory(StrFormatLogRecord)
logger = get_logger()
