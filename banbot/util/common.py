#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : utils.py
# Author: anyongjin
# Date  : 2023/2/11
import time
import sys
import logging
import collections.abc


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


class MeasureTime:
    def __init__(self, prefix: str = ''):
        from typing import List, Tuple
        self.prefix = prefix
        self.history: List[Tuple[str, float]] = []

    def start_for(self, name: str):
        self.history.append((self.prefix + name, time.monotonic()))

    def print_all(self, top_n: int = 0):
        self.history.append((self.prefix + 'end', time.monotonic()))
        cost_list = []
        for i in range(len(self.history) - 1):
            item, nt = self.history[i], self.history[i + 1]
            cost_list.append((item[0], nt[1] - item[1]))
        if len(cost_list) > 2:
            cost_list.append((self.prefix + 'total', self.history[-1][1] - self.history[0][1]))
        cost_list = sorted(cost_list, key=lambda x: x[1], reverse=True)
        max_name_len = max(len(n[0]) for n in cost_list)
        max_name_len = max(max_name_len + 3, 12)
        title, cost_t = 'action', 'cost(s)'
        print(f'{title:<{max_name_len}}{cost_t}')
        for i, item in enumerate(cost_list):
            if top_n and i >= top_n:
                break
            print(f'{item[0]:<{max_name_len}}{item[1]:.5f}')


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


def get_logger(level=logging.INFO) -> TradeLogger:
    log: TradeLogger = logging.getLogger('banbot')
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
    return log


logging.setLoggerClass(TradeLogger)
logging.setLogRecordFactory(StrFormatLogRecord)
logger = get_logger()
