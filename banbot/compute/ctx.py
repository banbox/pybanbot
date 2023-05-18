#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : ctx.py
# Author: anyongjin
# Date  : 2023/5/4
'''
上下文变量的归类：pair+timeframe
跟策略无关，不考虑策略。（即pair+timeframe对应多个策略时，也是同一个上下文环境）
请勿在以下协程方法中使用上下文变量，会创建新的上下文：
get_event_loop
create_task
wait_fir
wait
gather
'''
from contextvars import Context, ContextVar, copy_context
from numbers import Number
from typing import *

import numpy as np

# 所有对上下文变量的数据访问，均应先获取锁，避免协程交互访问时发生混乱
bar_num = ContextVar('bar_num')
symbol_tf = ContextVar('symbol_tf')
bar_arr = ContextVar('bar_arr')
fea_col_start = ContextVar('fea_col_start')
bar_time = ContextVar('bar_time', default=(0, 0))
_symbol_ctx: Dict[str, Context] = dict()
symbol_tf.set('')
# 几个常用列的索引
tcol, ocol, hcol, lcol, ccol, vcol = 0, 1, 2, 3, 4, 5


def _update_context(kwargs):
    for key, val in kwargs:
        key.set(val)


def get_cur_symbol(ctx: Optional[Context] = None) -> Tuple[str, str, str, str]:
    pair_tf = ctx[symbol_tf] if ctx else symbol_tf.get()
    base_symbol, quote_symbol, timeframe = pair_tf.split('/')
    return f'{base_symbol}/{quote_symbol}', base_symbol, quote_symbol, timeframe


def get_context(pair_tf: str) -> Context:
    '''
    此方法获取一个缓存的上下文环境。仅可用于读取，不可用于写入。
    :param pair_tf:
    :return:
    '''
    if symbol_tf.get() == pair_tf:
        return copy_context()
    return _symbol_ctx[pair_tf]


def set_context(symbol: str):
    '''
    设置交易对和时间单元上下文。
    :param symbol: BTC/USDT/1m
    :return:
    '''
    old_ctx = copy_context()
    if symbol_tf in old_ctx and symbol_tf.get() in _symbol_ctx:
        if symbol_tf.get() == symbol:
            # 上下文未变化，直接退出
            return
        # 保存旧的值到旧的上下文
        save_ctx = _symbol_ctx[symbol_tf.get()]
        save_ctx.run(_update_context, old_ctx.items())
    if not symbol:
        symbol_tf.set('')
        bar_num.set(0)
        bar_arr.set([])
        return
    if symbol not in _symbol_ctx:
        symbol_tf.set(symbol)
        bar_num.set(0)
        bar_arr.set([])
        _symbol_ctx[symbol] = copy_context()
    else:
        # 从新的上下文恢复上次的状态
        _update_context(_symbol_ctx[symbol].items())


class TempContext:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.is_locked = False
        self.back_symbol = None

    def __enter__(self):
        self.back_symbol = symbol_tf.get()
        if self.back_symbol != self.symbol:
            self.back_symbol = symbol_tf.get()
            self.is_locked = True
            set_context(self.symbol)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_locked:
            return
        set_context(self.back_symbol)
        self.is_locked = False


def avg_in_range(view_arr, start_rate=0.25, end_rate=0.75, is_abs=True) -> float:
    if is_abs:
        view_arr = np.abs(view_arr)
    sort_arr = np.sort(view_arr, axis=None)
    arr_len = len(sort_arr)
    start, end = round(arr_len * start_rate), round(arr_len * end_rate)
    return np.average(sort_arr[start: end])


class LongVar:
    '''
    更新频率较低的市场信息；同一指标，不同交易对不同周期，需要的参数可能不同。
    更新方法：一定周期的K线数据更新、API接口计算更新。
    （创建参数：更新函数、更新周期、）
    更新间隔：60s的整数倍
    '''
    _instances: Dict[str, 'LongVar'] = dict()
    create_fns: Dict[str, Callable] = dict()

    price_range = 'price_range'
    vol_avg = 'vol_avg'
    bar_len = 'bar_len'
    sub_malong = 'sub_malong'
    atr_low = 'atr_low'

    def __init__(self, calc_func: Callable, roll_len: int, update_step: int, old_rate=0.7):
        self.old_rate = old_rate
        self.calc_func = calc_func
        self.val = np.nan
        self.update_step = update_step
        self.roll_len = roll_len
        self.update_at = 0

    def on_bar(self, arr: np.ndarray):
        if arr.shape[0] < self.roll_len:
            return
        num_off = bar_num.get() - self.update_at
        cur_val = self.val
        if np.isnan(cur_val) or num_off == self.update_step:
            new_val = self.calc_func(arr[-self.roll_len:, :])
            if np.isnan(cur_val):
                cur_val = new_val
            else:
                cur_val = cur_val * self.old_rate + new_val * (1 - self.old_rate)
            self.val = cur_val
            self.update_at = bar_num.get()

    @classmethod
    def update(cls, arr: np.ndarray):
        symbol = symbol_tf.get()
        for key, obj in cls._instances.items():
            if not key.startswith(symbol):
                continue
            obj.on_bar(arr)

    @classmethod
    def _create(cls, key: str) -> 'LongVar':
        if key in cls.create_fns:
            return cls.create_fns[key]()
        elif key == cls.price_range:
            return LongVar(lambda view_arr: np.max(view_arr[:, hcol]) - np.min(view_arr[:, lcol]), 600, 600)
        elif key == cls.vol_avg:
            return LongVar(lambda view_arr: np.average(view_arr[:, vcol]), 600, 600)
        elif key == cls.bar_len:
            return LongVar(lambda arr: avg_in_range(arr[:, ocol] - arr[:, ccol]), 600, 600)
        else:
            raise ValueError(f'unknown long key: {key}')

    @classmethod
    def get(cls, key: str) -> 'LongVar':
        symbol = symbol_tf.get()
        cache_key = f'{symbol}_{key}'
        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls._create(key)
        return cls._instances[cache_key]


class CachedInd(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        cls_key = f'{symbol_tf.get()}_{cls.__name__}{args}{kwargs}'
        if cls_key not in cls._instances:
            cls._instances[cls_key] = super(CachedInd, cls).__call__(*args, **kwargs)
        return cls._instances[cls_key]


class BaseInd(metaclass=CachedInd):
    def __init__(self, cache_key: str = ''):
        self.cache_key = cache_key
        self.calc_bar = 0
        self.arr: List[float] = []

    def __getitem__(self, index):
        return self.arr[index]

    def __len__(self):
        return len(self.arr)


class BaseSimpleInd(BaseInd):
    def __init__(self, period: int, cache_key: str = ''):
        super(BaseSimpleInd, self).__init__(cache_key)
        self.period = period

    def __call__(self, *args, **kwargs):
        if self.calc_bar >= bar_num.get():
            return self.arr[-1]
        val = args[0]
        assert isinstance(val, Number)
        ind_val = self._compute(val)
        self.arr.append(ind_val)
        self.calc_bar = bar_num.get()
        return ind_val

    def _compute(self, val):
        raise NotImplementedError(f'_compute in {self.__class__.__name__}')

