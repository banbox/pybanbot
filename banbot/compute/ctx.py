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
from typing import *

import numpy as np
from banbot.storage.symbols import ExSymbol

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


def get_cur_symbol(ctx: Optional[Context] = None) -> Tuple[ExSymbol, str]:
    pair_tf = ctx[symbol_tf] if ctx else symbol_tf.get()
    exg_name, market, symbol, timeframe = pair_tf.split('_')
    exs = ExSymbol(exchange=exg_name, market=market, symbol=symbol)
    return exs, timeframe


def get_context(pair_tf: str) -> Context:
    '''
    此方法获取一个缓存的上下文环境。仅可用于读取，不可用于写入。
    :param pair_tf:
    :return:
    '''
    if symbol_tf.get() == pair_tf:
        return copy_context()
    return _symbol_ctx[pair_tf]


def set_context(pair_tf: str):
    '''
    设置交易对和时间单元上下文。
    :param pair_tf: binance/future/BTC/USDT/1m
    :return:
    '''
    old_ctx = copy_context()
    if symbol_tf in old_ctx and symbol_tf.get() in _symbol_ctx:
        if symbol_tf.get() == pair_tf:
            # 上下文未变化，直接退出
            return
        # 保存旧的值到旧的上下文
        save_ctx = _symbol_ctx[symbol_tf.get()]
        save_ctx.run(_update_context, old_ctx.items())
    if not pair_tf:
        reset_context('')
        return
    if pair_tf not in _symbol_ctx:
        reset_context(pair_tf)
        _symbol_ctx[pair_tf] = copy_context()
    else:
        # 从新的上下文恢复上次的状态
        _update_context(_symbol_ctx[pair_tf].items())


def reset_context(pair_tf: str):
    symbol_tf.set(pair_tf)
    bar_num.set(0)
    bar_arr.set([])
    bar_time.set((0, 0))
    if pair_tf:
        LongVar.reset(pair_tf)
    if pair_tf:
        CachedInd.reset(pair_tf)


class TempContext:
    def __init__(self, pair_tf: str):
        self.pair_tf = pair_tf
        self.is_locked = False
        self.back_symbol = None

    def __enter__(self):
        self.back_symbol = symbol_tf.get()
        if self.back_symbol != self.pair_tf:
            self.back_symbol = symbol_tf.get()
            self.is_locked = True
            set_context(self.pair_tf)

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

    @classmethod
    def reset(cls, ctx_key: str):
        for key in cls._instances:
            if key.startswith(ctx_key):
                del cls._instances[key]


class CachedInd(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        pair_tf = symbol_tf.get()
        if not pair_tf:
            raise ValueError('State Inds should be create with context!')
        cls_key = f'{pair_tf}_{cls.__name__}{args}{kwargs}'
        if cls_key not in cls._instances:
            cls._instances[cls_key] = super(CachedInd, cls).__call__(*args, **kwargs)
        return cls._instances[cls_key]

    @classmethod
    def reset(cls, ctx_key: str):
        del_keys = {key for key in cls._instances if key.startswith(ctx_key)}
        for key in del_keys:
            del cls._instances[key]


class BaseInd(metaclass=CachedInd):
    input_dim = 0  # 0表示输入一个数字，1表示输入一行，2表示输入二维数组
    keep_num = 600

    def __init__(self, cache_key: str = ''):
        self.cache_key = cache_key
        self.calc_bar = 0
        self.arr: List[Any] = []

    def __getitem__(self, index):
        return self.arr[index]

    def __len__(self):
        return len(self.arr)

    def __call__(self, in_val: Union[np.ndarray, float]):
        cur_bar_no = bar_num.get()
        if self.calc_bar >= cur_bar_no:
            return self.arr[-1]
        # 检查输入数据的有效性
        if isinstance(in_val, np.ndarray):
            if self.input_dim != in_val.ndim:
                raise ValueError(f'input val dim error: {in_val.ndim}, require: {self.input_dim}')
            ind_val = self._compute_arr(in_val)
        else:
            if self.input_dim:
                raise ValueError(f'input val dim error: 0, require: {self.input_dim}')
            ind_val = self._compute_val(in_val)
        self.arr.append(ind_val)
        if len(self.arr) > self.keep_num * 1.5:
            self.arr = self.arr[-self.keep_num:]
        self.calc_bar = cur_bar_no
        return ind_val

    def _compute_val(self, val: float):
        raise NotImplementedError(f'_compute in {self.__class__.__name__}')

    def _compute_arr(self, val: np.ndarray):
        raise NotImplementedError(f'_compute in {self.__class__.__name__}')
