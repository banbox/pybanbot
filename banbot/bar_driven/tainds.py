#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : tainds.py
# Author: anyongjin
# Date  : 2023/3/24
from numbers import Number
from contextvars import Context, ContextVar
from typing import *

import numpy as np

from banbot.util.num_utils import *


ohcl_arr = ContextVar('ohcl_arr')
bar_num = ContextVar('bar_num')
symbol_tf = ContextVar('symbol_tf')
bar_num.set(0)


def avg_in_range(view_arr, start_rate=0.25, end_rate=0.75, is_abs=True) -> float:
    if is_abs:
        view_arr = np.abs(view_arr)
    sort_arr = np.sort(view_arr, axis=None)
    arr_len = len(sort_arr)
    start, end = round(arr_len * start_rate), round(arr_len * end_rate)
    return np.average(sort_arr[start: end])


class LongVar:
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
            return LongVar(lambda view_arr: np.max(view_arr[:, 1]) - np.min(view_arr[:, 2]), 600, 600)
        elif key == cls.vol_avg:
            return LongVar(lambda view_arr: np.average(view_arr[:, 4]), 600, 600)
        elif key == cls.bar_len:
            return LongVar(lambda arr: avg_in_range(arr[:, 0] - arr[:, 3]), 600, 600)
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
        cls_key = f'{symbol_tf.get()}{cls.__name__}{args}{kwargs}'
        if cls_key not in cls._instances:
            cls._instances[cls_key] = super(CachedInd, cls).__call__(*args, **kwargs)
        return cls._instances[cls_key]


class BaseInd(metaclass=CachedInd):
    def __init__(self, cache_key: str = ''):
        self.cache_key = cache_key
        self.calc_bar = bar_num.get()
        self.arr: List[float] = []


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


class SMA(BaseSimpleInd):
    def __init__(self, period: int, cache_key: str = ''):
        super(SMA, self).__init__(period, cache_key)
        self.dep_vals: List[Number] = []

    def _compute(self, val):
        self.dep_vals.append(val / self.period)
        if len(self.dep_vals) < self.period:
            return np.nan
        self.dep_vals = self.dep_vals[-self.period:]
        return sum(self.dep_vals)


class EMA(BaseSimpleInd):
    def __init__(self, period: int, cache_key: str = ''):
        super(EMA, self).__init__(period, cache_key)
        self.mul = 2 / (period + 1)

    def _compute(self, val):
        if not self.arr or np.isnan(self.arr[-1]):
            ind_val = val
        else:
            ind_val = val * self.mul + self.arr[-1] * (1 - self.mul)
        return ind_val


class TR(BaseInd):
    def __init__(self, cache_key: str = ''):
        super(TR, self).__init__(cache_key)

    def __call__(self, *args, **kwargs):
        arr = args[0]
        assert isinstance(arr, np.ndarray)
        if self.calc_bar >= bar_num.get():
            return self.arr[-1]
        crow = arr[-1, :]
        if arr.shape[0] < 2:
            cur_tr = crow[1] - crow[2]
        else:
            prow = arr[-2, :]
            cur_tr = max(crow[1] - crow[2], abs(crow[1] - prow[3]), abs(crow[2] - prow[3]))
        self.arr = self.arr[-600:]
        self.calc_bar = bar_num.get()
        return cur_tr


class NATR(BaseInd):
    def __init__(self, period: int = 3, cache_key: str = ''):
        super(NATR, self).__init__(cache_key)
        self.period = period
        self.tr = TR(cache_key)

    def __call__(self, *args, **kwargs):
        arr = args[0]
        assert isinstance(arr, np.ndarray)
        if self.calc_bar >= bar_num.get():
            return self.arr[-1]
        tr_val = self.tr(arr)
        long_price_range = LongVar.get(LongVar.price_range)
        if arr.shape[0] < long_price_range.roll_len:
            ind_val = np.nan
        elif not self.arr or np.isnan(self.arr[-1]):
            ind_val = sum(self.tr.arr[-self.period:]) / self.period
        else:
            ind_val = (self.arr[-1] * (self.period - 1) + tr_val) / self.period
        self.arr.append(ind_val / long_price_range.val)
        self.arr = self.arr[-600:]
        self.calc_bar = bar_num.get()
        return self.arr[-1]


class TRRoll(BaseInd):
    def __init__(self, period: int = 4, cache_key: str = ''):
        super(TRRoll, self).__init__(cache_key)
        self.period = period

    def __call__(self, *args, **kwargs):
        arr = args[0]
        assert isinstance(arr, np.ndarray)
        if self.calc_bar >= bar_num.get():
            return self.arr[-1]
        high_col, low_col = arr[-self.period:, 1], arr[-self.period:, 2]
        max_id = np.argmax(high_col)
        roll_max = high_col[max_id]
        min_id = np.argmin(low_col)
        roll_min = low_col[min_id]

        prev_tr = roll_max - roll_min
        if self.period - min(max_id, min_id) <= 2:
            # 如果是从最后两个蜡烛计算得到的，则是重要的波动范围。
            ind_val = prev_tr
        else:
            # 从前面计算的TrueRange加权缩小，和最后两个蜡烛的TrueRange取最大值
            prev_tr *= (min(max_id, min_id) + 1) / (self.period * 2) + 0.5
            ind_val = max(prev_tr, np.max(high_col[-2:]) - np.min(low_col[-2:]))
        self.arr.append(ind_val)
        self.arr = self.arr[-600:]
        self.calc_bar = bar_num.get()
        return ind_val


class NTRRoll(BaseInd):
    def __init__(self, period: int = 4, roll_len: int = 4, cache_key: str = ''):
        super(NTRRoll, self).__init__(cache_key)
        self.tr_roll = TRRoll(period, cache_key)
        self.roll_len = roll_len

    def __call__(self, *args, **kwargs):
        arr = args[0]
        assert isinstance(arr, np.ndarray)
        if self.calc_bar >= bar_num.get():
            return self.arr[-1]
        tr_val = self.tr_roll(arr)
        self.arr.append(tr_val / LongVar.get(LongVar.price_range).val)
        self.arr = self.arr[-600:]
        self.calc_bar = bar_num.get()
        return self.arr[-1]


class NVol(BaseInd):
    def __init__(self, cache_key: str = ''):
        super(NVol, self).__init__(cache_key)

    def __call__(self, *args, **kwargs):
        arr = args[0]
        assert isinstance(arr, np.ndarray)
        if self.calc_bar >= bar_num.get():
            return self.arr[-1]
        self.arr.append(arr[-1, 4] / LongVar.get(LongVar.vol_avg).val)
        self.arr = self.arr[-600:]
        self.calc_bar = bar_num.get()
        return self.arr[-1]


class MACD(BaseInd):
    def __init__(self, short_period: int = 12, long_period: int = 26, smooth_period: int = 9,
                 cache_key: str = ''):
        super(MACD, self).__init__(cache_key)
        self.ema_short = EMA(short_period, cache_key)
        self.ema_long = EMA(long_period, cache_key)
        self.ema_sgl = EMA(smooth_period, cache_key + 'macd')
        self.macd_arr = []
        self.singal_arr = []

    def __call__(self, *args, **kwargs):
        val = args[0]
        assert isinstance(val, Number)
        if self.calc_bar >= bar_num.get():
            return [self.macd_arr[-1], self.singal_arr[-1]]
        macd = self.ema_short(val) - self.ema_long(val)
        self.macd_arr.append(macd)
        self.macd_arr = self.macd_arr[-600:]
        singal = self.ema_sgl(macd)
        self.singal_arr.append(singal)
        self.singal_arr = self.singal_arr[-600:]
        self.calc_bar = bar_num.get()
        return macd, singal


def _make_sub_malong():
    malong = SMA(120)

    def calc(arr):
        return abs(arr[-1, 3] - malong(arr[-1, 3]))
    return LongVar(calc, 900, 600)


def _make_atr_low():
    natr = NATR()

    def calc(arr):
        return avg_in_range(natr.arr, 0.1, 0.4)
    return LongVar(calc, 600, 600)


LongVar.create_fns.update({
    LongVar.sub_malong: _make_sub_malong,
    LongVar.atr_low: _make_atr_low
})

