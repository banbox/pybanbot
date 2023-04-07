#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : tainds.py
# Author: anyongjin
# Date  : 2023/3/24
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
from numbers import Number
from contextvars import Context, ContextVar, copy_context
from asyncio import Lock
from typing import *

import pandas as pd

from banbot.util.num_utils import *


# 所有对上下文变量的数据访问，均应先获取锁，避免协程交互访问时发生混乱
ctx_lock = Lock()
bar_num = ContextVar('bar_num')
symbol_tf = ContextVar('symbol_tf')
timeframe_secs = ContextVar('timeframe_secs')
pair_state = ContextVar('bar_state')
bar_arr = ContextVar('bar_arr')
fea_col_start = ContextVar('fea_col_start')
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
        pair_state.set(dict())
        bar_arr.set([])
        timeframe_secs.set(0)
        return
    if symbol not in _symbol_ctx:
        from banbot.exchange.exchange_utils import timeframe_to_seconds
        base_s, quote_s, tf = symbol.split('/')
        symbol_tf.set(symbol)
        bar_num.set(0)
        pair_state.set(dict())
        bar_arr.set([])
        timeframe_secs.set(timeframe_to_seconds(tf))
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

    async def __aenter__(self):
        self.back_symbol = symbol_tf.get()
        if self.back_symbol != self.symbol:
            # 仅当和当前环境的上下文不同时，才尝试获取协程锁；允许相同环境嵌套TempContext
            await ctx_lock.acquire()
            self.back_symbol = symbol_tf.get()
            self.is_locked = True
            set_context(self.symbol)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self.is_locked:
            return
        set_context(self.back_symbol)
        ctx_lock.release()
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


class StaSMA(BaseSimpleInd):
    def __init__(self, period: int, cache_key: str = ''):
        super(StaSMA, self).__init__(period, cache_key)
        self.dep_vals: List[Number] = []

    def _compute(self, val):
        self.dep_vals.append(val / self.period)
        if len(self.dep_vals) < self.period:
            return np.nan
        self.dep_vals = self.dep_vals[-self.period:]
        return sum(self.dep_vals)


def _to_nparr(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, (pd.Series, pd.DataFrame)):
        return arr.to_numpy()
    else:
        return np.array(arr)


def _nan_array(arr):
    result = np.zeros(len(arr))
    result[:] = np.nan
    return result


def SMA(arr: np.ndarray, period: int) -> np.ndarray:
    arr = _to_nparr(arr)
    assert isinstance(arr, np.ndarray) and len(arr.shape) == 1
    result = _nan_array(arr)
    div_arr = arr / period
    for i in range(period, len(arr)):
        result[i] = sum(div_arr[i - period: i])
    return result


class StaEMA(BaseSimpleInd):
    def __init__(self, period: int, cache_key: str = ''):
        super(StaEMA, self).__init__(period, cache_key)
        self.mul = 2 / (period + 1)

    def _compute(self, val):
        if not self.arr or np.isnan(self.arr[-1]):
            ind_val = val
        else:
            ind_val = val * self.mul + self.arr[-1] * (1 - self.mul)
        return ind_val


def EMA(arr: np.ndarray, period: int) -> np.ndarray:
    arr = _to_nparr(arr)
    assert isinstance(arr, np.ndarray) and len(arr.shape) == 1
    mul = 2 / (period + 1)
    result = _nan_array(arr)
    old_val = arr[0]
    result[0] = old_val
    for i in range(1, len(arr)):
        result[i] = arr[i] * mul + old_val * (1 - mul)
        old_val = result[i]
    return result


class StaTR(BaseInd):
    def __init__(self, cache_key: str = ''):
        super(StaTR, self).__init__(cache_key)

    def __call__(self, *args, **kwargs):
        arr = args[0]
        assert isinstance(arr, np.ndarray)
        if self.calc_bar >= bar_num.get():
            return self.arr[-1]
        crow = arr[-1, :]
        if arr.shape[0] < 2:
            cur_tr = crow[hcol] - crow[lcol]
        else:
            prow = arr[-2, :]
            cur_tr = max(crow[hcol] - crow[lcol], abs(crow[hcol] - prow[ccol]), abs(crow[lcol] - prow[ccol]))
        self.arr.append(cur_tr)
        self.arr = self.arr[-600:]
        self.calc_bar = bar_num.get()
        return cur_tr


def TR(arr) -> np.ndarray:
    arr = _to_nparr(arr)
    assert isinstance(arr, np.ndarray) and len(arr.shape) == 2 and arr.shape[1] >= 5
    result = _nan_array(arr)
    result[0] = arr[0, hcol] - arr[0, lcol]
    for i in range(1, result.shape[0]):
        crow, prow = arr[i, :], arr[i - 1, :]
        result[i] = max(crow[hcol] - crow[lcol], abs(crow[hcol] - prow[ccol]), abs(crow[lcol] - prow[ccol]))
    return result


class StaATR(BaseInd):
    def __init__(self, period: int = 3, cache_key: str = ''):
        super(StaATR, self).__init__(cache_key)
        self.period = period
        self.tr = StaTR(cache_key)

    def __call__(self, *args, **kwargs):
        arr = args[0]
        assert isinstance(arr, np.ndarray)
        if self.calc_bar >= bar_num.get():
            return self.arr[-1]
        tr_val = self.tr(arr)
        if not self.arr or np.isnan(self.arr[-1]):
            ind_val = sum(self.tr.arr[-self.period:]) / self.period
        else:
            ind_val = (self.arr[-1] * (self.period - 1) + tr_val) / self.period
        self.arr.append(ind_val)
        self.arr = self.arr[-600:]
        self.calc_bar = bar_num.get()
        return self.arr[-1]


def ATR(arr, period: int) -> np.ndarray:
    arr = _to_nparr(arr)
    assert isinstance(arr, np.ndarray) and len(arr.shape) == 2 and arr.shape[1] >= 5, f'{arr.shape} invalid'
    tr = TR(arr)
    result = _nan_array(arr)
    old_val = sum(tr[:period]) / period
    result[period] = old_val
    for i in range(period + 1, result.shape[0]):
        old_val = (tr[i] + old_val * (period - 1)) / period
        result[i] = old_val
    return result


class StaNATR(BaseInd):
    def __init__(self, period: int = 3, cache_key: str = ''):
        super(StaNATR, self).__init__(cache_key)
        self.period = period
        self.atr = StaATR(period, cache_key)

    def __call__(self, *args, **kwargs):
        arr = args[0]
        assert isinstance(arr, np.ndarray)
        if self.calc_bar >= bar_num.get():
            return self.arr[-1]
        ind_val = self.atr(arr)
        long_price_range = LongVar.get(LongVar.price_range)
        if arr.shape[0] < long_price_range.roll_len:
            ind_val = np.nan
        self.arr.append(ind_val / long_price_range.val)
        self.arr = self.arr[-600:]
        self.calc_bar = bar_num.get()
        return self.arr[-1]


class StaTRRoll(BaseInd):
    def __init__(self, period: int = 4, cache_key: str = ''):
        super(StaTRRoll, self).__init__(cache_key)
        self.period = period

    def __call__(self, *args, **kwargs):
        arr = args[0]
        assert isinstance(arr, np.ndarray)
        if self.calc_bar >= bar_num.get():
            return self.arr[-1]
        high_col, low_col = arr[-self.period:, hcol], arr[-self.period:, lcol]
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


class StaNTRRoll(BaseInd):
    def __init__(self, period: int = 4, roll_len: int = 4, cache_key: str = ''):
        super(StaNTRRoll, self).__init__(cache_key)
        self.tr_roll = StaTRRoll(period, cache_key)
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


class StaNVol(BaseInd):
    def __init__(self, cache_key: str = ''):
        super(StaNVol, self).__init__(cache_key)

    def __call__(self, *args, **kwargs):
        arr = args[0]
        assert isinstance(arr, np.ndarray)
        if self.calc_bar >= bar_num.get():
            return self.arr[-1]
        self.arr.append(arr[-1, vcol] / LongVar.get(LongVar.vol_avg).val)
        self.arr = self.arr[-600:]
        self.calc_bar = bar_num.get()
        return self.arr[-1]


class StaMACD(BaseInd):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, smooth_period: int = 9,
                 cache_key: str = ''):
        super(StaMACD, self).__init__(cache_key)
        self.ema_short = StaEMA(fast_period, cache_key)
        self.ema_long = StaEMA(slow_period, cache_key)
        self.ema_sgl = StaEMA(smooth_period, cache_key + 'macd')
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


def MACD(arr: np.ndarray, fast_period: int = 12, slow_period: int = 26, smooth_period: int = 9)\
        -> Tuple[np.ndarray, np.ndarray]:
    arr = _to_nparr(arr)
    ema_fast, ema_slow = EMA(arr, fast_period), EMA(arr, slow_period)
    macd = ema_fast - ema_slow
    signal = EMA(macd, smooth_period)
    return macd, signal


def _make_sub_malong():
    malong = StaSMA(120)

    def calc(arr):
        return abs(arr[-1, ccol] - malong(arr[-1, ccol]))
    return LongVar(calc, 900, 600)


def _make_atr_low():
    natr = StaNATR()

    def calc(arr):
        return avg_in_range(natr.arr, 0.1, 0.4)
    return LongVar(calc, 600, 600)


LongVar.create_fns.update({
    LongVar.sub_malong: _make_sub_malong,
    LongVar.atr_low: _make_atr_low
})

