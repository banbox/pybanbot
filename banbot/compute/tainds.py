#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : tainds.py
# Author: anyongjin
# Date  : 2023/5/4
from banbot.compute.ctx import *


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
    import pandas as pd
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


class StaRSI(BaseInd):
    def __init__(self, period: int, cache_key: str = ''):
        super(StaRSI, self).__init__(cache_key)
        self.period = period
        self.gain_avg = 0
        self.loss_avg = 0
        self.last_input = np.nan

    def __call__(self, *args, **kwargs):
        input_val = args[0]
        assert hasattr(input_val, '__sub__')
        if self.calc_bar >= bar_num.get():
            return self.arr[-1]
        if np.isnan(self.last_input):
            self.last_input = input_val
            self.arr.append(np.nan)
            return self.arr[-1]
        val_delta = input_val - self.last_input
        self.last_input = input_val
        if len(self.arr) > self.period:
            if val_delta >= 0:
                gain_delta, loss_delta = val_delta, 0
            else:
                gain_delta, loss_delta = 0, val_delta
            self.gain_avg = (self.gain_avg * (self.period - 1) + gain_delta) / self.period
            self.loss_avg = (self.loss_avg * (self.period - 1) + loss_delta) / self.period
            self.arr.append(self.gain_avg * 100 / (self.gain_avg - self.loss_avg))
        else:
            if val_delta >= 0:
                self.gain_avg += val_delta / self.period
            else:
                self.loss_avg += val_delta / self.period
            if len(self.arr) == self.period:
                self.arr.append(self.gain_avg * 100 / (self.gain_avg - self.loss_avg))
            else:
                self.arr.append(np.nan)
        self.arr = self.arr[-600:]
        return self.arr[-1]


def RSI(arr: np.ndarray, period: int):
    '''
    相对强度指数。0-100之间。
    价格变化有的使用变化率，大部分使用变化值。这里使用变化值：price_chg
    :param arr:
    :param period:
    :return:
    '''
    if len(arr) <= period:
        return np.array([np.nan] * len(arr))
    price_chg = np.diff(arr)
    gain_arr = np.maximum(price_chg, 0)
    loss_arr = np.abs(np.minimum(price_chg, 0))
    gain_avg = np.average(gain_arr[:period])
    loss_avg = np.average(loss_arr[:period])
    result = [np.nan] * period
    result.append(gain_avg * 100 / (gain_avg + loss_avg))
    for i in range(period, len(price_chg)):
        gain_avg = (gain_avg * (period - 1) + gain_arr[i]) / period
        loss_avg = (loss_avg * (period - 1) + loss_arr[i]) / period
        result.append(gain_avg * 100 / (gain_avg + loss_avg))
    return np.array(result)


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

