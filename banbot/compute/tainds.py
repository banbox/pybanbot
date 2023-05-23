#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : tainds.py
# Author: anyongjin
# Date  : 2023/5/4
from banbot.compute.ctx import *
from banbot.util.num_utils import *


class StaSMA(BaseInd):
    def __init__(self, period: int, cache_key: str = ''):
        super(StaSMA, self).__init__(cache_key)
        self.period = period
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
    elif hasattr(arr, 'to_numpy'):
        # 针对pandas.Series/DataFrame
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
    for i in range(period - 1, len(arr)):
        result[i] = sum(div_arr[i - period + 1: i + 1])
    return result


class _StaEWMA(BaseInd):
    def __init__(self, period: int, alpha: float, init_type: int, init_val=None, cache_key: str = ''):
        super(_StaEWMA, self).__init__(cache_key)
        self.period = period
        self.mul = alpha
        self._init_type = init_type
        self._init_first = init_val
        self._init_vals = []

    def _compute(self, val):
        if not np.isfinite(val):
            return val
        if not self.arr or not np.isfinite(self.arr[-1]):
            if self._init_first is not None:
                # 使用给定值作为计算第一个值的前置值
                ind_val = val * self.mul + self._init_first * (1 - self.mul)
            elif self._init_type == 0:
                # SMA作为第一个EMA值
                if len(self._init_vals) < self.period:
                    self._init_vals.append(val)
                    if len(self._init_vals) < self.period:
                        return np.nan
                ind_val = sum(self._init_vals) / self.period
            else:
                # 第一个有效值作为第一个EMA值
                ind_val = val
        else:
            ind_val = val * self.mul + self.arr[-1] * (1 - self.mul)
        return ind_val


def _EWMA(arr: np.ndarray, period: int, alpha: float, init_type: int, init_val=None) -> np.ndarray:
    arr = _to_nparr(arr)
    assert isinstance(arr, np.ndarray) and len(arr.shape) == 1
    result = _nan_array(arr)
    start_id = arg_valid_id(arr)
    if init_val is not None:
        # 使用给定的值作为第一个值计算的前置值
        old_val = init_val
        first_idx = start_id - 1
    elif init_type == 0:
        # SMA作为第一个EMA值
        old_val = np.sum(arr[start_id:start_id + period]) / period
        first_idx = start_id + period - 1
    else:
        # 第一个值作为EMA值
        old_val = arr[start_id]
        first_idx = start_id
    if start_id <= first_idx < len(result):
        result[first_idx] = old_val
    for i in range(first_idx + 1, len(arr)):
        result[i] = arr[i] * alpha + old_val * (1 - alpha)
        old_val = result[i]
    return result


class StaEMA(_StaEWMA):
    '''
    最近一个权重：2/(n+1)
    '''
    def __init__(self, period: int, init_type=0, cache_key: str = ''):
        super(StaEMA, self).__init__(period, 2 / (period + 1), init_type, cache_key=cache_key)


def EMA(arr: np.ndarray, period: int, init_type=0) -> np.ndarray:
    return _EWMA(arr, period, 2 / (period + 1), init_type)


class StaRMA(_StaEWMA):
    '''
    和StaEMA区别是：分子分母都减一
    最近一个权重：1/n
    '''
    def __init__(self, period: int, init_type=0, init_val=None, cache_key: str = ''):
        super(StaRMA, self).__init__(period, 1 / period, init_type, init_val, cache_key=cache_key)


def RMA(arr: np.ndarray, period: int, init_type=0, init_val=None) -> np.ndarray:
    return _EWMA(arr, period, 1 / period, init_type, init_val)


class StaTR(BaseInd):
    input_dim = 2

    def __init__(self, cache_key: str = ''):
        super(StaTR, self).__init__(cache_key)

    def _compute(self, val):
        crow = val[-1, :]
        if val.shape[0] < 2:
            cur_tr = np.nan  # crow[hcol] - crow[lcol]
        else:
            prow = val[-2, :]
            cur_tr = max(crow[hcol] - crow[lcol], abs(crow[hcol] - prow[ccol]), abs(crow[lcol] - prow[ccol]))
        return cur_tr


def TR(arr) -> np.ndarray:
    arr = _to_nparr(arr)
    assert isinstance(arr, np.ndarray) and len(arr.shape) == 2 and arr.shape[1] >= 5
    result = _nan_array(arr)
    # 和ta-lib保持一致，第一个nan
    # result[0] = arr[0, hcol] - arr[0, lcol]
    for i in range(1, result.shape[0]):
        crow, prow = arr[i, :], arr[i - 1, :]
        result[i] = max(crow[hcol] - crow[lcol], abs(crow[hcol] - prow[ccol]), abs(crow[lcol] - prow[ccol]))
    return result


class StaATR(BaseInd):
    input_dim = 2

    def __init__(self, period: int = 3, cache_key: str = ''):
        super(StaATR, self).__init__(cache_key)
        self.tr = StaTR(cache_key)
        self._rma = StaRMA(period, init_type=0, cache_key=cache_key + 'tr')

    def _compute(self, val):
        tr_val = self.tr(val)
        return self._rma(tr_val)


def ATR(arr, period: int) -> np.ndarray:
    arr = _to_nparr(arr)
    assert isinstance(arr, np.ndarray) and len(arr.shape) == 2 and arr.shape[1] >= 5, f'{arr.shape} invalid'
    tr = TR(arr)
    return RMA(tr, period, init_type=0)


class StaNATR(BaseInd):
    input_dim = 2

    def __init__(self, period: int = 3, cache_key: str = ''):
        super(StaNATR, self).__init__(cache_key)
        self.period = period
        self.atr = StaATR(period, cache_key)

    def _compute(self, val):
        ind_val = self.atr(val)
        long_price_range = LongVar.get(LongVar.price_range)
        if val.shape[0] < long_price_range.roll_len:
            ind_val = np.nan
        return ind_val / long_price_range.val


class StaTRRoll(BaseInd):
    input_dim = 2

    def __init__(self, period: int = 4, cache_key: str = ''):
        super(StaTRRoll, self).__init__(cache_key)
        self.period = period

    def _compute(self, val):
        high_col, low_col = val[-self.period:, hcol], val[-self.period:, lcol]
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
        return ind_val


class StaNTRRoll(BaseInd):
    input_dim = 2

    def __init__(self, period: int = 4, roll_len: int = 4, cache_key: str = ''):
        super(StaNTRRoll, self).__init__(cache_key)
        self.tr_roll = StaTRRoll(period, cache_key)
        self.roll_len = roll_len

    def _compute(self, val):
        tr_val = self.tr_roll(val)
        return tr_val / LongVar.get(LongVar.price_range).val


class StaNVol(BaseInd):
    input_dim = 2

    def __init__(self, cache_key: str = ''):
        super(StaNVol, self).__init__(cache_key)

    def _compute(self, val):
        return val[-1, vcol] / LongVar.get(LongVar.vol_avg).val


class StaMACD(BaseInd):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, smooth_period: int = 9, init_type=0,
                 cache_key: str = ''):
        '''
        带状态计算MACD指标。国外主流使用init_type=0，MyTT和国内主要使用init_type=1
        '''
        super(StaMACD, self).__init__(cache_key)
        self.ema_short = StaEMA(fast_period, init_type=init_type, cache_key=cache_key)
        self.ema_long = StaEMA(slow_period, init_type=init_type, cache_key=cache_key)
        self.ema_sgl = StaEMA(smooth_period, init_type=init_type, cache_key=cache_key + 'macd')

    def _compute(self, val):
        macd = self.ema_short(val) - self.ema_long(val)
        singal = self.ema_sgl(macd)
        return macd, singal


def MACD(arr: np.ndarray, fast_period: int = 12, slow_period: int = 26, smooth_period: int = 9, init_type=0)\
        -> Tuple[np.ndarray, np.ndarray]:
    '''
    计算MACD指标。国外主流使用init_type=0，MyTT和国内主要使用init_type=1
    '''
    arr = _to_nparr(arr)
    ema_fast, ema_slow = EMA(arr, fast_period, init_type=init_type), EMA(arr, slow_period, init_type=init_type)
    macd = ema_fast - ema_slow
    signal = EMA(macd, smooth_period, init_type=init_type)
    return macd, signal


class StaRSI(BaseInd):
    def __init__(self, period: int, cache_key: str = ''):
        super(StaRSI, self).__init__(cache_key)
        self.period = period
        self.gain_avg = 0
        self.loss_avg = 0
        self.last_input = np.nan

    def _compute(self, val):
        if not np.isfinite(self.last_input):
            self.last_input = val
            return np.nan
        val_delta = val - self.last_input
        self.last_input = val
        if len(self.arr) > self.period:
            if val_delta >= 0:
                gain_delta, loss_delta = val_delta, 0
            else:
                gain_delta, loss_delta = 0, val_delta
            self.gain_avg = (self.gain_avg * (self.period - 1) + gain_delta) / self.period
            self.loss_avg = (self.loss_avg * (self.period - 1) + loss_delta) / self.period
            return self.gain_avg * 100 / (self.gain_avg - self.loss_avg)
        else:
            if val_delta >= 0:
                self.gain_avg += val_delta / self.period
            else:
                self.loss_avg += val_delta / self.period
            if len(self.arr) == self.period:
                return self.gain_avg * 100 / (self.gain_avg - self.loss_avg)
            else:
                return np.nan


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


class StaKDJ(BaseInd):
    input_dim = 2

    def __init__(self, period: int = 9, sm1: int = 3, sm2: int = 3, cache_key: str = ''):
        super(StaKDJ, self).__init__(cache_key)
        self.period = period
        self._k = StaRMA(sm1, init_val=50., cache_key=cache_key+'kdj_k')
        self._d = StaRMA(sm2, init_val=50., cache_key=cache_key+'kdj_d')

    def _compute(self, val):
        if len(val) < self.period:
            return np.nan, np.nan
        hhigh = np.max(val[-self.period:, hcol])
        llow = np.min(val[-self.period:, lcol])
        rsv = (val[-1, ccol] - llow) / (hhigh - llow) * 100
        self._k(rsv)
        self._d(self._k[-1])
        return self._k[-1], self._d[-1]


def KDJ(arr: np.ndarray, period: int = 9, m1: int = 3, m2: int = 3):
    '''
    这里使用最流行最原始的方法计算。RMA作为平滑。
    mytt和国内一些：使用k = EMA(rsv, m1 * 2 - 1, init_type=1)
    '''
    clo = arr[:, ccol]
    rhigh = max_rolling(arr[:, hcol], period)
    rlow = min_rolling(arr[:, lcol], period)
    rsv = (clo - rlow) * 100 / (rhigh - rlow)
    # 计算K、D值
    k = RMA(rsv, m1, init_val=50.)
    d = RMA(k, m2, init_val=50.)
    # 计算J值
    # j = 3 * k - 2 * d
    return k, d


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

