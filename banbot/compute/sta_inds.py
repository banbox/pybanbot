#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : sta_inds.py
# Author: anyongjin
# Date  : 2023/5/4
from banbot.compute.ctx import *
from banbot.util.num_utils import *


class StaSMA(BaseInd):
    def __init__(self, period: int, cache_key: str = ''):
        super(StaSMA, self).__init__(cache_key)
        self.period = period
        self.dep_vals: List[float] = []

    def _compute_val(self, val: float):
        self.dep_vals.append(val / self.period)
        if len(self.dep_vals) < self.period:
            return np.nan
        while len(self.dep_vals) > self.period:
            self.dep_vals.pop(0)
        return sum(self.dep_vals)


class _StaEWMA(BaseInd):
    def __init__(self, period: int, alpha: float, init_type: int, init_val=None, cache_key: str = ''):
        super(_StaEWMA, self).__init__(cache_key)
        self.period = period
        self.mul = alpha
        self._init_type = init_type
        self._init_first = init_val
        self._init_vals = []

    def _compute_val(self, val: float):
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


class StaEMA(_StaEWMA):
    '''
    最近一个权重：2/(n+1)
    '''
    def __init__(self, period: int, init_type=0, cache_key: str = ''):
        super(StaEMA, self).__init__(period, 2 / (period + 1), init_type, cache_key=cache_key)


class StaRMA(_StaEWMA):
    '''
    和StaEMA区别是：分子分母都减一
    最近一个权重：1/n
    '''
    def __init__(self, period: int, init_type=0, init_val=None, cache_key: str = ''):
        super(StaRMA, self).__init__(period, 1 / period, init_type, init_val, cache_key=cache_key)


class StaTR(BaseInd):
    input_dim = 2

    def __init__(self, cache_key: str = ''):
        super(StaTR, self).__init__(cache_key)

    def _compute_arr(self, val: np.ndarray):
        crow = val[-1, :]
        if val.shape[0] < 2:
            cur_tr = np.nan  # crow[hcol] - crow[lcol]
        else:
            prow = val[-2, :]
            cur_tr = max(crow[hcol] - crow[lcol], abs(crow[hcol] - prow[ccol]), abs(crow[lcol] - prow[ccol]))
        return cur_tr


class StaATR(BaseInd):
    input_dim = 2

    def __init__(self, period: int = 3, cache_key: str = ''):
        super(StaATR, self).__init__(cache_key)
        self.tr = StaTR(cache_key)
        self._rma = StaRMA(period, init_type=0, cache_key=cache_key + 'tr')

    def _compute_arr(self, val: np.ndarray):
        tr_val = self.tr(val)
        return self._rma(tr_val)


class StaNATR(BaseInd):
    input_dim = 2

    def __init__(self, period: int = 3, cache_key: str = ''):
        super(StaNATR, self).__init__(cache_key)
        self.period = period
        self.atr = StaATR(period, cache_key)

    def _compute_arr(self, val: np.ndarray):
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

    def _compute_arr(self, val: np.ndarray):
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

    def _compute_arr(self, val: np.ndarray):
        tr_val = self.tr_roll(val)
        return tr_val / LongVar.get(LongVar.price_range).val


class StaNVol(BaseInd):
    input_dim = 2

    def __init__(self, cache_key: str = ''):
        super(StaNVol, self).__init__(cache_key)

    def _compute_arr(self, val: np.ndarray):
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

    def _compute_val(self, val: float):
        macd = self.ema_short(val) - self.ema_long(val)
        singal = self.ema_sgl(macd)
        return macd, singal


class StaRSI(BaseInd):
    def __init__(self, period: int, cache_key: str = ''):
        super(StaRSI, self).__init__(cache_key)
        self.period = period
        self.gain_avg = 0
        self.loss_avg = 0
        self.last_input = np.nan

    def _compute_val(self, val: float):
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


class StaKDJ(BaseInd):
    '''
    KDJ指标。也称为：Stoch随机指标。
    '''
    input_dim = 2

    def __init__(self, period: int = 9, sm1: int = 3, sm2: int = 3, smooth_type='rma', cache_key: str = ''):
        super(StaKDJ, self).__init__(cache_key)
        self.period = period
        if smooth_type == 'rma':
            self._k = StaRMA(sm1, init_val=50., cache_key=cache_key+'kdj_k')
            self._d = StaRMA(sm2, init_val=50., cache_key=cache_key+'kdj_d')
        elif smooth_type == 'sma':
            self._k = StaSMA(sm1, cache_key=cache_key + 'kdj_k')
            self._d = StaSMA(sm2, cache_key=cache_key + 'kdj_d')
        else:
            raise ValueError(f'unsupport smooth_type: {smooth_type} for StaKDJ')

    def _compute_arr(self, val: np.ndarray):
        if len(val) < self.period:
            return np.nan, np.nan
        hhigh = np.max(val[-self.period:, hcol])
        llow = np.min(val[-self.period:, lcol])
        max_chg = hhigh - llow
        if not max_chg:
            # 四价相同，RSV定为50
            rsv = 50
        else:
            rsv = (val[-1, ccol] - llow) / max_chg * 100
        self._k(rsv)
        self._d(self._k[-1])
        return self._k[-1], self._d[-1]


class StaStdDev(BaseInd):
    '''
    带状态的标准差计算
    '''
    def __init__(self, period: int, ddof=0, cache_key: str = ''):
        super(StaStdDev, self).__init__(cache_key)
        self.period = period
        self.factor = period - ddof
        self._mean = StaSMA(period, cache_key)
        self._win_arr = []  # 记录输入值

    def _compute_val(self, val: float):
        mean_val = self._mean(val)
        self._win_arr.append(val)
        if len(self._win_arr) < self.period:
            return np.nan, np.nan
        variance = sum((x - mean_val) ** 2 for x in self._win_arr) / self.factor
        stddev_val = variance ** 0.5
        self._win_arr.pop(0)
        return stddev_val, mean_val


class StaBBANDS(BaseInd):
    '''
    带状态的布林带指标
    '''

    def __init__(self, period: int, std_up: int, std_dn: int, cache_key: str = ''):
        super(StaBBANDS, self).__init__(cache_key)
        self.period = period
        self.std_up = std_up
        self.std_dn = std_dn
        self._std = StaStdDev(period, cache_key=cache_key)

    def _compute_val(self, val: float):
        dev_val, mean_val = self._std(val)
        if np.isnan(dev_val):
            return np.nan, np.nan, np.nan
        upper = mean_val + dev_val * self.std_up
        lower = mean_val - dev_val * self.std_dn
        return upper, mean_val, lower


class CrossTrace:
    def __init__(self):
        self.prev_valid = None
        self.state = 0
        self.hist = []

    def __call__(self, *args, **kwargs):
        '''
        0：未发生交叉
        1：vala向上穿越valb
        -1：vala向下穿越valb
        '''
        if len(args) == 2:
            cur_diff = args[0] - args[1]
        elif len(args) == 1:
            cur_diff = args[0]
        else:
            raise ValueError(f'wrong args len: {len(args)}, expect 1 or 2')
        self.state = 0
        if not self.prev_valid or not np.isfinite(self.prev_valid):
            self.prev_valid = cur_diff
        elif not cur_diff:
            pass
        else:
            factor = self.prev_valid * cur_diff
            if factor < 0:
                self.prev_valid = cur_diff
                self.state = 1 if cur_diff > 0 else -1
                self.hist.append((self.state, bar_num.get()))
        return self.state


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

