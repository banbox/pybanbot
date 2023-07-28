#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : vec_inds.py
# Author: anyongjin
# Date  : 2023/5/23
from banbot.util.num_utils import *
from banbot.compute.ctx import *


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


def EMA(arr: np.ndarray, period: int, init_type=0) -> np.ndarray:
    return _EWMA(arr, period, 2 / (period + 1), init_type)


def RMA(arr: np.ndarray, period: int, init_type=0, init_val=None) -> np.ndarray:
    return _EWMA(arr, period, 1 / period, init_type, init_val)


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


def ATR(arr, period: int) -> np.ndarray:
    arr = _to_nparr(arr)
    assert isinstance(arr, np.ndarray) and len(arr.shape) == 2 and arr.shape[1] >= 5, f'{arr.shape} invalid'
    tr = TR(arr)
    return RMA(tr, period, init_type=0)


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


def KDJ(arr: np.ndarray, period: int = 9, m1: int = 3, m2: int = 3):
    '''
    这里使用最流行最原始的方法计算。RMA作为平滑。
    mytt和国内软件没有实现RMA，使用k = EMA(rsv, m1 * 2 - 1, init_type=1)进行近似
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


def BBANDS(arr: np.ndarray, period: int, std_up: int, std_dn: int):
    '''
    计算布林带。
    :return rolling_mean, upper_band, lower_band
    '''
    rolling_mean = mean_rolling(arr, period)
    rolling_std = std_rolling(arr, period)

    upper_band = rolling_mean + (rolling_std * std_up)
    lower_band = rolling_mean - (rolling_std * std_dn)

    return upper_band, rolling_mean, lower_band
