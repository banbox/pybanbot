#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : sta_inds.py
# Author: anyongjin
# Date  : 2023/5/4
'''
带状态的常见指标实现。每个bar计算一次，实盘时速度更快。
函数式调用，传入序列变量，自动追踪历史结果。
按序号取值，res[0]表示当前值，res[1]表示前一个。。。
'''
from banbot.compute.ctx import *
from banbot.util.num_utils import *


def SMA(obj: SeriesVar, period: int) -> SeriesVar:
    '''
    带状态SMA。返回序列对象，res[0]即最新结果
    '''
    div_obj = obj / period
    if len(div_obj) < period:
        res_val = np.nan
    else:
        res_val = sum(div_obj[:period])
    return SeriesVar(obj.key + f'_sma{period}', res_val)


def _EWMA(obj: SeriesVar, res_key: str, period: int, alpha: float, init_type: int, init_val=None) -> SeriesVar:
    in_val = obj[0]
    res_obj = SeriesVar.get(res_key)
    if not np.isfinite(in_val):
        res_val = in_val
    elif not res_obj or not np.isfinite(res_obj[0]):
        if init_val is not None:
            # 使用给定值作为计算第一个值的前置值
            res_val = in_val * alpha + init_val * (1 - alpha)
        elif init_type == 0:
            # SMA作为第一个EMA值
            res_val = SMA(obj, period)[0]
        else:
            # 第一个有效值作为第一个EMA值
            res_val = in_val
    else:
        res_val = in_val * alpha + res_obj[0] * (1 - alpha)
    if res_obj is None:
        return SeriesVar(res_key, res_val)
    res_obj.val.append(res_val)
    return res_obj


def EMA(obj: SeriesVar, period: int, init_type=0) -> SeriesVar:
    '''
    指数移动均线
    最近一个权重：2/(n+1)
    '''
    res_key = obj.key + f'_ema{period}_{init_type}'
    return _EWMA(obj, res_key, period, 2 / (period + 1), init_type)


def RMA(obj: SeriesVar, period: int, init_type=0, init_val=None) -> SeriesVar:
    '''
    相对移动均线
    和EMA区别是：分子分母都减一
    最近一个权重：1/n
    '''
    res_key = obj.key + f'_rma{period}_{init_type}_{init_val}'
    return _EWMA(obj, res_key, period, 1 / period, init_type, init_val)


def TR(high: SeriesVar, low: SeriesVar, close: SeriesVar) -> SeriesVar:
    '''
    真实振幅
    '''
    if len(high) < 2:
        res_val = np.nan
    else:
        chigh, clow, cclose = high[0], low[0], close[0]
        phigh, plow, pclose = high[1], low[1], close[1]
        res_val = max(chigh - clow, abs(chigh - pclose), abs(clow - pclose))
    return SeriesVar(high.key + '_tr', res_val)


def ATR(high: SeriesVar, low: SeriesVar, close: SeriesVar, period: int) -> SeriesVar:
    '''
    平均真实振幅
    '''
    return RMA(TR(high, low, close), period)


def TRRoll(high: SeriesVar, low: SeriesVar, period: int) -> SeriesVar:
    high_col, low_col = high[:period], low[:period]
    max_id = np.argmax(high_col)
    roll_max = high_col[max_id]
    min_id = np.argmin(low_col)
    roll_min = low_col[min_id]

    prev_tr = roll_max - roll_min
    if period - min(max_id, min_id) <= 2:
        # 如果是从最后两个蜡烛计算得到的，则是重要的波动范围。
        res_val = prev_tr
    else:
        # 从前面计算的TrueRange加权缩小，和最后两个蜡烛的TrueRange取最大值
        prev_tr *= (min(max_id, min_id) + 1) / (period * 2) + 0.5
        res_val = max(prev_tr, np.max(high_col[-2:]) - np.min(low_col[-2:]))
    return SeriesVar(high.key + f'_troll{period}', res_val)


def NTRRoll(high: SeriesVar, low: SeriesVar, period: int = 4) -> SeriesVar:
    troll = TRRoll(high, low, period)
    res_val = troll[0] / LongChange.get()
    return SeriesVar(high.key + f'_ntroll{period}', res_val)


def NVol(vol: SeriesVar) -> SeriesVar:
    res_val = vol[0] / LongVolAvg.get()
    return SeriesVar(vol.key + f'_nvol', res_val)


def MACD(obj: SeriesVar, fast_period: int = 12, slow_period: int = 26,
         smooth_period: int = 9, init_type=0) -> SeriesVar:
    '''
    计算MACD指标。国外主流使用init_type=0，MyTT和国内主要使用init_type=1
    '''
    short = EMA(obj, fast_period, init_type=init_type)
    long = EMA(obj, slow_period, init_type=init_type)
    macd = short - long
    singal = EMA(macd, smooth_period, init_type=init_type)
    return SeriesVar(obj.key + '_macd', (macd, singal))


class SeriesRSI(SeriesVar):
    def __init__(self, key: str, data, last_val):
        super().__init__(key, data)
        self.gain_avg = 0
        self.loss_avg = 0
        self.last_input = last_val


def RSI(obj: SeriesVar, period: int) -> SeriesRSI:
    '''
    相对强度指数。0-100之间。
    价格变化有的使用变化率，大部分使用变化值。这里使用变化值：price_chg
    '''
    res_key = obj.key + f'_rsi{period}'
    res_obj: SeriesRSI = SeriesVar.get(res_key)
    cur_val = obj[0]
    if not res_obj:
        return SeriesRSI(res_key, np.nan, cur_val)
    if not np.isfinite(res_obj.last_input):
        res_obj.last_input = cur_val
        res_obj.append(np.nan)
        return res_obj
    val_delta = cur_val - res_obj.last_input
    res_obj.last_input = cur_val
    if len(res_obj) > period:
        if val_delta >= 0:
            gain_delta, loss_delta = val_delta, 0
        else:
            gain_delta, loss_delta = 0, val_delta
        res_obj.gain_avg = (res_obj.gain_avg * (period - 1) + gain_delta) / period
        res_obj.loss_avg = (res_obj.loss_avg * (period - 1) + loss_delta) / period
        res_val = res_obj.gain_avg * 100 / (res_obj.gain_avg - res_obj.loss_avg)
    else:
        if val_delta >= 0:
            res_obj.gain_avg += val_delta / period
        else:
            res_obj.loss_avg += val_delta / period
        if len(res_obj) == period:
            res_val = res_obj.gain_avg * 100 / (res_obj.gain_avg - res_obj.loss_avg)
        else:
            res_val = np.nan
    res_obj.append(res_val)
    return res_obj


def KDJ(high: SeriesVar, low: SeriesVar, close: SeriesVar,
        period: int = 9, sm1: int = 3, sm2: int = 3, smooth_type='rma') -> SeriesVar:
    '''
    KDJ指标。也称为：Stoch随机指标。返回k, d
    '''
    res_key = high.key + f'_kdj{period}_{sm1}_{sm2}_{smooth_type}'
    if len(high) < period:
        return SeriesVar(res_key, [np.nan, np.nan])
    hhigh = np.max(high[:period])
    llow = np.min(low[:period])
    max_chg = hhigh - llow
    if not max_chg:
        # 四价相同，RSV定为50
        rsv = 50
    else:
        rsv = (close[0] - llow) / max_chg * 100
    rsv_obj = SeriesVar(high.key + f'_rsv{period}', rsv)
    if smooth_type == 'rma':
        k = RMA(rsv_obj, sm1, init_val=50.)
        d = RMA(k, sm2, init_val=50.)
    elif smooth_type == 'sma':
        k = SMA(rsv_obj, sm1)
        d = SMA(k, sm2)
    else:
        raise ValueError(f'unsupport smooth_type: {smooth_type} for KDJ')
    return SeriesVar(res_key, (k, d))


class SeriesStdDev(SeriesVar):
    def __init__(self, key: str, data, last_val):
        super().__init__(key, data)
        self.win_arr = [last_val]  # 记录输入值


def StdDev(obj: SeriesVar, period: int, ddof=0) -> SeriesStdDev:
    '''
    标准差，返回：stddev，mean
    '''
    mean = SMA(obj, period)
    res_key = obj.key + f'_sdev{period}'
    res_obj: SeriesStdDev = SeriesStdDev.get(res_key)
    cur_val = obj[0]
    if not res_obj:
        return SeriesStdDev(res_key, (np.nan, np.nan), cur_val)
    res_obj.win_arr.append(cur_val)
    if len(res_obj.win_arr) < period:
        res_val = np.nan, np.nan
    else:
        mean_val = mean[0]
        variance = sum((x - mean_val) ** 2 for x in res_obj.win_arr) / (period - ddof)
        stddev_val = variance ** 0.5
        res_obj.win_arr.pop(0)
        res_val = stddev_val, mean_val
    res_obj.append(res_val)
    return res_obj


def BBANDS(obj: SeriesVar, period: int, std_up: int, std_dn: int) -> SeriesVar:
    '''
    布林带指标。返回：upper, mean, lower
    '''
    dev_val, mean_val = StdDev(obj, period)[0]
    res_key = obj.key + f'_bb{period}_{std_up}_{std_dn}'
    if np.isnan(dev_val):
        return SeriesVar(res_key, (np.nan, np.nan, np.nan))
    upper = mean_val + dev_val * std_up
    lower = mean_val - dev_val * std_dn
    return SeriesVar(res_key, (upper, mean_val, lower))


def TD(obj: SeriesVar):
    '''
    Tom DeMark Sequence。神奇九转。狄马克序列。
    9和13表示超买；-9和-13表示超卖
    '''
    sub4 = obj[0] - obj[4]
    res_key = obj.key + '_td'
    res_obj: SeriesVar = SeriesVar.get(res_key)
    if not res_obj:
        return SeriesVar(res_key, np.nan)
    if not np.isfinite(sub4):
        res_obj.append(np.nan)
        return res_obj
    pindex = res_obj[0]
    step = 1 if sub4 > 0 else (-1 if sub4 < 0 else 0)
    if np.isfinite(pindex) and pindex * sub4 > 0:
        res_val = pindex + step
    else:
        res_val = step
    res_obj.append(res_val)
    return res_obj

