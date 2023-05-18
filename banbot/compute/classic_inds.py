#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : ind_coms.py
# Author: anyongjin
# Date  : 2023/1/29
import numpy as np
import talib.abstract as ta
from pandas import Series, DataFrame

np.seterr(divide='ignore', invalid='ignore')


def super_smoother(col: Series, period, prev_nan: bool = True) -> np.ndarray:
    '''
    John F. Ehlers提出的SuperSmoother指标。来自数字信号处理领域的技术。前24个信号无效
    可看做低通滤波器。用于解决移动均线之后的问题。
    https://www.tradingview.com/script/6tSfPE3W-e2-Reflex-Trendflex/
    :param col:
    :param period:
    :param prev_nan: 前24个无效，是否替换为nan
    :return:
    '''
    from banbot.compute.utils import np_shift
    a1 = np.exp(-1.414 * np.pi / (0.5 * period))
    c2 = 2 * a1 * np.cos(1.414 * 180 / (0.5 * period))
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    result = np.zeros(len(col))
    col_arr = col.to_numpy()
    col_pre1 = np_shift(col_arr, 1, 0)
    col_avg_p1 = (col_arr + col_pre1) / 2
    for i in range(2, len(result)):
        result[i] = c1 * col_avg_p1[i] + c2 * result[i - 1] + c3 * result[i - 2]
    if prev_nan:
        result[:24] = np.nan
    return result


def re_trend_flex(col: Series, period: int, is_trend: bool = True):
    '''
    https://mp.weixin.qq.com/s?__biz=MzkyODI5ODcyMA==&mid=2247484458&idx=1&sn=c8f01ddeadd954f432b91bcfe00afc3b&scene=21
    TrendFlex指标。解决移动均线延迟的问题。来源于数字信号处理。前150个信号无效
    :param col:
    :param period:
    :param is_trend: 使用trendflex还是reflex
    :return:
    '''
    smooth = super_smoother(col, period, False)
    from banbot.compute.utils import np_shift
    prev_res = np_shift(smooth, period, 0)
    slope = (prev_res - smooth) / period
    arr_sum = np.zeros(len(smooth))
    for i in range(1, period):
        if is_trend:
            arr_sum += smooth - np_shift(smooth, i, 0)
        else:
            arr_sum += smooth + i * slope - np_shift(smooth, i, 0)
    arr_sum /= period
    # 改为矢量化：https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
    sum_pow2 = np.power(arr_sum, 2)
    ms = np.zeros(len(sum_pow2))
    ms[0] = sum_pow2[0]
    for i in range(1, len(sum_pow2)):
        ms[i] = 0.04 * sum_pow2[i] + 0.96 * ms[i - 1]
    xflex = arr_sum / np.sqrt(ms)
    xflex[~np.isfinite(xflex)] = 0
    xflex[:150] = np.nan
    return xflex


def chaikin_money_flow(df: DataFrame, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        df(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= df['volume']
    # CMF的值应该在-1~1之间，一些0交易的数据，可能导致计算溢出，这里强制vol不小于0.001
    vol_mom = df['volume'].rolling(n, min_periods=0).sum().clip(lower=0.001)
    cmf = (mfv.rolling(n, min_periods=0).sum() / vol_mom)
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')


def ConnorsRSI(df: DataFrame) -> Series:
    # CRSI (3, 2, 100)
    if 'close_chg' not in df:
        df['close_chg'] = df['close'] / df['close'].shift(1) - 1
    close_chg = df['close_chg']
    crsi_updown = np.where(close_chg.gt(0), 1.0, np.where(close_chg.lt(0), -1.0, 0.0))
    if 'rsi_3' not in df:
        df['rsi_3'] = ta.RSI(df['close'], timeperiod=3)
    if 'roc_100' not in df:
        df['roc_100'] = ta.ROC(df['close'], 100)
    return (df['rsi_3'] + ta.RSI(crsi_updown, timeperiod=2) + df['roc_100']) / 3


def natr(df, period=14, norm_window=100):
    '''
    归一化的ATR。
    :param df:
    :param period:
    :param norm_window: 受暴涨暴跌影响，值可能很大，几千。这里用窗口取最大绝对值，归一化到-1~1之间
    :return:
    '''
    raw_atr = ta.NATR(df, timeperiod=period)
    return raw_atr / raw_atr.rolling(norm_window).max().clip(lower=3)


def ewo(dataframe, sma1_length=5, sma2_length=35, norm_window=100):
    '''
    Elliot Wave Oscillator
    :param dataframe:
    :param sma1_length:
    :param sma2_length:
    :param norm_window: 受暴涨暴跌影响，值可能很大，几千。这里用窗口取最大绝对值，归一化到-1~1之间
    :return:
    '''
    sma1 = ta.EMA(dataframe, timeperiod=sma1_length)
    sma2 = ta.EMA(dataframe, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / dataframe['close']
    return smadif / smadif.abs().rolling(norm_window).max().clip(lower=1)


def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    r_factor = (highest_high - dataframe["close"]) / (highest_high - lowest_low)
    WR = Series(r_factor, name=f"{period} Williams %R")

    return WR * -100


def t3_average(dataframe, length=5):
    """
    T3 Average by HPotter on Tradingview
    https://www.tradingview.com/script/qzoC9H1I-T3-Average/
    """
    df = dataframe.copy()

    df['xe1'] = ta.EMA(df['close'], timeperiod=length)
    df['xe1'].fillna(0, inplace=True)
    df['xe2'] = ta.EMA(df['xe1'], timeperiod=length)
    df['xe2'].fillna(0, inplace=True)
    df['xe3'] = ta.EMA(df['xe2'], timeperiod=length)
    df['xe3'].fillna(0, inplace=True)
    df['xe4'] = ta.EMA(df['xe3'], timeperiod=length)
    df['xe4'].fillna(0, inplace=True)
    df['xe5'] = ta.EMA(df['xe4'], timeperiod=length)
    df['xe5'].fillna(0, inplace=True)
    df['xe6'] = ta.EMA(df['xe5'], timeperiod=length)
    df['xe6'].fillna(0, inplace=True)
    b = 0.7
    c1 = -b * b * b
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']

    return df['T3Average']
