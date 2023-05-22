#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_rsi.py
# Author: anyongjin
# Date  : 2023/4/12
import talib as ta
import pandas as pd
import pandas_ta as pta
from test.common import *
'''
比较自行实现的指标和talib、pandas_ta、MyTT等结果异同
SMA  EMA  RMA  TR  ATR  MACD  RSI  KDJ
不对比：NATR、TRRoll、NTRRoll、NVol

|  指标  | 状态指标 |  MyTT  |TA-lib Class|TA-lib Metastock| Pandas-TA |
======================================================================
|  SMA  |   ✔    |   T1   |    ✔       |       ✔        |     ✔     |
|  EMA  |   ✔    |   T1    |    ✔       |       T1       |    ✔     |
|  RMA  |   ✔    |   ✔    |     X      |       X        |     ✔     |
|  RSI  |   ✔    |   T1   |    ✔       |       ✔        |     T2    |
|  KDJ  |   ✔    |   T1   |     T2     |       T1        |     T3    |  # 只有细微差别
'''


def test_sma():
    '''
    StaSMA的结果和TA-lib、Pandas-TA结果一致。
    MyTT的SMA前120周期不准确
    '''
    period = 5
    ta.set_compatibility(1)
    ta_res = ta.SMA(close_arr, timeperiod=period)
    ta.set_compatibility(0)
    ta2_res = ta.SMA(close_arr, timeperiod=period)
    from banbot.compute.mytt import SMA as MySMA
    # mytt的SMA初始120周期不精确
    mtt_res = MySMA(close_arr, period)
    mtt_res = np.array(mtt_res)
    res = SMA(close_arr, period)
    sta_res = calc_state_ind(StaSMA(period), close_arr)
    pta_res = pta.sma(pd.Series(close_arr), period, talib=False).to_numpy()
    print(ta_res)
    print(ta2_res)
    print(mtt_res)
    print(res)
    print(sta_res)
    print(pta_res)
    assert_arr_equal(ta_res, res)
    assert_arr_equal(ta_res, sta_res)
    assert_arr_equal(ta_res, pta_res)


def test_ema():
    '''
    StaEMA和MyTT的计算结果一致
    talib和pandas-ta的EMA计算一致
    '''
    period = 5
    ta.set_compatibility(1)
    ta1_res = ta.EMA(close_arr, timeperiod=period)
    ta.set_compatibility(0)
    ta2_res = ta.EMA(close_arr, timeperiod=period)
    from banbot.compute.mytt import EMA as MyEMA
    mtt_res = MyEMA(close_arr, period)
    res = EMA(close_arr, period)
    sta_res = calc_state_ind(StaEMA(period), close_arr)
    pta_res = pta.ema(pd.Series(close_arr), period, talib=False).to_numpy()
    print('\nTA-lib Class:')
    print(ta2_res)
    print('TA-lib MetaStock:')
    print(ta1_res)
    print('mytt')
    print(mtt_res)
    print('Self')
    print(res)
    print('StateSelf')
    print(sta_res)
    print('PandasTA')
    print(pta_res)
    assert_arr_equal(mtt_res, res)
    assert_arr_equal(mtt_res, sta_res)


def test_rma():
    '''

    '''
    period = 5
    res = RMA(close_arr, period)
    sta_res = calc_state_ind(StaRMA(period), close_arr)
    pta_res = pta.rma(pd.Series(close_arr), period, talib=False).to_numpy()
    print('Self')
    print(res)
    print('StateSelf')
    print(sta_res)
    print('PandasTA')
    print(pta_res)
    assert_arr_equal(pta_res, res)
    assert_arr_equal(pta_res, sta_res)


def test_rsi():
    '''
    StaRSI的结果和Ta-lib、Pandas-TA的计算一致
    MyTT的RSI受SMA影响，前120周期不准确
    '''
    period = 14
    ta.set_compatibility(1)
    ta_res = ta.RSI(close_arr, timeperiod=period)
    ta.set_compatibility(0)
    ta2_res = ta.RSI(close_arr, timeperiod=period)
    # MyTT的RSI因SMA影响，初始120周期不精确
    from banbot.compute.mytt import RSI as MttRSI
    mtt_res = MttRSI(close_arr, period)
    self_res = RSI(close_arr, period)
    sta_res = calc_state_ind(StaRSI(period), close_arr)
    pta_res = pta.rsi(pd.Series(close_arr), period, talib=False).to_numpy()
    print(ta_res)
    print(ta2_res)
    print(np.array(mtt_res))
    print(self_res)
    print(sta_res)
    print(pta_res)
    assert_arr_equal(ta_res, self_res)
    assert_arr_equal(ta_res, sta_res)
    assert_arr_equal(ta_res, pta_res)


def test_kdj():
    '''
    最流行的KDJ算法中，平滑应该使用RMA
    国内主流软件和MyTT使用EMA(2*period-1)且init_type=1。
    ta-lib中KDJ的平滑支持很多种方式，通过slowk_matype指定，默认的0是SMA，1是EMA；
    https://developer.hs.net/thread/2321
    '''
    high_arr, low_arr, close_arr = ohlcv_arr[:, hcol], ohlcv_arr[:, lcol], ohlcv_arr[:, ccol]

    # 这里使用2*period-1，EMA平滑，保持和MyTT一致
    ta_kdj_args = dict(fastk_period=9, slowk_period=5, slowd_period=5, slowk_matype=1, slowd_matype=1)
    ta.set_compatibility(1)
    ta_k, ta_d = ta.STOCH(high_arr, low_arr, close_arr, **ta_kdj_args)
    ta.set_compatibility(0)
    ta2_k, ta2_d = ta.STOCH(high_arr, low_arr, close_arr, **ta_kdj_args)
    # 使用mytt计算
    from banbot.compute.mytt import KDJ as MtKDJ
    mk, mt, mj = MtKDJ(close_arr, high_arr, low_arr)
    sta_k, sta_d = calc_state_ind(StaKDJ(), ohlcv_arr)
    myk, myd = KDJ(np.array(candles))
    pta_df = pta.kdj(pd.Series(high_arr), pd.Series(low_arr), pd.Series(close_arr), 9, 3, talib=False)
    pta_k, pta_d, pta_j = pta_df['K_9_3'], pta_df['D_9_3'], pta_df['J_9_3']
    print('\n Ta-lib Class')
    print(ta2_k)
    print('Ta-lib MetaStock')
    print(ta_k)
    print('mytt')
    print(mk)
    print('Self')
    print(np.array(myk))
    print('StateSelf')
    print(sta_k)
    print('Pandas-TA')
    print(pta_k.to_numpy())
    assert_arr_equal(myk, sta_k)
    assert_arr_equal(myd, sta_d)
