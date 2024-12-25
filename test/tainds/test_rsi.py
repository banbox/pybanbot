#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_rsi.py
# Author: anyongjin
# Date  : 2023/4/12
import numpy as np
import pandas as pd
import talib as ta
import pandas_ta as pta
from test.common import *
from banbot.compute import mytt
from banbot.compute.vec_inds import *
from banbot.compute import sta_inds as sta
'''
比较自行实现的指标和talib、pandas_ta、MyTT等结果异同
SMA  EMA  RMA  TR  ATR  MACD  RSI  KDJ  BBANDS
不对比：NATR、TRRoll、NTRRoll、NVol

|  指标  | 状态指标 |  MyTT  |TA-lib Class|TA-lib Metastock| Pandas-TA |
======================================================================
|  SMA  |   ✔    |   T1   |    ✔       |       ✔        |     ✔     |
|  EMA  |   ✔    |   T1   |    T1      |       ✔        |     ✔     |
|  RMA  |   ✔    |   --   |     --     |       --       |     T1    |
|  TR   |   ✔    |   --   |     ✔      |       ✔        |     ✔     |
|  ATR  |   ✔    |   T1   |     ✔      |       ✔        |     T2    |
|  MACD |   ✔    |   T1   |     T2     |       T1        |     ✔     |
|  RSI  |   ✔    |   T1   |    ✔       |       ✔        |     T2    |
|  KDJ  |   ✔    |   T1   |     T2     |       T1        |     T3    |  # 只有细微差别
| BBANDS|   ✔    |   ✔    |     ✔     |       ✔         |     ✔     |
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
    # mytt的SMA初始120周期不精确
    mtt_res = mytt.SMA(close_arr, period)
    mtt_res = np.array(mtt_res)
    res = SMA(close_arr, period)
    sta_res = calc_state_func(lambda: sta.SMA(Bar.close, period))
    pta_res = pta.sma(close_col, period, talib=False)
    print_tares(res, sta_res, ta_res, ta2_res, mtt_res, pta_res)
    assert_arr_equal(ta_res, res)
    assert_arr_equal(ta_res, sta_res)
    assert_arr_equal(ta_res, pta_res)


def test_ema():
    '''
    StaEMA和MyTT的计算结果一致
    talib和pandas-ta的EMA计算一致
    '''
    period = 12
    ta.set_compatibility(1)
    ta1_res = ta.EMA(close_arr, timeperiod=period)
    ta.set_compatibility(0)
    ta2_res = ta.EMA(close_arr, timeperiod=period)
    mtt_res = mytt.EMA(close_arr, period)
    res = EMA(close_arr, period)
    sta_res = calc_state_func(lambda: sta.EMA(Bar.close, period))
    pta_res = pta.ema(close_col, period, talib=False)
    print_tares(res, sta_res, ta1_res, ta2_res, mtt_res, pta_res)
    assert_arr_equal(ta2_res, res)
    assert_arr_equal(ta2_res, sta_res)
    assert_arr_equal(ta2_res, pta_res)


def test_rma():
    '''
    跟EMA类似，但比EMA对最近一个权重低一些。1/n
    '''
    period = 12
    res = RMA(close_arr, period)
    # sta_res = calc_state_ind(StaRMA(period), close_arr)
    sta_res = calc_state_func(lambda: sta.RMA(Bar.close, period))
    pta_res = pta.rma(close_col, period, talib=False)
    print_tares(res, sta_res, pta_res=pta_res)
    assert_arr_equal(res, sta_res)


def test_tr():
    res = TR(ohlcv_arr)
    sta_res = calc_state_func(lambda: sta.TR(Bar.high, Bar.low, Bar.close))
    # sta_res = calc_state_ind(StaTR(), ohlcv_arr)
    ta_res = ta.TRANGE(high_arr, low_arr, close_arr)
    pta_res = pta.true_range(high_col, low_col, close_col, talib=False)
    print_tares(res, sta_res, ta_res, pta_res=pta_res)
    assert_arr_equal(ta_res, pta_res)
    assert_arr_equal(ta_res, res)
    assert_arr_equal(ta_res, sta_res)


def test_atr():
    period = 14
    res = ATR(ohlcv_arr, period)
    # sta_res = calc_state_ind(StaATR(period), ohlcv_arr)
    sta_res = calc_state_func(lambda: sta.ATR(Bar.high, Bar.low, Bar.close, period))
    mtt_res = mytt.ATR(close_arr, high_arr, low_arr, period)
    ta.set_compatibility(1)
    ta2_res = ta.ATR(high_arr, low_arr, close_arr, timeperiod=period)
    ta.set_compatibility(0)
    ta_res = ta.ATR(high_arr, low_arr, close_arr, timeperiod=period)
    pta_res = pta.atr(high_col, low_col, close_col, period, talib=False)
    print_tares(res, sta_res, ta_res, ta2_res, mytt_res=mtt_res, pta_res=pta_res)
    assert_arr_equal(ta_res, res)
    assert_arr_equal(ta_res, sta_res)


def test_macd():
    res = MACD(close_arr)[0]
    # sta_res = calc_state_ind(StaMACD(), ohlcv_arr)[0]
    sta_res = calc_state_func(lambda: sta.MACD(Bar.close))[0]
    ta.set_compatibility(1)
    ta_mres = ta.MACD(close_arr, fastperiod=12, slowperiod=26, signalperiod=9)[0]
    ta.set_compatibility(0)
    ta_cres = ta.MACD(close_arr, fastperiod=12, slowperiod=26, signalperiod=9)[0]
    mtt_res = mytt.MACD(close_arr)[0]
    pta_res = pta.macd(close_col, 12, 26, 9, talib=False)['MACD_12_26_9'].to_numpy()
    print_tares(res, sta_res, ta_cres, ta_mres, mtt_res, pta_res)
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
    mtt_res = mytt.RSI(close_arr, period)
    self_res = RSI(close_arr, period)
    # sta_res = calc_state_ind(StaRSI(period), close_arr)
    sta_res = calc_state_func(lambda: sta.RSI(Bar.close, period))
    pta_res = pta.rsi(pd.Series(close_arr), period, talib=False).to_numpy()
    print_tares(self_res, sta_res, ta2_res, ta_res, mtt_res, pta_res)
    assert_arr_equal(ta2_res, self_res)
    assert_arr_equal(ta2_res, sta_res)


def test_vwma():
    period = 9
    price_col = pta.hlc3(close_col, high_col, low_col)
    ta_res = pta.vwma(price_col, vol_col, period, talib=False).to_numpy()
    print(' vwma Pandas Res '.center(60, '='))
    print(ta_res)


def test_kdj():
    '''
    最流行的KDJ算法中，平滑应该使用RMA
    国内主流软件和MyTT使用EMA(2*period-1)且init_type=1。
    ta-lib中KDJ的平滑支持很多种方式，通过slowk_matype指定，默认的0是SMA，1是EMA；
    https://developer.hs.net/thread/2321
    '''
    # 这里使用2*period-1，EMA平滑，保持和MyTT一致
    ta_kdj_args = dict(fastk_period=9, slowk_period=5, slowd_period=5, slowk_matype=1, slowd_matype=1)
    ta.set_compatibility(1)
    ta_k, ta_d = ta.STOCH(high_arr, low_arr, close_arr, **ta_kdj_args)
    ta.set_compatibility(0)
    ta2_k, ta2_d = ta.STOCH(high_arr, low_arr, close_arr, **ta_kdj_args)
    # 使用mytt计算
    mk, mt, mj = mytt.KDJ(close_arr, high_arr, low_arr)
    # sta_k, sta_d = calc_state_ind(StaKDJ(), ohlcv_arr)
    sta_res = calc_state_func(lambda: sta.KDJ(Bar.high, Bar.low, Bar.close))
    sta_k, sta_d = sta_res[0], sta_res[1]
    myk, myd = KDJ(ohlcv_arr)
    pta_df = pta.kdj(high_col, low_col, close_col, 9, 3, talib=False)
    pta_k, pta_d, pta_j = pta_df['K_9_3'], pta_df['D_9_3'], pta_df['J_9_3']
    print_tares(myk, sta_k, ta2_k, ta_k, mk, pta_k)
    assert_arr_equal(myk, sta_k)
    assert_arr_equal(myd, sta_d)


def test_bband():
    '''
    对比测试布林带指标
    '''
    ta.set_compatibility(1)
    period, nbdevup, nbdevdn = 9, 2, 2
    ta_up, ta_md, ta_lo = ta.BBANDS(close_arr, timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn)
    ta.set_compatibility(0)
    ta2_up, ta2_md, ta2_lo = ta.BBANDS(close_arr, timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn)
    # 使用mytt计算
    m_up, m_md, m_lo = mytt.BOLL(close_arr, period, nbdevup)
    # pta 计算
    pta_df = pta.bbands(close_col, period, nbdevup, talib=False)
    pta_up, pta_md, pta_lo = pta_df['BBU_9_2.0'], pta_df['BBM_9_2.0'], pta_df['BBL_9_2.0']
    # 带状态
    # sta_up, sta_md, sta_lo = calc_state_ind(StaBBANDS(period, nbdevup, nbdevdn), close_arr)
    sta_res = calc_state_func(lambda: sta.BBANDS(Bar.close, period, nbdevup, nbdevdn))
    sta_up, sta_md, sta_lo = sta_res
    # 向量计算
    my_up, my_md, my_lo = BBANDS(close_arr, period, nbdevup, nbdevdn)
    # 对比
    print_tares(my_up, sta_up, ta2_up, ta_up, m_up, pta_up)
    # print_tares(my_md, sta_md, ta2_md, ta_md, m_md, pta_md)
    # print_tares(my_lo, sta_lo, ta2_lo, ta_lo, m_lo, pta_lo)
    assert_arr_equal(my_up, sta_up)
    assert_arr_equal(sta_up, ta_up)


def test_adx():
    ta.set_compatibility(1)
    period = 9
    ta_res = ta.ADX(high_arr, low_arr, close_arr, timeperiod=period)
    ta.set_compatibility(0)
    ta2_res = ta.ADX(high_arr, low_arr, close_arr, timeperiod=period)
    sta_res = calc_state_func(lambda: sta.ADX(Bar.high, Bar.close, Bar.close, period))[0]
    pta_res = pta.adx(high_col, low_col, close_col, period, talib=False).to_numpy()
    print_tares(None, sta_res, ta_res, ta2_res, None, pta_res)


def test_roc():
    ta.set_compatibility(1)
    period = 9
    mytt_res = mytt.ROC(close_arr, period)
    ta_res = ta.ROC(close_arr, timeperiod=period)[0]
    ta.set_compatibility(0)
    ta2_res = ta.ROC(close_arr, timeperiod=period)
    sta_res = calc_state_func(lambda: sta.ROC(Bar.close, period))
    print_tares(None, sta_res, ta_res, ta2_res, mytt_res)


def test_cci():
    ta.set_compatibility(1)
    period = 10
    mytt_res = mytt.CCI(close_arr, high_arr, low_arr, period)
    ta_res = ta.CCI(high_arr, low_arr, close_arr, timeperiod=period)
    ta.set_compatibility(0)
    ta2_res = ta.CCI(high_arr, low_arr, close_arr, timeperiod=period)
    pta_res = pta.cci(high_col, low_col, close_col, period, talib=False).to_numpy()
    print_tares(None, None, ta_res, ta2_res, mytt_res, pta_res)


def test_cmf():
    ta.set_compatibility(1)
    period = 10
    pta_res = pta.cmf(high_col, low_col, close_col, vol_col, length=period).to_numpy()
    print(pta_res)


def test_kama():
    ta.set_compatibility(1)
    period = 10
    ta_res = ta.KAMA(close_arr, timeperiod=period)
    ta.set_compatibility(0)
    ta2_res = ta.KAMA(close_arr, timeperiod=period)
    pta_res = pta.kama(close_col, length=period).to_numpy()
    print_tares(None, None, ta_res, ta2_res, None, pta_res)


def test_will_r():
    ta.set_compatibility(1)
    period = 10
    ta_res = ta.WILLR(high_arr, low_arr, close_arr, timeperiod=period)
    ta.set_compatibility(0)
    ta2_res = ta.WILLR(high_arr, low_arr, close_arr, timeperiod=period)
    pta_res = pta.willr(high_col, low_col, close_col, length=period).to_numpy()
    print_tares(None, None, ta_res, ta2_res, None, pta_res)


def test_stoch_rsi():
    ta.set_compatibility(1)
    period = 9
    rsi = ta.RSI(close_arr, timeperiod=period)
    ta_res = ta.STOCH(rsi, rsi, rsi, fastk_period=9, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[0]
    ta.set_compatibility(0)
    rsi = ta.RSI(close_arr, timeperiod=period)
    ta2_res = ta.STOCH(rsi, rsi, rsi, fastk_period=9, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[0]
    pta_res = pta.stochrsi(close_col, length=9, rsi_length=period, k=3, d=3).to_numpy()[:, 0]
    print_tares(None, None, ta_res, ta2_res, None, pta_res)


def test_mfi():
    ta.set_compatibility(1)
    period = 10
    ta_res = ta.MFI(high_arr, low_arr, close_arr, vol_arr, timeperiod=period)
    ta.set_compatibility(0)
    ta2_res = ta.MFI(high_arr, low_arr, close_arr, vol_arr, timeperiod=period)
    mytt_res = mytt.MFI(high_arr, low_arr, close_arr, vol_arr, period)
    pta_res = pta.mfi(high_col, low_col, close_col, vol_col, length=period, talib=False).to_numpy()
    print_tares(None, None, ta_res, ta2_res, mytt_res, pta_res)


def test_aroon():
    ta.set_compatibility(1)
    period = 9
    ta_res = ta.AROON(high_arr, low_arr, timeperiod=period)
    ta.set_compatibility(0)
    ta2_res = ta.AROON(high_arr, low_arr, timeperiod=period)
    arn_res = pta.aroon(high_col, low_col, period).to_numpy()
    print(ta_res)
    print(ta2_res)
    print(arn_res)


def test_any():
    res = calc_state_func(lambda: sta.HeikinAshi())
    print(res)
    # res = Std(ohlcv_arr, 9, 3, 3)[0]
    # print(res)


if __name__ == '__main__':
    test_mfi()
