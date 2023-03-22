#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : T20Extremum.py
# Author: anyongjin
# Date  : 2023/2/23
import operator
import logging
from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade, LocalTrade
from functools import reduce
from banbot.compute.classic_inds import *
from datetime import datetime
from typing import Union, Optional
import time

log = logging.getLogger(__name__)


class T20Extremum(IStrategy):
    '''
    https://mp.weixin.qq.com/s?__biz=MzkyODI5ODcyMA==&mid=2247484119&idx=1&sn=40636937af309698dd56a24a129dad8e&scene=21#wechat_redirect
    日内极值突破策略
    未复现策略的效果。
    '''
    INTERFACE_VERSION = 3
    version = '0.1.0'

    minimal_roi = {
        "0": 10
    }

    # 默认观察频率
    timeframe = '5m'

    # 复购、部分卖出
    position_adjustment_enable = False
    # 交易所宕机保护
    has_downtime_protection = False

    stoploss = -0.1
    use_custom_stoploss = False

    # Exit options
    use_exit_signal = True
    ignore_roi_if_entry_signal = True

    # 最少前置蜡烛数
    startup_candle_count: int = 150
    can_short = False

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # df['smo'] = super_smoother(df['close'], 20)
        # df['smo_sub2'] = df['smo'] - df['smo'].shift(2)
        # df['smo_sub2_2'] = df['smo_sub2'].shift(2)
        # # 查找极小值，做多
        # cross_up_ids = np.where((df['smo_sub2'] >= 0) & (df['smo_sub2_2'] < 0))[0]
        # df['rol3_max'] = df['close'].rolling(5).max().shift()
        # df['hh'] = df['rol3_max'][cross_up_ids]
        # print(df['hh'].dropna())
        # df['hh'].fillna(method='ffill', inplace=True)
        # df['hh_s1'] = df['hh'].shift()
        # # 查找极大值，做空
        # cross_down_ids = np.where((df['smo_sub2'] <= 0) & (df['smo_sub2_2'] > 0))[0]
        # df['rol3_min'] = df['close'].rolling(3).min().shift()
        # df['ll'] = df['rol3_min'][cross_up_ids]
        # df['ll_s1'] = df['ll'].shift()
        # df['ll'].fillna(method='ffill', inplace=True)


        df['close_s1'] = df['close'].shift()
        df['close_s2'] = df['close'].shift(2)
        df['slope'] = ta.LINEARREG_SLOPE(df['close_s1'], 18)
        df['slope_s1'] = df['slope'].shift()
        # 查找极小值，做多
        cross_up_ids = np.where((df['slope_s1'] <= 0) & (df['slope'] > 0))[0]
        df['hh'] = np.maximum(df.loc[cross_up_ids, 'close_s1'], df.loc[cross_up_ids, 'close_s2'])
        df['hh_s1'] = df['hh'].shift()
        # 查找极大值，做空
        cross_down_ids = np.where((df['slope_s1'] >= 0) & (df['slope'] < 0))[0]
        df['ll'] = np.minimum(df.loc[cross_down_ids, 'close_s1'], df.loc[cross_down_ids, 'close_s2'])
        df['ll_s1'] = df['ll'].shift()
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_filters = [
            df['close_s1'] > df['hh_s1'],
            df['hh_s1'] > df['ll_s1']
        ]
        long_ids = np.where(reduce(operator.and_, enter_filters))[0]
        df.loc[long_ids, "enter_long"] = 1
        df.loc[long_ids, "enter_tag"] = 'enter'
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_filters = [
            ((df['close_s1'] < df['ll_s1']) & (df['hh_s1'] > df['ll_s1']))
        ]
        short_ids = np.where(reduce(operator.or_, exit_filters))[0]
        df.loc[short_ids, "exit_long"] = 1
        df.loc[short_ids, "exit_tag"] = 'exit'
        return df
