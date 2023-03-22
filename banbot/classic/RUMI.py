#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : RUMI.py
# Author: anyongjin
# Date  : 2023/2/22
import operator
import logging
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series, concat
from functools import reduce
import pandas_ta as pta
import time

log = logging.getLogger(__name__)


class RUMI(IStrategy):
    '''
    Futures Truth Magazine杂志策略排行第6
    长短均线差再均线平滑。与0交叉时买入卖出
    https://mp.weixin.qq.com/s?__biz=MzkyODI5ODcyMA==&mid=2247484064&idx=1&sn=cfd99a47728f889692845ccb7b0a099d&scene=21#wechat_redirect
    该策略本质使用的是双均线，但不是我们平时的那种用法——“短线上穿长线做多，短线下穿长线做空”，
    而是计算双均线的离差值，并进行平滑处理，类似于微积分求X轴上下面积代数之和的简化方法，
    一定程度上过滤市场噪音，进而发出有效的开平仓信号
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
    startup_candle_count: int = 50
    can_short = False

    # 默认值是3 100 30
    peroid_short = 3
    period_long = 100
    period_sub = 30

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['ma_s'] = ta.SMA(df, timeperiod=self.peroid_short)
        df['ema_l'] = ta.EMA(df, timeperiod=self.period_long)
        df['ma_sub'] = ta.SMA(df['ma_s'] - df['ema_l'], timeperiod=self.period_sub)

        df['low4'] = df['low'].rolling(4).min()
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_filters = [
            df['ma_sub'] > 0,
        ]
        long_ids = np.where(reduce(operator.and_, enter_filters))[0]
        df.loc[long_ids, "enter_long"] = 1
        df.loc[long_ids, "enter_tag"] = 'enter'
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_filters = [
            df['ma_sub'] < 0,
            # 默认没有突破4周期最低退出。测试影响不大
            df['close'] < df['low4'],
        ]
        short_ids = np.where(reduce(operator.or_, exit_filters))[0]
        df.loc[short_ids, "exit_long"] = 1
        df.loc[short_ids, "exit_tag"] = 'exit'
        return df
