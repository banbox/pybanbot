#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : TrendModelSys.py
# Author: anyongjin
# Date  : 2023/2/21
import operator
import logging
import talib.abstract as ta
from banbot.strategy.base import *
from pandas import DataFrame, Series, concat
from functools import reduce
from banbot.util.num_utils import np_shift
import time

log = logging.getLogger(__name__)


class TrendModelSys(BaseStrategy):
    '''
    Futures Truth Magazine杂志策略排行第9
    https://mp.weixin.qq.com/s?__biz=MzkyODI5ODcyMA==&mid=2247484039&idx=1&sn=defcd9c0c03653ed1078ba98392af315&scene=21#wechat_redirect
    这个策略是一个趋势突破型的交易系统，利用MACD快(DIF)慢(DEA)线的金叉死叉，
    当证券价格突破过去N次金/死叉记录的“最高/低价±0.5倍ATR”时，开仓；
    若持有多头仓位，当证券价格回落至M根K线最低点平仓；
    若持有空头仓位，当证券价格上升至过去M根K线最高点平仓。

    个人感觉，这个策略值得借鉴最大的亮点就是“关键点位思想”，
    以前突破通道的确定往往是根据过去整一段时间的指标值/因子值，而这里只考虑关键点位的指标值/因子值。
    例如唐奇安通道上轨是过去N个交易日的最大值，那关键点位就是可以只考虑过去N个金叉对应的最高价的最大值，
    这里只是只是打个比方，类推就好了，就如同以下评论，点评得非常到位。
    '''

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['macd_f'], df['macd_s'], df['macd_hist'] = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['hi_tag'] = df['macd_hist'].apply(lambda x: 1 if x >= 0 else -1)

        num_cross, back_period = 4, 50
        # 找到MACD金叉或死叉
        df['hi_cross'] = df['hi_tag'].shift() + df['hi_tag']
        cross_ids = np.where(df['hi_cross'] == 0)[0]

        # 计算与前4个交叉的距离，查找距离不超过back_period的交叉位置
        cro_num_periods = cross_ids - np_shift(cross_ids, num_cross, -back_period)
        good_ids = np.where(cro_num_periods < back_period)[0]
        df['cross'] = np.nan
        df.loc[cross_ids, 'cross'] = 0
        df.loc[cross_ids[good_ids], 'cross'] = 1
        df['cross'].fillna(method='bfill', inplace=True)

        # 计算开多/开空的入场价格
        df['atr'] = ta.ATR(df, timeperiod=4)
        df['roll_high'] = df.loc[cross_ids, 'high'].rolling(num_cross).max() + df.loc[cross_ids, 'atr'] * 0.5
        # df['roll_low'] = df.loc[cross_ids, 'low'].rolling(num_cross).min() - df.loc[cross_ids, 'atr'] * 0.5
        df['roll_high'].fillna(method='bfill', inplace=True)
        # df['roll_low'].fillna(method='bfill', inplace=True)

        df.loc[df['close'] > df['roll_high'], 'cross'] += 1
        # df.loc[df['close'] < df['roll_low'], 'cross'] += 1

        df['low4'] = df['low'].rolling(6).min()
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_filters = [
            df['cross'] > 1,
            df['macd_f'] > df['macd_s']
        ]
        long_ids = np.where(reduce(operator.and_, enter_filters))[0]
        df.loc[long_ids, "enter_long"] = 1
        df.loc[long_ids, "enter_tag"] = 'cross'
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_filters = [
            df['close'] < df['low4'],
        ]
        short_ids = np.where(reduce(operator.or_, exit_filters))[0]
        df.loc[short_ids, "exit_long"] = 1
        df.loc[short_ids, "exit_tag"] = 'low_min4'
        return df
