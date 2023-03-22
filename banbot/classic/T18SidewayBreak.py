#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : T18SidewayBreak.py
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


class T18SidewayBreak(IStrategy):
    '''
    https://mp.weixin.qq.com/s?__biz=MzkyODI5ODcyMA==&mid=2247483852&idx=1&sn=00e3211e3ad821606e6d10b9bdc09b5a&scene=21#wechat_redirect
    波动率收敛突破【横盘突破】

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

    cus_trailing_stop = 75

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """
        跟踪止损。
        """
        elipsed_units = round((current_time - trade.open_date).total_seconds() // 60)
        if not elipsed_units:
            return None
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        open_pos = np.where(df['date'] == trade.open_date)
        if not len(open_pos) or len(df) <= open_pos[0] + 1:
            return None
        wait_df = df[int(open_pos[0]):]
        if len(wait_df) <= 5:
            if current_profit < -0.02:
                return 'loss_2%'
            return
        entry_highs = wait_df['high'].tolist()
        hhigh = entry_highs[0]
        sar, af, af_step = wait_df.iloc[0]['low'], 0.02, 0.02
        for val in entry_highs[1:]:
            if val > hhigh:
                hhigh = val
                if af + af_step <= 0.2:
                    af += af_step
            sar += (hhigh - sar) * af
        if wait_df.iloc[-1]['low'] < sar:
            return 'sar_stop'

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # df['stddev'] = ta.STDDEV(df, timeperiod=26)
        # df['ma20'] = ta.SMA(df['close'], timeperiod=20)
        # band_fac = 2
        # df['upband'] = df['ma20'] + df['stddev'] * band_fac
        # df['lowband'] = df['ma20'] - df['stddev'] * band_fac
        # band_period = round(std_period * 0.5)
        # df['max_up'] = df['upband'].rolling(band_period).max()
        # df['min_low'] = df['lowband'].rolling(band_period).min()
        std_period, ma_period = 20, 16
        df['ma20'] = ta.SMA(df['close'], timeperiod=ma_period)
        df['ma5'] = ta.SMA(df['close'], timeperiod=5)
        df['stddev'] = ta.STDDEV(df, timeperiod=std_period)
        df['nstddev'] = df['stddev'] * 1000 / df['ma20']
        df['stddev_rol_avg'] = df['nstddev'].rolling(std_period * 5).mean()  # .shift(round(std_period * 0.5))
        df['std_norm'] = df['nstddev'] / df['stddev_rol_avg'] - 1

        # df['dir'] = df.apply(lambda x: 1 if x[4] >= x[1] else -1, axis=1)
        # df['dir_s5'] = df['dir'].rolling(5).sum()


        df['ma3'] = ta.SMA(df['close'], timeperiod=3)
        df['subs'] = df['ma3'] - df['ma3'].shift()
        df['subs_ma'] = df['subs'].rolling(4).mean()
        df['subs'] = df[['subs', 'subs_ma']].min(axis=1)
        df['chg5'] = df['subs'].rolling(5).sum() * 100 / df['ma5']
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_filters = [
            df['std_norm'] > 0.1,
            df['ma5'] > df['ma5'].shift(),
            df['close'] > df['close'].shift(),
            df['chg5'] >= 0.0005
        ]
        long_ids = np.where(reduce(operator.and_, enter_filters))[0]
        df.loc[long_ids, "enter_long"] = 1
        df.loc[long_ids, "enter_tag"] = 'enter'
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # exit_filters = [
        #     df['close'].shift() <= df['min_low'].shift()
        # ]
        # short_ids = np.where(reduce(operator.or_, exit_filters))[0]
        # df.loc[short_ids, "exit_long"] = 1
        # df.loc[short_ids, "exit_tag"] = 'exit'
        return df
