#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : Trendflex.py
# Author: anyongjin
# Date  : 2023/2/22
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


class Trendflex(IStrategy):
    '''
    https://mp.weixin.qq.com/s?__biz=MzkyODI5ODcyMA==&mid=2247484458&idx=1&sn=c8f01ddeadd954f432b91bcfe00afc3b&scene=21#wechat_redirect
    ReFlex指标的算法：
        使用低通滤波器平滑价格曲线。
        从N根K线前的价格到当前价格绘制一条线。
        取线到价格的平均垂直距离。
        将结果除以标准偏差。

    TrendFlex指标的算法：
        使用低通滤波器平滑价格曲线。
        将最后N个价格与当前价格的平均差。
        将结果除以标准偏差。
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
        wait_df = df[int(open_pos[0]) + 1:]
        max_low = wait_df['low'].max()
        cur_candle = df.iloc[-1].squeeze()
        time_coef = max(0.3, 1 - 0.1 * len(wait_df))
        thres = cur_candle['open'] * time_coef * self.cus_trailing_stop / 1000
        stoploss = max_low - thres
        if cur_candle['low'] <= stoploss:
            return 'trail_stop'

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        start = time.time()
        df['smo'] = super_smoother(df['close'], 80)
        df['trend_flex'] = re_trend_flex(df['close'], 80)
        # df['re_flex'] = re_trend_flex(df['close'], 80, False)
        cost = time.time() - start
        log.warning(f'super_smoother cost: {cost: .3f} for {len(df)}')
        df['low4'] = df['low'].rolling(4).min()
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_filters = [
            # df['close'] > df['smo'],
            # df['close'].shift() > df['smo'],
            df['trend_flex'] > df['trend_flex'].rolling(20).max().shift(1)
            # (df['trend_flex'] > df['trend_flex'].shift() | df['re_flex'] > df['re_flex'].shift())
        ]
        long_ids = np.where(reduce(operator.and_, enter_filters))[0]
        df.loc[long_ids, "enter_long"] = 1
        df.loc[long_ids, "enter_tag"] = 'enter'
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # exit_filters = [
        #     df['close'] < df['smo'],
        #     # 默认没有突破4周期最低退出。测试影响不大
        #     df['close'] < df['low4'],
        # ]
        # short_ids = np.where(reduce(operator.or_, exit_filters))[0]
        # df.loc[short_ids, "exit_long"] = 1
        # df.loc[short_ids, "exit_tag"] = 'exit'
        return df
