#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : wacther.py
# Author: anyongjin
# Date  : 2023/4/30
import sys
from typing import List, Tuple, Callable
from banbot.util import btime
from banbot.util.common import logger


class PairTFCache:
    def __init__(self, timeframe: str, tf_decs: int):
        self.timeframe = timeframe
        self.tf_secs = tf_decs
        self.wait_bar = None  # 记录尚未完成的bar。已完成时应置为None
        self.latest = None  # 记录最新bar数据，可能未完成，可能已完成

    def set_bar(self, bar_row):
        self.wait_bar = bar_row
        self.latest = bar_row


class Watcher:
    def __init__(self, pair: str, callback: Callable):
        self.pair = pair
        self.callback = callback

    def _on_state_ohlcv(self, state: PairTFCache, ohlcvs: List[Tuple], last_finish: bool, do_fire=True) -> list:
        finish_bars = []
        for i in range(len(ohlcvs)):
            new_bar = ohlcvs[i]
            if state.wait_bar and state.wait_bar[0] < new_bar[0]:
                finish_bars.append(state.wait_bar)
            state.set_bar(new_bar)
        if last_finish:
            finish_bars.append(state.wait_bar)
            state.wait_bar = None
        if finish_bars and do_fire:
            self._fire_callback(finish_bars, state.timeframe, state.tf_secs)
        return finish_bars

    def _fire_callback(self, bar_arr, timeframe: str, tf_secs: int):
        for bar_row in bar_arr:
            if btime.run_mode not in btime.LIVE_MODES:
                btime.cur_timestamp = bar_row[0] / 1000 + tf_secs
            self.callback(self.pair, timeframe, bar_row)
        if btime.run_mode in btime.LIVE_MODES:
            bar_delay = btime.time() - bar_arr[-1][0] // 1000 - tf_secs
            if bar_delay > tf_secs:
                # 当蜡烛的触发时间过于滞后时，输出错误信息
                logger.error('{0}/{1} bar is too late, delay:{2}', self.pair, timeframe, bar_delay)
