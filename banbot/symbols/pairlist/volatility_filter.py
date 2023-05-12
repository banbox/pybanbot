#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : volatility_filter.py
# Author: anyongjin
# Date  : 2023/4/19
import sys

import numpy as np
from cachetools import TTLCache
from banbot.symbols.pairlist.base import *
from banbot.compute.tainds import ccol
from banbot.util.misc import parallel_jobs


class VolatilityFilter(PairList):
    '''
    波动性过滤器。计算的波动性分值和实际波动百分比接近，在0~1之间
max: 1.122  min: 0.726  avg: 1.006  avg_log: 0.032 std_log: 0.049 score: 0.154
max: 1.144  min: 0.827  avg: 1.004  avg_log: 0.019 std_log: 0.063 score: 0.200
max: 1.024  min: 0.978  avg: 1.000  avg_log: 0.002 std_log: 0.012 score: 0.039
max: 1.050  min: 0.910  avg: 1.001  avg_log: 0.009 std_log: 0.026 score: 0.081
max: 1.088  min: 0.745  avg: 1.005  avg_log: 0.029 std_log: 0.034 score: 0.108
max: 1.207  min: 0.833  avg: 1.005  avg_log: 0.006 std_log: 0.099 score: 0.312
max: 1.209  min: 0.724  avg: 1.008  avg_log: 0.032 std_log: 0.072 score: 0.229
max: 1.080  min: 0.834  avg: 1.003  avg_log: 0.018 std_log: 0.045 score: 0.141
max: 1.238  min: 0.756  avg: 1.008  avg_log: 0.028 std_log: 0.090 score: 0.286
max: 1.049  min: 0.924  avg: 1.001  avg_log: 0.008 std_log: 0.025 score: 0.080
max: 1.122  min: 0.805  avg: 1.004  avg_log: 0.022 std_log: 0.052 score: 0.163
max: 1.190  min: 0.541  avg: 1.024  avg_log: 0.061 std_log: 0.111 score: 0.351
max: 1.146  min: 0.769  avg: 1.006  avg_log: 0.026 std_log: 0.066 score: 0.210
max: 1.057  min: 0.869  avg: 1.001  avg_log: 0.014 std_log: 0.030 score: 0.095
max: 1.116  min: 0.757  avg: 1.005  avg_log: 0.028 std_log: 0.044 score: 0.140
max: 1.050  min: 0.941  avg: 1.000  avg_log: 0.004 std_log: 0.027 score: 0.086
    '''
    def __init__(self, manager, exchange: CryptoExchange,
                 config: Config, handler_cfg: Dict[str, Any]):
        super(VolatilityFilter, self).__init__(manager, exchange, config, handler_cfg)

        self.backdays = handler_cfg.get('back_days', 10)
        self.min_val = handler_cfg.get('min', 0)
        self.max_val = handler_cfg.get('max', sys.maxsize)
        self.refresh_period = handler_cfg.get('refresh_period', 1440)

        self.pair_cache = TTLCache(maxsize=1000, ttl=self.refresh_period)

        assert self.backdays > 0, 'RangeStabilityFilter back_days should >= 1'
        candle_limit = exchange.ohlcv_candle_limit('1d')
        assert self.backdays <= candle_limit, f'RangeStabilityFilter back_days should <= {candle_limit}'

    async def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        new_pairs = [p for p in pairlist if p not in self.pair_cache]
        for p in new_pairs:
            candles = await auto_fetch_ohlcv(self.exchange, p, '1d', limit=self.backdays)
            if not self._validate_pair_loc(p, candles):
                pairlist.remove(p)
        return pairlist

    def _validate_pair_loc(self, pair: str, candles: List):
        if not candles:
            logger.info('remove %s in VolatilityFilter, no candles found', pair)
            return False
        close_arr = np.array(candles)[:, ccol]
        close_s1 = np.roll(close_arr, -1)
        ratio_arr = close_arr / close_s1
        log_ratios = np.log(ratio_arr)  # 平滑，并将均值改为0
        log_ratios[-1:] = 0

        arr_len = len(candles)
        volatility_avg = log_ratios.std() * np.sqrt(arr_len)

        result = True
        if volatility_avg < self.min_val or volatility_avg > self.max_val:
            result = False
            logger.info(f'remove {pair} as volatility {volatility_avg:.3f} not in range {self.min_val}~{self.max_val}')
        self.pair_cache[pair] = result
        return result
