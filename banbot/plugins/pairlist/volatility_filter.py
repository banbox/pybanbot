#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : volatility_filter.py
# Author: anyongjin
# Date  : 2023/4/19
import sys

import numpy as np
from cachetools import TTLCache
from banbot.plugins.pairlist.base import *
from banbot.bar_driven.tainds import ccol
from banbot.util.misc import parallel_jobs


class VolatilityFilter(PairList):
    '''
    波动性过滤器。
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
        since = int(btime.time() - self.backdays * 86_400)
        args_list = [(self.exchange.name, p, '1d', since) for p in new_pairs]
        pair_res = parallel_jobs(auto_fetch_ohlcv, args_list)
        for job in pair_res:
            job_res = await job
            candles, p = job_res['data'], job_res['args'][0]
            if not self._validate_pair_loc(p, candles):
                pairlist.remove(p)
        return pairlist

    def _validate_pair_loc(self, pair: str, candles: List):
        if not candles:
            logger.info('remove %s in RangeStabilityFilter, no candles found', pair)
            return False
        close_arr = np.array(candles)[:, ccol]
        close_s1 = np.roll(close_arr, -1)
        returns = np.log(close_arr / close_s1)
        returns[-1:] = 0

        std_dev = np.empty(len(returns))
        for i in range(self.backdays - 1, len(returns)):
            std_dev[i] = returns[i - self.backdays + 1:i + 1].std()
        std_dev = std_dev[self.backdays - 1:]

        # Scale by square root of period
        std_dev *= np.sqrt(self.backdays)

        volatility_avg = np.mean(std_dev)
        result = True
        if volatility_avg < self.min_val or volatility_avg > self.max_val:
            result = False
            logger.info(f'remove {pair} as volatility {volatility_avg} not in range {self.min_val}~{self.max_val}')
        self.pair_cache[pair] = result
        return result
