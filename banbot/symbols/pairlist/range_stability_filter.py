#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : range_stability_filter.py
# Author: anyongjin
# Date  : 2023/4/19
import numpy as np
from cachetools import TTLCache

from banbot.compute.sta_inds import hcol, lcol
from banbot.symbols.pairlist.base import *


class RangeStabilityFilter(PairList):

    def __init__(self, manager, exchange: CryptoExchange,
                 config: Config, handler_cfg: Dict[str, Any]):
        super(RangeStabilityFilter, self).__init__(manager, exchange, config, handler_cfg)

        self.backdays = handler_cfg.get('back_days', 10)
        self.min_roc = handler_cfg.get('min_chg_rate', 0.01)
        self.max_roc = handler_cfg.get('max_chg_rate')
        self.refresh_period = handler_cfg.get('refresh_period', 1440)

        self.pair_cache = TTLCache(maxsize=1000, ttl=self.refresh_period)

        assert self.backdays > 0, 'RangeStabilityFilter back_days should >= 1'
        candle_limit = exchange.ohlcv_candle_limit('1d')
        assert self.backdays <= candle_limit, f'RangeStabilityFilter back_days should <= {candle_limit}'

    async def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        if not self.enable:
            return pairlist

        def kline_cb(candles, exs: ExSymbol, timeframe, **kwargs):
            if not self._validate_pair_loc(exs.symbol, candles):
                pairlist.remove(exs.symbol)

        new_pairs = [pair for pair in pairlist if pair not in self.pair_cache]
        down_args = dict(limit=self.backdays)
        await bulk_ohlcv_do(self.exchange, new_pairs, '1d', down_args, kline_cb)
        return pairlist

    def _validate_pair_loc(self, pair: str, candles: List):
        if not candles:
            logger.info('remove %s in RangeStabilityFilter, no candles found', pair)
            return False
        arr = np.array(candles)
        hhigh, llow = arr[:, hcol].max(), arr[:, lcol].min()
        pct_change = ((hhigh - llow) / llow) if llow else 0
        result = True
        if pct_change < self.min_roc:
            logger.info('remove %s because rate of change over %d days is %f < %f', pair,
                        self.backdays, pct_change, self.min_roc)
            result = False
        if self.max_roc and pct_change > self.max_roc:
            logger.info('remove %s because rate of change over %d days is %f > %f', pair,
                        self.backdays, pct_change, self.min_roc)
            result = False
        self.pair_cache[pair] = result
        return result
