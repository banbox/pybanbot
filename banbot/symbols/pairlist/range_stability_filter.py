#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : range_stability_filter.py
# Author: anyongjin
# Date  : 2023/4/19
from cachetools import TTLCache
from banbot.symbols.pairlist.base import *
from banbot.compute.tainds import hcol, lcol


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
        new_pairs = [pair for pair in pairlist if pair not in self.pair_cache]
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
