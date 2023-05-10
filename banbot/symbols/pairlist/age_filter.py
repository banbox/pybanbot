#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : age_filter.py
# Author: anyongjin
# Date  : 2023/4/17
from banbot.symbols.pairlist.base import *
from banbot.util.cache import *


class AgeFilter(PairList):
    def __init__(self, manager, exchange: CryptoExchange,
                 config: Config, handler_cfg: Dict[str, Any]):
        super(AgeFilter, self).__init__(manager, exchange, config, handler_cfg)

        # Checked symbols cache (dictionary of ticker symbol => timestamp)
        self._checked: Dict[str, int] = {}
        self._failed = PunctualCache(maxsize=1000, ttl=86_400)

        self.min_days = handler_cfg.get('min', 10)
        assert 1 <= self.min_days <= 1000, f'min days should be range[1, 1000], cur: {self.min_days}'
        self.max_days = handler_cfg.get('max', 0)
        if self.max_days:
            assert self.max_days <= 1000, f'max days should be range[1, 1000], cur: {self.max_days}'

    async def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        result = await self._do_filter(pairlist)
        if len(result) != len(pairlist):
            logger.info(f'{len(pairlist)} pairs left {len(result)} after applying AgeFilter')
        return result

    async def _do_filter(self, pairlist: List[str]) -> List[str]:
        nofails = [p for p in pairlist if p not in self._failed]
        new_pairs = [p for p in nofails if p not in self._checked]
        if not new_pairs:
            # Remove pairs that have been removed before
            return nofails
        if not self.enable:
            return pairlist
        since_days = (self.max_days if self.max_days else self.min_days) + 1
        for pair in new_pairs:
            candles = await auto_fetch_ohlcv(self.exchange, pair, '1d', limit=since_days)
            knum = len(candles)
            cur_ms = btime.time() * 1000
            if knum >= self.min_days and (not self.max_days or knum <= self.max_days):
                self._checked[pair] = cur_ms
            else:
                self._failed[pair] = cur_ms
                nofails.remove(pair)

        return nofails
