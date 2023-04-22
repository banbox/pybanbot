#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : offset_filter.py
# Author: anyongjin
# Date  : 2023/4/18
from banbot.plugins.pairlist.base import *


class OffsetFilter(PairList):
    def __init__(self, manager, exchange: CryptoExchange, data_mgr: DataProvider,
                 config: Config, handler_cfg: Dict[str, Any]):
        super(OffsetFilter, self).__init__(manager, exchange, data_mgr, config, handler_cfg)

        self.offset = handler_cfg.get('offset', 0)
        self.limit = handler_cfg.get('limit', 0)
        assert self.offset >= 0

    async def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        result = pairlist[self.offset:]

        if self.limit:
            result = result[:self.limit]

        if len(result) != len(pairlist):
            logger.info(f'{len(pairlist)} pairs left {len(result)} after applying OffsetFilter')

        return result
