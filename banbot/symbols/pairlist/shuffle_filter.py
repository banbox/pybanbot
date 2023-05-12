#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : shuffle_filter.py
# Author: anyongjin
# Date  : 2023/4/19
import random

from banbot.symbols.pairlist.base import *


class ShuffleFilter(PairList):
    def __init__(self, manager, exchange: CryptoExchange,
                 config: Config, handler_cfg: Dict[str, Any]):
        super(ShuffleFilter, self).__init__(manager, exchange, config, handler_cfg)

        if btime.run_mode in LIVE_MODES:
            self.seed = None
        else:
            self.seed = handler_cfg.get('seed')
            logger.info('apply seed for pairlist: %s', self.seed)

        self.random = random.Random(self.seed)

    async def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        self.random.shuffle(pairlist)
        return pairlist
