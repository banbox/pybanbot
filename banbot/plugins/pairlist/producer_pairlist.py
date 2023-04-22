#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : producer_pairlist.py
# Author: anyongjin
# Date  : 2023/4/19
from banbot.plugins.pairlist.base import *


class ProducerPairList(PairList):
    '''
    接收外部传入的交易对。用于分布式机器人。
    Usage:
        "pairlists": [
            {
                "method": "ProducerPairList",
                "limit": 5,
                "producer": "default",
            }
        ],
    '''

    def __init__(self, manager, exchange: CryptoExchange, data_mgr: DataProvider,
                 config: Config, handler_cfg: Dict[str, Any]):
        super(ProducerPairList, self).__init__(manager, exchange, data_mgr, config, handler_cfg)

        self.limit = handler_cfg.get('limit', 0)
        self.from_name = handler_cfg.get('producer', 'default')

    def _filter_pairlist(self, pairlist: Optional[List[str]]):
        add_pairlist = self.data_mgr.get_producer_pairs(self.from_name)
        if pairlist is None:
            pairlist = []

        pairs = list(dict.fromkeys(pairlist + add_pairlist))
        if self.limit:
            pairs = pairs[:self.limit]

        return pairs

    async def gen_pairlist(self, tickers: Tickers) -> List[str]:
        pairs = self._filter_pairlist(None)
        logger.debug('received pairs: %s', pairs)
        result = self.manager.verify_whitelist(pairs)
        return self._filter_unactive(result)

    async def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        return self._filter_pairlist(pairlist)

