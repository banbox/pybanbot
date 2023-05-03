#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : static_pairlist.py
# Author: anyongjin
# Date  : 2023/4/19
from banbot.plugins.pairlist.base import *


class StaticPairList(PairList):
    def __init__(self, manager, exchange: CryptoExchange,
                 config: Config, handler_cfg: Dict[str, Any]):
        super(StaticPairList, self).__init__(manager, exchange, config, handler_cfg)
        self.exg_pairs = AppConfig.get_exchange(config).get('pair_whitelist')
        assert self.exg_pairs, '`pair_whitelist` is requied in exchange for StaticPairList'

    async def gen_pairlist(self, tickers: Tickers) -> List[str]:
        result = self.manager.verify_whitelist(self.exg_pairs)
        return self._filter_unactive(result)

    async def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        old_dic = OrderedDict.fromkeys(pairlist)
        exg_dic = OrderedDict.fromkeys(self.exg_pairs)
        old_dic.update(exg_dic)
        return list(old_dic.keys())
