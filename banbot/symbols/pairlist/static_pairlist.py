#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : static_pairlist.py
# Author: anyongjin
# Date  : 2023/4/19
from banbot.symbols.pairlist.base import *


class StaticPairList(PairList):
    is_generator = True

    def __init__(self, manager, exchange: CryptoExchange,
                 config: Config, handler_cfg: Dict[str, Any]):
        super(StaticPairList, self).__init__(manager, exchange, config, handler_cfg)
        self.exg_pairs = exchange.exg_config.get('pair_whitelist')
        assert self.exg_pairs, '`pair_whitelist` is requied in exchange for StaticPairList'

    async def gen_pairlist(self, tickers: Tickers) -> List[str]:
        result = self.manager.verify_whitelist(self.exg_pairs)
        ava_symbols = self.manager.avaiable_symbols
        result = [p for p in result if p in ava_symbols]
        return result

    async def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        old_dic = OrderedDict.fromkeys(pairlist)
        exg_dic = OrderedDict.fromkeys(self.exg_pairs)
        old_dic.update(exg_dic)
        return list(old_dic.keys())
