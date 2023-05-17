#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : base.py
# Author: anyongjin
# Date  : 2023/4/17
from banbot.exchange.types import *
from banbot.data.provider import *
from banbot.config.consts import *
from typing import *
from copy import deepcopy


class PairList:
    need_tickers = False

    def __init__(self, manager, exchange: CryptoExchange, config: Config, handler_cfg: Dict[str, Any]):
        self.exchange = exchange
        self.manager = manager
        self.config = config
        self.handler_cfg = handler_cfg
        self.enable = bool(handler_cfg.get('enable', True))

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _validate_pair(self, pair: str, ticker: Optional[Ticker]) -> bool:
        raise NotImplementedError()

    async def gen_pairlist(self, tickers: Tickers) -> List[str]:
        raise RuntimeError('This Pairlist should not be used as first')

    async def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        if not self.enable:
            return pairlist
        for p in deepcopy(pairlist):
            if not self._validate_pair(p, tickers.get(p)):
                pairlist.remove(p)
        return pairlist

