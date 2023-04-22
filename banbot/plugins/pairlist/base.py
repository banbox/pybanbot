#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : base.py
# Author: anyongjin
# Date  : 2023/4/17
from banbot.exchange.types import *
from banbot.data.data_provider import *
from banbot.config.consts import *
from typing import *
from copy import deepcopy


class PairList:
    need_tickers = False

    def __init__(self, manager, exchange: CryptoExchange, data_mgr: DataProvider,
                 config: Config, handler_cfg: Dict[str, Any]):
        self.exchange = exchange
        self.manager = manager
        self.data_mgr = data_mgr
        self.config = config
        self.handler_cfg = handler_cfg
        self.stake_currency: Set[str] = set(config.get('stake_currency', []))
        self.enable = bool(handler_cfg.get('enable', True))

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _validate_pair(self, pair: str, ticker: Optional[Ticker]) -> bool:
        raise NotImplementedError()

    async def gen_pairlist(self, tickers: Tickers) -> List[str]:
        raise RuntimeError('This Pairlist should not be used as first')

    async def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        if self.enable:
            for p in deepcopy(pairlist):
                if not self._validate_pair(p, tickers.get(p)):
                    pairlist.remove(p)
        return pairlist

    def _filter_unactive(self, pairlist: List[str]) -> List[str]:
        markets = self.exchange.markets
        if not markets:
            raise RuntimeError('markets not loaded')

        result = OrderedDict()
        not_founds, not_tradables, not_currency, not_active = [], [], [], []
        for pair in pairlist:
            if pair not in markets:
                not_founds.append(pair)
                continue
            market = markets[pair]

            if not self.exchange.market_tradable(market):
                not_tradables.append(pair)
                continue

            if market.get('quote') not in self.stake_currency:
                not_currency.append(pair)
                continue

            if not market.get('active', True):
                not_active.append(pair)
                continue
            result[pair] = 1

        exg_name = self.exchange.name
        if not_founds:
            logger.warning('%d not found in %s, removing %s', len(not_founds), exg_name, not_founds)
        if not_tradables:
            logger.warning('%d not tradable in %s, removing %s', len(not_tradables), exg_name, not_tradables)
        if not_currency:
            logger.warning('%d not currency in %s, removing %s', len(not_currency), exg_name, not_currency)
        if not_active:
            logger.warning('%d not active in %s, removing %s', len(not_active), exg_name, not_active)

        return list(result.keys())

