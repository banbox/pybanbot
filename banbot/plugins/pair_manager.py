#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : pair_manager.py
# Author: anyongjin
# Date  : 2023/4/17
from banbot.plugins.pair_resolver import *
from banbot.plugins.pairlist.helper import *
from banbot.exchange.crypto_exchange import CryptoExchange
from cachetools import TTLCache, cached


class PairManager:
    '''
    暂时不建议在回测中使用：VolumePairlist, SpreadFilter, PriceFilter
    '''

    def __init__(self, config: Config, exchange: CryptoExchange, data_mgr: DataProvider):
        self.exchange = exchange
        self.config = config
        self.handlers = PairResolver.load_handlers(exchange, self, config, data_mgr)
        self._whitelist = config['exchange'].get('pair_whitelist')
        self._blacklist = config['exchange'].get('pair_blacklist', [])
        if not self.handlers:
            raise RuntimeError('no pairlist defined')
        ticker_names = [h.name for h in self.handlers if h.need_tickers]
        self.need_tickers = bool(ticker_names)
        if not exchange.has_api('fetchTickers') and ticker_names:
            raise ValueError(f'exchange not support fetchTickers, affect: {ticker_names}')
        self.ticker_cache = TTLCache(maxsize=1, ttl=1800)

    @property
    def symbols(self):
        return self._whitelist

    async def refresh_pairlist(self):
        tickers = self.ticker_cache.get('tickers')
        if not tickers and self.need_tickers:
            tickers = await self.exchange.fetch_tickers()
            self.ticker_cache['tickers'] = tickers

        pairlist = await self.handlers[0].gen_pairlist(tickers)
        for handler in self.handlers[1:]:
            pairlist = await handler.filter_pairlist(pairlist, tickers)

        pairlist = self.verify_blacklist(pairlist)
        self._whitelist = pairlist

    def verify_whitelist(self, pairlist: List[str], keep_invalid: bool = False) -> List[str]:
        try:
            all_pairs = list(self.exchange.markets.keys())
            return search_pairlist(pairlist, all_pairs, keep_invalid)
        except ValueError as e:
            logger.error(f'pair list contains invalid wildcard {e}')
            return []

    def verify_blacklist(self, pairlist: List[str]) -> List[str]:
        if not self._blacklist:
            return pairlist
        try:
            all_pairs = list(self.exchange.markets.keys())
            all_blacks = search_pairlist(self._blacklist, all_pairs)
        except ValueError as e:
            logger.error(f'pair list contains invalid wildcard {e}')
            return []
        del_keys = set(pairlist).intersection(all_blacks)
        if del_keys:
            for k in del_keys:
                pairlist.remove(k)
        return pairlist
