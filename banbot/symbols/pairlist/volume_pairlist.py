#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : volume_pairlist.py
# Author: anyongjin
# Date  : 2023/4/19
import numpy as np
from cachetools import TTLCache

from banbot.compute.sta_inds import hcol, lcol, ccol, vcol
from banbot.symbols.pairlist.base import *

SORT_VALUES = ['quoteVolume']


class VolumePairList(PairList):
    '''
    交易量倒排产品列表
    '''
    def __init__(self, manager, exchange: CryptoExchange,
                 config: Config, handler_cfg: Dict[str, Any]):
        super(VolumePairList, self).__init__(manager, exchange, config, handler_cfg)

        self.limit = handler_cfg.get('limit', sys.maxsize)
        self.sort_key: str = handler_cfg.get('sort_key') or 'quoteVolume'
        self.min_value = handler_cfg.get('min_value', 0)
        self.refresh_secs = handler_cfg.get('refresh_secs', 7200)  # in secs
        self.pair_cache = TTLCache(maxsize=1, ttl=self.refresh_secs)
        self.backtf = handler_cfg.get('back_timeframe', '1d')
        if self.backtf not in KLine.agg_map:
            raise RuntimeError(f'`back_timeframe` must in {KLine.agg_map.keys()}')
        self.backperiod = handler_cfg.get('back_period', 1)

        back_secs = tf_to_secs(self.backtf) * self.backperiod
        self.use_tickers = back_secs <= 0 or back_secs == 86400
        if self.use_tickers and btime.run_mode not in LIVE_MODES:
            self.use_tickers = False
            self.backperiod = self.backperiod or 1

        if self.use_tickers and not (self.exchange.has_api('fetchTickers') and
                                     self.exchange.get_option('tickers_have_quoteVolume')):
            raise RuntimeError(f'exchange {self.exchange.name} not support tickers required by VolumePairList')

        if self.sort_key not in SORT_VALUES:
            raise RuntimeError(f'unsupport sort_key: {self.sort_key}, all supported: {SORT_VALUES}')

        candle_limit = exchange.ohlcv_candle_limit('1d')
        assert 0 <= self.backperiod <= candle_limit, f'back_days should in range [1, {candle_limit}]'

    @property
    def need_tickers(self):
        return self.use_tickers

    async def gen_pairlist(self, tickers: Tickers) -> List[str]:
        pairlist = self.pair_cache.get('pairlist')
        if pairlist:
            return pairlist.copy()

        if self.use_tickers and tickers:
            pairlist = [v['symbol'] for k, v in tickers.items() if v.get(self.sort_key)]
        else:
            pairlist = list(self.manager.avaiable_symbols)

        pairlist = await self.filter_pairlist(pairlist, tickers)
        self.pair_cache['pairlist'] = pairlist
        return pairlist

    async def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        if not self.use_tickers:
            res_list = []

            def kline_cb(data, exs: ExSymbol, timeframe, **kwargs):
                item = dict(symbol=exs.symbol)
                contract_size = self.exchange.markets[exs.symbol].get('contractSize', 1.0) or 1.0
                if not data:
                    item['quoteVolume'] = 0
                else:
                    arr = np.array(data)
                    if self.exchange.get_option('ohlcv_volume_currency') == 'base':
                        typical = (arr[:, hcol] + arr[:, lcol] + arr[:, ccol]) / 3
                        quote_vol = typical * arr[:, vcol] * contract_size
                    else:
                        quote_vol = arr[:, vcol]
                    item['quoteVolume'] = np.sum(quote_vol[-self.backperiod:])
                res_list.append(item)
            down_args = dict(limit=self.backperiod)
            await bulk_ohlcv_do(self.exchange, pairlist, self.backtf, down_args, kline_cb)
        else:
            # Tickers mode - filter based on incoming pairlist.
            res_list = [dict(symbol=k, quoteVolume=v.get('quoteVolume', 0))
                        for k, v in tickers.items() if k in pairlist]

        if self.min_value:
            res_list = [v for v in res_list if v[self.sort_key] > self.min_value]

        res_list = sorted(res_list, key=lambda x: x[self.sort_key], reverse=True)
        res_pairs = [p['symbol'] for p in res_list]

        ava_symbols = self.manager.avaiable_symbols
        res_pairs = [p for p in res_pairs if p in ava_symbols]
        res_pairs = res_pairs[:self.limit]

        return res_pairs
