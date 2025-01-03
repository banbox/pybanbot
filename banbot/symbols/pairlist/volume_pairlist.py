#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : volume_pairlist.py
# Author: anyongjin
# Date  : 2023/4/19

from banbot.compute.sta_inds import ccol, vcol
from banbot.symbols.pairlist.base import *

SORT_VALUES = ['quoteVolume']


class VolumePairList(PairList):
    is_generator = True
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
            # 非实时模式、实时模式非按天更新
            res_list = []

            def kline_cb(data, exs: ExSymbol, timeframe, **kwargs):
                item = dict(symbol=exs.symbol)
                if not data:
                    item['quoteVolume'] = 0
                else:
                    item['quoteVolume'] = sum(bar[ccol] * bar[vcol] for bar in data[-self.backperiod:])
                res_list.append(item)
            down_args = dict(limit=self.backperiod)
            if not BotGlobal.live_mode:
                down_args['start_ms'] = self.config['timerange'].startts * 1000
            await fast_bulk_ohlcv(self.exchange, pairlist, self.backtf, callback=kline_cb, **down_args)
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
