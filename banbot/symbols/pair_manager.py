#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : pair_manager.py
# Author: anyongjin
# Date  : 2023/4/17

from cachetools import TTLCache

from banbot.symbols.pair_resolver import *
from banbot.symbols.pairlist.helper import *
from banbot.symbols.tfscaler import calc_symboltf_scales


class PairManager:
    '''
    交易对管理器，自动刷新筛选要交易的货币。
    同时计算交易对的最佳交易周期。
    暂时不建议在回测中使用：SpreadFilter
    '''

    def __init__(self, config: Config, exchange: CryptoExchange):
        self.exchange = exchange
        self.config = config
        self.market_type = config.get('market_type') or 'spot'
        self.stake_currency: Set[str] = set(config.get('stake_currency') or [])
        self._whitelist = exchange.exg_config.get('pair_whitelist')
        self._blacklist = exchange.exg_config.get('pair_blacklist', [])
        self.handlers = PairResolver.load_handlers(exchange, self, config)
        if not self.handlers:
            raise RuntimeError('no pairlist defined')
        ticker_names = [h.name for h in self.handlers if h.need_tickers]
        self.need_tickers = bool(ticker_names)
        if not exchange.has_api('fetchTickers') and ticker_names:
            raise ValueError(f'exchange not support fetchTickers, affect: {ticker_names}')
        self.ticker_cache = TTLCache(maxsize=1, ttl=1800)
        pair_cfg = config.get('paircfg') or dict()
        self.refresh_secs = pair_cfg.get('refresh_mins', 720) * 60  # 交易对定期刷新间隔
        self._ava_at = 0
        self._ava_symbols = None
        self.pair_tfscores: Dict[str, List[Tuple[str, float]]] = dict()  # 记录每个交易对的周期质量分数

    @property
    def symbols(self):
        return self._whitelist

    async def refresh_pairlist(self):
        if self.config.get('pairs'):
            # 回测模式传入pairs
            pairlist = self.config['pairs']
        else:
            tickers = self.ticker_cache.get('tickers') or dict()
            if not tickers and self.need_tickers and btime.run_mode in LIVE_MODES:
                ava_symbols = list(self.avaiable_symbols)
                tickers = await self.exchange.fetch_tickers(ava_symbols)
                self.ticker_cache['tickers'] = tickers

            pairlist = await self.handlers[0].gen_pairlist(tickers)
            logger.info(f'get {len(pairlist)} symbols from {self.handlers[0].name}')
            for handler in self.handlers[1:]:
                pairlist = await handler.filter_pairlist(pairlist, tickers)
                logger.info(f'left {len(pairlist)} symbols after {handler.name}')

        # 计算交易对各维度K线质量分数
        back_num = 300
        if hasattr(self.handlers[0], 'refresh_secs'):
            back_num = self.handlers[0].refresh_secs // 30
        self.pair_tfscores = await calc_symboltf_scales(self.exchange, pairlist, back_num)

        self._whitelist = pairlist
        BotGlobal.pairs = set(pairlist)

    @property
    def avaiable_symbols(self) -> Set[str]:
        if self._ava_symbols and btime.utctime() - self._ava_at < 300:
            return self._ava_symbols
        ava_markets = self.exchange.get_cur_markets()
        all_symbols = list(ava_markets.keys())
        self._ava_symbols = set(self.verify_blacklist(all_symbols))
        self._ava_at = btime.utctime()
        return self._ava_symbols

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
