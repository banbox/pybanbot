#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : precision_filter.py
# Author: anyongjin
# Date  : 2023/4/18
from banbot.compute.sta_inds import ccol
from banbot.symbols.pairlist.base import *


class PriceFilter(PairList):
    '''
    价格过滤器。
    precision: 0.001，按价格精度过滤交易对，默认要求价格变动最小单位是0.1%
    min_price: 最低价格
    max_price: 最高价格
    max_unit_value: 最大允许的单位价格变动对应的价值(针对定价货币，一般是USDT)。
    '''
    need_tickers = True

    def __init__(self, manager, exchange: CryptoExchange,
                 config: Config, handler_cfg: Dict[str, Any]):
        super(PriceFilter, self).__init__(manager, exchange, config, handler_cfg)
        self.precision = handler_cfg.get('precision', 0.001)
        self.min_price = handler_cfg.get('min_price', 0)
        self.max_price = handler_cfg.get('max_price', 0)
        self.max_fee = handler_cfg.get('max_fee', 0.002)
        self.max_unit_value = handler_cfg.get('max_unit_value', 0)
        self.stoploss = 0.95

    async def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        if not self.enable:
            return pairlist
        res_pairs = []
        if btime.run_mode in LIVE_MODES:
            if not tickers:
                logger.warning('no tickers, price_filter skipped')
                return pairlist
            for p in pairlist:
                tk = tickers.get(p)
                if not tk or not tk.get('last'):
                    continue
                if self._validate_price(p, tk.get('last')):
                    res_pairs.append(p)
        else:
            has = dict()

            def kline_cb(candles, exs: ExSymbol, timeframe, **kwargs):
                if not candles:
                    return
                if self._validate_price(exs.symbol, candles[-1][ccol]):
                    has[exs.symbol] = True
            await fast_bulk_ohlcv(self.exchange, pairlist, '1h', limit=1, callback=kline_cb)
            for pair in pairlist:
                if not (pair in has):
                    continue
                res_pairs.append(pair)
        return res_pairs

    def _validate_price(self, pair: str, price: float) -> bool:
        if self.precision > 0:
            compare = self.exchange.price_get_one_pip(pair)
            changeprec = compare / price
            if changeprec > self.precision:
                logger.info(f'remove {pair} because 1 unit is {changeprec:.2%}')
                return False

        if self.max_unit_value > 0:
            market = self.exchange.markets[pair]
            min_precision = market['precision']['amount']
            if min_precision is not None:
                if self.exchange.precisionMode != 4:
                    min_precision = pow(0.1, min_precision)
                unit_value = min_precision * price

                if unit_value > self.max_unit_value:
                    logger.info(f"Removed {pair} because min value change of {unit_value} > {self.max_unit_value}.")
                    return False

        if self.max_fee:
            fee_rate = self.exchange.calc_fee(pair, OrderType.Market.value)['rate']
            if fee_rate > self.max_fee:
                logger.info(f'Removed {pair} as fee rate {fee_rate} > {self.max_fee}')
                return False

        if self.min_price > 0 and price < self.min_price:
            logger.info(f'remove {pair} because last price < {self.min_price:.8f}')
            return False

        if 0 < self.max_price < price:
            logger.info('remove %s because last price > %f', pair, self.max_price)
            return False

        return True
