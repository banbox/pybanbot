#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : precision_filter.py
# Author: anyongjin
# Date  : 2023/4/18
from banbot.symbols.pairlist.base import *
from banbot.compute.tainds import ccol


class PriceFilter(PairList):
    '''
    价格过滤器。
    precision: 0.001，按价格精度过滤交易对，默认要求价格变动最小单位是0.1%
    unit_ratio：0 价格变动单位比率。作用跟precision类似
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
        self.max_unit_value = handler_cfg.get('max_unit_value', 0)
        self.enable = self.enable and (self.precision > 0 or self.min_price > 0
                                       or self.max_price > 0 or self.max_unit_value > 0)
        self.stoploss = 0.95

    def _validate_pair(self, pair: str, ticker: Optional[Ticker]) -> bool:
        if btime.run_mode in TRADING_MODES:
            if not ticker or not ticker.get('last'):
                return False
            price = ticker['last']
        else:
            since_ts, to_ts = get_back_ts(300, 1, in_ms=False)
            loop = asyncio.get_running_loop()
            candles = loop.run_until_complete(auto_fetch_ohlcv(self.exchange.name, pair, '5m', since_ts))
            price = candles[-1][ccol]

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

        if self.min_price > 0 and price < self.min_price:
            logger.info(f'remove {pair} because last price < {self.min_price:.8f}')
            return False

        if 0 < self.max_price < price:
            logger.info('remove %s because last price > %f', pair, self.max_price)
            return False

        return True