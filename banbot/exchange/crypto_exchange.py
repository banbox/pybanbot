#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : crypto_exchange.py
# Author: anyongjin
# Date  : 2023/3/29
import asyncio

import ccxt
import ccxt.async_support as ccxt_async
import ccxt.pro as ccxtpro
import os
import time
from banbot.exchange.exchange_utils import *
from banbot.util.common import logger
from banbot.bar_driven.tainds import *
from banbot.config.consts import *
from banbot.util import btime
from banbot.util.misc import *
import numpy as np
from typing import *


def loop_forever(func):

    async def wrap(*args, **kwargs):
        while True:
            try:
                await run_async(func, *args, **kwargs)
            except ccxt.errors.NetworkError as e:
                if str(e) == '1006':
                    logger.warning(f'[{args}] watch balance get 1006, retry...')
                    continue
                raise e
    return wrap


def _create_exchange(module, cfg: dict):
    exg_cfg = cfg['exchange']
    exg_class = getattr(module, exg_cfg['name'])
    run_env = cfg["env"]
    credit = exg_cfg[f'credit_{run_env}']
    has_proxy = bool(exg_cfg.get('proxies'))
    exchange = exg_class(dict(
        apiKey=credit['api_key'],
        secret=credit['api_secret'],
        trust_env=has_proxy
    ))
    if run_env == 'test':
        exchange.set_sandbox_mode(True)
        logger.warning('running in TEST mode!!!')
    if has_proxy:
        exchange.proxies = exg_cfg['proxies']
        exchange.aiohttp_proxy = exg_cfg['proxies']['http']
    return exchange


def _init_exchange(cfg: dict, with_ws=False) -> Tuple[ccxt.Exchange, ccxt_async.Exchange, Optional[ccxtpro.Exchange]]:
    exg_cfg = cfg['exchange']
    has_proxy = bool(exg_cfg.get('proxies'))
    if has_proxy:
        os.environ['HTTP_PROXY'] = exg_cfg['proxies']['http']
        os.environ['HTTPS_PROXY'] = exg_cfg['proxies']['https']
        os.environ['WS_PROXY'] = exg_cfg['proxies']['http']
        os.environ['WSS_PROXY'] = exg_cfg['proxies']['http']
        logger.warning(f"[PROXY] {exg_cfg['proxies']}")
    exchange = _create_exchange(ccxt, cfg)
    exchange_async = _create_exchange(ccxt_async, cfg)
    if not with_ws:
        return exchange, exchange_async, None
    run_env = cfg["env"]
    credit = exg_cfg[f'credit_{run_env}']
    exg_class = getattr(ccxtpro, exg_cfg['name'])
    exchange_ws = exg_class(dict(
        newUpdates=True,
        apiKey=credit['api_key'],
        secret=credit['api_secret'],
        aiohttp_trust_env=has_proxy
    ))
    if run_env == 'test':
        exchange_ws.set_sandbox_mode(True)
    if has_proxy:
        exchange_ws.aiohttp_proxy = exg_cfg['proxies']['http']
    return exchange, exchange_async, exchange_ws


def _copy_markets(exg_src, exg_dst):
    exg_dst.reloading_markets = True
    exg_dst.markets_by_id = exg_src.markets_by_id
    exg_dst.markets = exg_src.markets
    exg_dst.symbols = exg_src.symbols
    exg_dst.ids = exg_src.ids
    if exg_src.currencies:
        exg_dst.currencies = exg_src.currencies
    if exg_src.baseCurrencies:
        exg_dst.baseCurrencies = exg_src.baseCurrencies
        exg_dst.quoteCurrencies = exg_src.quoteCurrencies
    if exg_src.currencies_by_id:
        exg_dst.currencies_by_id = exg_src.currencies_by_id
    exg_dst.codes = exg_src.codes
    exg_dst.reloading_markets = False


class CryptoExchange:
    def __init__(self, config: dict):
        self.api, self.api_async, self.api_ws = _init_exchange(config, True)
        self.name = self.api.name
        self.quote_prices: Dict[str, float] = dict()
        self.quote_base = config.get('quote_base', 'USDT')
        self.quote_symbols = {p.split('/')[1] for p, _ in config.get('pairlist')}

    async def load_markets(self):
        if btime.run_mode not in TRADING_MODES:
            return
        markets = await self.api_async.load_markets()
        self.api_ws.markets_by_id = self.api_async.markets_by_id
        _copy_markets(self.api_async, self.api_ws)
        _copy_markets(self.api_async, self.api)
        logger.info(f'{len(markets)} markets loaded for {self.api.name}')

    def calc_funding_fee(self, symbol: str, od_type: str, side: str, amount: float, price: float, is_taker=True):
        taker_maker = 'maker' if is_taker else 'taker'
        return self.api_async.calculate_fee(symbol, od_type, side, amount, price, taker_maker)

    async def fetch_ohlcv(self, symbol, timeframe='1m', since=None, limit=None, params=None):
        if params is None:
            params = {}
        return await self.api_async.fetch_ohlcv(symbol, timeframe, since, limit, params)

    async def watch_trades(self, symbol, since=None, limit=None, params={}):
        return await self.api_ws.watch_trades(symbol, since, limit, params)

    async def fetch_balance(self, params={}):
        return await self.api_async.fetch_balance(params)

    async def fetch_order_book(self, symbol, limit=None, params={}):
        return await self.api_async.fetch_order_book(symbol, limit, params)

    async def watch_balance(self, params={}):
        '''
        没有新数据时此方法每1分钟发出异常：ccxt.base.errors.NetworkError: 1006，需要在try catch中重试
        :param params:
        :return:
        '''
        return await self.api_ws.watch_balance(params)

    async def watch_orders(self, symbol=None, since=None, limit=None, params={}):
        '''
        监听订单更新，来自和watch_my_trades相同的数据触发。建议使用watch_my_trades，更细粒度
        没有新数据时此方法每1分钟发出异常：ccxt.base.errors.NetworkError: 1006，需要在try catch中重试
        :param symbol:
        :param since:
        :param limit:
        :param params:
        :return:
        '''
        return await self.api_ws.watch_orders(symbol, since, limit, params)

    async def watch_my_trades(self, symbol=None, since=None, limit=None, params={}):
        '''
        监听订单的部分成交变化。
        没有新数据时此方法每1分钟发出异常：ccxt.base.errors.NetworkError: 1006，需要在try catch中重试
        :param symbol:
        :param since:
        :param limit:
        :param params:
        :return:
        '''
        return await self.api_ws.watch_my_trades(symbol, since, limit, params)

    async def fetch_ohlcv_plus(self, pair: str, timeframe: str, since=None, limit=None,
                               force_sub=False, min_last_ratio=0.799):
        '''
        某些时间维度不能直接从交易所得到，需要从更细粒度计算。如3s的要从1s计算。2m的要从1m的计算
        :param pair:
        :param timeframe:
        :param since:
        :param limit:
        :param force_sub: 是否强制使用更细粒度的时间帧，即使当前时间帧支持
        :param min_last_ratio: 最后一个蜡烛的最低完成度
        :return:
        '''
        if (not force_sub or timeframe == '1s') and timeframe in self.api.timeframes:
            return await self.api_async.fetch_ohlcv(pair, timeframe, since=since, limit=limit)
        sub_tf, sub_tf_secs = max_sub_timeframe(self.api.timeframes.keys(), timeframe, force_sub)
        cur_tf_secs = timeframe_to_seconds(timeframe)
        if not limit:
            sub_arr = await self.api_async.fetch_ohlcv(pair, sub_tf, since=since)
            ohlc_arr = build_ohlcvc(sub_arr, cur_tf_secs)
            if sub_arr[-1][0] / 1000 / cur_tf_secs % 1 < min_last_ratio:
                ohlc_arr = ohlc_arr[:-1]
            return ohlc_arr
        fetch_num = limit * round(cur_tf_secs / sub_tf_secs)
        count = 0
        if not since:
            cur_time = int(math.floor(time.time() / cur_tf_secs) * cur_tf_secs)
            since = (cur_time - fetch_num * sub_tf_secs) * 1000
        result = []
        while count < fetch_num:
            sub_arr = await self.api_async.fetch_ohlcv(pair, sub_tf, since=since, limit=MAX_FETCH_NUM)
            if not sub_arr:
                break
            result.extend(sub_arr)
            count += len(sub_arr)
            since = sub_arr[-1][0] + 1
        ohlc_arr = build_ohlcvc(result[-fetch_num:], cur_tf_secs)
        if result[-1][0] / 1000 / cur_tf_secs % 1 < min_last_ratio:
            ohlc_arr = ohlc_arr[:-1]
        return ohlc_arr

    async def update_quote_price(self):
        if btime.run_mode not in TRADING_MODES:
            return
        for symbol in self.quote_symbols:
            if symbol.find('USD') >= 0:
                self.quote_prices[symbol] = 1
            else:
                od_books = await self.api_async.fetch_order_book(f'{symbol}/USDT', limit=5)
                self.quote_prices[symbol] = od_books['bids'][0][0] + od_books['asks'][0][0]

    async def create_limit_order(self, symbol, side, amount, price, params={}):
        if btime.run_mode != btime.RunMode.LIVE:
            raise RuntimeError(f'create_order is unavaiable in {btime.run_mode}')
        return await self.api_async.create_limit_order(symbol, side, amount, price, params)

    async def close(self):
        await self.api_async.close()
        await self.api_ws.close()

    def __str__(self):
        return self.name

