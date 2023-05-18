#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : crypto_exchange.py
# Author: anyongjin
# Date  : 2023/3/29
import asyncio
import random
import time

import six

import ccxt
from ccxt import TICK_SIZE
import ccxt.async_support as ccxt_async
import ccxt.pro as ccxtpro
import os

import orjson

from banbot.config.appconfig import AppConfig
from banbot.util.misc import *
from banbot.data.tools import *
from banbot.storage.orders import Order
from typing import *


_market_keys = ['markets_by_id', 'markets', 'symbols', 'ids', 'currencies', 'baseCurrencies',
                'quoteCurrencies', 'currencies_by_id', 'codes']
exg_map: Dict[str, 'CryptoExchange'] = dict()


def loop_forever(func):

    async def wrap(*args, **kwargs):
        fname = func.__qualname__
        while True:
            try:
                await run_async(func, *args, **kwargs)
                continue
            except ccxt.errors.NetworkError as e:
                if str(e) == '1006':
                    logger.warning('[%s] watch balance get 1006, retry...', fname)
                    continue
                logger.exception(f'{fname} network error')
                await asyncio.sleep(3)
            except Exception:
                logger.exception(f'{fname} loop exception')
                await asyncio.sleep(1)
    return wrap


def _create_exchange(module, cfg: dict):
    exg_cfg = AppConfig.get_exchange(cfg)
    exg_class = getattr(module, exg_cfg['name'])
    run_env = cfg["env"]
    credit = exg_cfg[f'credit_{run_env}']
    has_proxy = bool(exg_cfg.get('proxies'))
    exg_args = dict(
        apiKey=credit['api_key'],
        secret=credit['api_secret'],
        trust_env=has_proxy,
    )
    if exg_cfg.get('options'):
        exg_args['options'] = exg_cfg.get('options')
    exchange = exg_class(exg_args)
    if run_env == 'test':
        exchange.set_sandbox_mode(True)
        logger.warning('running in TEST mode!!!')
    if has_proxy:
        exchange.proxies = exg_cfg['proxies']
        exchange.aiohttp_proxy = exg_cfg['proxies']['http']
    return exchange


def _init_exchange(cfg: dict, with_ws=False) -> Tuple[ccxt.Exchange, ccxt_async.Exchange, Optional[ccxtpro.Exchange]]:
    exg_cfg = AppConfig.get_exchange(cfg)
    has_proxy = bool(exg_cfg.get('proxies'))
    if has_proxy:
        os.environ['HTTP_PROXY'] = exg_cfg['proxies']['http']
        os.environ['HTTPS_PROXY'] = exg_cfg['proxies']['https']
        os.environ['WS_PROXY'] = exg_cfg['proxies']['http']
        os.environ['WSS_PROXY'] = exg_cfg['proxies']['http']
        logger.warning("[PROXY] %s", exg_cfg['proxies'])
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


def _restore_markets(exg, cache_dir: str) -> float:
    cache_path = os.path.join(cache_dir, f'{exg.name.lower()}.json')
    if not os.path.isfile(cache_path):
        return 0
    with open(cache_path, 'rb') as fdata:
        cache = orjson.loads(fdata.read())
    exg.reloading_markets = True
    for key in _market_keys:
        cache_val = cache.get(key)
        if cache_val is None:
            continue
        setattr(exg, key, cache_val)
    exg.reloading_markets = False
    return cache.get('timestamp', 0)


def _save_markets(exg, cache_dir: str):
    cache_path = os.path.join(cache_dir, f'{exg.name.lower()}.json')
    result = dict()
    for key in _market_keys:
        item_val = getattr(exg, key, None)
        if item_val is None:
            continue
        result[key] = item_val
    result['timestamp'] = time.time()
    with open(cache_path, 'wb') as fout:
        fout.write(orjson.dumps(result, option=orjson.OPT_INDENT_2))


def net_retry(func):
    import random
    retry_num: int = 2

    async def wrapper(self, *args, **kwargs):
        if self.conti_error > retry_num and random.random() < 0.7:
            # 当不可用时，30%的概率重试，70%概率直接错误
            raise ccxt.ExchangeNotAvailable(self.name)
        while True:
            try:
                ret_val = await func(self, *args, **kwargs)
                self.conti_error = 0
                return ret_val
            except (ccxt.ExchangeNotAvailable, ccxt.RequestTimeout):
                self.conti_error += 1
                if self.conti_error > retry_num:
                    raise
                await asyncio.sleep(0.03)
    return wrapper


class CryptoExchange:

    _ft_has: Dict = {
        "stoploss_on_exchange": False,
        "order_time_in_force": ["GTC"],
        "time_in_force_parameter": "timeInForce",
        "ohlcv_params": {},
        "ohlcv_candle_limit": 500,
        "ohlcv_has_history": True,  # Some exchanges (Kraken) don't provide history via ohlcv
        "ohlcv_partial_candle": True,
        "ohlcv_require_since": False,
        # Check https://github.com/ccxt/ccxt/issues/10767 for removal of ohlcv_volume_currency
        "ohlcv_volume_currency": "base",  # "base" or "quote"
        "tickers_have_quoteVolume": True,
        "tickers_have_bid_ask": True,  # bid / ask empty for fetch_tickers
        "tickers_have_price": True,
        "trades_pagination": "time",  # Possible are "time" or "id"
        "trades_pagination_arg": "since",
        "l2_limit_range": None,
        "l2_limit_range_required": True,  # Allow Empty L2 limit (kucoin)
        "mark_ohlcv_price": "mark",
        "mark_ohlcv_timeframe": "8h",
        "ccxt_futures_name": "swap",
        "fee_cost_in_contracts": False,  # Fee cost needs contract conversion
        "needs_trading_fees": False,  # use fetch_trading_fees to cache fees
        "order_props_in_contracts": ['amount', 'cost', 'filled', 'remaining'],
    }
    _ft_has_futures: Dict = {}

    def __init__(self, config: dict):
        self.config = config
        self.api, self.api_async, self.api_ws = _init_exchange(config, True)
        self.name = self.api.name.lower()
        self.bot_name = config.get('name', 'noname')
        self.trade_mode = config.get('trade_mode')
        self.quote_prices: Dict[str, float] = dict()
        self.quote_base = config.get('quote_base', 'USDT')
        self.quote_symbols: Set[str] = set(config.get('stake_currency') or [])
        self.markets: Dict = {}
        # 记录每个交易对最近一次交易的费用类型，费率
        self.pair_fees: Dict[str, Tuple[str, float]] = dict()
        self.markets_at = time.time() - 7200
        self.market_dir = os.path.join(config['data_dir'], 'exg_markets')
        self.pair_fee_limits = AppConfig.get_exchange(config).get('pair_fee_limits')
        self.conti_error = 0
        if not os.path.isdir(self.market_dir):
            os.mkdir(self.market_dir)

    async def init(self, pairs: List[str]):
        await self.update_quote_price()
        await self.cancel_open_orders(pairs)

    @net_retry
    async def load_markets(self):
        if time.time() - self.markets_at < 1800:
            logger.warning('load_markets too freq, skip')
            return
        self.markets_at = time.time()
        restore_ts, markets = 0, []
        if btime.run_mode not in LIVE_MODES:
            restore_ts = _restore_markets(self.api_async, self.market_dir)
            markets = self.api_async.markets or []
        if time.time() - restore_ts > 604800:
            # 非实时模式，缓存的交易对7天有效
            if restore_ts:
                logger.warning('exchange markets expired, renew...')
            markets = await self.api_async.load_markets(True)
            _save_markets(self.api_async, self.market_dir)
        self.api_ws.markets_by_id = self.api_async.markets_by_id
        _copy_markets(self.api_async, self.api_ws)
        _copy_markets(self.api_async, self.api)
        logger.info('%d markets loaded for %s', len(markets), self.api.name)
        if not self.markets:
            # 首次加载，输出统计的交易对信息
            from banbot.optmize.reports import text_markets
            print(text_markets(self.api_async.markets, 30))
        self.markets = self.api_async.markets

    def calc_funding_fee(self, od: Order):
        '''
        {'type': 'taker', 'currency': 'USDT', 'rate': 0.001, 'cost': 0.029733297910755734}
        :param od:
        :return:
        '''
        taker_maker = 'maker' if od.order_type == 'limit' else 'taker'
        cache = self.pair_fees.get(f'{od.symbol}_{taker_maker}')
        if cache:
            return dict(rate=cache[1], currency=od.symbol.split('/')[0])
        if self.pair_fee_limits and btime.run_mode != RunMode.PROD:
            fee_rate = self.pair_fee_limits.get(od.symbol)
            if fee_rate is not None:
                return dict(rate=fee_rate, currency=od.symbol.split('/')[0])
        return self.api_async.calculate_fee(od.symbol, od.order_type, od.side, od.amount, od.price, taker_maker)

    def price_to_precision(self, symbol: str, price: float):
        return self.api_async.price_to_precision(symbol, price)

    def market_tradable(self, market: Dict[str, Any]) -> bool:
        return (
            market.get('quote') is not None
            and market.get('base') is not None
            and (self.precisionMode != TICK_SIZE or market.get('precision', {}).get('price', 0) > 1e-11)
            and bool(market.get(self.trade_mode))
        )

    def market_is_future(self, market: Dict[str, Any]) -> bool:
        return (
            market.get(self._ft_has["ccxt_futures_name"], False) is True and
            market.get('linear', False) is True
        )

    def get_markets(self, quote_currs=None, base_currs=None, trade_modes: Union[str, Set[str], List[str]] = None,
                    tradable_only: bool = True, active_only: bool = True) -> Dict[str, Any]:
        """
        Return exchange ccxt markets, filtered out by base currency and quote currency
        if this was requested in parameters.
        """
        markets = self.markets
        if not markets:
            raise RuntimeError("Markets were not loaded.")

        spot_only, margin_only, futures_only = False, False, False
        if trade_modes:
            if isinstance(trade_modes, six.string_types):
                trade_modes = {trade_modes}
            else:
                trade_modes = set(trade_modes)
            spot_only = 'spot' in trade_modes
            margin_only = 'margin' in trade_modes
            futures_only = 'future' in trade_modes

        def ia_valid(v: dict):
            if base_currs and v['base'] not in base_currs:
                return False
            if quote_currs and v['quote'] not in quote_currs:
                return False
            if tradable_only and not self.market_tradable(v):
                return False
            if active_only and not v.get('active', True):
                return False
            if spot_only and not v.get('spot'):
                return False
            if margin_only and not v.get('margin'):
                return False
            if futures_only and not self.market_is_future(v):
                return False
            return True

        return {k: v for k, v in markets.items() if ia_valid(v)}

    @property
    def precisionMode(self) -> int:
        """exchange ccxt precisionMode"""
        return self.api_async.precisionMode

    def price_get_one_pip(self, pair: str) -> float:
        """
        Get's the "1 pip" value for this pair.
        Used in PriceFilter to calculate the 1pip movements.
        """
        precision = self.markets[pair]['precision']['price']
        if self.precisionMode == TICK_SIZE:
            return precision
        else:
            return 1 / pow(10, precision)

    def get_pair_quote_currency(self, pair: str) -> str:
        """ Return a pair's quote currency (base/quote:settlement) """
        return self.markets.get(pair, {}).get('quote', '')

    def ohlcv_candle_limit(self, timeframe: str):
        return int(self._ft_has.get('ohlcv_candle_limit_per_timeframe', {})
                   .get(timeframe, self._ft_has.get('ohlcv_candle_limit')))

    def get_option(self, param: str, default: Optional[Any] = None) -> Any:
        """
        Get parameter value from _ft_has
        """
        return self._ft_has.get(param, default)

    def has_api(self, endpoint: str) -> bool:
        '''
        检查交易所是否支持指定的api
        :param endpoint:
        :return:
        '''
        return endpoint in self.api_async.has and self.api_async.has[endpoint]

    @net_retry
    async def fetch_ticker(self, symbol, params={}):
        return await self.api_async.fetch_ticker(symbol, params)

    @net_retry
    async def fetch_tickers(self, symbols=None, params={}):
        return await self.api_async.fetch_tickers(symbols, params)

    @net_retry
    async def fetch_ohlcv(self, symbol, timeframe='1m', since=None, limit=None, params=None):
        if params is None:
            params = {}
        return await self.api_async.fetch_ohlcv(symbol, timeframe, since, limit, params)

    async def watch_trades(self, symbol, since=None, limit=None, params={}):
        return await self.api_ws.watch_trades(symbol, since, limit, params)

    @net_retry
    async def fetch_balance(self, params={}):
        return await self.api_async.fetch_balance(params)

    @net_retry
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
                               force_sub=False):
        '''
        某些时间维度不能直接从交易所得到，需要从更细粒度计算。如3s的要从1s计算。2m的要从1m的计算
        :param pair:
        :param timeframe:
        :param since:
        :param limit:
        :param force_sub: 是否强制使用更细粒度的时间帧，即使当前时间帧支持
        :return:
        '''
        if (not force_sub or timeframe == '1s') and timeframe in self.api.timeframes:
            return await self.api_async.fetch_ohlcv(pair, timeframe, since=since, limit=limit)
        sub_tf, sub_tf_secs = max_sub_timeframe(self.api.timeframes.keys(), timeframe, force_sub)
        cur_tf_secs = tf_to_secs(timeframe)
        if not limit:
            sub_arr = await self.api_async.fetch_ohlcv(pair, sub_tf, since=since)
            ohlc_arr, last_finish = build_ohlcvc(sub_arr, cur_tf_secs)
            if not last_finish:
                ohlc_arr = ohlc_arr[:-1]
            return ohlc_arr
        fetch_num = limit * round(cur_tf_secs / sub_tf_secs)
        count = 0
        if not since:
            cur_time = time.time() // cur_tf_secs * cur_tf_secs
            since = (cur_time - fetch_num * sub_tf_secs) * 1000
        result = []
        while count < fetch_num:
            sub_arr = await self.api_async.fetch_ohlcv(pair, sub_tf, since=since, limit=MAX_FETCH_NUM)
            if not sub_arr:
                break
            result.extend(sub_arr)
            count += len(sub_arr)
            since = sub_arr[-1][0] + 1
        ohlc_arr, last_finish = build_ohlcvc(result[-fetch_num:], cur_tf_secs)
        if not last_finish:
            ohlc_arr = ohlc_arr[:-1]
        return ohlc_arr

    async def update_quote_price(self):
        if btime.run_mode not in LIVE_MODES:
            return
        for symbol in self.quote_symbols:
            if symbol.find('USD') >= 0:
                self.quote_prices[symbol] = 1
            else:
                od_books = await self.api_async.fetch_order_book(f'{symbol}/USDT', limit=5)
                self.quote_prices[symbol] = od_books['bids'][0][0] + od_books['asks'][0][0]

    async def edit_limit_order(self, id, symbol, side, amount, price=None, params={}):
        return await self.api_async.edit_limit_order(id, symbol, side, amount, price, params)

    async def cancel_order(self, id: str, symbol: str, params={}):
        return await self.api_async.cancel_order(id, symbol, params)

    async def create_limit_order(self, symbol, side, amount, price, params={}):
        if btime.run_mode != btime.RunMode.PROD:
            raise RuntimeError(f'create_order is unavaiable in {btime.run_mode}')
        if self.name == 'binance':
            params['clientOrderId'] = f'{self.bot_name}_{random.randint(0, 999999)}'
        return await self.api_async.create_limit_order(symbol, side, amount, price, params)

    async def cancel_open_orders(self, symbols: List[str]):
        # 查询数据库的订单，删除未创建成功的入场订单
        from banbot.storage import InOutOrder, db, InOutStatus
        op_ods = InOutOrder.open_orders()
        open_pairs = set()
        if op_ods:
            sess = db.session
            for od in op_ods:
                if od.status == InOutStatus.Init and (not od.enter or not od.enter.order_id):
                    sess.delete(od)
                    if od.enter:
                        sess.delete(od.enter)
                if od.status in {InOutStatus.Init, InOutStatus.PartExit, InOutStatus.PartEnter}:
                    open_pairs.add(od.symbol)
            sess.commit()
        success_cnt = 0
        for symbol in open_pairs:
            # 这里不要使用fetch_open_orders不带参数一次性获取，有1200s频率限制
            orders = await self.api_async.fetch_open_orders(symbol)
            for od in orders:
                if self.name == 'binance' and not od['clientOrderId'].startswith(self.bot_name):
                    continue
                try:
                    res = await self.api_async.cancel_order(od['id'], symbol)
                    if res['status'] != 'canceled':
                        logger.error('cancel order fail: %s', res)
                        continue
                    success_cnt += 1
                except ccxt.OrderNotFound:
                    continue
        if success_cnt:
            logger.warning('canceled %d unfill orders', success_cnt)

    async def close(self):
        await self.api_async.close()
        await self.api_ws.close()

    def __str__(self):
        return self.name


def get_exchange(name: Optional[str] = None) -> CryptoExchange:
    '''
    获取交易所实例，全局缓存
    '''
    if name is None:
        config = AppConfig.get()
        name = config['exchange']['name']
        if name not in exg_map:
            exg_map[name] = CryptoExchange(config)
    elif name not in exg_map:
        config = AppConfig.get()
        bak_exg_name = config['exchange']['name']
        config['exchange']['name'] = name
        exg_map[name] = CryptoExchange(config)
        config['exchange']['name'] = bak_exg_name
    return exg_map[name]
