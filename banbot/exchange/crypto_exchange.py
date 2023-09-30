#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : crypto_exchange.py
# Author: anyongjin
# Date  : 2023/3/29
import random

import ccxt.async_support as ccxt_async
import ccxt.pro as ccxtpro
import orjson
import six
import os
from ccxt import TICK_SIZE
from asyncio import Lock
from cachetools import TTLCache

from banbot.config.appconfig import AppConfig
from banbot.util.misc import *
from banbot.util.common import logger
from banbot.config.consts import *
from banbot.exchange.exchange_utils import *
from banbot.main.addons import MarketPrice
from banbot.exchange.ccxt_exts import get_pro_overrides, get_asy_overrides
from banbot.types import *

_market_keys = ['markets_by_id', 'markets', 'symbols', 'ids', 'currencies', 'baseCurrencies',
                'quoteCurrencies', 'currencies_by_id', 'codes']
exg_map: Dict[str, 'CryptoExchange'] = dict()
exg_fut_map: Dict[str, str] = dict(
    binance='binanceusdm',
    kraken='krakenfutures',
    kucoin='kucoinfutures',
    poloniex='poloniexfutures'
)
exg_mtype_map = dict(
    future='contract'
)
pro_overrides = get_pro_overrides()
asy_overrides = get_asy_overrides()
api_overrides = dict()


def loop_forever(func):

    async def wrap(*args, **kwargs):
        fname = func.__qualname__
        while True:
            try:
                result = await run_async(func, *args, **kwargs)
                if result == 'exit':
                    break
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


def _get_credits(exg_cfg: dict, run_env: str) -> dict:
    credit = exg_cfg.get(f'credit_{run_env}')
    ret_data = dict()
    if credit:
        ret_data['apiKey'] = credit['api_key']
        ret_data['secret'] = credit['api_secret']
    return ret_data


def _create_exchange(module, override_map: dict, cfg: dict, exg_name: str = None, market_type: str = None):
    exg_cfg = AppConfig.get_exchange(cfg, exg_name)
    exg_name = exg_cfg['name']
    if not market_type:
        market_type = cfg['market_type']
    if market_type == 'future':
        exg_name = exg_fut_map.get(exg_name) or exg_name
    if exg_name in override_map:
        exg_class = override_map[exg_name]
    else:
        exg_class = getattr(module, exg_name)
    run_env = cfg["env"]
    has_proxy = bool(exg_cfg.get('proxies'))
    exg_args = dict(trust_env=has_proxy)
    if exg_cfg.get('options'):
        exg_args['options'] = exg_cfg.get('options')
    cred_args = _get_credits(exg_cfg, run_env)
    exchange = exg_class(dict(**exg_args, **cred_args))
    if run_env == 'test':
        exchange.set_sandbox_mode(True)
        logger.warning('running in TEST mode!!!')
    if has_proxy:
        exchange.proxies = exg_cfg['proxies']
        exchange.aiohttp_proxy = exg_cfg['proxies']['http']
    logger.info(f'Create Exg: {module.__name__}.{exchange.__class__.__name__} {run_env} '
                f'proxy:{exchange.proxies}  {exg_args}')
    return exchange


def _init_exchange(cfg: dict, with_ws=False, exg_name: str = None, market_type: str = None)\
        -> Tuple[ccxt_async.Exchange, Optional[ccxtpro.Exchange]]:
    exg_cfg = AppConfig.get_exchange(cfg, exg_name)
    exg_name = exg_cfg['name']
    has_proxy = bool(exg_cfg.get('proxies'))
    if has_proxy:
        os.environ['HTTP_PROXY'] = exg_cfg['proxies']['http']
        os.environ['HTTPS_PROXY'] = exg_cfg['proxies']['https']
        os.environ['WS_PROXY'] = exg_cfg['proxies']['http']
        os.environ['WSS_PROXY'] = exg_cfg['proxies']['http']
        logger.warning("[PROXY] %s", exg_cfg['proxies'])
    # exchange = _create_exchange(ccxt, api_overrides, cfg, exg_name, market_type)
    exchange_async = _create_exchange(ccxt_async, asy_overrides, cfg, exg_name, market_type)
    exchange_async.options['warnOnFetchOpenOrdersWithoutSymbol'] = False
    if not with_ws:
        return exchange_async, None
    run_env = cfg["env"]
    if market_type == 'future':
        exg_name = exg_fut_map.get(exg_name) or exg_name
    if exg_name in pro_overrides:
        exg_class = pro_overrides[exg_name]
    else:
        exg_class = getattr(ccxtpro, exg_name)
    exg_args = dict(newUpdates=True, aiohttp_trust_env=has_proxy)
    cred_args = _get_credits(exg_cfg, run_env)
    exchange_ws = exg_class(dict(**exg_args, **cred_args))
    if run_env == 'test':
        exchange_ws.set_sandbox_mode(True)
    if has_proxy:
        exchange_ws.aiohttp_proxy = exg_cfg['proxies']['http']
    return exchange_async, exchange_ws


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
    result['timestamp'] = btime.utctime()
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

    def __init__(self, config: dict, exg_name: str = None, market_type: str = None):
        self.config = config
        self.api_async, self.api_ws = _init_exchange(config, True, exg_name, market_type)
        self.exg_config = AppConfig.get_exchange(config, exg_name)
        self.name = self.exg_config['name']
        self.bot_name = config.get('name', 'noname')
        self.market_type = market_type or config.get('market_type')
        self.mtype = exg_mtype_map.get(self.market_type) or self.market_type
        self.quote_symbols: Set[str] = set(config.get('stake_currency') or [])
        self.markets: Dict = {}
        self._leverage = config.get('leverage')
        self.leverages: Dict[str, LeverageTiers] = dict()  # 记录每个币种的杠杆
        # 记录每个交易对最近一次交易的费用类型，费率
        self.pair_fees: Dict[str, Tuple[str, float]] = dict()
        self.markets_at = btime.utctime() - 7200
        self.market_dir = os.path.join(config['data_dir'], 'exg_markets')
        self.pair_fee_limits = self.exg_config.get('pair_fee_limits') or dict()
        # Cache for 10 minutes ...
        self._cache_lock = Lock()
        self._fetch_tickers_cache: TTLCache = TTLCache(maxsize=2, ttl=60 * 10)

        self.conti_error = 0
        if not os.path.isdir(self.market_dir):
            os.mkdir(self.market_dir)

    async def init(self, pairs: List[str]):
        await self._check_fee_limits()
        await self.cancel_invalid_orders()
        await self.cancel_open_orders(pairs)
        await self.update_symbol_leverages(pairs)

    @net_retry
    async def load_markets(self):
        if btime.utctime() - self.markets_at < 1800:
            logger.warning('load_markets too freq, skip')
            return
        load_new = False
        if btime.run_mode in btime.LIVE_MODES:
            load_new = True
        else:
            restore_ts = _restore_markets(self.api_async, self.market_dir)
            exp_secs = 604800
            if btime.utctime() - restore_ts > exp_secs:
                load_new = True
                logger.warning('exchange markets expired, renew...')
        if load_new:
            # 实时模式，或缓存的交易所信息过期
            markets = await self.api_async.load_markets(True)
            await self.api_ws.load_markets(True)
            _save_markets(self.api_async, self.market_dir)
        else:
            markets = self.api_async.markets or []
            self.api_ws.markets_by_id = self.api_async.markets_by_id
            _copy_markets(self.api_async, self.api_ws)
        logger.info('%d markets loaded for %s', len(markets), self.api_async.name)
        if not self.markets:
            # 首次加载，输出统计的交易对信息
            from banbot.exchange.exchange_utils import text_markets
            print(text_markets(self.api_async.markets, 30))
        self.markets = self.api_async.markets
        self.markets_at = btime.utctime()

    def get_market_by_id(self, symbol_id: str):
        '''
        传入BTCUSDT，不带/
        '''
        return self.api_async.market(symbol_id)

    def exchange_has(self, endpoint: str) -> bool:
        """
        Checks if exchange implements a specific API endpoint.
        Wrapper around ccxt 'has' attribute
        :param endpoint: Name of endpoint (e.g. 'fetchOHLCV', 'fetchTickers')
        :return: bool
        """
        return endpoint in self.api_async.has and self.api_async.has[endpoint]

    async def _check_fee_limits(self):
        exg_fees = await self.api_async.fetch_trading_fees()
        for symbol, fee in exg_fees.items():
            currency = symbol.split('/')[-1].split(':')[0]
            if currency not in self.quote_symbols:
                continue
            if 'maker' in fee:
                self.pair_fees[f'{symbol}_maker'] = currency, fee['maker']
            if 'taker' in fee:
                self.pair_fees[f'{symbol}_taker'] = currency, fee['taker']
        if not self.pair_fee_limits:
            return
        for symbol, limit in self.pair_fee_limits.items():
            fee_rate = self.pair_fees.get(f'{symbol}_taker') or self.pair_fees.get(f'{symbol}_maker')
            if not fee_rate:
                raise ValueError(f'no fee found for {symbol}, limit: {limit}')
            if fee_rate[1] > limit:
                raise ValueError(f'fee {fee_rate[1]} exceed limit {limit}, symbol: {symbol}')

    def calc_fee(self, symbol: str, order_type: str, side: str = 'buy', amount=None, price=None):
        '''
        计算交易费用。
        返回值目前只用到：rate + currency
        返回：{'type': 'taker', 'currency': 'USDT', 'rate': 0.001, 'cost': 0.029733297910755734}
        '''
        taker_maker = 'maker' if order_type == 'limit' else 'taker'
        cache = self.pair_fees.get(f'{symbol}_{taker_maker}')
        if cache:
            return dict(rate=cache[1], currency=cache[0])
        fee_limit = self.pair_fee_limits.get(symbol)
        if fee_limit is not None and btime.run_mode != RunMode.PROD:
            # 非生产模式，直接使用指定手续费限制作为手续费率
            return dict(rate=fee_limit, currency=symbol.split('/')[0].split(':')[0])
        calc_fee = self.api_async.calculate_fee(symbol, order_type, side, amount, price, taker_maker)
        if fee_limit is not None and calc_fee['rate'] > fee_limit:
            # 有手续费限制，且计算的手续费超过限制。
            raise ValueError(f'{self.name}/{symbol} fee rate exceed limit {fee_limit}')
        return calc_fee

    def market_tradable(self, market: Dict[str, Any]) -> bool:
        return (
            market.get('quote') is not None
            and market.get('base') is not None
            and (self.precisionMode != TICK_SIZE or market.get('precision', {}).get('price', 0) > 1e-11)
            and bool(market.get(self.mtype))
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
            return True

        return {k: v for k, v in markets.items() if ia_valid(v)}

    def get_cur_markets(self):
        return self.get_markets(quote_currs=self.quote_symbols, trade_modes=[self.market_type])

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

    def pres_cost(self, symbol, cost):
        return float(self.api_async.cost_to_precision(symbol, cost))

    def pres_price(self, symbol, price):
        return float(self.api_async.price_to_precision(symbol, price))

    def pres_amount(self, symbol, amount):
        return float(self.api_async.amount_to_precision(symbol, amount))

    def pres_fee(self, symbol, fee):
        return float(self.api_async.fee_to_precision(symbol, fee))

    def min_margin(self, quote_amount: float, coin_base: bool = False):
        if self.mtype != 'contract':
            return 0
        if self.name == 'binance':
            # https://www.binance.com/zh-CN/futures/trading-rules/perpetual/leverage-margin
            if coin_base:
                raise ValueError('not support coin base margin currently')
            if quote_amount < 50000:
                mg_rate, cut_amount = 0.004, 0
            elif quote_amount < 250000:
                mg_rate, cut_amount = 0.005, 50
            elif quote_amount < 3000000:
                mg_rate, cut_amount = 0.01, 1300
            elif quote_amount < 20000000:
                mg_rate, cut_amount = 0.025, 46300
            elif quote_amount < 40000000:
                mg_rate, cut_amount = 0.05, 546300
            elif quote_amount < 100000000:
                mg_rate, cut_amount = 0.1, 2546300
            elif quote_amount < 120000000:
                mg_rate, cut_amount = 0.125, 5046300
            elif quote_amount < 200000000:
                mg_rate, cut_amount = 0.15, 8046300
            elif quote_amount < 300000000:
                mg_rate, cut_amount = 0.25, 28046300
            elif quote_amount < 500000000:
                mg_rate, cut_amount = 0.5, 103046300
            else:
                raise ValueError(f'invalid quote amount: {quote_amount}')
            return quote_amount * mg_rate - cut_amount
        else:
            raise ValueError(f'unsupport exchange: {self.name}')

    @net_retry
    async def fetch_ticker(self, symbol, params={}):
        return await self.api_async.fetch_ticker(symbol, params)

    @net_retry
    async def fetch_ohlcv(self, symbol, timeframe='1m', since=None, limit=None, params=None):
        if params is None:
            params = {}
        # 币安期货返回的ohlcv包含未完成的bar，指定时间截取掉
        cut_end = self.name == 'binance' and self.mtype == 'contract'
        if cut_end:
            tf_msecs = tf_to_secs(timeframe) * 1000
            cur_time = btime.utcstamp()
            end_ts = cur_time // tf_msecs * tf_msecs - 1
            if since and since >= end_ts:
                # 避免开始时间>结束时间，接口返回错误
                return []
            params['endTime'] = end_ts
        return await self.api_async.fetch_ohlcv(symbol, timeframe, since, limit, params)

    async def watch_trades(self, symbol, since=None, limit=None, params={}):
        return await self.api_ws.watch_trades(symbol, since, limit, params)

    async def watch_ohlcv(self, symbol: str, timeframe: str, since=None, limit=None, params={}):
        return await self.api_ws.watch_ohlcv(symbol, timeframe, since, limit, params)

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

    async def watch_account_update(self, params={}):
        if not hasattr(self.api_ws, 'watch_account_config'):
            raise ValueError(f'watch_account_config not found in {self.api_ws.name}')
        return await self.api_ws.watch_account_config(params)

    async def watch_mark_prices(self, params={}):
        if not hasattr(self.api_ws, 'watch_mark_prices'):
            raise ValueError(f'watch_mark_prices not found in {self.api_ws.name}')
        result = await self.api_ws.watch_mark_prices(params)
        return [item for item in result if item['symbol'] in self.markets]

    async def fetch_orders(self, symbol: Optional[str] = None, since: Optional[int] = None,
                           limit: Optional[int] = None, params={}) -> List[dict]:
        return await self.api_async.fetch_orders(symbol, since, limit, params)

    async def fetch_account_positions(self, symbols=None, params={}):
        if not hasattr(self.api_async, 'fetch_account_positions'):
            raise ValueError(f'fetch_account_positions not support in {self.api_async.name}')
        return await self.api_async.fetch_account_positions(symbols, params)

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
        from banbot.data.tools import build_ohlcvc
        if (not force_sub or timeframe == '1s') and timeframe in self.api_async.timeframes:
            return await self.api_async.fetch_ohlcv(pair, timeframe, since=since, limit=limit)
        sub_tf, sub_tf_secs = max_sub_timeframe(self.api_async.timeframes.keys(), timeframe, force_sub)
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
            cur_time = btime.utctime() // cur_tf_secs * cur_tf_secs
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

    @net_retry
    async def fetch_tickers(self, symbols: Optional[List[str]] = None, cached: bool = False) -> Tickers:
        """
        :param cached: Allow cached result
        :return: fetch_tickers result
        """
        tickers: Tickers
        if not self.exchange_has('fetchTickers'):
            return {}
        if cached:
            async with self._cache_lock:
                tickers = self._fetch_tickers_cache.get('fetch_tickers')  # type: ignore
            if tickers:
                return tickers
        try:
            tickers = await self.api_async.fetch_tickers(symbols)
            async with self._cache_lock:
                self._fetch_tickers_cache['fetch_tickers'] = tickers
            return tickers
        except ccxt.NotSupported as e:
            raise OperateError(
                f'Exchange {self.api_async.name} does not support fetching tickers in batch. '
                f'Message: {e}') from e

    async def update_prices(self):
        '''
        更新所有币种的价格。
        此方法运行一段时间后会卡住，请使用watch_mark_prices
        '''
        from banbot.storage import BotGlobal
        if not BotGlobal.live_mode or (btime.time_ms() - BotGlobal.last_bar_ms) > 20000:
            # 只在收到一个bar的后续20s内允许更新，足够处理订单，否则太频繁容易被封ip
            return
        try:
            prices: Dict[str, Dict] = await self.api_async.fetch_last_prices()
        except ccxt.NetworkError as e:
            logger.error(f'watch_prices net error: {e}')
            return
        for key, item in prices.items():
            MarketPrice.set_new_price(key, item['price'])

    async def edit_limit_order(self, id, symbol, side, amount, price=None, params={}):
        return await self.api_async.edit_limit_order(id, symbol, side, amount, price, params)

    async def cancel_order(self, id: str, symbol: str, params={}):
        return await self.api_async.cancel_order(id, symbol, params)

    async def create_order(self, symbol, od_type, side, amount, price, params={}):
        if btime.run_mode != btime.RunMode.PROD:
            raise RuntimeError(f'create_order is unavaiable in {btime.run_mode}')
        if self.name == 'binance':
            params['clientOrderId'] = f'{self.bot_name}_{random.randint(0, 999999)}'
        logger.debug('create exg order, %s %s %s %s %s %s', symbol, od_type, side, amount, price, params)
        return await self.api_async.create_order(symbol, od_type, side, amount, price, params)

    async def set_leverage(self, leverage: int, symbol: str, params={}):
        item = self.leverages.get(symbol)
        if not item:
            await self.update_symbol_leverages([symbol])
            item = self.leverages.get(symbol)
        leverage = min(leverage, item.max_leverage)
        if item.leverage == leverage:
            return item
        if not hasattr(self.api_async, 'set_leverage'):
            raise ValueError(f'exchange {self.name}.{self.market_type} not support set_leverage')
        res = await self.api_async.set_leverage(leverage, symbol, params)
        item.leverage = int(res['leverage'])
        return item

    async def update_symbol_leverages(self, symbols=None, cur_val=-1):
        '''
        获取持仓风险。future市场可用。同时更新币种的杠杆倍数。
        '''
        tiers = await self.api_async.fetch_leverage_tiers(symbols)
        for pair, raw_list in tiers.items():
            self.leverages[pair] = LeverageTiers(raw_list)
        if cur_val < 0:
            risks = await self.api_async.fetch_positions_risk(symbols, {})
            for risk in risks:
                self.leverages[risk['symbol']].leverage = risk['leverage']
        else:
            for pair in tiers:
                self.leverages[pair].leverage = cur_val

    def get_leverage(self, symbol: str, with_def=True):
        item = self.leverages.get(symbol)
        if item and item.leverage:
            return item.leverage
        return self._leverage if with_def else 0

    async def _open_orders(self, all_pairs=True) -> List[dict]:
        if all_pairs:
            try:
                return await self.api_async.fetch_open_orders()
            except Exception as e:
                logger.error(f'fetch_open_orders for all fail: {e}')
        from banbot.storage import BotGlobal
        result = []
        for symbol in BotGlobal.pairs:
            orders = await self.api_async.fetch_open_orders(symbol)
            result.extend(orders)
        return result

    async def cancel_invalid_orders(self):
        '''
        有些策略会挂止损止盈单，且每个bar删除重新下。
        机器人可能有些旧的止损止盈单没有及时撤销，长时间运行后导致超出最大订单限制。
        故调用此方法检查所有打开的订单，如果没有和订单关联，则取消
        '''
        from banbot.storage import InOutOrder, db, InOutStatus, BotGlobal
        op_ods = InOutOrder.open_orders(pairs=list(BotGlobal.pairs))
        valid_odids = set()
        for od in op_ods:
            if od.enter.order_id:
                valid_odids.add(od.enter.order_id)
            if od.exit and od.exit.order_id:
                valid_odids.add(od.exit.order_id)
            sl_oid = od.get_info('stoploss_oid')
            tp_oid = od.get_info('takeprofit_oid')
            if sl_oid:
                valid_odids.add(sl_oid)
            if tp_oid:
                valid_odids.add(tp_oid)
        exg_orders = await self._open_orders()
        exp_types = {'stop', 'take_limit', 'stop_market', 'take_market'}
        cancel_num = 0
        for eod in exg_orders:
            if not eod['clientOrderId'].startswith(self.bot_name):
                # 只取消当前机器人下的订单
                continue
            if eod['id'] in valid_odids:
                continue
            if eod['type'] not in exp_types:
                # 只取消：止损单、止盈单
                continue
            try:
                await self.api_async.cancel_order(eod['id'], eod['symbol'])
                cancel_num += 1
            except ccxt.OrderNotFound:
                pass
            except Exception as e:
                logger.error(f'cancel invalid order fail: {e} {eod}')
        logger.warning(f'canceled {cancel_num} invalid orders')

    async def cancel_open_orders(self, symbols: List[str]):
        # 查询数据库的订单，删除未创建成功的入场订单
        from banbot.storage import InOutOrder, db, InOutStatus
        op_ods = InOutOrder.open_orders(pairs=symbols)
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

    async def reconnect(self):
        try:
            self.api_async.close()
        except Exception:
            pass
        try:
            self.api_ws.close()
        except Exception:
            pass
        self.api_async, self.api_ws = _init_exchange(config, True, exg_name, market_type)
        await self.load_markets()

    def __str__(self):
        return self.name


def get_exchange(name: Optional[str] = None, market: str = None) -> CryptoExchange:
    '''
    获取交易所实例，全局缓存
    '''
    config = AppConfig.get()
    if name is None:
        name = config['exchange']['name']
    if not market:
        market = config['market_type']
    cache_key = f'{name}.{market}'
    if cache_key not in exg_map:
        logger.warning(f'No Cache Exchange, Create For {cache_key}')
        exg_map[cache_key] = CryptoExchange(config, name, market)
    return exg_map[cache_key]
