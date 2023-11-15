#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : crawler.py
# Author: anyongjin
# Date  : 2023/10/24
import asyncio
import os.path
import pickle
import time
import six
from ccxt import TICK_SIZE
from ccxt.async_support.base.exchange import Exchange
from ccxt.pro.binanceusdm import binanceusdm
from ccxt.async_support.base.ws.client import Client
from typing import *
from banbot.util.common import logger


global market_type, ccxt_mtype, running, task_num
market_type: Optional[str] = None
ccxt_mtype: Optional[str] = None
running = False
task_num = 0
batch_size = 1000

exg_mtype_map = dict(
    future='contract'
)


def set_market_type(market: str):
    global market_type, ccxt_mtype
    market_type = market
    ccxt_mtype = exg_mtype_map.get(market) or market


def market_tradable(self: Exchange, market: Dict[str, Any]) -> bool:
    return (
            market.get('quote') is not None
            and market.get('base') is not None
            and (self.precisionMode != TICK_SIZE or market.get('precision', {}).get('price', 0) > 1e-11)
            and bool(market.get(ccxt_mtype))
    )


def get_markets(self: Exchange, quote_currs=None, base_currs=None, trade_modes: Union[str, Set[str], List[str]] = None,
                tradable_only: bool = True, active_only: bool = True) -> Dict[str, Any]:
    """
    Return exchange ccxt markets, filtered out by base currency and quote currency
    if this was requested in parameters.
    """
    markets = self.markets
    if not markets:
        raise RuntimeError("Markets were not loaded.")

    spot_only, future_only = False, False
    if trade_modes:
        if isinstance(trade_modes, six.string_types):
            trade_modes = {trade_modes}
        else:
            trade_modes = set(trade_modes)
        spot_only = 'spot' in trade_modes
        future_only = 'future' in trade_modes

    def ia_valid(v: dict):
        if base_currs and v['base'] not in base_currs:
            return False
        if quote_currs and v['quote'] not in quote_currs:
            return False
        if tradable_only and not market_tradable(self, v):
            return False
        if active_only and not v.get('active', True):
            return False
        if spot_only and not v.get('spot'):
            return False
        if future_only and not v.get('swap'):
            # 期货模式下，只交易永续合约。
            # margin: 保证金  future: 短期期货  swap: 永续期货  option: 期权  contract: future/swap/option
            return False
        return True

    return {k: v for k, v in markets.items() if ia_valid(v)}


class my_binanceusdm(binanceusdm):

    def __init__(self, config: dict):
        super().__init__(config)
        stamp = int(time.time())  # 本次下载的时间戳
        self.data_dir = os.path.join(config.get('data_dir'), 'binanceusdm', str(stamp))
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.file_ods: Dict[str, IO] = dict()
        self.pair_odmsgs: Dict[str, List[dict]] = dict()
        self.file_tds: Dict[str, IO] = dict()
        self.pair_tdmsgs: Dict[str, List[dict]] = dict()

    async def watch_order_book_for_symbols(self, symbols: List[str], limit: Optional[int] = None, params={}):
        """订阅订单簿，初始化要保存的文件对象"""
        if not self.file_ods:
            for symbol in symbols:
                safe_pair = symbol.replace(':', '_').replace('/', '_')
                fname = f'od_{safe_pair}.pkl'
                out_path = os.path.join(self.data_dir, fname)
                self.file_ods[symbol] = open(out_path, 'wb')
                self.pair_odmsgs[symbol] = []
        await super().watch_order_book_for_symbols(symbols, limit, params)

    async def fetch_rest_order_book_safe(self, symbol, limit=None, params={}):
        """获取订单簿快照。缓存到文件"""
        orderbook = await super().fetch_rest_order_book_safe(symbol, limit, params)
        file = self.file_ods[symbol]
        file.write(pickle.dumps(orderbook))
        file.flush()
        logger.info(f'dump od shot to file: {symbol}')
        return orderbook

    def handle_order_book(self, client: Client, message):
        """处理ws收到的订单簿更新消息"""
        isTestnetSpot = client.url.find('testnet') > 0
        isSpotMainNet = client.url.find('/stream.binance.') > 0
        isSpot = isTestnetSpot or isSpotMainNet
        marketType = 'spot' if isSpot else 'contract'
        marketId = self.safe_string(message, 's')
        market = self.safe_market(marketId, None, None, marketType)
        symbol = market['symbol']
        books = self.pair_odmsgs[symbol]
        books.append(message)
        if len(books) > batch_size:
            saves = books[:batch_size]
            self.pair_odmsgs[symbol] = books[batch_size:]
            file = self.file_ods[symbol]
            file.write(pickle.dumps(saves))
            file.flush()
            logger.info(f'dump 1000 od msg to file: {symbol}')
        super().handle_order_book(client, message)

    async def watch_trades_for_symbols(self, symbols: List[str], since: Optional[int] = None,
                                       limit: Optional[int] = None, params={}):
        """监听币种的交易流，初始化要保存的文件对象"""
        if not self.file_tds:
            for symbol in symbols:
                safe_pair = symbol.replace(':', '_').replace('/', '_')
                fname = f'trade_{safe_pair}.pkl'
                out_path = os.path.join(self.data_dir, fname)
                self.file_tds[symbol] = open(out_path, 'wb')
                self.pair_tdmsgs[symbol] = []
                print(f'init {symbol}')
        await super().watch_trades_for_symbols(symbols, since, limit, params)

    def handle_trade(self, client: Client, message):
        # the trade streams push raw trade information in real-time
        # each trade has a unique buyer and seller
        isSpot = ((client.url.find('/stream') > -1) or (client.url.find('/testnet.binance') > -1))
        marketType = 'spot' if (isSpot) else 'contract'
        marketId = self.safe_string(message, 's')
        market = self.safe_market(marketId, None, None, marketType)
        symbol = market['symbol']
        trades = self.pair_tdmsgs[symbol]
        trades.append(message)
        if len(trades) > batch_size:
            saves = trades[:batch_size]
            self.pair_tdmsgs[symbol] = trades[batch_size:]
            file = self.file_tds[symbol]
            file.write(pickle.dumps(saves))
            file.flush()
            logger.info(f'dump 1000 trades to file: {symbol}')
        super().handle_trade(client, message)


async def run_watch_trades(exchange: Exchange, symbols: List[str]):
    global running, task_num
    task_num += 1
    try:
        while running:
            await exchange.watch_trades_for_symbols(symbols)
    except Exception:
        logger.exception('watch trades fail')
    task_num -= 1
    running = task_num > 0


async def run_watch_ods(exchange: Exchange, symbols: List[str]):
    global running, task_num
    task_num += 1
    try:
        while running:
            await exchange.watch_order_book_for_symbols(symbols, limit=1000)
    except Exception:
        logger.exception('watch orderbook fail')
    task_num -= 1
    running = task_num > 0


async def down_ws(args: Dict[str, Any]):
    global running
    from banbot.config import AppConfig
    config = AppConfig.get()
    data_dir = AppConfig.get_data_dir()
    exg_cfg = AppConfig.get_exchange(config)
    exg_name = exg_cfg['name']
    set_market_type(config['market_type'])
    has_proxy = bool(exg_cfg.get('proxies'))
    if has_proxy:
        os.environ['HTTP_PROXY'] = exg_cfg['proxies']['http']
        os.environ['HTTPS_PROXY'] = exg_cfg['proxies']['https']
        os.environ['WS_PROXY'] = exg_cfg['proxies']['http']
        os.environ['WSS_PROXY'] = exg_cfg['proxies']['http']
    exg_args = dict(newUpdates=True, aiohttp_trust_env=has_proxy, data_dir=data_dir)
    if exg_name == 'binance' and market_type == 'future':
        exchange = my_binanceusdm(exg_args)
    else:
        raise ValueError(f'unsupport exchange: {exg_name}')
    if has_proxy:
        exchange.aiohttp_proxy = exg_cfg['proxies']['http']
    quote_codes = ['USDT']
    await exchange.load_markets()
    markets = get_markets(exchange, quote_codes, trade_modes=[market_type])
    symbols = list(markets.keys())
    running = True
    logger.info(f'start down orderbook and trades for {exg_name}')
    asyncio.create_task(run_watch_trades(exchange, symbols))
    asyncio.create_task(run_watch_ods(exchange, symbols))
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        running = False
        await exchange.close()
