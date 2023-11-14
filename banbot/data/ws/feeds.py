#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : feeds.py
# Author: anyongjin
# Date  : 2023/11/4
import os
import pickle
import asyncio
from typing import *
from ccxt.pro.binanceusdm import binanceusdm
from ccxt.async_support.base.ws.client import Client

from banbot.config.consts import BotState
from banbot.storage import BotGlobal
from banbot.data.feeder import DataFeeder
from banbot.util import btime
from banbot.util.common import logger


def get_ws_path(exg_name: str) -> Tuple[str, int]:
    from banbot.config import AppConfig
    config = AppConfig.get()
    data_dir = os.path.join(config.get('data_dir'), exg_name)
    if not os.path.isdir(data_dir):
        raise ValueError(f'{exg_name} data dir not exits: {data_dir}')
    ws_stamp = config.get('ws_stamp')
    if ws_stamp:
        if len(str(ws_stamp)) == 13:
            ws_stamp = int(ws_stamp / 1000)
        dir_path = os.path.join(data_dir, str(ws_stamp))
        if not os.path.isdir(dir_path):
            raise ValueError(f'ws data not found: {dir_path}')
        return dir_path, int(ws_stamp)
    names = os.listdir(data_dir)
    names = [n for n in names if len(n) == 10 and n[0] == '1']
    if not names:
        raise ValueError(f'ws data is empty: {data_dir}')
    return os.path.join(data_dir, names[0]), int(names[0])


class FakeClient(Client):

    def __init__(self):
        super().__init__('', None, None, None, None)

    def resolve(self, result, message_hash):
        pass

    def reject(self, result, message_hash=None):
        pass

    def open(self, session, backoff_delay=0):
        pass

    def connect(self, session, backoff_delay=0):
        pass


class my_binanceusdm(binanceusdm):
    """回测使用的交易所对象，请勿调用watch_等方法"""
    loop_intv = 100
    '更新websocket数据流的间隔，单位：毫秒'

    def __init__(self, config: dict):
        super().__init__(config)
        self.data_dir, self.start_ts = get_ws_path('binanceusdm')
        self.last_ms = self.start_ts * 1000
        self.pairs = set()
        self._pairs_init = set()
        self._pair_map = dict()
        self.io_trades: Dict[str, IO] = dict()
        self.io_books: Dict[str, IO] = dict()
        self._q_trades: Dict[str, List[dict]] = dict()
        '读取的交易缓存，等待排序触发'
        self._q_books: Dict[str, List[dict]] = dict()
        '读取的订单簿缓存，等待排序触发'
        self.done_trades: Dict[str, List[dict]] = dict()
        '已就绪可读取的交易流'
        self.emit_q: Optional[asyncio.Queue] = None

    def sub_pairs(self, symbols: List[str]):
        for symbol in symbols:
            if symbol in self.pairs:
                continue
            safe_pair = symbol.replace(':', '_').replace('/', '_')
            book_path = os.path.join(self.data_dir, f'od_{safe_pair}.pkl')
            trade_path = os.path.join(self.data_dir, f'trade_{safe_pair}.pkl')
            if not os.path.isfile(trade_path):
                logger.error(f'sub pair ws skip, no data found: {symbol}')
                self.pairs.add(symbol)
                continue
            self.io_books[symbol] = open(book_path, 'rb')
            self.io_trades[symbol] = open(trade_path, 'rb')
            subscription = {
                'id': str(len(self.pairs)),
                'messageHash': 'messageHash',
                'name': 'depth',
                'symbol': symbol,
                'limit': 1000,
                'type': type,
                'params': {},
            }
            try:
                client = FakeClient()
                client.subscriptions['messageHash'] = subscription
                self.handle_order_book_subscription(client, '', subscription)
            except Exception as e:
                logger.error(f"init odbook fail: {symbol}: {e}")
                del self.io_books[symbol]
                del self.io_trades[symbol]
            self.pairs.add(symbol)

    async def fetch_rest_order_book_safe(self, symbol, limit=None, params={}):
        stream = self.io_books.get(symbol)
        if not stream:
            raise ValueError(f'odbook not init: {symbol}')
        self._pairs_init.add(symbol)
        snapshot = pickle.load(stream)
        if not isinstance(snapshot, dict):
            raise ValueError(f'invalid odbook snapshot: {symbol}')
        return snapshot

    async def run_loop(self):
        btime.cur_timestamp = 1
        while True:
            if not self._pairs_init:
                await asyncio.sleep(0.05)
                continue
            stop_ms = self.last_ms
            next_end = stop_ms + self.loop_intv
            # 预加载下一波的数据
            load_num = self._load_caches(next_end)
            if load_num < 0:
                # 全部交易对遍历完毕，退出
                break
            # 按顺序触发数据
            items = []
            for pair in self._pairs_init:
                trades = self._q_trades.get(pair)
                while trades:
                    trade = trades.pop(0)
                    if trade['timestamp'] > stop_ms:
                        trades.insert(0, trade)
                        break
                    items.append((trade['timestamp'], trade['symbol'], trade))
                books = self._q_books.get(pair)
                while books:
                    book = books.pop(0)
                    if book['E'] > stop_ms:
                        books.insert(0, book)
                        break
                    items.append((book['E'], book['s'], book))
            await self._emit_datas(items)
            self.last_ms = next_end
        logger.info(f'trades for {len(self._pairs_init)} pairs loop done')
        BotGlobal.state = BotState.STOPPED

    def watch_trades(self, symbol: str, since: Optional[int] = None, limit: Optional[int] = None, params={}):
        raise NotImplementedError('watch_trades is not support in backtest')

    def watch_trades_for_symbols(self, symbols: List[str], since: Optional[int] = None, limit: Optional[int] = None, params={}):
        raise NotImplementedError('watch_trades_for_symbols is not support in backtest')

    def watch_order_book(self, symbol: str, limit: Optional[int] = None, params={}):
        raise NotImplementedError('watch_order_book is not support in backtest')

    def watch_order_book_for_symbols(self, symbols: List[str], limit: Optional[int] = None, params={}):
        raise NotImplementedError('watch_order_book_for_symbols is not support in backtest')

    def _load_caches(self, until_ms: int):
        load_num = 0
        done_num = 0
        for pair in self._pairs_init:
            trades = self._q_trades.get(pair)
            if not trades or trades[-1]['timestamp'] < until_ms:
                if self.io_trades.get(pair):
                    try:
                        new_trades = pickle.load(self.io_trades[pair])
                        new_trades = self._parse_trades(new_trades)
                        if not trades:
                            self._q_trades[pair] = new_trades
                        else:
                            self._q_trades[pair].extend(new_trades)
                        load_num += len(new_trades)
                    except EOFError:
                        del self.io_trades[pair]
                        del self.io_books[pair]
                        last_ts = trades[-1]['timestamp'] if trades else None
                        logger.debug(f'read trade eof: %s, last: %s', pair, last_ts)
                else:
                    done_num += 1
            books = self._q_books.get(pair)
            if not books or books[-1]['E'] < until_ms:
                if self.io_books.get(pair):
                    try:
                        new_books = pickle.load(self.io_books[pair])
                        if isinstance(new_books, dict):
                            # 订单簿被重置
                            orderbook = self.safe_value(self.orderbooks, pair)
                            if orderbook is None:
                                raise EOFError
                            orderbook.reset(new_books)
                        else:
                            if not books:
                                self._q_books[pair] = new_books
                            else:
                                self._q_books[pair].extend(new_books)
                            load_num += len(new_books)
                    except EOFError:
                        del self.io_trades[pair]
                        del self.io_books[pair]
                        last_ts = books[-1]['E'] if books else None
                        logger.debug('read odbook eof: %s, last: %s', pair, last_ts)
                else:
                    done_num += 1
        if not load_num and done_num == len(self._pairs_init):
            return -1
        return load_num

    def _parse_trades(self, trades: List[dict]):
        """将交易所原始推送的数据转为cctx格式，要求输入的必须是统一标的的交易流"""
        if not trades:
            return []
        isSpot = False
        marketType = 'spot' if isSpot else 'contract'
        marketId = self.safe_string(trades[0], 's')
        market = self.safe_market(marketId, None, None, marketType)
        return [self.parse_trade(message, market) for message in trades]

    async def _emit_datas(self, items: List[Tuple[int, str, dict]]):
        items = sorted(items, key=lambda x: (x[0], x[1]))
        cache_key, cache_ms = None, None
        cache_trades = []
        for item in items:
            cur_ms, cur_pair, data = item
            if data.get('info') and data['info']['e'] == 'trade':
                if not cache_key or cur_pair == cache_key and cur_ms - cache_ms < 100:
                    # 相同pair的交易，缓冲到一起，仅限100ms内
                    cache_key = cur_pair
                    cache_trades.append(data)
                    if not cache_ms:
                        cache_ms = cur_ms
                    continue
                if cache_trades:
                    await self._emit_trades(cache_key, cache_trades)
                cache_key, cache_ms = cur_pair, cur_ms
                cache_trades = [data]
            elif data.get('e') == 'depthUpdate':
                if cache_trades:
                    await self._emit_trades(cache_key, cache_trades)
                    cache_key, cache_ms = None, None
                    cache_trades = []
                await self.handle_order_book_msg(None, data)
            else:
                raise ValueError(f'unknown data type: {data}')
        if cache_trades:
            await self._emit_trades(cache_key, cache_trades)

    async def _emit_trades(self, pair: str, trades: List[dict]):
        await self.emit_q.put(('trade', pair, trades))

    def _parse_pair(self, marketId: str):
        if marketId not in self._pair_map:
            isSpot = False
            marketType = 'spot' if isSpot else 'contract'
            market = self.safe_market(marketId, None, None, marketType)
            self._pair_map[marketId] = market['symbol']
        return self._pair_map[marketId]

    async def handle_order_book_msg(self, client: Client, message):
        btime.cur_timestamp = message['E'] / 1000
        marketId = self.safe_string(message, 's')
        pair = self._parse_pair(marketId)
        orderbook = self.safe_value(self.orderbooks, pair)
        if not orderbook:
            return
        self.handle_order_book_message(client, message, orderbook)
        return await self._emit_books(pair, orderbook)

    async def _emit_books(self, pair: str, book):
        await self.emit_q.put(('book', pair, book))


def get_local_wsexg(config: dict):
    exg_cfg = config['exchange']
    exg_name = exg_cfg['name']
    market = config['market_type']
    if exg_name == 'binance':
        if market == 'future':
            return exg_name, my_binanceusdm
    raise ValueError(f'unsupport local exchange: {exg_name}.{market}')


class LiveWSFeeder(DataFeeder):
    def __init__(self, pair: str, tf_warms: Dict[str, int], callback: Callable):
        super().__init__(pair, tf_warms, callback)

    async def on_new_data(self, trades: List[dict]):
        await self.callback(self.pair, trades)
