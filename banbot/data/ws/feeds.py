#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : feeds.py
# Author: anyongjin
# Date  : 2023/11/4
import collections
import os
import pickle
import asyncio
from typing import *
from ccxt.pro.binanceusdm import binanceusdm
from ccxt.async_support.base.ws.client import Client
from asyncio import Future

from banbot.config.consts import BotState
from banbot.storage import BotGlobal
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
        self._q_books: Dict[str, List[dict]] = dict()
        self.trade_waits: Dict[str, Deque[Future]] = dict()
        self.book_watis: Dict[str, Deque[Future]] = dict()
        self.done_trades: Dict[str, List[dict]] = dict()
        '已就绪可读取的交易流'

    def _sub_pairs(self, symbols: List[str]):
        new_pairs = []
        for symbol in symbols:
            if symbol in self.pairs:
                continue
            safe_pair = symbol.replace(':', '_').replace('/', '_')
            book_name = f'od_{safe_pair}.pkl'
            trade_name = f'trade_{safe_pair}.pkl'
            self.io_books[symbol] = open(os.path.join(self.data_dir, book_name), 'rb')
            self.io_trades[symbol] = open(os.path.join(self.data_dir, trade_name), 'rb')
            new_pairs.append(symbol)
        if not new_pairs:
            return
        subscription = {
            'id': str(len(self.pairs)),
            'messageHash': 'messageHash',
            'name': 'depth',
            'symbols': new_pairs,
            'limit': 1000,
            'type': type,
            'params': {},
        }
        self.handle_order_book_subscription(FakeClient(), '', subscription)
        self.pairs.update(new_pairs)

    async def watch_order_book(self, symbol: str, limit: Optional[int] = None, params={}):
        self._sub_pairs([symbol])
        if symbol not in self.book_watis:
            self.book_watis[symbol] = collections.deque()
        fut = self.asyncio_loop.create_future()
        self.book_watis[symbol].append(fut)
        return await fut

    async def watch_order_book_for_symbols(self, symbols: List[str], limit: Optional[int] = None, params={}):
        self._sub_pairs(symbols)
        fut = self.asyncio_loop.create_future()
        for symbol in symbols:
            if symbol not in self.book_watis:
                self.book_watis[symbol] = collections.deque()
            self.book_watis[symbol].append(fut)
        return await fut

    async def fetch_rest_order_book_safe(self, symbol, limit=None, params={}):
        stream = self.io_books.get(symbol)
        if not stream:
            raise ValueError(f'odbook not init: {symbol}')
        self._pairs_init.add(symbol)
        return pickle.load(stream)

    async def watch_trades(self, symbol: str, since: Optional[int] = None, limit: Optional[int] = None, params={}):
        self._sub_pairs([symbol])
        cache = self.done_trades.get(symbol)
        if cache:
            # 有缓存内容，直接返回，无需等待
            del self.done_trades[symbol]
            return cache
        if symbol not in self.trade_waits:
            self.trade_waits[symbol] = collections.deque()
        fut = self.asyncio_loop.create_future()
        self.trade_waits[symbol].append(fut)
        return await fut

    async def watch_trades_for_symbols(self, symbols: List[str], since: Optional[int] = None,
                                       limit: Optional[int] = None, params={}):
        self._sub_pairs(symbols)
        for symbol in symbols:
            cache = self.done_trades.get(symbol)
            if cache:
                # 有缓存内容，直接返回，无需等待
                del self.done_trades[symbol]
                return cache
        fut = self.asyncio_loop.create_future()
        for symbol in symbols:
            if symbol not in self.trade_waits:
                self.trade_waits[symbol] = collections.deque()
            self.trade_waits[symbol].append(fut)
        return await fut

    async def run_loop(self):
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
                    items.append((trade['timestamp'], trade))
                books = self._q_books.get(pair)
                while books:
                    book = books.pop(0)
                    if book['E'] > stop_ms:
                        books.insert(0, book)
                        break
                    items.append((book['E'], book))
            await self._emit_datas(items)
            self.last_ms = next_end
        logger.info(f'trades for {len(self._pairs_init)} pairs loop done')
        BotGlobal.state = BotState.STOPPED

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
                        logger.info(f'read trade eof: {pair}, last: {last_ts}')
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
                        logger.info(f'read odbook eof: {pair}, last: {last_ts}')
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

    async def _emit_datas(self, items: List[Tuple[int, dict]]):
        items = sorted(items, key=lambda x: x[0])
        cache_key, cache_ms = None, None
        cache_trades = []
        for item in items:
            cur_ms, data = item
            if data.get('info') and data['info']['e'] == 'trade':
                cur_pair = data['symbol']
                if not cache_key or cur_pair == cache_key and cur_ms - cache_ms < 100:
                    # 相同pair的交易，缓冲到一起，仅限100ms内
                    cache_key = cur_pair
                    cache_trades.append(data)
                    if not cache_ms:
                        cache_ms = cur_ms
                    continue
                if cache_trades:
                    if self._emit_trades(cache_key, cache_trades):
                        await asyncio.sleep(0)
                cache_key, cache_ms = cur_pair, cur_ms
                cache_trades = [data]
            elif data.get('e') == 'depthUpdate':
                if self.handle_order_book_msg(None, data):
                    await asyncio.sleep(0)
            else:
                raise ValueError(f'unknown data type: {data}')
        if cache_trades:
            if self._emit_trades(cache_key, cache_trades):
                await asyncio.sleep(0)

    def _emit_trades(self, pair: str, trades: List[dict]):
        waits = self.trade_waits.get(pair)
        is_copied = False
        if waits:
            btime.cur_timestamp = trades[-1]['timestamp'] / 1000
            self.trade_waits[pair] = collections.deque()
            for fut in waits:
                if fut.done():
                    continue
                fut.set_result(trades)
                is_copied = True
        if not is_copied:
            if pair not in self.done_trades:
                self.done_trades[pair] = []
            self.done_trades[pair].extend(trades)
        return is_copied

    def _parse_pair(self, marketId: str):
        if marketId not in self._pair_map:
            isSpot = False
            marketType = 'spot' if isSpot else 'contract'
            market = self.safe_market(marketId, None, None, marketType)
            self._pair_map[marketId] = market['symbol']
        return self._pair_map[marketId]

    def handle_order_book_msg(self, client: Client, message):
        btime.cur_timestamp = message['E'] / 1000
        marketId = self.safe_string(message, 's')
        pair = self._parse_pair(marketId)
        orderbook = self.safe_value(self.orderbooks, pair)
        if not orderbook:
            return
        self.handle_order_book_message(client, message, orderbook)
        return self._emit_books(pair, orderbook)

    def _emit_books(self, pair: str, book):
        waits = self.book_watis.get(pair)
        is_copied = False
        if waits:
            del self.book_watis[pair]
            for fut in waits:
                if fut.done():
                    continue
                fut.set_result(book)
                is_copied = True
        return is_copied


def get_local_wsexg(config: dict):
    exg_cfg = config['exchange']
    exg_name = exg_cfg['name']
    market = config['market_type']
    if exg_name == 'binance':
        if market == 'future':
            return exg_name, my_binanceusdm
    raise ValueError(f'unsupport local exchange: {exg_name}.{market}')
