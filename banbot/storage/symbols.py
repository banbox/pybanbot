#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : symbols.py
# Author: anyongjin
# Date  : 2023/4/24
import re
from typing import *

import ccxt

from banbot.storage.base import *
from banbot.util import btime
re_symbol = re.compile(r'^(\w+)/(\w+)(([:.])(\S+))?')
re_symbol_tv = re.compile(r'^(\w+)(USDT|TUSD|USDC|BUSD)((\.)(\S+))?')  # TradingView的交易对格式


class ExSymbol(BaseDbModel):
    __tablename__ = 'symbol'
    _object_map: ClassVar[Dict[str, 'ExSymbol']] = dict()
    _id_map: ClassVar[Dict[int, 'ExSymbol']] = dict()

    id = Column(sa.Integer, primary_key=True)
    exchange = Column(sa.String(50))
    symbol = Column(sa.String(20))  # BTC/USDT  BTC/USDT:USDT  BTC/USDT:USDT-230630
    market = Column(sa.String(20))
    list_dt = Column(type_=sa.TIMESTAMP(timezone=True))
    delist_dt = Column(type_=sa.TIMESTAMP(timezone=True))

    @orm.reconstructor
    def __init__(self, **kwargs):
        self.quote_code = ''
        self.base_code = ''
        symbol: str = kwargs.get('symbol') or self.symbol
        if symbol:
            pair_arr = symbol.split('/')
            if len(pair_arr) != 2:
                raise ValueError(f'invalid symbol: {kwargs}')
            self.base_code, quote_part = pair_arr
            self.quote_code = quote_part.split(':')[0]
        super(ExSymbol, self).__init__(**kwargs)

    def client_dict(self) -> dict:
        data = self.dict()
        data['short_name'] = to_short_symbol(self.symbol)
        return data

    def quote_suffix(self) -> str:
        """
        返回定价后缀，可用于判断其他交易对是否和当前交易对同属一个定价币。
        """
        pair_arr = self.symbol.split('/')
        suffix = '/'.join(pair_arr[1:])
        return '/' + suffix

    def __eq__(self, other):
        if not isinstance(other, ExSymbol):
            return False
        return (self.id == other.id and self.exchange == other.exchange and self.market == other.market and
                self.symbol == other.symbol)

    def __str__(self):
        return f'[{self.id}] {self.exchange}:{self.symbol}:{self.market}'

    def __repr__(self):
        return f'[{self.id}] {self.exchange}:{self.symbol}:{self.market}'

    @classmethod
    async def load_all(cls, sess: SqlSession, more_than: int = 0):
        fts = [ExSymbol.id > more_than, ExSymbol.delist_dt.is_(None)]
        stmt = select(ExSymbol).where(*fts)
        records: Iterable[ExSymbol] = (await sess.scalars(stmt)).all()
        for r in records:
            rkey = f'{r.exchange}:{r.market}:{r.symbol}'
            if rkey in cls._object_map:
                old_id = cls._object_map[rkey].id
                if old_id != r.id:
                    logger.error(f'duplicate symbol found {rkey}, id: {old_id} {r.id}')
                continue
            detach_obj(sess, r)
            cls._object_map[rkey] = r
            cls._id_map[r.id] = r

    @classmethod
    def get(cls, exg_name: str, market: str, symbol: str) -> 'ExSymbol':
        key = f'{exg_name}:{market}:{symbol}'
        obj = cls._object_map.get(key)
        if not obj:
            raise ValueError(f'{key} not exist in {len(cls._object_map)} cache')
        return obj

    @classmethod
    async def ensures(cls, exg_name: str, market: str, symbols: Iterable[str]):
        sess = dba.session
        fail_pairs = set(symbols).difference(cls._object_map.keys())
        if fail_pairs:
            await cls.load_all(sess)
        result = []
        add_items = []
        for symbol in symbols:
            key = f'{exg_name}:{market}:{symbol}'
            cache_val = cls._object_map.get(key)
            if cache_val:
                result.append(cache_val)
                continue
            obj = ExSymbol(exchange=exg_name, symbol=symbol, market=market)
            sess.add(obj)
            result.append(obj)
            add_items.append((key, obj))
        await sess.flush()
        for key, obj in add_items:
            detach_obj(sess, obj)
            cls._object_map[key] = obj
            cls._id_map[obj.id] = obj
        return result

    @classmethod
    def get_id(cls, exg_name: str, market: str, symbol: str) -> int:
        return cls.get(exg_name, market, symbol).id

    @classmethod
    def get_by_id(cls, sid: int) -> 'ExSymbol':
        cache_val = cls._id_map.get(sid)
        if not cache_val:
            raise ValueError(f'invalid sid: {sid}, from {len(cls._id_map)} items')
        return cache_val

    @classmethod
    def search(cls, keyword: str) -> List['ExSymbol']:
        if not keyword:
            return [item for key, item in cls._object_map.items()]
        upp_text = keyword.upper()
        return [item for key, item in cls._object_map.items() if key.find(upp_text) >= 0]

    async def get_valid_start(self, start_ms: int):
        if not self.list_dt:
            await self.init_list_dt()
        list_ms = btime.to_utcstamp(self.list_dt, ms=True, cut_int=True)
        start_ms = max(list_ms, start_ms)
        return start_ms

    async def init_list_dt(self):
        if self.list_dt:
            return
        sess = dba.session
        inst: ExSymbol = await sess.get(ExSymbol, self.id)
        if not inst.list_dt:
            from banbot.exchange.crypto_exchange import get_exchange
            exchange = get_exchange(self.exchange, inst.market)
            candles = await exchange.fetch_ohlcv(self.symbol, '1m', 1325376000, limit=10)
            if not candles:
                logger.warning(f'no candles found for {self.exchange}/{self.symbol}')
                return
            inst.list_dt = btime.to_datetime(candles[0][0])
        self.list_dt = inst.list_dt

    @classmethod
    async def init(cls):
        '''
        遍历symbols，填充未计算的list_dt，在机器人启动时执行
        '''
        sess = dba.session
        start = time.monotonic()
        await cls.load_all(sess)
        for key, obj in cls._object_map.items():
            if obj.list_dt:
                continue
            try:
                await obj.init_list_dt()
            except ccxt.BadSymbol:
                obj.delist_dt = btime.now()
        cost = time.monotonic() - start
        if cost > 0.5:
            logger.info(f'fill_list_dts cost: {cost:.2f} s')


def to_short_symbol(symbol: str) -> str:
    short_name = symbol
    mat = re_symbol.search(short_name)
    base_s, quote_s, spliter, suffix = mat.group(1), mat.group(2), mat.group(4), mat.group(5)
    if suffix and spliter == ':':
        if quote_s == suffix:
            # 后缀和定价币相同，是永续合约
            short_name = f'{base_s}/{quote_s}.P'
        elif suffix.startswith(quote_s):
            # 后缀包含定价币，是短时合约  BTC/USDT:USDT-230630
            clean_sx = suffix[len(quote_s):].strip('-')
            if clean_sx:
                short_name = f'{base_s}/{quote_s}.{clean_sx}'
        elif suffix:
            logger.warning(f'unknown symbol format: {symbol}')
    return short_name


def get_symbol_market(symbol_short: str) -> Tuple[str, str]:
    '''
    根据传入的symbol，解析得到正式使用的symbol和market
    :param symbol_short: 可以是short_name，也可以是完整的symbol
    '''
    symbol, market = symbol_short, 'spot'
    mat = re_symbol.search(symbol_short)
    if not mat:
        mat = re_symbol_tv.search(symbol_short)
    if not mat:
        raise ValueError(f'unsupport symbol format: {symbol_short}')
    base_s, quote_s, spliter, suffix = mat.group(1), mat.group(2), mat.group(4), mat.group(5)
    if suffix:
        # 带后缀的，默认是期货
        market = 'future'
    if suffix and spliter == '.':
        # 分隔符是.表示是short_name
        symbol = f'{base_s}/{quote_s}:{quote_s}'
        if suffix.lower() != 'p':
            # 非永续合约，带上后缀
            symbol += '-' + suffix
    return symbol, market


def split_symbol(symbol: str):
    base_s, quote_s = symbol.split('/')
    quote_s = quote_s.split(':')[0]
    return base_s, quote_s
