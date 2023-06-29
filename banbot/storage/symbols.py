#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : symbols.py
# Author: anyongjin
# Date  : 2023/4/24

from typing import ClassVar,Tuple

from banbot.storage.base import *


class ExSymbol(BaseDbModel):
    __tablename__ = 'symbol'
    _object_map: ClassVar[Dict[str, 'ExSymbol']] = dict()

    id = Column(sa.Integer, primary_key=True)
    exchange = Column(sa.String(50))
    symbol = Column(sa.String(20))
    market = Column(sa.String(20))
    list_dt = Column(sa.DateTime)

    def __str__(self):
        return f'{self.exchange}:{self.symbol}:{self.market}'

    def __repr__(self):
        return f'{self.exchange}:{self.symbol}:{self.market}'

    @classmethod
    def _load_objects(cls, sess: SqlSession, more_than: int = 0):
        records = sess.query(ExSymbol).filter(ExSymbol.id > more_than).all()
        for r in records:
            rkey = f'{r.exchange}:{r.symbol}:{r.market}'
            if rkey in cls._object_map:
                old_id = cls._object_map[rkey].id
                if old_id != r.id:
                    logger.error(f'duplicate symbol found {rkey}, id: {old_id} {r.id}')
                continue
            detach_obj(sess, r)
            cls._object_map[rkey] = r

    @classmethod
    def get(cls, exg_name: str, symbol: str, market='spot') -> 'ExSymbol':
        key = f'{exg_name}:{symbol}:{market}'
        cache_val = cls._object_map.get(key)
        if cache_val:
            return cache_val
        sess = db.session
        cls._load_objects(sess)
        if key in cls._object_map:
            return cls._object_map[key]
        logger.info(f'{key} not found in cache, create new, cache: {cls._object_map.keys()}')
        obj = ExSymbol(exchange=exg_name, symbol=symbol, market=market)
        sess.add(obj)
        sess.commit()
        logger.info(f'create symbol: {key}, id: {obj.id}')
        detach_obj(sess, obj)
        cls._object_map[key] = obj
        return obj

    @classmethod
    def get_id(cls, exg_name: str, symbol: str, market='spot') -> int:
        return cls.get(exg_name, symbol, market).id

    @classmethod
    def search(cls, keyword: str) -> List['ExSymbol']:
        if not cls._object_map:
            sess = db.session
            cls._load_objects(sess)
        if not keyword:
            return [item for key, item in cls._object_map.items()]
        upp_text = keyword.upper()
        return [item for key, item in cls._object_map.items() if key.find(upp_text) >= 0]

    async def get_valid_start(self, start_ms: int):
        if not self.list_dt:
            await self.init_list_dt()
        list_ms = btime.to_utcstamp(self.list_dt, ms=True, round_int=True)
        start_ms = max(list_ms, start_ms)
        return start_ms

    async def init_list_dt(self):
        if self.list_dt:
            return
        sess = db.session
        inst: ExSymbol = sess.query(ExSymbol).get(self.id)
        if not inst.list_dt:
            from banbot.exchange.crypto_exchange import get_exchange
            exchange = get_exchange(self.exchange, inst.market)
            candles = await exchange.fetch_ohlcv(self.symbol, '1m', 1325376000, limit=10)
            if not candles:
                logger.warning(f'no candles found for {self.exchange}/{self.symbol}')
                return
            inst.list_dt = btime.to_datetime(candles[0][0])
            sess.commit()
        self.list_dt = inst.list_dt

    @classmethod
    async def fill_list_dts(cls):
        '''
        遍历symbols，填充未计算的list_dt，在机器人启动时执行
        '''
        sess = db.session
        start = time.monotonic()
        cls._load_objects(sess)
        for key, obj in cls._object_map.items():
            if obj.list_dt:
                continue
            await obj.init_list_dt()
        cost = time.monotonic() - start
        if cost > 0.5:
            logger.info(f'fill_list_dts cost: {cost:.2f} s')


def get_symbol_market(symbol_com: str) -> Tuple[str, str]:
    symbol, market = symbol_com, 'spot'
    split_id = symbol_com.rfind('.')
    if split_id > 0:
        market_short = symbol_com[split_id + 1:]
        if market_short == 'P':
            market = 'future'
        elif market_short:
            logger.error(f'unknown market type: {market_short} {symbol_com}')
        symbol = symbol_com[:split_id]
    return symbol, market
