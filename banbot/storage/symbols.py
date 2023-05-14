#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : symbols.py
# Author: anyongjin
# Date  : 2023/4/24
import time

from banbot.storage.base import *
from typing import Dict, ClassVar


class ExSymbol(BaseDbModel):
    __tablename__ = 'symbol'
    _object_map: ClassVar[Dict[str, 'ExSymbol']] = dict()

    id = Column(sa.Integer, primary_key=True)
    exchange = Column(sa.String(50))
    symbol = Column(sa.String(20))
    list_dt = Column(sa.DateTime)

    def get_valid_start(self, start_ms: int):
        if self.list_dt:
            list_ms = btime.to_utcstamp(self.list_dt, ms=True, round_int=True)
            start_ms = max(list_ms, start_ms)
        return start_ms

    async def init_list_dt(self):
        from banbot.exchange.crypto_exchange import get_exchange
        sess = db.session
        sess.refresh(self)
        if self.list_dt:
            return
        exchange = get_exchange(self.exchange)
        candles = await exchange.fetch_ohlcv(self.symbol, '1s', 1325376000, limit=10)
        if not candles:
            logger.warning(f'no candles found for {self.exchange}/{self.symbol}')
            return
        self.list_dt = btime.to_datetime(candles[0][0])
        sess.commit()

    @classmethod
    def _load_objects(cls, sess: SqlSession, more_than: int = 0):
        records = sess.query(ExSymbol).filter(ExSymbol.id > more_than).all()
        for r in records:
            rkey = f'{r.exchange}:{r.symbol}'
            cls._object_map[rkey] = r

    @classmethod
    def get(cls, exg_name: str, symbol: str) -> 'ExSymbol':
        key = f'{exg_name}:{symbol}'
        cache_val = cls._object_map.get(key)
        if cache_val:
            return cache_val
        old_mid = 0
        if cls._object_map:
            old_mid = max(list(map(lambda x: x.id, cls._object_map.values())))
        sess = db.session
        cls._load_objects(sess, old_mid)
        if key in cls._object_map:
            return cls._object_map[key]
        obj = ExSymbol(exchange=exg_name, symbol=symbol)
        sess.add(obj)
        sess.flush()
        cls._object_map[key] = obj
        sess.commit()
        return obj

    @classmethod
    def get_id(cls, exg_name: str, symbol: str) -> int:
        return cls.get(exg_name, symbol).id

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
        logger.info(f'fill_list_dts cost: {cost:.2f} s')
