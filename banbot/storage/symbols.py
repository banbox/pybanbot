#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : symbols.py
# Author: anyongjin
# Date  : 2023/4/24

from typing import ClassVar

from banbot.storage.base import *


class ExSymbol(BaseDbModel):
    __tablename__ = 'symbol'
    _object_map: ClassVar[Dict[str, 'ExSymbol']] = dict()

    id = Column(sa.Integer, primary_key=True)
    exchange = Column(sa.String(50))
    symbol = Column(sa.String(20))
    list_dt = Column(sa.DateTime)

    @classmethod
    def _load_objects(cls, sess: SqlSession, more_than: int = 0):
        records = sess.query(ExSymbol).filter(ExSymbol.id > more_than).all()
        for r in records:
            rkey = f'{r.exchange}:{r.symbol}'
            detach_obj(sess, r)
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
        sess.commit()
        logger.info(f'create symbol: {exg_name}:{symbol}, id: {obj.id}')
        detach_obj(sess, obj)
        cls._object_map[key] = obj
        return obj

    @classmethod
    def get_id(cls, exg_name: str, symbol: str) -> int:
        return cls.get(exg_name, symbol).id

    @classmethod
    def search(cls, keyword: str) -> List['ExSymbol']:
        if not cls._object_map:
            sess = db.session
            cls._load_objects(sess)
        upp_text = keyword.upper()
        return [item for key, item in cls._object_map.items() if key.find(upp_text) >= 0]

    @classmethod
    async def get_valid_start(cls, exg_name: str, symbol: str, start_ms: int):
        obj = cls.get(exg_name, symbol)
        if not obj.list_dt:
            await obj.init_list_dt()
        list_ms = btime.to_utcstamp(obj.list_dt, ms=True, round_int=True)
        start_ms = max(list_ms, start_ms)
        return start_ms

    async def init_list_dt(self):
        if self.list_dt:
            return
        sess = db.session
        inst: ExSymbol = sess.query(ExSymbol).get(self.id)
        if not inst.list_dt:
            from banbot.exchange.crypto_exchange import get_exchange
            exchange = get_exchange(self.exchange)
            candles = await exchange.fetch_ohlcv(self.symbol, '1s', 1325376000, limit=10)
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
