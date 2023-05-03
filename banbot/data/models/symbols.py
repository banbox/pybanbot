#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : symbols.py
# Author: anyongjin
# Date  : 2023/4/24
from banbot.data.models.base import *
from typing import Dict, ClassVar


class SymbolTF(BaseDbModel):
    __tablename__ = 'symboltf'
    _object_map: ClassVar[Dict[str, int]] = dict()

    id = Column(sa.Integer, primary_key=True)
    exchange = Column(sa.String(50))
    symbol = Column(sa.String(20))
    timeframe = Column(sa.String(5))
    tf_secs = Column(sa.Integer)

    @classmethod
    def get_id(cls, exg_name: str, symbol: str, timeframe: str = '1m') -> int:
        key = f'{exg_name}:{symbol}:{timeframe}'
        cache_val = cls._object_map.get(key)
        if cache_val:
            return cache_val
        old_mid = max(cls._object_map.values()) if cls._object_map else 0
        with db_sess() as sess:
            records = sess.query(SymbolTF).filter(SymbolTF.id > old_mid).all()
            for r in records:
                rkey = f'{r.exchange}:{r.symbol}:{r.timeframe}'
                cls._object_map[rkey] = r.id
            if key in cls._object_map:
                return cls._object_map[key]
            from banbot.exchange.exchange_utils import timeframe_to_seconds
            tf_secs = timeframe_to_seconds(timeframe)
            obj = SymbolTF(exchange=exg_name, symbol=symbol, timeframe=timeframe, tf_secs=tf_secs)
            sess.add(obj)
            sess.flush()
            cls._object_map[key] = obj.id
            sess.commit()
            return obj.id
