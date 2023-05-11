#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : symbols.py
# Author: anyongjin
# Date  : 2023/4/24
from banbot.storage.base import *
from typing import Dict, ClassVar


class SymbolTF(BaseDbModel):
    __tablename__ = 'symboltf'
    _object_map: ClassVar[Dict[str, int]] = dict()

    id = Column(sa.Integer, primary_key=True)
    exchange = Column(sa.String(50))
    symbol = Column(sa.String(20))

    @classmethod
    def get_id(cls, exg_name: str, symbol: str) -> int:
        key = f'{exg_name}:{symbol}'
        cache_val = cls._object_map.get(key)
        if cache_val:
            return cache_val
        old_mid = max(cls._object_map.values()) if cls._object_map else 0
        sess = db.session
        records = sess.query(SymbolTF).filter(SymbolTF.id > old_mid).all()
        for r in records:
            rkey = f'{r.exchange}:{r.symbol}'
            cls._object_map[rkey] = r.id
        if key in cls._object_map:
            return cls._object_map[key]
        obj = SymbolTF(exchange=exg_name, symbol=symbol)
        sess.add(obj)
        sess.flush()
        cls._object_map[key] = obj.id
        sess.commit()
        return obj.id
