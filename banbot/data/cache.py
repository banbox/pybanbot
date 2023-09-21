#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : cache.py
# Author: anyongjin
# Date  : 2023/9/20
from typing import Dict, Any
from banbot.util import btime
from cachetools import TTLCache


class BanCache:
    data: Dict[str, Any] = dict()
    expires: Dict[str, float] = dict()

    @classmethod
    def set(cls, key: str, data: Any, expires_secs: float = None):
        if data is None:
            if key in cls.data:
                del cls.data[key]
            if key in cls.expires:
                del cls.expires[key]
        else:
            cls.data[key] = data
            if expires_secs:
                cls.expires[key] = expires_secs + btime.time()

    @classmethod
    def get(cls, key: str, def_val=None):
        if not key:
            return def_val
        exp_at = cls.expires.get(key)
        cur_ts = btime.time()
        if exp_at and cur_ts >= exp_at:
            if key in cls.data:
                del cls.data[key]
            if key in cls.expires:
                del cls.expires[key]
            return def_val
        if key in cls.data:
            return cls.data.get(key)
        return def_val

    @classmethod
    def ttl(cls, key: str):
        if not key or key not in cls.expires:
            return -1
        exp_at = cls.expires.get(key) or 0
        return exp_at - btime.time()


class PunctualCache(TTLCache):
    '''
    整点过期缓存。
    比如ttl是3600时，将会在每小时的0分过期。
    '''

    def __init__(self, maxsize, ttl, getsizeof=None):
        from datetime import datetime, timezone

        def local_timer():
            ts = datetime.now(timezone.utc).timestamp()
            offset = (ts % ttl)
            return ts - offset

        # Init with smlight offset
        super().__init__(maxsize=maxsize, ttl=ttl - 1e-5, timer=local_timer, getsizeof=getsizeof)
