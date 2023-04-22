#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : cache.py
# Author: anyongjin
# Date  : 2023/4/18
from cachetools import TTLCache


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
