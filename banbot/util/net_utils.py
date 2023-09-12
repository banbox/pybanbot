#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : net_utils.py
# Author: anyongjin
# Date  : 2023/9/12

import aiohttp
import orjson
from typing import *

_sess_map: Dict[str, aiohttp.ClientSession] = dict()


class HTTPError(IOError):
    def __init__(self, status: int, content: str):
        super(HTTPError, self).__init__()
        self.status = status
        self.content = content

    def __str__(self):
        return f'[HTTPError:{self.status}]{self.content}'

    def __repr__(self):
        return f'[HTTPError:{self.status}]{self.content}'


def split_origin_path(url: str):
    dot_idx = url.find('.')
    cut_pos = url.find('/', dot_idx)
    if cut_pos > 0:
        return url[:cut_pos], url[cut_pos:]
    return url, ''


async def get_http_sess(host: str, timeout: int = 20):
    global _sess_map
    host = split_origin_path(host)[0]
    import threading
    cache_key = f'{host}_{threading.current_thread().ident}'
    sess = _sess_map.get(cache_key)
    if sess and not sess.closed:
        return sess
    timeout_ = aiohttp.ClientTimeout(total=timeout)
    sess = aiohttp.ClientSession(host, timeout=timeout_)
    _sess_map[cache_key] = sess
    return sess


async def parse_http_rsp(rsp: aiohttp.ClientResponse) -> dict:
    rsp_data = await rsp.content.read()
    rsp_text = rsp_data.decode('utf-8')
    if rsp.status // 100 != 2:
        raise HTTPError(rsp.status, rsp_text)
    return orjson.loads(rsp_text)
