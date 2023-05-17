#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_ohlcvs.py
# Author: anyongjin
# Date  : 2023/5/17
from banbot.data.tools import *
from banbot.storage.base import init_db, db
from banbot.config import AppConfig


async def test_down():
    from banbot.exchange.crypto_exchange import get_exchange
    exg = get_exchange('binance')
    start_ms, stop_ms = 1684214100000, 1684215000000
    arr = await fetch_api_ohlcv(exg, 'AVAX/USDT', '1m', start_ms, stop_ms)
    print(arr)


AppConfig.init_by_args(dict(config=[r'E:\trade\banbot\banbot\config\config.json']))
init_db()
with db():
    asyncio.run(test_down())
