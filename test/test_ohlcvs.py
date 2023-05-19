#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_ohlcvs.py
# Author: anyongjin
# Date  : 2023/5/17
import asyncio
from banbot.data.tools import *
from banbot.storage.base import init_db, db
from banbot.config import AppConfig


async def test_down():
    from banbot.exchange.crypto_exchange import get_exchange
    from banbot.storage import KLine
    exg = get_exchange('binance')
    start_ms, stop_ms = 1684490340000, 1684490400000
    symbol, timeframe = 'BTC/TUSD', '1m'
    arr = await fetch_api_ohlcv(exg, symbol, timeframe, start_ms, stop_ms)
    KLine.insert(exg.name, symbol, timeframe, arr)
    print(arr)


AppConfig.init_by_args(dict(config=[r'E:\trade\banbot\banbot\config\config.json']))
init_db()
with db():
    asyncio.run(test_down())
