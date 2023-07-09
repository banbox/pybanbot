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
    exg = get_exchange('binance', 'future')
    symbol, timeframe = 'BTC/USDT:USDT', '1m'
    tf_msecs = tf_to_secs(timeframe) * 1000
    stop_ms = btime.utcstamp()
    start_ms = stop_ms - tf_msecs * 3
    arr = await fetch_api_ohlcv(exg, symbol, timeframe, start_ms, stop_ms)
    # sid = ExSymbol.get_id(exg.name, symbol, exg.market_type)
    # KLine.insert(sid, timeframe, arr)
    [print(r) for r in arr]
    bar_end = stop_ms // tf_msecs * tf_msecs
    print(bar_end)


AppConfig.init_by_args()
init_db()
with db():
    asyncio.run(test_down())
