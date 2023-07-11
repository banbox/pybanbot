#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_ohlcvs.py
# Author: anyongjin
# Date  : 2023/5/17
import asyncio
import random

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


def test_kline_insert():
    from banbot.storage import KLine
    KLine.sync_timeframes()
    insert_num = 3000
    open_price = 30000
    ohlcvs = []
    cur_stamp = btime.utcstamp() - 1000 * insert_num * 60
    for i in range(insert_num):
        oprice = open_price + random.random() - 0.5
        hprice = oprice + 1
        lprice = oprice - 1
        cprice = oprice + random.random()
        bar = (cur_stamp, oprice, hprice, lprice, cprice, random.random() * 1000)
        ohlcvs.append(bar)
        open_price = cprice
        cur_stamp += 60 * 1000
    KLine.insert(14, '1m', ohlcvs)


if __name__ == '__main__':
    AppConfig.init_by_args()
    with db():
        test_kline_insert()
