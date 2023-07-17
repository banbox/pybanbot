#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_ohlcvs.py
# Author: anyongjin
# Date  : 2023/5/17
import asyncio
import csv
import random
import time

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
    insert_num = 300
    open_price = 30000
    ohlcvs = []
    cur_stamp = btime.utcstamp() - 1000 * insert_num * 60
    cost_list = []
    for i in range(insert_num):
        oprice = open_price + random.random() - 0.5
        hprice = oprice + 1
        lprice = oprice - 1
        cprice = oprice + random.random()
        bar = (cur_stamp, oprice, hprice, lprice, cprice, random.random() * 1000)
        ohlcvs.append(bar)
        open_price = cprice
        cur_stamp += 60 * 1000
        start = time.monotonic()
        KLine.insert(14, '1m', [bar])
        cost = time.monotonic() - start
        if i > 10:
            cost_list.append(cost)
    print(sum(cost_list) / len(cost_list), max(cost_list), min(cost_list))


async def test_get_kline():
    from banbot.exchange.crypto_exchange import get_exchange
    exg = get_exchange('binance', 'future')
    exs = ExSymbol.get('binance', 'BTC/USDT:USDT', 'future')
    end_ms = btime.utcstamp()
    start_ms = end_ms - 30 * 24 * 3600 * 1000
    data = await auto_fetch_ohlcv(exg, exs, '1w', start_ms, end_ms, with_unfinish=True)
    print(data[-1])


async def test_trade_agg():
    '''
    从原始交易流生成1m的ohlcv，然后从交易所获取1m的ohlcv，都输出到文件，进行比较。
    需要给TradesWatcher的__init__添加下面：
    cln_p = pair.replace('/', '_').replace(':', '_')
    out_path = f'E:/Data/temp/{exg_name}_{market}_{cln_p}.csv'
    import csv
    wt_mode = 'a' if os.path.isfile(out_path) else 'w'
    self.out_file = open(out_path, wt_mode, newline='')
    self.writer = csv.writer(self.out_file)
    self.save_ts = time.time()
    通过watch_trades获取到交易后，用下面代码存储到文件
    csv_rows = [(t['info']['T'], t['info']['E'], t['price'], t['amount'], t['side']) for t in details]
    self.writer.writerows(csv_rows)
    if time.time() - self.save_ts > 10:
        print(f'flush: {self.pair}')
        self.out_file.flush()
        self.save_ts = time.time()
    '''
    data_dir = 'E:/Data/temp/'
    coin_list = ['1000FLOKI', '1000LUNC', '1000PEPE', '1000SHIB', '1000XEC', '1INCH', 'AAVE', 'ACH', 'ADA',
                 'AGIX', 'BTC']
    for coin in coin_list:
        trade_path = data_dir + f'binance_future_{coin}_USDT_USDT.csv'
        fdata = open(trade_path, 'r')
        trades = []
        for line in fdata:
            time_ts, _, price, amount, side = line.strip().split(',')
            trades.append(dict(timestamp=int(time_ts), price=float(price), amount=float(amount)))
        from banbot.data.tools import build_ohlcvc, trades_to_ohlcv
        ohlcv_arr = trades_to_ohlcv(trades)
        ohlcv_arr, _ = build_ohlcvc(ohlcv_arr, 60, with_count=False)
        start_ms, end_ms = ohlcv_arr[1][0], ohlcv_arr[-1][0]
        agg_path = data_dir + f'{coin}_agg.csv'
        with open(agg_path, 'w', newline='') as fout:
            agg_writer = csv.writer(fout)
            agg_writer.writerows(ohlcv_arr)

        from banbot.exchange.crypto_exchange import get_exchange
        exg = get_exchange('binance', 'future')
        ohlcv_true = await fetch_api_ohlcv(exg, f'{coin}/USDT:USDT', '1m', start_ms, end_ms)
        agg_path = data_dir + f'{coin}_true.csv'
        with open(agg_path, 'w', newline='') as fout:
            agg_writer = csv.writer(fout)
            agg_writer.writerows(ohlcv_true)
        print(f'{coin} ok')


if __name__ == '__main__':
    AppConfig.init_by_args()
    with db():
        # test_kline_insert()
        asyncio.run(test_trade_agg())
