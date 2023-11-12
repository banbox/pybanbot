#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_exchange.py
# Author: anyongjin
# Date  : 2023/8/16

import logging
import time

from banbot.exchange.crypto_exchange import *
from banbot.main.wallets import CryptoWallet

# logging.basicConfig(level=logging.DEBUG)


async def test_watch_trades():
    AppConfig.init_by_args()
    config = AppConfig.get()
    exchange = get_exchange()
    # wallet = CryptoWallet(config, exchange)
    # data_hd = LiveDataProvider(config, lambda x: x)
    # od_manager = LiveOrderManager(config, exchange, wallet, data_hd, lambda x: x)
    # await wallet.watch_balance_forever()
    stop = time.time() + 10
    while time.time() < stop:
        datas = await exchange.watch_order_book_for_symbols(['ADA/USDT:USDT', 'ETC/USDT:USDT'])
        print(datas)


async def test_ccxt_future():
    from banbot.exchange.crypto_exchange import get_exchange
    exchange = get_exchange('binance', 'future').api_async

    # 下单
    async def create_order(symbol, side, quantity, price, params: dict):
        order = await exchange.create_order(
            symbol=symbol,
            type='market',
            side=side,
            amount=quantity,
            price=price,
            params=params
        )
        return order

    # 使用示例
    symbol = 'BTC/USDT'
    quantity = 0.001
    side = 'buy'

    # 下单
    pos_side = 'LONG' if side == 'buy' else 'SHORT'
    params = dict(positionSide=pos_side)
    order = await create_order(symbol, side, quantity, 30000, params)
    print('enter:', order)
    enter_price = order['average']

    await asyncio.sleep(5)
    # 平仓
    price = enter_price * 0.95
    params = dict(closePosition=True, triggerPrice=price, positionSide=pos_side)
    # params = dict(positionSide=pos_side)
    side = 'sell'
    od_res = await create_order(symbol, side, quantity, price, params)
    print('exit:', od_res)


async def test_watch_mark_prices():
    AppConfig.init_by_args()
    exchange = get_exchange()
    await exchange.load_markets()
    while True:
        res = await exchange.watch_mark_prices()
        logger.info(f'mark price: {len(res)}')


async def test_calc_fee():
    AppConfig.init_by_args()
    exchange = get_exchange()
    await exchange.load_markets()
    symbol, order_type, side, amount, price, taker_maker = 'UNFI/USDT:USDT', None, 'buy', 14.4, 6.908, 'maker'
    fee = exchange.calc_fee(symbol, order_type, side, amount, price)
    print(fee)


if __name__ == '__main__':
    asyncio.run(test_calc_fee())
    # asyncio.run(test_watch_mark_prices())
