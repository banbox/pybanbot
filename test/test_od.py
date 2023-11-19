#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_od.py
# Author: anyongjin
# Date  : 2023/11/19
import asyncio

from banbot.main.itrader import *
from banbot.main.od_manager import *
from banbot.rpc import Notify, NotifyType, start_api
from banbot.storage import *
from banbot.symbols.pair_manager import PairManager
from banbot.util import btime
from banbot.util.misc import *
from banbot.exchange import get_exchange
from banbot.config import Config
from banbot.main.wallets import CryptoWallet
from banbot.util.common import logger
from banbot.util.support import BanEvent
from banbot.storage.orders import OrderJob, InOutTracer, ORMTracer


async def test_ent_od():
    config = AppConfig.get()
    exchange = get_exchange()
    wallets = CryptoWallet(config, exchange)
    order_mgr = LiveOrderMgr.init(config, exchange, wallets, None, lambda x, y: x)
    pair = 'ADA/USDT:USDT'
    btime.run_mode = btime.RunMode.PROD
    async with dba():
        await ExSymbol.init()
        await exchange.load_markets()
    BotGlobal.state = BotState.RUNNING
    ctx_key = f'{exchange.name}_{exchange.market_type}_{pair}_ws'
    bar = [1700366400000, 0.3768, 0.3776, 0.3730, 0.3737, 19910245541]
    with TempContext(ctx_key):
        append_new_bar(bar, 300)
        MarketPrice.set_bar_price(pair, float(bar[ccol]))
    ctx = get_context(ctx_key)
    ent_d = dict(tag='long', short=False, stoploss_price=0.37, enter_order_type=OrderType.Limit.value,
                 enter_price=0.384, legal_cost=10)
    async with dba():
        od = await order_mgr.enter_order(ctx, 'MALimit', ent_d, False)
        od_id = od.id
    async with LocalLock(f'iod_{od_id}', 5, force_on_fail=True):
        tester = ORMTracer()
        async with dba():
            tracer = InOutTracer()
            od = await InOutOrder.get(od_id)
            tester.trace([od])
            tracer.trace([od])
            await order_mgr._exec_order_enter(od)
            dup_od = od.clone()
            # 这里的status修改不会影响od
            dup_od.status = InOutStatus.FullExit
            print('status: ', od.status, dup_od.status)
            tester.update()
            await tracer.save()
        async with dba():
            od = await InOutOrder.get(od_id)
            # 这里输出的status依然是2
            print(od.status, od.enter.filled, od.enter.order_id)
            new_od = await dup_od.attach(dba.session)
            # 这里输出的status是两个4，并且会被保存到数据库
            print(new_od.status, od.status)
            await tester.test()
        async with dba():
            od = await InOutOrder.get(od_id)
            # 这里输出的是4
            print(od.status, od.enter.filled, od.enter.order_id)
    # await order_mgr.exec_od_job(OrderJob(od.id, OrderJob.ACT_ENTER))


if __name__ == '__main__':
    asyncio.run(test_ent_od())
