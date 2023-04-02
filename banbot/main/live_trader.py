#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : whack_mole.py
# Author: anyongjin
# Date  : 2023/2/28
import asyncio
from banbot.main.itrader import *
from banbot.exchange.crypto_exchange import *
from banbot.storage.od_manager import *
from banbot.util.misc import *
from banbot.config import *
from banbot.util import btime
from banbot.rpc.rpc_manager import RPCManager, RPCMessageType


class LiveTrader(Trader):
    '''
    实盘交易、实时模拟。模式:DRY_RUN LIVE
    '''

    def __init__(self, config: Config):
        super(LiveTrader, self).__init__(config)
        self.exchange = CryptoExchange(config)
        self.data_hold = LiveDataProvider(config, self.exchange)
        self.wallets = CryptoWallet(config, self.exchange)
        self.order_hold = LiveOrderManager(config, self.exchange, self.wallets)
        self.data_hold.set_callback(self._make_invoke())
        self.rpc = RPCManager(self)
        self._run_tasks: List[asyncio.Task] = []

    def order_callback(self, od: InOutOrder, is_enter: bool):
        msg_type = RPCMessageType.ENTRY if is_enter else RPCMessageType.EXIT
        msg = dict(
            type=msg_type,
            enter_tag=od.enter_tag,
            exit_tag=od.exit_tag,
            amount=od.enter.amount,
            price=od.enter.price,
            strategy=od.strategy,
            pair=od.symbol,
            profit=od.profit,
            profit_rate=od.profit_rate
        )
        self.rpc.send_msg(msg)

    async def init(self):
        self._load_strategies()
        await self.exchange.load_markets()
        await self.exchange.update_quote_price()
        await self.wallets.init()
        await init_longvars(self.exchange, self.pairlist)
        self.order_hold.callbacks.append(self.order_callback)
        logger.info('banbot init complete')
        self.rpc.startup_messages()

    async def run(self):
        # 初始化
        await self.init()
        # 启动异步任务
        await self._start_tasks()
        # 轮训执行任务
        cur_time = time.time()
        await self._loop_tasks([
            # 轮询函数，轮询间隔(s)，下次执行时间
            [self.data_hold.process, self.data_hold.min_interval, cur_time],
            # 定时更新定价货币的价格
            [self.exchange.update_quote_price, 60, cur_time],
            # 定时检查整体损失是否触发限制
            [self.order_hold.check_fatal_stop, 300, cur_time + 300]
        ])

    def _bar_callback(self):
        if btime.run_mode != RunMode.LIVE:
            self.order_hold.fill_pending_orders(bar_arr.get())

    async def _start_tasks(self):
        if btime.run_mode == RunMode.LIVE:
            # 仅实盘交易模式，监听钱包和订单状态更新
            self._run_tasks.extend([
                # 监听钱包更新
                asyncio.create_task(self.wallets.update_forever()),
                # 监听订单更新
                asyncio.create_task(self.order_hold.update_forever())
            ])

    async def cleanup(self):
        logger.info('Cleaning up trading...')
        self.rpc.cleanup()
        await self.exchange.close()
