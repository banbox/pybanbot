#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : whack_mole.py
# Author: anyongjin
# Date  : 2023/2/28
import asyncio
from banbot.main.itrader import *
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
        self.data_hold = LiveDataProvider(config, self.exchange, self._pair_row_callback)
        self.wallets = CryptoWallet(config, self.exchange)
        self.order_hold = LiveOrderManager(config, self.exchange, self.wallets, self.data_hold)
        self.order_hold.callbacks.append(self.order_callback)
        self.rpc = RPCManager(self)

    async def order_callback(self, od: InOutOrder, is_enter: bool):
        msg_type = RPCMessageType.ENTRY if is_enter else RPCMessageType.EXIT
        sub_od = od.enter if is_enter else od.exit
        msg = dict(
            type=msg_type,
            enter_tag=od.enter_tag,
            exit_tag=od.exit_tag,
            side=sub_od.side,
            amount=sub_od.amount,
            price=sub_od.price,
            value=sub_od.amount * sub_od.price,
            strategy=od.strategy,
            pair=od.symbol,
            profit=od.profit,
            profit_rate=od.profit_rate
        )
        await self.rpc.send_msg(msg)

    async def init(self):
        await self.exchange.load_markets()
        await self.exchange.cancel_open_orders()
        await self.exchange.update_quote_price()
        await self.wallets.init()
        logger.info('banbot init complete')
        await self.rpc.startup_messages()

    async def run(self):
        self.start_heartbeat_check(3)
        # 初始化
        await self.init()
        # 启动异步任务
        await self._start_tasks()
        # 需要预热数据
        btime.run_mode = RunMode.OTHER
        btime.cur_timestamp = time.time() - self.data_hold.set_warmup(self._warmup_num)
        cur_time = btime.time()
        loop_tasks = [
            # 轮询函数，轮询间隔(s)，下次执行时间
            [self.data_hold.process, self.data_hold.min_interval, cur_time],
            # 两小时更新一次货币行情信息
            [self.exchange.load_markets, 7200, cur_time + 7200],
            # 定时更新定价货币的价格
            [self.exchange.update_quote_price, 60, cur_time],
            # 定时检查整体损失是否触发限制
            [self.order_hold.check_fatal_stop, 300, cur_time + 300]
        ]
        # 轮训执行任务
        await self._loop_tasks(loop_tasks)

    async def _bar_callback(self):
        if btime.run_mode != RunMode.LIVE:
            await self.order_hold.fill_pending_orders(bar_arr.get())

    async def _start_tasks(self):
        if btime.run_mode == RunMode.LIVE:
            # 仅实盘交易模式，监听钱包和订单状态更新
            self._run_tasks.extend([
                # 监听钱包更新
                asyncio.create_task(self.wallets.update_forever()),
                # 监听订单更新
                asyncio.create_task(self.order_hold.listen_orders_forever()),
                # 跟踪监听未成交订单，及时更新价格确保成交
                asyncio.create_task(self.order_hold.trail_open_orders_forever())
            ])
            logger.info('listen websocket , watch wallets and order updates ...')

    async def cleanup(self):
        exit_ods = await self.order_hold.exit_open_orders('force_exit', 0)
        if exit_ods:
            logger.info(f'exit {len(exit_ods)} open trades')
        await self.rpc.send_msg(dict(
            type=RPCMessageType.STATUS,
            status='Bot stopped'
        ))
        await self.rpc.cleanup()
        await self.exchange.close()
