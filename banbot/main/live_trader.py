#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : whack_mole.py
# Author: anyongjin
# Date  : 2023/2/28
from banbot.main.itrader import *
from banbot.main.od_manager import *
from banbot.util.misc import *
from banbot.config import *
from banbot.util import btime
from banbot.rpc.rpc_manager import RPCManager, RPCMessageType
from banbot.symbols.pair_manager import PairManager
from banbot.storage import *


class LiveTrader(Trader):
    '''
    实盘交易、实时模拟。模式:DRY_RUN LIVE
    '''

    def __init__(self, config: Config):
        super(LiveTrader, self).__init__(config)
        self.exchange = get_exchange()
        self.data_mgr = LiveDataProvider(config, self.on_data_feed)
        self.pair_mgr = PairManager(config, self.exchange)
        self.wallets = CryptoWallet(config, self.exchange)
        self.order_mgr = LiveOrderManager(config, self.exchange, self.wallets, self.data_mgr, self.order_callback)
        self.rpc = RPCManager(self)

    def order_callback(self, od: InOutOrder, is_enter: bool):
        msg_type = RPCMessageType.ENTRY if is_enter else RPCMessageType.EXIT
        sub_od = od.enter if is_enter else od.exit
        if sub_od.status != OrderStatus.Close:
            return
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
        asyncio.create_task(self.rpc.send_msg(msg))

    async def init(self):
        await self.exchange.load_markets()
        await self.pair_mgr.refresh_pairlist()
        await self.wallets.init(self.pair_mgr.symbols)
        await self.exchange.init(self.pair_mgr.symbols)
        pair_tfs = self._load_strategies(self.pair_mgr.symbols)
        with db():
            BotTask.init()
            self.data_mgr.sub_pairs(pair_tfs)
        await self.rpc.startup_messages()

    async def run(self):
        self.start_heartbeat_check(3)
        # 初始化
        await self.init()
        # 启动异步任务
        await self._start_tasks()
        # 轮训执行任务
        await self._loop_tasks([
            # 两小时更新一次货币行情信息
            [self.exchange.load_markets, 7200, 7200],
            # 定时更新定价货币的价格
            [self.exchange.update_quote_price, 60, 0],
            # 定时检查整体损失是否触发限制
            [self.order_mgr.check_fatal_stop, 300, 300]
        ])

    async def _start_tasks(self):
        # 监听实时数据推送
        self._run_tasks.append(asyncio.create_task(LiveDataProvider.watch_ohlcvs()))
        if btime.run_mode == RunMode.LIVE:
            # 仅实盘交易模式，监听钱包和订单状态更新
            self._run_tasks.extend([
                # 监听钱包更新
                asyncio.create_task(self.wallets.update_forever()),
                # 监听订单更新
                asyncio.create_task(self.order_mgr.listen_orders_forever()),
                # 跟踪监听未成交订单，及时更新价格确保成交
                asyncio.create_task(self.order_mgr.trail_open_orders_forever()),
                # 订单异步消费队列
                asyncio.create_task(self.order_mgr.consume_queue())
            ])
            logger.info('listen websocket , watch wallets and order updates ...')

    async def cleanup(self):
        with db():
            exit_ods = self.order_mgr.exit_open_orders('force_exit', 0)
            if exit_ods:
                logger.info('exit %d open trades', len(exit_ods))
        await self.order_mgr.order_q.join()
        await self.rpc.send_msg(dict(
            type=RPCMessageType.STATUS,
            status='Bot stopped'
        ))
        await self.rpc.cleanup()
        await self.exchange.close()
