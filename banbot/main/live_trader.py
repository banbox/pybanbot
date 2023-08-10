#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : whack_mole.py
# Author: anyongjin
# Date  : 2023/2/28
from banbot.main.itrader import *
from banbot.main.od_manager import *
from banbot.rpc.rpc_manager import RPCManager, RPCMessageType
from banbot.storage import *
from banbot.symbols.pair_manager import PairManager
from banbot.util import btime
from banbot.util.misc import *


class LiveTrader(Trader):
    '''
    实盘交易、实时模拟。模式:DRY_RUN PROD
    '''

    def __init__(self, config: Config):
        super(LiveTrader, self).__init__(config)
        self.exchange = get_exchange()
        self.data_mgr = LiveDataProvider(config, self.on_data_feed)
        self.pair_mgr = PairManager(config, self.exchange)
        self.wallets = CryptoWallet(config, self.exchange)
        self.order_mgr = LiveOrderManager(config, self.exchange, self.wallets, self.data_mgr, self.order_callback)
        self.rpc = RPCManager(config)

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
        from banbot.data.toolbox import sync_timeframes
        await self.exchange.load_markets()
        with db():
            BotTask.init()
            sync_timeframes()
            await ExSymbol.fill_list_dts()
            await self.pair_mgr.refresh_pairlist()
            await self.wallets.init(self.pair_mgr.symbols)
            await self.exchange.init(self.pair_mgr.symbols)
            pair_tfs = self._load_strategies(self.pair_mgr.symbols, self.pair_mgr.pair_tfscores)
            await self.data_mgr.sub_warm_pairs(pair_tfs)
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
            # 定时更新货币的价格
            [self.exchange.update_symbol_prices, 60, 0],
            # 定时检查整体损失是否触发限制
            [self.order_mgr.check_fatal_stop, 300, 300]
        ])

    async def _start_tasks(self):
        # 监听实时数据推送
        self._run_tasks.append(asyncio.create_task(LiveDataProvider.run()))
        # 定期刷新交易对
        self._run_tasks.append(asyncio.create_task(self.loop_refresh_pairs()))
        if btime.prod_mode():
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

    async def refresh_pairs(self):
        '''
        定期刷新交易对
        '''
        logger.info("start refreshing symbols")
        old_symbols = set(self.pair_mgr.symbols)
        await self.pair_mgr.refresh_pairlist()
        now_symbols = self.pair_mgr.symbols
        await self.wallets.init(now_symbols)
        # 移除已删除的交易对
        del_symbols = list(old_symbols.difference(now_symbols))
        if del_symbols:
            logger.info(f"remove symbols: {del_symbols}")
            await self.exchange.cancel_open_orders(del_symbols)
            self.order_mgr.exit_open_orders(dict(tag='pair_del'), pairs=del_symbols, od_dir='both')
            await self.data_mgr.unsub_pairs(del_symbols)
        # 处理新增的交易对
        add_symbols = list(set(now_symbols).difference(old_symbols))
        if add_symbols:
            logger.info(f"listen new symbols: {add_symbols}")
            pair_tfs = self._load_strategies(add_symbols, self.pair_mgr.pair_tfscores)
            await self.data_mgr.sub_warm_pairs(pair_tfs)

    async def loop_refresh_pairs(self):
        refresh_intv = self.pair_mgr.refresh_secs
        if not refresh_intv:
            return
        while True:
            await asyncio.sleep(refresh_intv)
            try:
                with db():
                    await self.refresh_pairs()
            except Exception:
                logger.exception('loop refresh pairs error')

    async def cleanup(self):
        await self.order_mgr.cleanup()
        await self.rpc.send_msg(dict(
            type=RPCMessageType.STATUS,
            status='Bot stopped'
        ))
        await self.rpc.cleanup()
        await self.exchange.close()
