#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : whack_mole.py
# Author: anyongjin
# Date  : 2023/2/28
from banbot.main.itrader import *
from banbot.main.od_manager import *
from banbot.rpc.notify_mgr import Notify, NotifyType
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
        self.rpc = Notify(config)

    def order_callback(self, od: InOutOrder, is_enter: bool):
        msg_type = NotifyType.ENTRY if is_enter else NotifyType.EXIT
        sub_od = od.enter if is_enter else od.exit
        if sub_od.status != OrderStatus.Close:
            return
        if is_enter:
            action = '开空' if od.short else '开多'
        else:
            action = '平空' if od.short else '平多'
        msg = dict(
            type=msg_type,
            action=action,
            enter_tag=od.enter_tag,
            exit_tag=od.exit_tag,
            side=sub_od.side,
            short=od.short,
            leverage=od.leverage,
            amount=sub_od.amount,
            price=sub_od.price,
            value=sub_od.amount * sub_od.price,
            cost=sub_od.amount * sub_od.price / od.leverage,
            strategy=od.strategy,
            pair=od.symbol,
            timeframe=od.timeframe,
            profit=od.profit,
            profit_rate=od.profit_rate,
            **(od.infos or dict())
        )
        asyncio.create_task(self.rpc.send_msg(msg))

    async def init(self):
        from banbot.data.toolbox import sync_timeframes
        await self.exchange.load_markets()
        with db():
            BotTask.init()
            sync_timeframes()
            await ExSymbol.fill_list_dts()
            # 先更新所有未平仓订单的状态
            old_num, new_num, del_num, open_ods = await self.order_mgr.sync_orders_with_exg()
            add_pairs = {od.symbol for od in open_ods}
            await self.pair_mgr.refresh_pairlist(add_pairs)
            cur_pairs = self.pair_mgr.symbols
            await self.wallets.init(cur_pairs)
            await self.exchange.init(cur_pairs)
            pair_tfs = self._load_strategies(cur_pairs, self.pair_mgr.pair_tfscores)
            logger.info(f'warm pair_tfs: {pair_tfs}')
            await self.data_mgr.sub_warm_pairs(pair_tfs)
        await self.rpc.startup_messages()
        await self.rpc.send_msg(dict(
            type=NotifyType.STATUS,
            status=f'订单同步：恢复{old_num}，删除{del_num}，新增{new_num}，已开启{len(open_ods)}单'
        ))

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
            # 更新所有货币的价格 5s一次
            [self.exchange.update_prices, 5, 0],
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
                asyncio.create_task(self.wallets.watch_balance_forever()),
                # 监听订单更新
                asyncio.create_task(self.order_mgr.listen_orders_forever()),
                # 跟踪监听未成交订单，及时更新价格确保成交
                asyncio.create_task(self.order_mgr.trail_open_orders_forever()),
                # 处理未匹配订单，跟踪用户下单
                asyncio.create_task(self.order_mgr.trail_unmatches_forever()),
                # 跟踪账户杠杆倍数和保证金配置
                asyncio.create_task(self.order_mgr.watch_leverage_forever()),
                # 订单异步消费队列
                asyncio.create_task(self.order_mgr.consume_queue()),
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
            ext_dic = dict(tag=ExitTags.pair_del)
            exit_ods = self.order_mgr.exit_open_orders(ext_dic, pairs=del_symbols, od_dir='both', is_force=True)
            logger.info(f'exit orders: {exit_ods}')
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
            type=NotifyType.STATUS,
            status='Bot stopped'
        ))
        await self.rpc.cleanup()
        await self.exchange.close()
