#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : whack_mole.py
# Author: anyongjin
# Date  : 2023/2/28
from banbot.main.itrader import *
from banbot.main.od_manager import *
from banbot.rpc import Notify, NotifyType, start_api
from banbot.storage import *
from banbot.util import btime
from banbot.util.misc import *
from banbot.config import Config
from banbot.main.wallets import CryptoWallet
from banbot.util.common import logger


class LiveTrader(Trader):
    '''
    实盘交易、实时模拟。模式:DRY_RUN PROD
    '''

    def __init__(self, config: Config):
        super(LiveTrader, self).__init__(config)
        self.data_mgr = self._init_data_mgr()
        self.wallets = CryptoWallet(config, self.exchange)
        self.order_mgr = LiveOrderMgr.init(config, self.exchange, self.wallets, self.data_mgr, self.order_callback)
        self.loop = None

    def order_callback(self, od: InOutOrder, is_enter: bool):
        msg_type = NotifyType.ENTRY if is_enter else NotifyType.EXIT
        sub_od = od.enter if is_enter else od.exit
        if sub_od.status != OrderStatus.Close or not sub_od.amount:
            return
        if is_enter:
            action = '开空' if od.short else '开多'
        else:
            action = '平空' if od.short else '平多'
        Notify.send(
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

    def _init_data_mgr(self):
        from banbot.data.ws import LiveWSProvider
        from banbot.data.provider import LiveDataProvider
        if self.is_ws_mode():
            return LiveWSProvider(self.config, self.on_pair_trades)
        return LiveDataProvider(self.config, self.on_data_feed)

    async def init(self):
        BotGlobal.set_loop(asyncio.get_running_loop())
        from banbot.data.toolbox import sync_timeframes
        await self.exchange.load_markets()
        async with dba():
            await BotTask.init()
            await sync_timeframes()
            await ExSymbol.init()
            # 先更新所有未平仓订单的状态
            old_num, new_num, del_num, open_ods = await self.order_mgr.sync_orders_with_exg()
            add_pairs = {od.symbol for od in open_ods if od.timeframe}
            await self.pair_mgr.refresh_pairlist(add_pairs)
            cur_pairs = self.pair_mgr.symbols
            await self.wallets.init(cur_pairs)
            await self.exchange.init(cur_pairs)
            pair_tfs = self._load_strategies(cur_pairs, self.pair_mgr.pair_tfscores)
            await self._init_orders(open_ods)
            await self._init_strategies()
            await self.data_mgr.sub_warm_pairs(pair_tfs)
        Notify.startup_msg()
        Notify.send(
            type=NotifyType.STATUS,
            status=f'订单同步：恢复{old_num}，删除{del_num}，新增{new_num}，已开启{len(open_ods)}单'
        )

    async def _init_orders(self, open_ods: List[InOutOrder]):
        sess: SqlSession = dba.session
        for od in open_ods:
            if od.timeframe:
                continue
            tf = next((p[2] for p in BotGlobal.stg_symbol_tfs if p[0] == od.strategy and p[1] == od.symbol), None)
            if not tf:
                logger.warning(f'Order Trace canceled, job not found: {od}')
                await sess.delete(od)
            else:
                od.timeframe = tf

    async def _init_strategies(self):
        open_ods = await InOutOrder.open_orders()
        for stg_name, pair, tf in BotGlobal.stg_symbol_tfs:
            stg_list = BotGlobal.pairtf_stgs[f'{pair}_{tf}']
            stg = next((s for s in stg_list if s.name == stg_name), None)
            if not stg:
                logger.error(f'stg not found: {stg_name}')
                continue
            cur_ods = [od for od in open_ods if od.symbol == pair and od.strategy == stg_name]
            if not cur_ods:
                continue
            stg.enter_num = len(cur_ods)
            for od in cur_ods:
                od.timeframe = tf

    async def run(self):
        self.start_heartbeat_check(3)
        self.loop = asyncio.get_running_loop()
        # 监听实时数据推送，这里要先连接，因为init中需要分布式锁
        self._run_tasks.append(asyncio.create_task(self.data_mgr.loop_main()))
        # 初始化
        await self.init()
        BotGlobal.state = BotState.RUNNING
        # 启动restapi
        start_api(self)
        # 启动异步任务
        await self._start_tasks()
        # 轮训执行任务
        await self._loop_tasks([
            # 两小时更新一次货币行情信息
            [self.exchange.load_markets, 7200, 7200],
            # 定时检查整体损失是否触发限制
            [self.order_mgr.check_fatal_stop, 300, 300]
        ])

    async def _start_tasks(self):
        # 定期刷新交易对
        self._run_tasks.append(asyncio.create_task(self.loop_refresh_pairs()))
        # 执行事件
        self._run_tasks.append(asyncio.create_task(BanEvent.run_forever()))
        # 监听交易对变化
        BanEvent.on('set_pairs', self.add_del_pairs, with_db=True)
        # 监听K线是否超时
        self._run_tasks.append(asyncio.create_task(BotCache.run_bar_waiter()))
        # 定期输出收到K线概况
        self._run_tasks.append(asyncio.create_task(BotCache.run_bar_summary()))
        if btime.prod_mode():
            # 仅实盘交易模式，监听钱包和订单状态更新
            self._run_tasks.extend([
                # 监听钱包更新
                asyncio.create_task(self.wallets.watch_balance_forever()),
                # 跟踪监听未成交订单，及时更新价格确保成交
                asyncio.create_task(self.order_mgr.trail_unfill_orders_forever()),
                # 跟踪账户杠杆倍数和保证金配置
                asyncio.create_task(self.order_mgr.watch_leverage_forever()),
                # 订单异步消费队列
                asyncio.create_task(self.order_mgr.consume_order_queue()),
                # 监听交易所用户订单，更新本地订单状态
                asyncio.create_task(self.order_mgr.watch_my_exg_trades()),
                # 处理未匹配订单，跟踪用户下单
                asyncio.create_task(self.order_mgr.trail_unmatches_forever()),
            ])
            logger.info('listen websocket , watch wallets and order updates ...')

    async def loop_refresh_pairs(self):
        reset_ctx()
        while True:
            wait_secs = self.pair_mgr.get_refresh_wait()
            if not wait_secs:
                return
            await Sleeper.sleep(wait_secs)
            if BotGlobal.state != BotState.RUNNING:
                break
            await self.refresh_pairs()

    async def cleanup(self):
        await Sleeper.cancel_all()
        await self.order_mgr.cleanup()
        Notify.send(type=NotifyType.STATUS, status='Bot stopped')
        await Notify.cleanup()
        await self.exchange.close()
