#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : whack_mole.py
# Author: anyongjin
# Date  : 2023/2/28
from banbot.main.itrader import *
from banbot.main.od_manager import *
from banbot.rpc import Notify, NotifyType, start_api
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

    async def init(self):
        BotGlobal.bot_loop = asyncio.get_running_loop()
        from banbot.data.toolbox import sync_timeframes
        await self.exchange.load_markets()
        async with dba():
            await BotTask.init()
            await sync_timeframes()
            await ExSymbol.init()
            # 先更新所有未平仓订单的状态
            old_num, new_num, del_num, open_ods = await self.order_mgr.sync_orders_with_exg()
            add_pairs = {od.symbol for od in open_ods}
            await self.pair_mgr.refresh_pairlist(add_pairs)
            cur_pairs = self.pair_mgr.symbols
            await self.wallets.init(cur_pairs)
            await self.exchange.init(cur_pairs)
            pair_tfs = self._load_strategies(cur_pairs, self.pair_mgr.pair_tfscores)
            await self._init_strategies()
            logger.info(f'warm pair_tfs: {pair_tfs}')
            await self.data_mgr.sub_warm_pairs(pair_tfs)
        Notify.startup_msg()
        Notify.send(
            type=NotifyType.STATUS,
            status=f'订单同步：恢复{old_num}，删除{del_num}，新增{new_num}，已开启{len(open_ods)}单'
        )

    async def _init_strategies(self):
        open_ods = await InOutOrder.open_orders()
        for stg_name, pair, tf in BotGlobal.stg_symbol_tfs:
            cur_ods = [od for od in open_ods if od.symbol == pair and od.timeframe == tf]
            if not cur_ods:
                continue
            stg_list = BotGlobal.pairtf_stgs[f'{pair}_{tf}']
            stg = next((s for s in stg_list if s.name == stg_name), None)
            if not stg:
                logger.error(f'stg not found: {stg_name}')
                continue
            stg.enter_num = len(cur_ods)

    async def run(self):
        self.start_heartbeat_check(3)
        self.loop = asyncio.get_running_loop()
        # 初始化
        await self.init()
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
        # 监听实时数据推送
        self._run_tasks.append(asyncio.create_task(LiveDataProvider.run()))
        # 定期刷新交易对
        self._run_tasks.append(asyncio.create_task(self.loop_refresh_pairs()))
        if btime.prod_mode():
            # 仅实盘交易模式，监听钱包和订单状态更新
            self._run_tasks.extend([
                # K线延迟预警，预期时间内未收到发出错误
                asyncio.create_task(self.data_mgr.run_checks_forever()),
                # 监听钱包更新
                asyncio.create_task(self.wallets.watch_balance_forever()),
                # 跟踪监听未成交订单，及时更新价格确保成交
                asyncio.create_task(self.order_mgr.trail_open_orders_forever()),
                # 跟踪账户杠杆倍数和保证金配置
                asyncio.create_task(self.order_mgr.watch_leverage_forever()),
                # 跟踪所有币的最新价格
                asyncio.create_task(self.order_mgr.watch_price_forever()),
                # 订单异步消费队列
                asyncio.create_task(self.order_mgr.consume_queue()),
                # 监听订单更新
                asyncio.create_task(self.order_mgr.listen_orders_forever()),
                # 处理未匹配订单，跟踪用户下单
                asyncio.create_task(self.order_mgr.trail_unmatches_forever()),
            ])
            logger.info('listen websocket , watch wallets and order updates ...')

    async def refresh_pairs(self):
        '''
        定期刷新交易对
        '''
        logger.info("start refreshing symbols")
        old_symbols = set(self.pair_mgr.symbols)
        await self.pair_mgr.refresh_pairlist()
        await self.add_del_pairs(old_symbols)

    async def add_del_pairs(self, old_symbols: Set[str]):
        now_symbols = set(self.pair_mgr.symbols)
        BotGlobal.pairs = now_symbols
        await self.wallets.init(now_symbols)
        del_symbols = list(old_symbols.difference(now_symbols))
        add_symbols = list(now_symbols.difference(old_symbols))
        # 检查删除的交易对是否有订单，有则添加回去
        if del_symbols:
            open_ods = await InOutOrder.open_orders(pairs=del_symbols)
            for od in open_ods:
                if od.symbol in del_symbols:
                    del_symbols.remove(od.symbol)
                    self.pair_mgr.symbols.append(od.symbol)
            if del_symbols:
                logger.info(f"remove symbols: {del_symbols}")
                await self.data_mgr.unsub_pairs(del_symbols)
        # 处理新增的交易对
        if add_symbols:
            calc_keys = [s for s in add_symbols if s not in self.pair_mgr.pair_tfscores]
            if calc_keys:
                # 如果是rpc添加的，这里需要计算tfscores
                from banbot.symbols.tfscaler import calc_symboltf_scales
                tfscores = await calc_symboltf_scales(self.exchange, calc_keys)
                self.pair_mgr.pair_tfscores.update(**tfscores)
            logger.info(f"listen new symbols: {add_symbols}")
            pair_tfs = self._load_strategies(add_symbols, self.pair_mgr.pair_tfscores)
            await self.data_mgr.sub_warm_pairs(pair_tfs)

    async def loop_refresh_pairs(self):
        reset_ctx()
        while True:
            wait_secs = self.pair_mgr.get_refresh_wait()
            if not wait_secs:
                return
            await asyncio.sleep(wait_secs)
            try:
                async with dba():
                    await self.refresh_pairs()
            except Exception:
                logger.exception('loop refresh pairs error')

    async def cleanup(self):
        await self.order_mgr.cleanup()
        Notify.send(type=NotifyType.STATUS, status='Bot stopped')
        await Notify.cleanup()
        await self.exchange.close()
