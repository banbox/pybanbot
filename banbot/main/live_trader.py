#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : whack_mole.py
# Author: anyongjin
# Date  : 2023/2/28
import asyncio
from banbot.main.itrader import *
from banbot.exchange.crypto_exchange import *
from banbot.data.live_provider import LiveDataProvider
from banbot.storage.od_manager import *
from banbot.util.misc import *
from banbot.util import btime


class LiveTrader(Trader):
    '''
    实盘交易、实时模拟。模式:DRY_RUN LIVE
    '''

    def __init__(self):
        super(LiveTrader, self).__init__()
        self.exchange = CryptoExchange(cfg)
        self.data_hold = LiveDataProvider(self.exchange)
        self.wallets = CryptoWallet(cfg, self.exchange)
        self.order_hold = LiveOrderManager(cfg, self.exchange, self.wallets)
        self.data_hold.set_callback(self._make_invoke())

    def _make_invoke(self):
        def invoke_pair(pair, timeframe, row):
            set_context(f'{pair}/{timeframe}')
            logger.info(f'{pair}/{timeframe} {row}')
            self.on_data_feed(np.array(row))
        return invoke_pair

    async def init(self):
        self._load_strategies()
        await self.exchange.load_markets()
        await self.exchange.update_quote_price()
        await self.wallets.init()
        await init_longvars(self.exchange, self.pairlist)
        logger.info('banbot init complete')

    async def run(self):
        # 初始化
        await self.init()
        # 启动异步任务
        await self._start_tasks()
        # 轮训执行任务
        await self._loop_tasks()

    def _bar_callback(self):
        if btime.run_mode != RunMode.LIVE:
            self.order_hold.fill_pending_orders(bar_arr.get())

    async def _start_tasks(self):
        if btime.run_mode == RunMode.LIVE:
            # 仅实盘交易模式，监听钱包和订单状态更新
            # 监听钱包更新
            asyncio.create_task(self.wallets.update_forever())
            # 监听订单更新
            asyncio.create_task(self.order_hold.update_forever())

    async def _loop_tasks(self):
        '''
        这里不能执行耗时的异步任务（比如watch_balance），
        :return:
        '''
        cur_time = time.time()
        biz_list = [
            # 轮询函数，轮询间隔(s)，下次执行时间
            [self.data_hold.process, self.data_hold.min_interval, cur_time],
            # 定时更新定价货币的价格
            [self.exchange.update_quote_price, 60, cur_time],
            # 定时检查整体损失是否触发限制
            [self.order_hold.check_fatal_stop, 300, cur_time + 3]
        ]
        while True:
            wait_list = sorted(biz_list, key=lambda x: x[2])
            biz_func, interval, next_start = wait_list[0]
            wait_secs = next_start - time.time()
            if wait_secs > 0:
                await asyncio.sleep(wait_secs)
            start_time = time.time()
            if inspect.iscoroutinefunction(biz_func):
                try:
                    future = biz_func()
                    if future:
                        await asyncio.wait_for(future, interval)
                except TimeoutError:
                    raise RuntimeError(f'{biz_func.__qualname__} rum timeout: {interval:.3f}s')
            else:
                biz_func()
            exec_cost = time.time() - start_time
            if exec_cost >= interval * 0.9:
                logger.warning(f'{biz_func.__qualname__} cost {exec_cost:.3f} > interval: {interval:.3f}')
                interval = exec_cost * 1.5
                wait_list[0][1] = interval
            wait_list[0][2] += interval


if __name__ == '__main__':
    btime.run_mode = btime.RunMode(cfg.get('run_mode', 'dry_run'))
    logger.warning(f"Run Mode: {btime.run_mode.value}")
    trader = LiveTrader()
    call_async(trader.run)
