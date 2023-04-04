#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : itrader.py
# Author: anyongjin
# Date  : 2023/3/17
from __future__ import annotations

from banbot.storage.common import *
from banbot.strategy.base import *
from banbot.storage.od_manager import *
from banbot.data.data_provider import *
from banbot.strategy.resolver import load_run_jobs


class Trader:
    def __init__(self, config: Config):
        BotGlobal.state = BotState.RUNNING
        self.config = config
        self.pairlist: List[Tuple[str, str]] = config.get('pairlist')
        self.wallets: WalletsLocal = None
        self.order_hold: OrderManager = None
        self.data_hold: DataProvider = None
        self.symbol_stgs: Dict[str, List[BaseStrategy]] = dict()
        self._warmup_num = 0
        self._job_exp_end = btime.time() + 5
        self._run_tasks: List[asyncio.Task] = []
        self._load_strategies()

    def _load_strategies(self):
        run_jobs = load_run_jobs(self.config)
        for pair, timeframe, cls_list in run_jobs:
            symbol = f'{pair}/{timeframe}'
            set_context(symbol)
            self.symbol_stgs[symbol] = [cls(self.config) for cls in cls_list]
            self._warmup_num = max(self._warmup_num, *[cls.warmup_num for cls in cls_list])

    async def _pair_row_callback(self, pair, timeframe, row):
        set_context(f'{pair}/{timeframe}')
        await self.on_data_feed(np.array(row))

    async def on_data_feed(self, row: np.ndarray):
        strategy_list = self.symbol_stgs[symbol_tf.get()]
        pair, base_s, quote_s, timeframe = get_cur_symbol()
        pair_arr = append_new_bar(row)
        await self._bar_callback()
        self.order_hold.update_by_bar(pair_arr)
        start_time = time.time()
        ext_tags: Dict[str, str] = dict()
        enter_list, exit_list = [], []
        for strategy in strategy_list:
            stg_name = strategy.name
            strategy.state = dict()
            strategy.on_bar(pair_arr)
            # 调用策略生成入场和出场信号
            entry_tag = strategy.on_entry(pair_arr)
            exit_tag = strategy.on_exit(pair_arr)
            if entry_tag and (not strategy.skip_enter_on_exit or not exit_tag):
                cost = strategy.custom_cost(entry_tag)
                enter_list.append((stg_name, entry_tag, cost))
            if not strategy.skip_exit_on_enter or not entry_tag:
                if exit_tag:
                    exit_list.append((stg_name, exit_tag))
                ext_tags.update(self.order_hold.calc_custom_exits(pair_arr, strategy))
        calc_end = time.time()
        calc_cost = (calc_end - start_time) * 1000
        if calc_cost >= 1:
            logger.trade_info(f'calc with {len(strategy_list)} strategies, cost: {calc_cost:.1f} ms')
        if enter_list or exit_list or ext_tags:
            logger.trade_info(f'bar tags: {enter_list}  {exit_list}  {ext_tags}')
            enter_ods, exit_ods = await self.order_hold.enter_exit_pair_orders(pair, enter_list, exit_list, ext_tags)
            if enter_ods or exit_ods:
                post_cost = (time.time() - calc_end) * 1000
                logger.trade_info(f'enter: {len(enter_ods)} exit: {len(exit_ods)} cost: {post_cost:.1f} ms')

    async def _bar_callback(self):
        pass

    async def run(self):
        raise NotImplementedError('`run` is not implemented')

    async def cleanup(self):
        pass

    def start_heartbeat_check(self, min_intv: float):
        from threading import Thread

        def handle():
            while True:
                time.sleep(min_intv * 0.3)
                if self._job_exp_end < btime.time():
                    logger.error('check loop tasks heartbeat fail, task stucked')
                    time.sleep(30)

        Thread(target=handle, daemon=True).start()

    async def _loop_tasks(self, biz_list):
        '''
        这里不能执行耗时的异步任务（比如watch_balance），
        :return:
        '''
        live_mode = btime.run_mode in TRADING_MODES
        while BotGlobal.state == BotState.RUNNING:
            wait_list = sorted(biz_list, key=lambda x: x[2])
            biz_func, interval, next_start = wait_list[0]
            wait_secs = next_start - btime.time()
            self._job_exp_end = next_start + interval * 2
            func_name = biz_func.__qualname__
            if wait_secs > 0:
                if wait_secs > 30:
                    logger.info(f'sleep {wait_secs} : {func_name}')
                await btime.sleep(wait_secs)
            job_start = time.time()
            try:
                await run_async(biz_func)
            except EOFError:
                # 任意一个发出EOF错误时，终止循环
                break
            exec_cost = time.time() - job_start
            if live_mode and exec_cost >= interval * 0.9 and not is_debug():
                logger.warning(f'{func_name} cost {exec_cost:.3f} > interval: {interval:.3f}')
                interval = exec_cost * 1.5
                wait_list[0][1] = interval
            wait_list[0][2] += interval
