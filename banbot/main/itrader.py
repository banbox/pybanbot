#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : itrader.py
# Author: anyongjin
# Date  : 2023/3/17
from __future__ import annotations

import time

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

    def _load_strategies(self):
        run_jobs = load_run_jobs(self.config)
        for pair, timeframe, cls_list in run_jobs:
            symbol = f'{pair}/{timeframe}'
            set_context(symbol)
            self.symbol_stgs[symbol] = [cls(self.config) for cls in cls_list]

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
            stg_name = strategy.__class__.__name__
            strategy.state = dict()
            strategy.on_bar(pair_arr)
            # 调用策略生成入场和出场信号
            entry_tag = strategy.on_entry(pair_arr)
            exit_tag = strategy.on_exit(pair_arr)
            if entry_tag or exit_tag:
                logger.trade_info(f'[{stg_name}] enter: {entry_tag}  exit: {exit_tag}')
            if entry_tag and (not strategy.skip_enter_on_exit or not exit_tag):
                cost = strategy.custom_cost(entry_tag)
                enter_list.append((stg_name, entry_tag, cost))
            if not strategy.skip_exit_on_enter or not entry_tag:
                if exit_tag:
                    exit_list.append((stg_name, exit_tag))
                ext_tags.update(self.order_hold.calc_custom_exits(pair_arr, strategy))
        calc_end = time.time()
        calc_cost = calc_end - start_time
        if btime.run_mode in btime.TRADING_MODES:
            logger.info(f'{pair} {timeframe} calc with {len(strategy_list)} strategies, cost: {calc_cost:.f} ms')
        if enter_list or exit_list or ext_tags:
            await self.order_hold.enter_exit_pair_orders(pair, enter_list, exit_list, ext_tags)
        cost = time.time() - calc_end
        if cost > 0.001 and btime.run_mode in btime.TRADING_MODES:
            logger.info(f'handle bar {bar_end_time.get()} cost: {cost * 1000:.1f} ms')

    async def _bar_callback(self):
        pass

    def run(self):
        raise NotImplementedError('`run` is not implemented')

    def cleanup(self):
        pass

    async def _loop_tasks(self, biz_list):
        '''
        这里不能执行耗时的异步任务（比如watch_balance），
        :return:
        '''
        live_mode = btime.run_mode in TRADING_MODES
        while True:
            wait_list = sorted(biz_list, key=lambda x: x[2])
            biz_func, interval, next_start = wait_list[0]
            wait_secs = next_start - btime.time()
            if wait_secs > 0:
                await btime.sleep(wait_secs)
            start_time = time.time()
            try:
                await run_async(biz_func)
            except EOFError:
                # 任意一个发出EOF错误时，终止循环
                break
            exec_cost = time.time() - start_time
            if live_mode and exec_cost >= interval * 0.9 and not is_debug():
                logger.warning(f'{biz_func.__qualname__} cost {exec_cost:.3f} > interval: {interval:.3f}')
                interval = exec_cost * 1.5
                wait_list[0][1] = interval
            wait_list[0][2] += interval
