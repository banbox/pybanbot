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

    def _load_strategies(self):
        run_jobs = load_run_jobs(self.config)
        for pair, timeframe, cls_list in run_jobs:
            symbol = f'{pair}/{timeframe}'
            set_context(symbol)
            self.symbol_stgs[symbol] = [cls(self.config) for cls in cls_list]

    def _make_invoke(self):
        def invoke_pair(pair, timeframe, row):
            set_context(f'{pair}/{timeframe}')
            self.on_data_feed(np.array(row))
        return invoke_pair

    def on_data_feed(self, row: np.ndarray):
        strategy_list = self.symbol_stgs[symbol_tf.get()]
        pair, base_s, quote_s, timeframe = get_cur_symbol()
        pair_arr = append_new_bar(row)
        self._bar_callback()
        self.order_hold.update_by_bar(pair_arr)
        start_time = time.time()
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
                price = pair_arr[-1][ccol]
                self.order_hold.enter_order(stg_name, pair, entry_tag, cost, price)
            if not strategy.skip_exit_on_enter or not entry_tag:
                if exit_tag:
                    self.order_hold.exit_open_orders(stg_name, pair, exit_tag)
                self.order_hold.check_custom_exits(pair_arr, strategy)
        cost = time.time() - start_time
        if cost > 0.05 and btime.run_mode in btime.TRADING_MODES:
            logger.info(f'handle bar {bar_end_time.get()} cost: {cost * 1000:.1f} ms')

    def _bar_callback(self):
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
                if live_mode:
                    await asyncio.sleep(wait_secs)
                else:
                    btime.add_secs(wait_secs)
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(biz_func):
                    await biz_func()
                else:
                    biz_func()
            except EOFError:
                # 任意一个发出EOF错误时，终止循环
                break
            exec_cost = time.time() - start_time
            if live_mode and exec_cost >= interval * 0.9 and not is_debug():
                logger.warning(f'{biz_func.__qualname__} cost {exec_cost:.3f} > interval: {interval:.3f}')
                interval = exec_cost * 1.5
                wait_list[0][1] = interval
            wait_list[0][2] += interval
