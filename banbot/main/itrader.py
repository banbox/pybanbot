#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : itrader.py
# Author: anyongjin
# Date  : 2023/3/17
from __future__ import annotations

from banbot.compute.tools import append_new_bar
from banbot.main.od_manager import *
from banbot.data.provider import *
from banbot.strategy.resolver import StrategyResolver


class Trader:
    def __init__(self, config: Config):
        BotGlobal.state = BotState.RUNNING
        self.config = config
        self.name = config.get('name', 'noname')
        if btime.run_mode in TRADING_MODES:
            logger.info('started bot:   >>>  %s  <<<', self.name)
        self.wallets: WalletsLocal = None
        self.order_mgr: OrderManager = None
        self.data_mgr: DataProvider = None
        self.symbol_stgs: Dict[str, List[BaseStrategy]] = dict()
        self._job_exp_end = btime.time() + 5
        self._run_tasks: List[asyncio.Task] = []

    def _load_strategies(self, pairlist: List[str]) -> Dict[Tuple[str, int], Set[str]]:
        run_jobs = StrategyResolver.load_run_jobs(self.config, pairlist)
        pair_tfs = dict()
        for pair, (warm_secs, tf_dic) in run_jobs.items():
            for timeframe, stg_set in tf_dic.items():
                res_key = pair, warm_secs
                if res_key not in pair_tfs:
                    pair_tfs[res_key] = set()
                pair_tfs[res_key].add(timeframe)
                symbol = f'{pair}/{timeframe}'
                with TempContext(symbol):
                    self.symbol_stgs[symbol] = [cls(self.config) for cls in stg_set]
        return pair_tfs

    def on_data_feed(self, pair, timeframe, row: list):
        logger.debug('data_feed %s %s %s', pair, timeframe, row)
        row = np.array(row)
        pair_tf = f'{pair}/{timeframe}'
        tf_secs = tf_to_secs(timeframe)
        # 超过1分钟或周期的一半，认为bar延迟，不可下单
        delay = btime.time() - (row[0] // 1000 + tf_secs)
        bar_expired = delay >= max(60., tf_secs * 0.5)
        is_live_mode = btime.run_mode == RunMode.LIVE
        if bar_expired and is_live_mode:
            logger.warning(f'{pair}/{timeframe} delay {delay:.2}s, enter order is disabled')
        with TempContext(pair_tf):
            # 策略计算部分，会用到上下文变量
            strategy_list = self.symbol_stgs[pair_tf]
            pair_arr = append_new_bar(row, tf_secs)
            self.order_mgr.update_by_bar(pair_arr)
            start_time = time.monotonic()
            ext_tags: Dict[int, str] = dict()
            enter_list, exit_list = [], []
            for strategy in strategy_list:
                stg_name = strategy.name
                strategy.state = dict()
                strategy.on_bar(pair_arr)
                # 调用策略生成入场和出场信号
                entry_tag = strategy.on_entry(pair_arr)
                exit_tag = strategy.on_exit(pair_arr)
                if entry_tag and not bar_expired and (not strategy.skip_enter_on_exit or not exit_tag):
                    cost = strategy.custom_cost(entry_tag)
                    enter_list.append((stg_name, entry_tag, cost))
                if not strategy.skip_exit_on_enter or not entry_tag:
                    if exit_tag:
                        exit_list.append((stg_name, exit_tag))
                    ext_tags.update(self.order_mgr.calc_custom_exits(pair_arr, strategy))
            calc_end = time.monotonic()
        calc_cost = (calc_end - start_time) * 1000
        if calc_cost >= 10 and btime.run_mode in TRADING_MODES:
            logger.info('calc with {0} strategies, cost: {1:.1f} ms', len(strategy_list), calc_cost)
        if not is_live_mode:
            # 模拟模式，填充未成交订单
            self.order_mgr.fill_pending_orders(pair, timeframe, row)
        if enter_list or exit_list or ext_tags:
            logger.debug('bar signals: %s %s %s', enter_list, exit_list, ext_tags)
            self.order_mgr.process_pair_orders(pair_tf, enter_list, exit_list, ext_tags)
        return enter_list, exit_list, ext_tags

    async def run(self):
        raise NotImplementedError('`run` is not implemented')

    async def cleanup(self):
        pass

    def start_heartbeat_check(self, min_intv: float):
        from threading import Thread

        def handle():
            time.sleep(5)
            while True:
                time.sleep(min_intv * 0.3)
                if self._job_exp_end < btime.time():
                    logger.error('check loop tasks heartbeat fail, task stucked')
                    time.sleep(30)

        Thread(target=handle, daemon=True).start()

    async def _loop_tasks(self, biz_list: List[List[Callable, float, float]]):
        '''
        这里不能执行耗时的异步任务（比如watch_balance）最好单次执行时长不超过1s。
        :param biz_list: [(func, interval, start_delay), ...]
        :return:
        '''
        # 将第三个参数改为期望下次执行时间
        cur_time = btime.time()
        for job in biz_list:
            job[2] += cur_time
        # 轮询执行任务
        while BotGlobal.state == BotState.RUNNING:
            live_mode = btime.run_mode in TRADING_MODES
            wait_list = sorted(biz_list, key=lambda x: x[2])
            biz_func, interval, next_start = wait_list[0]
            wait_secs = next_start - btime.time()
            self._job_exp_end = next_start + interval * 2
            func_name = biz_func.__qualname__
            if wait_secs > 0:
                await asyncio.sleep(wait_secs)
            job_start = time.monotonic()
            # 执行任务
            await run_async(biz_func)
            exec_cost = time.monotonic() - job_start
            if live_mode and exec_cost >= interval * 0.9 and not is_debug():
                logger.warning('loop task timeout {0} cost {1:.3f} > {2:.3f}', func_name, exec_cost, interval)
            wait_list[0][2] += interval
