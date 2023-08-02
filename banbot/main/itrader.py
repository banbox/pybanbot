#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : itrader.py
# Author: anyongjin
# Date  : 2023/3/17
from __future__ import annotations

from banbot.compute.tools import append_new_bar
from banbot.main.od_manager import *
from banbot.strategy.resolver import StrategyResolver


class Trader:
    def __init__(self, config: Config):
        BotGlobal.state = BotState.RUNNING
        self.config = config
        self.name = config.get('name', 'noname')
        if btime.prod_mode():
            logger.info('started bot:   >>>  %s  <<<', self.name)
        self.wallets: WalletsLocal = None
        self.order_mgr: OrderManager = None
        self.data_mgr: DataProvider = None
        self.symbol_stgs: Dict[str, List[BaseStrategy]] = dict()
        self._job: Tuple[str, float, float] = None
        self._run_tasks: List[asyncio.Task] = []

    def _load_strategies(self, pairlist: List[str], pair_tfscores: Dict[str, List[Tuple[str, float]]])\
            -> Dict[str, Dict[str, int]]:
        run_jobs = StrategyResolver.load_run_jobs(self.config, pairlist, pair_tfscores)
        if not run_jobs:
            raise ValueError('no run jobs found')
        pair_tfs = dict()
        stg_pairs = []
        for pair, timeframe, warm_num, stg_set in run_jobs:
            if pair not in pair_tfs:
                pair_tfs[pair] = dict()
            pair_tfs[pair][timeframe] = warm_num
            symbol = f'{self.data_mgr.exg_name}_{self.data_mgr.market}_{pair}_{timeframe}'
            with TempContext(symbol):
                self.symbol_stgs[symbol] = [cls(self.config) for cls in stg_set]
                stg_pairs.extend([(cls.__name__, pair, timeframe) for cls in stg_set])
        from itertools import groupby
        stg_pairs = sorted(stg_pairs, key=lambda x: x[:2])
        sp_groups = groupby(stg_pairs, key=lambda x: x[0])
        for key, gp in sp_groups:
            items = ' '.join([f'{it[1]}/{it[2]}' for it in gp])
            logger.info(f'{key}: {items}')
        return pair_tfs

    def on_data_feed(self, pair, timeframe, row: list):
        pair_tf = f'{self.data_mgr.exg_name}_{self.data_mgr.market}_{pair}_{timeframe}'
        if not BotGlobal.is_wramup:
            logger.debug('data_feed %s %s %s', pair, timeframe, row)
        tf_secs = tf_to_secs(timeframe)
        # 超过1分钟或周期的一半，认为bar延迟，不可下单
        delay = btime.time() - (row[0] // 1000 + tf_secs)
        bar_expired = delay >= max(60., tf_secs * 0.5)
        is_live_mode = btime.run_mode == RunMode.PROD
        if bar_expired and is_live_mode and not BotGlobal.is_wramup:
            logger.warning(f'{pair}/{timeframe} delay {delay:.2}s, enter order is disabled')
        with TempContext(pair_tf):
            # 策略计算部分，会用到上下文变量
            strategy_list = self.symbol_stgs[pair_tf]
            pair_arr = append_new_bar(row, tf_secs)
            if not BotGlobal.is_wramup:
                self.order_mgr.update_by_bar(row)
            start_time = time.monotonic()
            ext_tags: Dict[int, dict] = dict()
            enter_list, exit_list = [], []
            for strategy in strategy_list:
                stg_name = strategy.name
                strategy.on_bar(pair_arr)
                if not BotGlobal.is_wramup:
                    # 调用策略生成入场和出场信号
                    sigin = strategy.on_entry(pair_arr)
                    sigout = strategy.on_exit(pair_arr)
                    if sigin and not bar_expired and (not strategy.skip_enter_on_exit or not sigout):
                        if 'legal_cost' not in sigin:
                            sigin['legal_cost'] = strategy.custom_cost(sigin)
                        enter_list.append((stg_name, sigin))
                    if not strategy.skip_exit_on_enter or not sigin:
                        if sigout:
                            exit_list.append((stg_name, sigout))
                        ext_tags.update(self.order_mgr.calc_custom_exits(pair_arr, strategy))
            calc_cost = (time.monotonic() - start_time) * 1000
            if calc_cost >= 20 and btime.run_mode in LIVE_MODES:
                logger.info('{2} calc with {0} strategies at {3}, cost: {1:.1f} ms',
                            len(strategy_list), calc_cost, symbol_tf.get(), bar_num.get())
        ent_ods, ext_ods = self.order_mgr.process_orders(pair_tf, enter_list, exit_list, ext_tags)
        if ent_ods or ext_ods:
            for stgy in strategy_list:
                stgy.update_orders(ent_ods, ext_ods)
        return enter_list, exit_list, ext_tags

    async def run(self):
        raise NotImplementedError('`run` is not implemented')

    async def cleanup(self):
        pass

    def start_heartbeat_check(self, min_intv: float):
        from threading import Thread
        from banbot.util.redis_helper import SyncRedis
        import socket
        import os
        bot_key = f'bot_{self.config["name"]}'
        bot_id = f'{socket.gethostname()}:{os.getpid()}'
        if not self.config['cluster']:
            # 仅单机模式，检查bot是否已启动
            with SyncRedis() as redis:
                old_id = redis.get(bot_key)
                if old_id:
                    raise RuntimeError(f'{bot_key} already running at {old_id}')
                redis.set(bot_key, bot_id, 5)
        slp_intv = min_intv * 0.3
        key_ex_secs = round(slp_intv + 1)

        def handle():
            last_tip_at = 0
            while True:
                time.sleep(slp_intv)
                with SyncRedis() as sredis:
                    sredis.set(bot_key, bot_id, key_ex_secs)
                if self._job:
                    cur_time = btime.time()
                    if self._job[-1] < cur_time and cur_time - last_tip_at > 30:
                        last_tip_at = cur_time
                        start_at = btime.to_datestr(self._job[1])
                        logger.error(f'loop tasks stucked: {self._job[0]}, start at {start_at}')

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
        logger.info('start run loop tasks...')
        while BotGlobal.state == BotState.RUNNING:
            live_mode = btime.run_mode in LIVE_MODES
            wait_list = sorted(biz_list, key=lambda x: x[2])
            biz_func, interval, next_start = wait_list[0]
            wait_secs = next_start - btime.time()
            func_name = biz_func.__qualname__
            if wait_secs > 0:
                await asyncio.sleep(wait_secs)
            cur_time = btime.time()
            self._job = (func_name, cur_time, cur_time + interval * 2)
            job_start = time.monotonic()
            # 执行任务
            await run_async(biz_func)
            exec_cost = time.monotonic() - job_start
            if live_mode and exec_cost >= interval * 0.9 and not is_debug():
                logger.warning('loop task timeout {0} cost {1:.3f} > {2:.3f}', func_name, exec_cost, interval)
            wait_list[0][2] += interval
