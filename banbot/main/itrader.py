#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : itrader.py
# Author: anyongjin
# Date  : 2023/3/17
from __future__ import annotations
from sqlalchemy import exc
from banbot.compute.tools import append_new_bar
from banbot.main.od_manager import *
from banbot.strategy.resolver import StrategyResolver
from banbot.strategy.base import BaseStrategy
from banbot.data.provider import *
from banbot.storage import *
from banbot.main.wallets import *
from banbot.compute.ctx import *


class Trader:
    def __init__(self, config: Config):
        BotGlobal.state = BotState.INIT
        self.config = config
        self.name = config.get('name', 'noname')
        if btime.prod_mode():
            logger.info('started bot:   >>>  %s  <<<', self.name)
        self.wallets: WalletsLocal = None
        self.order_mgr: OrderManager = None
        self.data_mgr: DataProvider = None
        self._job: Tuple[str, float, float] = None
        self._run_tasks: List[asyncio.Task] = []
        self.last_process = 0
        '上次处理bar的毫秒时间戳，用于判断是否工作正常'
        self.last_check_hang = 0
        '上次检查没有新数据的时间戳'
        self.check_hang_intv = 60
        '全局检查没有新数据的间隔，单位：秒'

    def _load_strategies(self, pairlist: List[str], pair_tfscores: Dict[str, List[Tuple[str, float]]])\
            -> Dict[str, Dict[str, int]]:
        from io import StringIO
        from banbot.symbols.utils import group_symbols
        run_jobs = StrategyResolver.load_run_jobs(self.config, pairlist, pair_tfscores)
        if not run_jobs:
            raise ValueError(f'no run jobs found: {pairlist} {pair_tfscores}')
        pair_tfs = dict()
        for pair, timeframe, warm_num, stg_set in run_jobs:
            if pair not in pair_tfs:
                pair_tfs[pair] = dict()
            pair_tfs[pair][timeframe] = warm_num
            ctx_key = f'{self.data_mgr.exg_name}_{self.data_mgr.market}_{pair}_{timeframe}'
            with TempContext(ctx_key):
                pair_tf_key = f'{pair}_{timeframe}'
                stg_insts = BotGlobal.pairtf_stgs.get(pair_tf_key) or []
                for cls in stg_set:
                    job = (cls.__name__, pair, timeframe)
                    if job in BotGlobal.stg_symbol_tfs:
                        continue
                    stg_insts.append(self._load_stg(cls, pair_tfs, pair, timeframe))
                    BotGlobal.stg_symbol_tfs.append(job)
                BotGlobal.pairtf_stgs[pair_tf_key] = stg_insts
        # 输出生效的任务
        groups = dict()
        for stg, pair, tf in BotGlobal.stg_symbol_tfs:
            warm_num = (pair_tfs.get(pair) or dict()).get(tf) or 0
            gp_key = f'{stg}_{tf}_{warm_num}'
            if gp_key not in groups:
                groups[gp_key] = []
            groups[gp_key].append(pair)
        out_io = StringIO()
        out_io.write('bot runing jobs:\n')
        for key, pairs in groups.items():
            pairs_gp = group_symbols(pairs)
            out_io.write(f'【{key}】\n')
            for quote, codes in pairs_gp.items():
                code_text = ' '.join(codes)
                out_io.write(f'{quote}: {code_text}\n')
        logger.info(out_io.getvalue())
        if BotGlobal.run_tf_secs and BotGlobal.run_tf_secs[0][1]:
            self.check_hang_intv = BotGlobal.run_tf_secs[0][1]
        return pair_tfs

    def _load_stg(self, cls: Type[BaseStrategy], pair_tfs: dict, pair: str, timeframe: str):
        stg_obj = cls(self.config)
        # 恢复配置
        stg_obj.restore_config()
        # 记录策略额外订阅的信息
        for info in cls.pair_infos:
            info_pair = pair if info.pair == '_cur_' else info.pair
            cur_ptf = f"{info_pair}_{info.timeframe}"
            if cur_ptf not in BotGlobal.info_pairtfs:
                BotGlobal.info_pairtfs[cur_ptf] = [stg_obj]
            else:
                BotGlobal.info_pairtfs[cur_ptf].append(stg_obj)
            # 额外订阅数据登记
            if info_pair not in pair_tfs:
                pair_tfs[info_pair] = dict()
            tfwarms = pair_tfs[info_pair]
            if info.timeframe not in tfwarms:
                tfwarms[info.timeframe] = info.warmup_num
            else:
                tfwarms[info.timeframe] = max(info.warmup_num, tfwarms[info.timeframe])
        return stg_obj

    async def on_data_feed(self, pair: str, timeframe: str, row: list):
        if not BotGlobal.is_warmup and btime.run_mode in btime.LIVE_MODES:
            logger.info('data_feed %s %s %s %s', pair, timeframe, btime.to_datestr(row[0]), row)
            self.last_process = btime.utcstamp()
        tf_secs = tf_to_secs(timeframe)
        # 超过1分钟或周期的一半，认为bar延迟，不可下单
        delay = btime.time() - (row[0] // 1000 + tf_secs)
        bar_expired = delay >= max(60., tf_secs * 0.5)
        is_live_mode = btime.run_mode == RunMode.PROD
        if bar_expired and is_live_mode and not BotGlobal.is_warmup:
            logger.warning(f'{pair}/{timeframe} delay {delay:.2}s, enter order is disabled')
        # 更新最新价格
        MarketPrice.set_bar_price(pair, float(row[ccol]))
        async with dba():
            try:
                await self._run_bar(pair, timeframe, row, tf_secs, bar_expired)
            except exc.SQLAlchemyError:
                logger.exception('itrader run_bar SQLAlchemyError %s %s', pair, timeframe)

    async def on_pair_trades(self, pair: str, trades: List[dict]):
        if not BotGlobal.is_warmup and btime.run_mode in btime.LIVE_MODES:
            self.last_process = btime.utcstamp()
        # 更新最新价格
        cur_price = trades[-1]['price']
        MarketPrice.set_new_price(pair, cur_price)
        if pair in BotGlobal.forbid_pairs:
            return
        all_open_ods = list(BotCache.open_ods.values())
        if not BotGlobal.is_warmup:
            tracer = InOutTracer(all_open_ods)
            bar = self.data_mgr.get_latest_ohlcv(pair)
            self.order_mgr.update_by_bar(all_open_ods, pair, 'ws', bar)
            chg_ods = tracer.get_changes()
            if chg_ods or BotCache.updod_at + 60 < btime.time():
                all_open_ods = await self._flush_cache_orders(chg_ods)
        pair_ods = [od for od in all_open_ods if od.symbol == pair]

        enter_list, exit_list = self._run_ws(pair_ods, pair, trades)

        if enter_list or exit_list:
            if not BotGlobal.live_mode:
                await self._apply_signals(pair, enter_list, exit_list, all_open_ods)
            else:
                async with dba():
                    await self._apply_signals(pair, enter_list, exit_list, all_open_ods)

    async def _flush_cache_orders(self, chg_ods: List[InOutOrder]):
        BotCache.updod_at = btime.time()
        if not BotGlobal.live_mode:
            for od in chg_ods:
                od.save_mem()
            open_ods = await InOutOrder.open_orders()
            BotCache.open_ods = {od.id: od for od in open_ods}
            return list(BotCache.open_ods.values())
        async with dba():
            sess: SqlSession = dba.session
            old_keys = BotCache.open_keys()
            open_ods = await InOutOrder.open_orders()
            if chg_ods:
                for od in chg_ods:
                    db_od = await od.attach(sess)
                    await db_od.save()
                await sess.flush()
                open_ods = [od for od in open_ods if od.status < InOutStatus.FullExit]
            BotCache.open_ods = {od.id: od.detach(sess) for od in open_ods}
            BotCache.print_chgs(old_keys, '_flush_cache_orders')
            return list(BotCache.open_ods.values())

    async def _apply_signals(self, pair: str, enter_list, exit_list, all_open_ods):
        ctx_key = f'{self.data_mgr.exg_name}_{self.data_mgr.market}_{pair}_ws'
        await self.order_mgr.process_orders(ctx_key, enter_list, exit_list)
        if self.last_check_hang + self.check_hang_intv < btime.time_ms():
            await self._check_pair_hang(all_open_ods)

    async def _run_bar(self, pair: str, timeframe: str, row: list, tf_secs: int, bar_expired: bool):
        ctx_key = f'{self.data_mgr.exg_name}_{self.data_mgr.market}_{pair}_{timeframe}'
        edit_triggers = []
        with TempContext(ctx_key):
            # 策略计算部分，会用到上下文变量
            pair_tf = f'{pair}_{timeframe}'
            strategy_list = BotGlobal.pairtf_stgs.get(pair_tf) or []
            try:
                pair_arr = append_new_bar(row, tf_secs)
            except ValueError as e:
                logger.info(f'skip invalid bar: {e}')
                return [], [], []
            start_time = time.monotonic()
            enter_list, exit_list = [], []
            if not BotGlobal.is_warmup:
                open_ods = await InOutOrder.open_orders()
                tracer = InOutTracer(open_ods)
                self.order_mgr.update_by_bar(open_ods, pair, timeframe, row)
                await tracer.save()
                open_ods = [od for od in open_ods if od.symbol == pair and od.status < InOutStatus.FullExit]
            else:
                open_ods = await InOutOrder.open_orders(pairs=pair)
            for strategy in strategy_list:
                stg_name = strategy.name
                strategy.init_bar(open_ods)
                strategy.on_bar(pair_arr)
                # 调用策略生成入场和出场信号
                if not bar_expired:
                    enter_list.extend([(stg_name, d) for d in strategy.entrys])
                if not BotGlobal.is_warmup:
                    cur_edits = strategy.check_custom_exits(pair_arr)
                    edit_triggers.extend(cur_edits)
                exit_list.extend([(stg_name, d) for d in strategy.exits])
            calc_cost = (time.monotonic() - start_time) * 1000
            if calc_cost >= 20 and btime.run_mode in LIVE_MODES:
                logger.info('{2} calc with {0} strategies at {3}, cost: {1:.1f} ms',
                            len(strategy_list), calc_cost, symbol_tf.get(), bar_num.get())
            # 更新辅助订阅的数据
            if pair_tf in BotGlobal.info_pairtfs:
                for stg in BotGlobal.info_pairtfs[pair_tf]:
                    stg.on_info_bar(pair, timeframe, pair_arr)
        if not BotGlobal.is_warmup:
            edit_tgs = list(set(edit_triggers))
            await self.order_mgr.process_orders(ctx_key, enter_list, exit_list, edit_tgs)
        if self.last_check_hang + self.check_hang_intv < btime.time_ms():
            await self._check_pair_hang()
        return enter_list, exit_list

    def _run_ws(self, open_ods: List[InOutOrder], pair: str, trades: List[dict]):
        pair_tf = f'{pair}_ws'
        strategy_list = BotGlobal.pairtf_stgs.get(pair_tf) or []
        enter_list, exit_list = [], []
        with TempContext(f'{self.data_mgr.exg_name}_{self.data_mgr.market}_{pair_tf}'):
            for strategy in strategy_list:
                stg_name = strategy.name
                strategy.init_bar(open_ods)
                strategy.on_trades(trades)
                # 调用策略生成入场和出场信号
                enter_list.extend([(stg_name, d) for d in strategy.entrys])
                exit_list.extend([(stg_name, d) for d in strategy.exits])
        if not BotGlobal.is_warmup:
            return enter_list, exit_list
        return [], []

    async def _check_pair_hang(self, open_ods: List[InOutOrder] = None):
        """检查是否有交易对数据长期未收到卡住，如有则平掉订单"""
        self.last_check_hang = btime.time_ms()
        if not open_ods:
            open_ods = await InOutOrder.open_orders()
        prefix = f'{self.data_mgr.exg_name}_{self.data_mgr.market}'
        for od in open_ods:
            stg_list = BotGlobal.pairtf_stgs.get(f'{od.symbol}_{od.timeframe}')
            if not stg_list:
                continue
            stg: BaseStrategy = next((stg for stg in stg_list if stg.name == od.strategy), None)
            if not stg:
                continue
            exp_intv = max(tf_to_secs(od.timeframe), 30) * 1000
            if stg.check_ms + exp_intv * 2 < btime.time_ms():
                ctx_key = f'{prefix}_{od.symbol}_{od.timeframe}'
                exit_d = dict(tag=ExitTags.data_stuck, short=od.short, order_id=od.id)
                await self.order_mgr.process_orders(ctx_key, [], [(od.strategy, exit_d)])

    def is_ws_mode(self):
        tfs = self.config.get('run_timeframes')
        if tfs and 'ws' in tfs:
            return True
        for item in self.config.get('run_policy'):
            if 'ws' in item['run_timeframes']:
                return True
        return False

    async def run(self):
        raise NotImplementedError('`run` is not implemented')

    async def cleanup(self):
        pass

    def start_heartbeat_check(self, min_intv: float):
        from threading import Thread
        slp_intv = min_intv * 0.3

        def handle():
            last_tip_at = 0
            while True:
                time.sleep(slp_intv)
                if not self._job:
                    continue
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
            try:
                await run_async(biz_func, timeout=10)
            except Exception:
                logger.exception(f'run loop task error: {func_name}')
            exec_cost = time.monotonic() - job_start
            tip_timeout = min(interval * 0.9, 2)
            if live_mode and exec_cost >= tip_timeout and not is_debug():
                logger.warning('loop task timeout {0} cost {1:.3f} > {2:.3f}', func_name, exec_cost, interval)
            wait_list[0][2] += interval
