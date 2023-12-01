#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : itrader.py
# Author: anyongjin
# Date  : 2023/3/17
from __future__ import annotations

from banbot.compute.tools import append_new_bar
from banbot.main.od_manager import *
from banbot.strategy.resolver import StrategyResolver
from banbot.strategy.base import BaseStrategy
from banbot.data.provider import *
from banbot.storage import *
from banbot.main.wallets import *
from banbot.compute.ctx import *
from banbot.util.support import BanEvent
from banbot.symbols.pair_manager import PairManager


class Trader:
    def __init__(self, config: Config):
        BotGlobal.state = BotState.INIT
        self.config = config
        self.name = config.get('name', 'noname')
        if btime.prod_mode():
            logger.info('started bot:   >>>  %s  <<<', self.name)
        self.exchange = get_exchange()
        self.wallets: WalletsLocal = None
        self.order_mgr: OrderManager = None
        self.data_mgr: DataProvider = None
        self._job: Tuple[str, float, float] = None
        self.pair_mgr = PairManager(config, self.exchange)
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
                    if cls.watch_book:
                        BotGlobal.book_pairs.add(pair)
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

    async def refresh_pairs(self):
        '''
        定期刷新交易对
        '''
        try:
            async with dba():
                logger.info("start refreshing symbols")
                old_symbols = set(self.pair_mgr.symbols)
                await self.pair_mgr.refresh_pairlist()
                await self.add_del_pairs(old_symbols)
        except Exception:
            logger.exception('loop refresh pairs error')

    async def add_del_pairs(self, old_symbols: Set[str]):
        now_symbols = set(self.pair_mgr.symbols)
        BotGlobal.pairs = now_symbols
        if hasattr(self.wallets, 'init'):
            await self.wallets.init(now_symbols)
        del_symbols = list(old_symbols.difference(now_symbols))
        add_symbols = list(now_symbols.difference(old_symbols))
        # 检查删除的交易对是否有订单，有则添加回去
        errors = dict()
        if del_symbols:
            open_ods = [od for _, od in BotCache.open_ods.items() if od.symbol in del_symbols]
            for od in open_ods:
                if od.symbol in del_symbols:
                    errors[od.symbol] = 'has open order, remove fail'
                    del_symbols.remove(od.symbol)
                    self.pair_mgr.symbols.append(od.symbol)
            if del_symbols:
                logger.info(f"remove symbols: {del_symbols}")
                fut = self.data_mgr.unsub_pairs(del_symbols)
                if fut and asyncio.isfuture(fut):
                    asyncio.ensure_future(fut)
                jobs = BotGlobal.get_jobs(del_symbols)
                BotGlobal.remove_jobs(jobs)

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
            if hasattr(self.data_mgr, 'sub_warm_pairs'):
                await self.data_mgr.sub_warm_pairs(pair_tfs)
            else:
                self.data_mgr.sub_pairs(pair_tfs)
        BanEvent.set('set_pairs_res', errors)
        return errors

    async def on_data_feed(self, pair: str, timeframe: str, row: list):
        tf_secs = tf_to_secs(timeframe)
        if not BotGlobal.is_warmup and BotGlobal.live_mode:
            self.last_process = btime.utcstamp()
        MarketPrice.set_bar_price(pair, float(row[ccol]))
        if pair in BotGlobal.forbid_pairs:
            return
        # 超过1分钟或周期的一半，认为bar延迟，不可下单
        delay = btime.time() - (row[0] // 1000 + tf_secs)
        bar_expired = tf_secs >= 60 and delay >= max(60., tf_secs * 0.5)
        is_live_mode = btime.run_mode == RunMode.PROD
        if bar_expired and is_live_mode and not BotGlobal.is_warmup:
            logger.warning(f'{pair}/{timeframe} delay {delay:.2}s, enter order is disabled')

        ctx_key = f'{self.data_mgr.exg_name}_{self.data_mgr.market}_{pair}_{timeframe}'
        with TempContext(ctx_key):
            try:
                pair_arr = append_new_bar(row, tf_secs, tf_secs >= 60)
            except ValueError as e:
                logger.info(f'skip invalid bar: {e}')
                return
            all_open_ods, open_ods = await self._update_orders(pair, timeframe, row)
            enters, exits, triggers = self._run_bar(open_ods, pair, timeframe, pair_arr, bar_expired)

        if enters or exits or triggers:
            edit_tgs = list(set(triggers))
            if not BotGlobal.live_mode:
                await self._apply_signals(all_open_ods, pair, timeframe, enters, exits, edit_tgs)
            else:
                async with dba():
                    await self._apply_signals(all_open_ods, pair, timeframe, enters, exits, edit_tgs)

    async def on_pair_trades(self, pair: str, trades: List[dict]):
        if not BotGlobal.is_warmup and btime.run_mode in btime.LIVE_MODES:
            self.last_process = btime.utcstamp()
        MarketPrice.set_new_price(pair, trades[-1]['price'])
        if pair in BotGlobal.forbid_pairs:
            return
        timeframe = 'ws'
        with TempContext(f'{self.data_mgr.exg_name}_{self.data_mgr.market}_{pair}_{timeframe}'):
            bar = self.data_mgr.get_latest_ohlcv(pair)
            bar_time.set((trades[0]['timestamp'], trades[-1]['timestamp']))
            all_open_ods, pair_ods = await self._update_orders(pair, timeframe, bar)
            enter_list, exit_list = self._run_ws(pair_ods, pair, trades)

        if enter_list or exit_list:
            if not BotGlobal.live_mode:
                await self._apply_signals(all_open_ods, pair, timeframe, enter_list, exit_list)
            else:
                async with dba():
                    await self._apply_signals(all_open_ods, pair, timeframe, enter_list, exit_list)

    async def _update_orders(self, pair: str, timeframe: str, bar):
        all_open_ods = list(BotCache.open_ods.values())
        if not BotGlobal.is_warmup:
            tracer = InOutTracer(all_open_ods)
            self.order_mgr.update_by_bar(all_open_ods, pair, timeframe, bar)
            chg_ods = tracer.get_changes()
            if chg_ods or BotCache.updod_at + 60 < btime.time():
                all_open_ods = await self._flush_cache_orders(chg_ods)
        return all_open_ods, [od for od in all_open_ods if od.symbol == pair]

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
            open_ods = await InOutOrder.open_orders()
            if chg_ods:
                for od in chg_ods:
                    db_od = await od.attach(sess)
                    await db_od.save()
                await sess.flush()
                open_ods = [od for od in open_ods if od.status < InOutStatus.FullExit]
            BotCache.open_ods = {od.id: od.clone() for od in open_ods}
            return list(BotCache.open_ods.values())

    async def _apply_signals(self, all_open_ods, pair: str, timeframe: str, enter_list, exit_list, triggers=None):
        ctx_key = f'{self.data_mgr.exg_name}_{self.data_mgr.market}_{pair}_{timeframe}'
        if not BotGlobal.is_warmup:
            await self.order_mgr.process_orders(ctx_key, enter_list, exit_list, triggers)
        if self.last_check_hang + self.check_hang_intv < btime.time_ms():
            await self._check_pair_hang(all_open_ods)

    def _run_bar(self, open_ods: List[InOutOrder], pair: str, timeframe: str, pair_arr: np.ndarray, bar_expired: bool):
        edit_triggers = []
        pair_tf = f'{pair}_{timeframe}'
        strategy_list = BotGlobal.pairtf_stgs.get(pair_tf) or []
        start_time = time.monotonic()
        enter_list, exit_list = [], []
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
        return enter_list, exit_list, edit_triggers

    def _run_ws(self, open_ods: List[InOutOrder], pair: str, trades: List[dict]):
        pair_tf = f'{pair}_ws'
        strategy_list = BotGlobal.pairtf_stgs.get(pair_tf) or []
        enter_list, exit_list = [], []
        for strategy in strategy_list:
            stg_name = strategy.name
            strategy.init_bar(open_ods)
            strategy.on_trades(trades)
            # 调用策略生成入场和出场信号
            enter_list.extend([(stg_name, d) for d in strategy.entrys])
            exit_list.extend([(stg_name, d) for d in strategy.exits])
        return enter_list, exit_list

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
