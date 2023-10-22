#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : watch_job.py
# Author: anyongjin
# Date  : 2023/7/29
'''
爬虫端运行策略监听市场行情。
'''
import asyncio
import time

from banbot.storage import *
from banbot.strategy.resolver import get_strategy, BaseStrategy
from banbot.compute.ctx import *
from banbot.compute.tools import append_new_bar
from banbot.config import AppConfig
from banbot.util.common import logger
from banbot.data import auto_fetch_ohlcv
from banbot.util.tf_utils import *
from banbot.util import btime

_tf_stgy_cls: Dict[str, List[Type[BaseStrategy]]] = dict()  # 记录所有周期，及其运行的策略
_state_map: Dict[str, 'WatchState'] = dict()
wait_jobs = []
_num_tip_time = 0


class WatchState:
    def __init__(self, exg_name: str, market: str, symbol: str, timeframe: str):
        self.exg_name = exg_name
        self.market = market
        self.symbol = symbol
        self.timeframe = timeframe
        self.tf_msecs = tf_to_secs(timeframe) * 1000
        self.end_ms = 0
        '上次计算的截止时间戳，13位'
        config = AppConfig.get()
        cls_list = _tf_stgy_cls.get(timeframe)
        self.stgs = [cls(config) for cls in cls_list]


def _get_state(exg_name: str, market: str, symbol: str, tf: str):
    '''
    获取策略实例，每个symbol+tf+strategy对应一个策略实例
    必须在context上下文中调用
    '''
    cache_key = f'{exg_name}_{market}_{symbol}_{tf}'
    if cache_key not in _state_map:
        state = WatchState(exg_name, market, symbol, tf)
        _state_map[cache_key] = state
    else:
        state = _state_map[cache_key]
    return state


async def run_on_bar(state: WatchState):
    '''
    计算某个币，在某个周期下，运行指定策略；（结果可以是通知、或者发送消息给机器人等，但不交易）
    此方法应始终在爬虫进程调用，防止多个进程调用时，策略状态不连续，导致计算错误
    '''
    from banbot.exchange import get_exchange
    tf_msecs = state.tf_msecs
    cur_end = align_tfmsecs(btime.utcstamp(), tf_msecs)
    if state.end_ms and state.end_ms + tf_msecs > cur_end:
        # 如果没有新数据，直接退出
        return
    ctx_key = f'{state.exg_name}_{state.market}_{state.symbol}_{state.timeframe}'
    exs = ExSymbol.get(state.exg_name, state.market, state.symbol)
    exchange = get_exchange(state.exg_name, state.market)
    with TempContext(ctx_key):
        # 获取期间的蜡烛数据
        if not state.end_ms:
            # 初次计算，需要预热
            BotGlobal.is_warmup = True
            warm_num = max([stg.warmup_num for stg in state.stgs])
            fetch_start = cur_end - tf_msecs * warm_num
            ohlcvs = await auto_fetch_ohlcv(exchange, exs, state.timeframe, fetch_start, cur_end)
        else:
            fetch_start = state.end_ms
            ohlcvs = await KLine.query(exs, state.timeframe, fetch_start, cur_end)
        if not ohlcvs:
            return
        for i in range(len(ohlcvs)):
            ohlcv_arr = append_new_bar(ohlcvs[i], tf_msecs // 1000)
            [stg.on_bar(ohlcv_arr) for stg in state.stgs]
        state.end_ms = bar_time.get()[1]
        BotGlobal.is_warmup = False


async def _consume_jobs():
    reset_ctx()
    while True:
        if not wait_jobs:
            await asyncio.sleep(1)
            continue
        exg_name, market, symbol = wait_jobs.pop(0)
        try:
            cur_ms = btime.utcstamp()
            async with dba():
                for tf in _tf_stgy_cls:
                    state = _get_state(exg_name, market, symbol, tf)
                    cur_end = cur_ms // state.tf_msecs * state.tf_msecs
                    if cur_end > state.end_ms:
                        start = time.time()
                        await run_on_bar(state)
                        cost = time.time() - start
                        if cost > 0.1:
                            logger.info(f'done run_on_bar {symbol} {tf}, cost: {cost:.2f} s')
        except Exception:
            logger.exception(f'_run_watch_job error: {exg_name} {market} {symbol}')


async def run_watch_jobs():
    '''
    运行策略信号更新任务。
    此任务应和爬虫在同一个进程。以便读取到爬虫Kline发出的异步事件。
    '''
    global _num_tip_time
    config = AppConfig.get()
    jobs: dict = config.get('watch_jobs')
    if not jobs:
        return
    # 加载任务到_tf_stgy_cls
    for tf, stf_list in jobs.items():
        cur_list = []
        for name in stf_list:
            stgy_cls = get_strategy(name)
            if not stgy_cls:
                continue
            cur_list.append(stgy_cls)
        if not cur_list:
            continue
        _tf_stgy_cls[tf] = cur_list
    if not _tf_stgy_cls:
        return
    from banbot.storage.base import init_db
    init_db()
    logger.info(f'start kline watch_jobs: {_tf_stgy_cls}')
    asyncio.create_task(_consume_jobs())
    while True:
        # 监听数据库K线写入事件，只监听1m级别
        exg_name, market, symbol, _ = await KLine.wait_bars('*', '*', '*', '1m')
        job_item = (exg_name, market, symbol)
        if job_item not in wait_jobs:
            wait_jobs.append(job_item)
            if len(wait_jobs) > 1000 and btime.time() - _num_tip_time > 1000:
                logger.error(f'watch jobs queue full: {len(wait_jobs)}')
                _num_tip_time = btime.time()

