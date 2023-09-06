#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : watch_job.py
# Author: anyongjin
# Date  : 2023/7/29
'''
爬虫端运行策略监听市场行情。
'''
from banbot.storage import *
from banbot.exchange.exchange_utils import *
from banbot.strategy.resolver import get_strategy, BaseStrategy
from banbot.compute.ctx import *
from banbot.compute.tools import append_new_bar
from banbot.config import AppConfig
from banbot.util.common import logger
from banbot.data import auto_fetch_ohlcv

_tf_stgy_cls: Dict[str, List[Type[BaseStrategy]]] = dict()  # 记录所有周期，及其运行的策略
_state_map: Dict[str, 'WatchState'] = dict()


class WatchState:
    def __init__(self, exs: ExSymbol, timeframe: str):
        self.exs = exs
        self.timeframe = timeframe
        self.tf_secs = tf_to_secs(timeframe)
        self.end_ms = 0
        '上次计算的截止时间戳，13位'
        config = AppConfig.get()
        cls_list = _tf_stgy_cls.get(timeframe)
        self.stgs = [cls(config) for cls in cls_list]


def _get_state(exs: ExSymbol, tf: str):
    '''
    获取策略实例，每个symbol+tf+strategy对应一个策略实例
    必须在context上下文中调用
    '''
    cache_key = f'{exs.exchange}_{exs.market}_{exs.symbol}_{tf}'
    if cache_key not in _state_map:
        state = WatchState(exs, tf)
        _state_map[cache_key] = state
    else:
        state = _state_map[cache_key]
    return state


async def run_on_bar(exs: ExSymbol, timeframe: str, end_ms: int):
    '''
    计算某个币，在某个周期下，运行指定策略；（结果可以是通知、或者发送消息给机器人等，但不交易）
    此方法应始终在爬虫进程调用，防止多个进程调用时，策略状态不连续，导致计算错误
    '''
    from banbot.exchange import get_exchange
    state = _get_state(exs, timeframe)
    tf_msecs = state.tf_secs * 1000
    cur_end = end_ms // tf_msecs * tf_msecs
    if state.end_ms and state.end_ms + tf_msecs > cur_end:
        # 如果没有新数据，直接退出
        return
    ctx_key = f'{exs.exchange}_{exs.market}_{exs.symbol}_{timeframe}'
    exchange = get_exchange(exs.exchange, exs.market)
    with TempContext(ctx_key):
        # 获取期间的蜡烛数据
        if not state.end_ms:
            # 初次计算，需要预热
            BotGlobal.is_warmup = True
            warm_num = max([stg.warmup_num for stg in state.stgs])
            fetch_start = cur_end - tf_msecs * warm_num
            ohlcvs = await auto_fetch_ohlcv(exchange, exs, timeframe, fetch_start, end_ms)
        else:
            fetch_start = state.end_ms
            ohlcvs = KLine.query(exs, timeframe, fetch_start, end_ms)
        if not ohlcvs:
            return
        for i in range(len(ohlcvs)):
            ohlcv_arr = append_new_bar(ohlcvs[i], tf_msecs // 1000)
            [stg.on_bar(ohlcv_arr) for stg in state.stgs]
        state.end_ms = ohlcvs[-1][0] + tf_msecs
        BotGlobal.is_warmup = False


async def _run_watch_job(exg_name: str, market: str, symbol: str, timeframe: str, end_ms: int):
    try:
        with db():
            exs = ExSymbol.get(exg_name, market, symbol)
            await run_on_bar(exs, timeframe, end_ms)
    except Exception:
        logger.exception(f'_run_watch_job error: {exg_name} {market} {symbol} {timeframe}')


async def run_watch_jobs():
    '''
    运行策略信号更新任务。
    此任务应和爬虫在同一个进程。以便读取到爬虫Kline发出的异步事件。
    '''
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
    tbl_list = [item.tbl for item in KLine.agg_list]
    with db():
        KLine.pause_compress(tbl_list)
    logger.info(f'start kline watch_jobs: {_tf_stgy_cls}')
    while True:
        exg_name, market, symbol, timeframe = await KLine.wait_bars('*', '*', '*', '*')
        if tf_to_secs(timeframe) < 60 or timeframe not in _tf_stgy_cls:
            continue
        await _run_watch_job(exg_name, market, symbol, timeframe, btime.utcstamp())

