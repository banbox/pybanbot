#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : wacther.py
# Author: anyongjin
# Date  : 2023/4/30
from typing import List, Tuple, Callable, Optional

from banbot.storage import BotGlobal
from banbot.util import btime
from banbot.util.common import logger
from dataclasses import dataclass


@dataclass
class PairTFCache:
    timeframe: str
    tf_secs: int
    wait_bar: tuple = None  # 记录尚未完成的bar。已完成时应置为None
    latest: tuple = None  # 记录最新bar数据，可能未完成，可能已完成

    def set_bar(self, bar_row):
        self.wait_bar = bar_row
        self.latest = bar_row


class Watcher:
    def __init__(self, callback: Callable):
        self.callback = callback

    def _on_state_ohlcv(self, pair: str, state: PairTFCache, ohlcvs: List[Tuple], last_finish: bool, do_fire=True) -> list:
        finish_bars = []
        for i in range(len(ohlcvs)):
            new_bar = ohlcvs[i]
            if state.wait_bar and state.wait_bar[0] < new_bar[0]:
                finish_bars.append(state.wait_bar)
            state.set_bar(new_bar)
        if last_finish:
            finish_bars.append(state.wait_bar)
            state.wait_bar = None
        if finish_bars and do_fire:
            self._fire_callback(finish_bars, pair, state.timeframe, state.tf_secs)
        return finish_bars

    def _fire_callback(self, bar_arr, pair: str, timeframe: str, tf_secs: int):
        for bar_row in bar_arr:
            if btime.run_mode not in btime.LIVE_MODES:
                btime.cur_timestamp = bar_row[0] / 1000 + tf_secs
            self.callback(pair, timeframe, bar_row)
        if btime.run_mode in btime.LIVE_MODES and not BotGlobal.is_wramup:
            bar_delay = btime.time() - bar_arr[-1][0] // 1000 - tf_secs
            if bar_delay > tf_secs:
                # 当蜡烛的触发时间过于滞后时，输出错误信息
                logger.warning('{0}/{1} bar is too late, delay:{2}', pair, timeframe, bar_delay)


@dataclass
class WatchParam:
    exchange: str
    symbol: str
    market: str
    timeframe: str = None
    since: int = None


class KlineLiveConsumer:
    '''
    这是通用的从爬虫端监听K线数据的消费者端。
    用于：
    机器人的实时数据反馈：LiveDataProvider
    网站端实时K线推送：KlineMonitor
    '''
    _obj: Optional['KlineLiveConsumer'] = None

    def __init__(self):
        from banbot.util.redis_helper import AsyncRedis
        self.redis = AsyncRedis()
        self.conn = self.redis.pubsub()

    async def watch_klines(self, *jobs: WatchParam):
        from banbot.data.spider import LiveSpider, SpiderJob
        job_list = []
        for job in jobs:
            args = [job.exchange, job.symbol, job.market, job.timeframe, job.since]
            job_list.append(SpiderJob('watch_ohlcv', *args))
            key = f'{job.exchange}_{job.market}_{job.symbol}'
            await self.conn.subscribe(key)
        # 发送消息给爬虫，实时抓取数据
        await LiveSpider.send(*job_list)

    async def unwatch_klines(self, *jobs: WatchParam):
        from banbot.data.spider import LiveSpider, SpiderJob
        cache_key, symbol_list = (None, None), []
        for job in jobs:
            await self.conn.unsubscribe(f'{job.exchange}_{job.market}_{job.symbol}')
            cur_key = job.exchange, job.market
            if symbol_list and cache_key != cur_key:
                await LiveSpider.send(SpiderJob('unwatch_pairs', cache_key[0], cache_key[1], symbol_list))
                symbol_list = []
            cache_key = cur_key
            symbol_list.append(job.symbol)
        if symbol_list:
            await LiveSpider.send(SpiderJob('unwatch_pairs', cache_key[0], cache_key[1], symbol_list))

    @classmethod
    async def run(cls):
        import orjson
        assert cls._obj, '`KlineLiveConsumer` is not initialized yet!'
        logger.info(f'start watching ohlcvs from {cls.__name__}...')
        # 监听个假的，防止立刻退出
        await cls._obj.conn.subscribe('fake')
        async for msg in cls._obj.conn.listen():
            if msg['type'] != 'message':
                continue
            try:
                exg_name, market, pair = msg['channel'].decode().split('_')
                ohlc_arr, fetch_tfsecs = orjson.loads(msg['data'])
                cls._on_ohlcv_msg(exg_name, market, pair, ohlc_arr, fetch_tfsecs)
            except Exception:
                logger.exception(f'handle live ohlcv listen error: {msg}')

    @classmethod
    def _on_ohlcv_msg(cls, exg_name, market, pair, ohlc_arr, fetch_tfsecs):
        raise NotImplementedError
