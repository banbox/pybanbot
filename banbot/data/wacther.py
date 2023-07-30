#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : wacther.py
# Author: anyongjin
# Date  : 2023/4/30
import asyncio
import six
from typing import *
import re

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


class RedisChannel:
    '''
    通过redis进行多进程通信的通道。
    '''
    _obj: Optional['RedisChannel'] = None

    def __init__(self):
        RedisChannel._obj = self
        from banbot.util.redis_helper import AsyncRedis
        self.redis = AsyncRedis()
        self.conn = self.redis.pubsub()
        self.listeners: List[Tuple[Union[str, re.Pattern], Callable]] = []

    @classmethod
    async def subscribe(cls, matcher: Union[str, re.Pattern], handler: Callable):
        if not cls._obj:
            raise ValueError('RedisChannel not start')
        listeners = cls._obj.listeners
        for i in range(len(listeners)):
            ptn, func = listeners[i]
            if ptn == matcher:
                if handler == func:
                    return
                listeners.pop(i)
                break
        listeners.append((matcher, handler))
        if isinstance(matcher, six.string_types):
            await cls._obj.conn.subscribe(matcher)

    @classmethod
    def unsubscribe(cls, matcher: Union[str, re.Pattern]):
        if not cls._obj:
            raise ValueError('RedisChannel not start')
        listeners = cls._obj.listeners
        for i in range(len(listeners)):
            ptn, func = listeners[i]
            if ptn == matcher:
                listeners.pop(i)
                break

    @classmethod
    async def call_remote(cls, action: str, data, timeout: int = 10):
        '''
        调用远程过程调用。通过Redis的SubPub机制。
        发送一次消息，监听一次，收到返回消息后取消监听
        '''
        if not cls._obj:
            raise ValueError('RedisChannel not start')
        future = asyncio.get_running_loop().create_future()

        def call_back(msg_key: str, msg_data):
            if not future.done():
                future.set_result((msg_key, msg_data))

        reply_key = f'{action}_{data}'
        await cls.subscribe(reply_key, call_back)
        await cls._obj.conn.subscribe(reply_key)
        await cls._obj.redis.publish(action, data)
        try:
            ret_key, ret_data = await asyncio.wait_for(future, timeout=timeout)
            cls.unsubscribe(reply_key)
            return ret_data
        except asyncio.TimeoutError:
            logger.warning(f'wait redis channel complete timeout: {action} {data} {timeout}')

    @classmethod
    async def run(cls):
        import orjson
        assert cls._obj, '`RedisChannel` is not initialized yet!'
        logger.info(f'run RedisChannel ...')
        # 监听个假的，防止立刻退出
        await cls._obj.conn.subscribe('fake')
        async for msg in cls._obj.conn.listen():
            if msg['type'] != 'message':
                continue
            try:
                msg_key = msg['channel'].decode()
                msg_data = msg['data'].decode()
                try:
                    msg_data = orjson.loads(msg_data)
                except Exception:
                    pass
                handle_func = None
                for key, func in cls._obj.listeners:
                    if isinstance(key, re.Pattern):
                        if key.fullmatch(msg_key):
                            handle_func = func
                            break
                    elif key == msg_key:
                        handle_func = func
                        break
                if handle_func:
                    ret_data = handle_func(msg_key, msg_data)
                    if ret_data and len(ret_data) == 2:
                        await cls._obj.redis.publish(ret_data[0], ret_data[1])
                    elif ret_data:
                        logger.info(f'invalid ret data from {handle_func}: {ret_data}')
                else:
                    logger.info(f'unhandle redis channel msg: {msg_key}')
            except Exception:
                logger.exception(f'handle RedisChannel msg error: {msg}')

@dataclass
class WatchParam:
    exchange: str
    symbol: str
    market: str
    timeframe: str = None
    since: int = None


class KlineLiveConsumer(RedisChannel):
    '''
    这是通用的从爬虫端监听K线数据的消费者端。
    用于：
    机器人的实时数据反馈：LiveDataProvider
    网站端实时K线推送：KlineMonitor
    '''

    def __init__(self):
        super(KlineLiveConsumer, self).__init__()
        self.listeners.append((re.compile(r'^ohlcv_.*'), self._handle_ohlcv))

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
    def _handle_ohlcv(cls, msg_key: str, msg_data):
        _, exg_name, market, pair = msg_key.split('_')
        ohlc_arr, fetch_tfsecs = msg_data
        cls._on_ohlcv_msg(exg_name, market, pair, ohlc_arr, fetch_tfsecs)

    @classmethod
    def _on_ohlcv_msg(cls, exg_name, market, pair, ohlc_arr, fetch_tfsecs):
        raise NotImplementedError
