#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : wacther.py
# Author: anyongjin
# Date  : 2023/4/30
import asyncio
from typing import *

from banbot.storage import BotGlobal
from banbot.util import btime
from banbot.util.common import logger
from banbot.util.banio import ClientIO
from banbot.config import AppConfig
from dataclasses import dataclass
from banbot.exchange import tf_to_secs
from banbot.data.tools import build_ohlcvc


@dataclass
class PairTFCache:
    timeframe: str
    tf_secs: int
    next_ms: int = 0  # 下一个需要的13位时间戳。一般和wait_bar不应该同时使用
    wait_bar: tuple = None  # 记录尚未完成的bar。已完成时应置为None
    latest: tuple = None  # 记录最新bar数据，可能未完成，可能已完成


def get_finish_ohlcvs(job: PairTFCache, ohlcvs: List[Tuple], last_finish: bool) -> List[Tuple]:
    if not ohlcvs:
        return ohlcvs
    job.wait_bar = None
    if not last_finish:
        job.wait_bar = ohlcvs[-1]
        ohlcvs = ohlcvs[:-1]
    return ohlcvs


class Watcher:
    def __init__(self, callback: Callable):
        self.callback = callback

    async def _on_state_ohlcv(self, pair: str, state: PairTFCache, ohlcvs: List[Tuple], last_finish: bool, do_fire=True) -> list:
        finish_bars = ohlcvs if last_finish else ohlcvs[:-1]
        if state.wait_bar and state.wait_bar[0] < ohlcvs[0][0]:
            finish_bars.insert(0, state.wait_bar)
        state.wait_bar = None if last_finish else ohlcvs[-1]
        state.latest = ohlcvs[-1]
        if finish_bars and do_fire:
            await self._fire_callback(finish_bars, pair, state.timeframe, state.tf_secs)
        return finish_bars

    async def _fire_callback(self, bar_arr, pair: str, timeframe: str, tf_secs: int):
        for bar_row in bar_arr:
            if btime.run_mode not in btime.LIVE_MODES:
                btime.cur_timestamp = bar_row[0] / 1000 + tf_secs
            await self.callback(pair, timeframe, bar_row)
        if btime.run_mode in btime.LIVE_MODES and not BotGlobal.is_warmup:
            bar_delay = btime.time() - bar_arr[-1][0] // 1000 - tf_secs
            if bar_delay > tf_secs:
                # 当蜡烛的触发时间过于滞后时，输出错误信息
                logger.warning('{0}/{1} bar is too late, delay:{2}', pair, timeframe, bar_delay)


@dataclass
class WatchParam:
    symbol: str
    timeframe: str
    since: int = None


class KlineLiveConsumer(ClientIO):
    '''
    这是通用的从爬虫端监听K线数据的消费者端。
    用于：
    机器人的实时数据反馈：LiveDataProvider
    网站端实时K线推送：KlineMonitor
    '''
    _starting_spider = False

    def __init__(self, realtime=False):
        self.config = AppConfig.get()
        super(KlineLiveConsumer, self).__init__(self.config.get('spider_addr'))
        self.jobs: Dict[str, PairTFCache] = dict()
        self.realtime = realtime
        self.prefix = 'uohlcv' if realtime else 'ohlcv'
        self.listens[self.prefix] = self.on_spider_bar
        self._inits: List[Tuple[str, Any]] = []

    async def init_conn(self):
        for item in self._inits:
            await self.write_msg(*item)

    async def connect(self):
        if self.writer and self.reader:
            return
        try:
            await super().connect()
            await self.init_conn()
            logger.info(f'spider connected at {self.remote}')
        except Exception:
            allow_start = self.config.get('with_spider')
            if not allow_start:
                logger.info('spider not running, wait 5s and retry...')
                await asyncio.sleep(5)
                await self.connect()
                return
            if self._starting_spider:
                while self._starting_spider:
                    await asyncio.sleep(0.3)
                await self.connect()
                return
            logger.info('spider not running, starting...')
            self._starting_spider = True
            import multiprocessing
            from banbot.data.spider import run_spider_prc
            prc = multiprocessing.Process(target=run_spider_prc, daemon=True)
            prc.start()
            start_ts = btime.time()
            while btime.time() - start_ts < 10:
                await asyncio.sleep(0.3)
                try:
                    await super().connect()
                    await self.init_conn()
                    break
                except Exception:
                    pass
            self._starting_spider = False
            if not self.writer:
                raise RuntimeError('wait spider timeout')
            logger.info(f'spider connected at {self.remote}')

    async def watch_klines(self, exg_name: str, market_type: str, *jobs: WatchParam):
        pairs = [job.symbol for job in jobs]
        tags = [f'{self.prefix}_{exg_name}_{market_type}_{p}' for p in pairs]
        for job in jobs:
            if job.symbol in self.jobs:
                continue
            tfsecs = tf_to_secs(job.timeframe)
            if tfsecs < 60:
                raise ValueError(f'spider not support {job.timeframe} currently')
            self.jobs[job.symbol] = PairTFCache(job.timeframe, tfsecs, job.since or 0)
        await self.write_msg('subscribe', tags)
        args = (exg_name, market_type, pairs)
        await self.write_msg('watch_pairs', args)
        self._inits.extend([
            ('subscribe', tags),
            ('watch_pairs', args)
        ])

    async def unwatch_klines(self, exg_name: str, market_type: str, pairs: List[str]):
        tags = [f'{self.prefix}_{exg_name}_{market_type}_{p}' for p in pairs]
        for p in pairs:
            if p in self.jobs:
                del self.jobs[p]
        await self.write_msg('unsubscribe', tags)
        # 其他端可能还需要此数据，这里不能直接取消。
        # TODO: spider端应保留引用计数，没有客户端需要的才可删除
        # args = (exg_name, market_type, pairs)
        # await self.write_msg('unwatch_pairs', args)

    async def on_spider_bar(self, msg_key: str, msg_data):
        logger.debug('receive ohlcv: %s %s', msg_key, msg_data)
        _, exg_name, market, pair = msg_key.split('_')
        if pair not in self.jobs:
            return False
        if self.realtime:
            ohlc_arr, fetch_tfsecs, update_tfsecs = msg_data
            await self._on_ohlcv_msg(exg_name, market, pair, ohlc_arr, fetch_tfsecs, update_tfsecs)
            return True
        ohlc_arr, fetch_tfsecs = msg_data
        job = self.jobs[pair]
        if fetch_tfsecs < job.tf_secs:
            old_ohlcvs = [job.wait_bar] if job.wait_bar else []
            # 和旧的bar_row合并更新，判断是否有完成的bar
            in_msecs = fetch_tfsecs * 1000
            ohlcvs, last_finish = build_ohlcvc(ohlc_arr, job.tf_secs, ohlcvs=old_ohlcvs, in_tf_msecs=in_msecs)
            ohlcvs = get_finish_ohlcvs(job, ohlcvs, last_finish)
        else:
            ohlcvs, last_finish = ohlc_arr, True
        if ohlcvs:
            logger.debug('finish ohlcv: %s %s', msg_key, ohlcvs)
            await self._on_ohlcv_msg(exg_name, market, pair, ohlcvs, job.tf_secs, job.tf_secs)
        return True

    @classmethod
    async def _on_ohlcv_msg(cls, exg_name: str, market: str, pair: str, ohlc_arr: list,
                      fetch_tfsecs: int, update_tfsecs: float):
        '''
        监听spider的蜡烛数据，收到时会回调此方法。
        :param fetch_tfsecs: 返回的蜡烛数据bar时间戳间隔
        :param update_tfsecs: 蜡烛数据更新间隔，可能小于fetch_tfsecs
        '''
        raise NotImplementedError
