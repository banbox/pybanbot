#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : wacther.py
# Author: anyongjin
# Date  : 2023/4/30
import asyncio

from banbot.storage import BotGlobal, BotCache
from banbot.util import btime
from banbot.util.common import logger
from banbot.util.banio import ClientIO
from banbot.config import AppConfig
from dataclasses import dataclass
from banbot.util.tf_utils import *
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
        exg_market = f'{BotGlobal.exg_name}_{BotGlobal.market_type}'
        self.prefix = 'uohlcv' if realtime else 'ohlcv'
        self.listens[f'{self.prefix}_{exg_market}'] = self.on_spider_bar
        self.listens[f'update_price_{exg_market}'] = self.update_price
        self.listens[f'trade_{exg_market}'] = self.on_spider_trade
        self.listens[f'book_{exg_market}'] = self.on_spider_book

        self._inits: List[Tuple[str, Any]] = []

    async def init_conn(self):
        for item in self._inits:
            await self.write_msg(*item)

    async def connect(self):
        # 使用锁防止同时连接
        from banbot.util.misc import LocalLock
        async with LocalLock('conn', 50):
            await self._connect()

    async def _connect(self):
        if self.writer and self.reader:
            return
        try:
            await super().connect()
            await self.init_conn()
            logger.info(f'spider connected at {self.remote}')
        except Exception as e:
            allow_start = self.config.get('with_spider')
            if not allow_start:
                logger.info(f'[{type(e)}] spider not running, {e}, wait 5s and retry...')
                await asyncio.sleep(5)
                await self._connect()
                return
            if self._starting_spider:
                while self._starting_spider:
                    await asyncio.sleep(0.3)
                await self._connect()
                return
            logger.info(f'[{type(e)}] spider not running, {e}, starting...')
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
        """从爬虫订阅K线。调用此方法前，须确保已调用ExSymbol.ensures"""
        ws_jobs = [j for j in jobs if j.timeframe == 'ws']
        tf_jobs = [j for j in jobs if j.timeframe != 'ws']
        if ws_jobs:
            await self._write_pairs(exg_name, market_type, 'ws', ws_jobs)
        if tf_jobs:
            await self._write_pairs(exg_name, market_type, 'bar', tf_jobs)

    def _get_prefixs(self, exg_name: str, market_type: str, is_ws: bool):
        exg_market = f'{exg_name}_{market_type}'
        if is_ws:
            prefixs = [f'trade_{exg_market}', f'book_{exg_market}']
        else:
            prefixs = [f'{self.prefix}_{exg_market}']
        return prefixs

    async def _write_pairs(self, exg_name: str, market_type: str, jtype: str, jobs: List[WatchParam]):
        prefixs = self._get_prefixs(exg_name, market_type, jtype == 'ws')
        tags, pairs = [], []
        for job in jobs:
            if jtype == 'ws':
                job_key = f'{job.symbol}_ws'
                tf_secs = 0
            else:
                job_key = job.symbol
                tf_secs = tf_to_secs(job.timeframe)
                if tf_secs < 60:
                    raise ValueError(f'spider not support {job.timeframe} currently')
            if job_key in self.jobs:
                continue
            for prefix in prefixs:
                tags.append(f'{prefix}_{job.symbol}')
            pairs.append(job.symbol)
            self.jobs[job_key] = PairTFCache(job.timeframe, tf_secs, job.since or 0)
        await self.write_msg('subscribe', tags)
        args = (exg_name, market_type, jtype, pairs)
        await self.write_msg('watch_pairs', args)
        self._inits.extend([
            ('subscribe', tags),
            ('watch_pairs', args)
        ])

    async def unwatch_klines(self, exg_name: str, market_type: str, pairs: List[str], is_ws: bool = False):
        prefixs = self._get_prefixs(exg_name, market_type, is_ws)
        tags = [f'{prefix}_{p}' for p in pairs for prefix in prefixs]
        for p in pairs:
            job_key = p if not is_ws else f'{p}_ws'
            if job_key in self.jobs:
                del self.jobs[job_key]
            if p in BotCache.pair_copied_at:
                del BotCache.pair_copied_at[p]
        await self.write_msg('unsubscribe', tags)
        # 其他端可能还需要此数据，这里不能直接取消。
        # TODO: spider端应保留引用计数，没有客户端需要的才可删除
        # args = (exg_name, market_type, pairs)
        # await self.write_msg('unwatch_pairs', args)

    async def update_price(self, price_list):
        print(f'update_price: {price_list}')
        from banbot.main.addons import MarketPrice
        for item in price_list:
            MarketPrice.set_new_price(item['symbol'], item['markPrice'])

    async def on_spider_bar(self, msg_key: str, msg_data):
        logger.debug('receive ohlcv: %s %s', msg_key, msg_data)
        _, exg_name, market, pair = msg_key.split('_')
        if pair not in self.jobs:
            return False
        ohlc_arr, fetch_tfsecs = msg_data[:2]
        if ohlc_arr:
            bar_ms = int(ohlc_arr[-1][0])
            BotCache.set_pair_ts(pair, bar_ms, fetch_tfsecs * 1000)
        if self.realtime:
            await self._on_ohlcv_msg(exg_name, market, pair, ohlc_arr, fetch_tfsecs, msg_data[2])
            return True
        if ohlc_arr:
            bar_ms = int(ohlc_arr[-1][0])
            BotCache.set_pair_ts(pair, bar_ms, fetch_tfsecs * 1000)
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

    async def on_spider_trade(self, msg_key: str, msg_data):
        # logger.debug('receive trade: %s %s', msg_key, len(msg_data))
        if not msg_data:
            return False
        _, exg_name, market, pair = msg_key.split('_')
        job_key = f'{pair}_ws'
        if job_key not in self.jobs:
            return False
        await self._on_trades(exg_name, market, pair, msg_data)
        return True

    async def on_spider_book(self, msg_key: str, msg_data):
        # logger.debug('receive odbook: %s', msg_key)
        if not msg_data:
            return False
        _, exg_name, market, pair = msg_key.split('_')
        job_key = f'{pair}_ws'
        if job_key not in self.jobs:
            return False
        BotCache.odbooks[pair] = msg_data
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

    @classmethod
    async def _on_trades(cls, exg_name: str, market: str, pair: str, trades: List[dict]):
        raise NotImplementedError
