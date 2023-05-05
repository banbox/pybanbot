#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : spider.py
# Author: anyongjin
# Date  : 2023/4/25
import asyncio
import os.path
import time

import orjson

from banbot.config.appconfig import AppConfig, Config
from banbot.util.misc import parallel_jobs
from banbot.exchange.crypto_exchange import get_exchange
from banbot.util.redis_helper import AsyncRedis
from banbot.data.wacther import *
from banbot.data.tools import *


def get_check_interval(timeframe_secs: int) -> float:
    '''
    根据监听的交易对和时间帧。计算最小检查间隔。
    <60s的通过WebSocket获取数据，检查更新间隔可以比较小。
    1m及以上的通过API的秒级接口获取数据，3s更新一次
    :param timeframe_secs:
    :return:
    '''
    if timeframe_secs <= 3:
        check_interval = 0.2
    elif timeframe_secs <= 10:
        check_interval = 0.5
    elif timeframe_secs < 60:
        check_interval = 1
    else:
        return 3
    return check_interval


async def down_pairs_by_config(config: Config):
    '''
    根据配置文件和解析的命令行参数，下载交易对数据（到数据库或文件）
    此方法由命令行调用。
    '''
    from banbot.storage.klines import KLine
    await KLine.fill_holes()
    pairs = config['pairs']
    timerange = config['timerange']
    start_ms = round(timerange.startts * 1000)
    end_ms = round(timerange.stopts * 1000)
    cur_ms = round(time.time() * 1000)
    end_ms = min(cur_ms, end_ms) if end_ms else cur_ms
    exchange = get_exchange()
    if config['medium'] == 'db':
        tr_text = btime.to_datestr(start_ms) + ' - ' + btime.to_datestr(end_ms)
        args_list = [(exchange, pair, start_ms, end_ms) for pair in pairs]
        for job in parallel_jobs(download_to_db, args_list):
            pair = (await job)['args'][0]
            logger.warning(f'{pair} down {tr_text} complete')
    else:
        data_dir = config['data_dir']
        timeframes = config['timeframes']
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        tr_text = btime.to_datestr(start_ms) + ' - ' + btime.to_datestr(end_ms)
        args_list = [(exchange, pair, tf, start_ms, end_ms, data_dir) for pair in pairs for tf in timeframes]
        for job in parallel_jobs(download_to_file, args_list):
            pair, tf = (await job)['args'][:2]
            logger.warning(f'{pair}/{tf} down {tr_text} complete')
    await exchange.close()


def run_down_pairs(args: Dict[str, Any]):
    '''
    解析命令行参数并下载交易对数据
    '''
    config = AppConfig.init_by_args(args)

    async def run_download():
        from banbot.storage import db
        with db():
            await down_pairs_by_config(config)

    asyncio.run(run_download())


class LiveMiner(Watcher):
    '''
    交易对实时数据更新，仅用于实盘。
    '''
    def __init__(self, exg_name, pair: str, since: Optional[int] = None):
        super(LiveMiner, self).__init__(pair, self._on_bar_finish)
        self.exchange = get_exchange(exg_name)
        timeframe, tf_secs = '1m', 60  # 固定1m间隔更新数据
        self.tf_secs = tf_secs
        self.state = PairTFCache(timeframe, tf_secs)
        self.check_intv = get_check_interval(tf_secs)  # 3s更新一次
        self.key = f'{exg_name}_{pair}'  # 取消标志，启动时设置为tf_secs，删除时表示取消任务
        self.next_since = since if since else None
        self.auto_prefire = AppConfig.get().get('prefire')

    async def run(self):
        with AsyncRedis() as redis:
            await redis.set(self.key, self.tf_secs)
        while True:
            with AsyncRedis() as redis:
                if await redis.get(self.key) != self.tf_secs:
                    break
            await self._try_update()
            await asyncio.sleep(self.check_intv)

    def _on_bar_finish(self, pair: str, timeframe: str, bar_arr: List[Tuple]):
        from banbot.storage import KLine
        KLine.insert(self.exchange.name, pair, bar_arr)

    async def _try_update(self):
        import ccxt
        try:
            if self.check_intv > 2:
                # 这里不设置limit，如果外部修改了更新间隔，这里能及时输出期间所有的数据，避免出现delay
                details = await self.exchange.fetch_ohlcv(self.pair, '1s', since=self.next_since)
                self.next_since = details[-1][0] + 1
            else:
                details = await self.exchange.watch_trades(self.pair)
                details = trades_to_ohlcv(details)
            # 合并得到数据库周期维度1m的数据
            prefire = 0.06 if self.auto_prefire else 0
            ohlcvs_upd, _ = build_ohlcvc(details, self.tf_secs, prefire)
            ohlcvs = [self.state.wait_bar] if self.state.wait_bar else []
            # 和旧的bar_row合并更新，判断是否有完成的bar
            ohlcvs, last_finish = build_ohlcvc(ohlcvs_upd, self.tf_secs, ohlcvs=ohlcvs)
            # 检查是否有完成的bar。写入到数据库
            self._on_state_ohlcv(self.state, ohlcvs, last_finish)
            # 发布间隔数据到redis订阅方
            sre_data = orjson.dumps((ohlcvs_upd, self.tf_secs))
            with AsyncRedis() as redis:
                await redis.publish(self.key, sre_data)
        except ccxt.NetworkError:
            logger.exception(f'get live data exception: {self.pair} {self.tf_secs}')


class SpiderJob:
    def __init__(self, action: str, *args, **kwargs):
        self.action = action
        self.args = args
        self.kwargs = kwargs

    def dumps(self) -> bytes:
        data = dict(action=self.action, args=self.args, kwargs=self.kwargs)
        return orjson.dumps(data)


def start_spider():
    spider = LiveSpider()
    asyncio.run(spider.run_listeners())


class LiveSpider:
    _key = 'spider'

    '''
    实时数据爬虫；仅用于实盘。负责：实时K线、订单簿等公共数据监听
    历史数据下载请直接调用对应方法，效率更高。
    '''
    def __init__(self):
        self.redis = AsyncRedis()  # 需要持续监听redis，单独占用一个连接
        self.conn = self.redis.pubsub()
        self.conn.subscribe(self._key)

    async def run_listeners(self):
        asyncio.create_task(self._heartbeat())
        async for msg in self.conn.listen():
            if msg['type'] != 'message':
                continue
            kwargs = orjson.loads(msg['data'])
            asyncio.create_task(self._run_job(kwargs))

    async def _run_job(self, params: dict):
        from banbot.storage import db
        action, args, kwargs = params['action'], params['args'], params['kwargs']
        try:
            with db():
                if hasattr(self, action):
                    await getattr(self, action)(*args, **kwargs)
                else:
                    logger.error(f'unknown spider job: {params}')
                    return
        except Exception:
            logger.exception(f'run spider job error: {params}')

    async def _heartbeat(self):
        while True:
            await self.redis.set(self._key, expire_time=5)
            await asyncio.sleep(4)

    async def watch_ohlcv(self, exg_name: str, pair: str, since: Optional[int] = None):
        key = f'{exg_name}_{pair}'
        if await self.redis.get(key):
            return
        miner = LiveMiner(exg_name, pair, since)
        await miner.run()

    @classmethod
    async def send(cls, *job_list: SpiderJob):
        '''
        发送命令到爬虫。不等待执行完成；发送成功即返回。
        '''
        redis = AsyncRedis()
        if not await redis.get(cls._key):
            async with redis.lock(cls._key + '_lock'):
                if not await redis.get(cls._key):
                    from threading import Thread
                    td = Thread(target=start_spider, daemon=True)
                    td.start()
        for job in job_list:
            await redis.publish(cls._key, job.dumps())
