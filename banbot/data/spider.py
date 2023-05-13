#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : spider.py
# Author: anyongjin
# Date  : 2023/4/25
import asyncio
import os.path
import time

import orjson

from banbot.config.appconfig import AppConfig
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
    timeframes = config['timeframes']
    tr_text = btime.to_datestr(start_ms) + ' - ' + btime.to_datestr(end_ms)
    if config['medium'] == 'db':
        tf = timeframes[0]
        if len(timeframes) > 1:
            logger.error('only one timeframe should be given to download into db')
            return
        if tf not in {'1m', '1h'}:
            logger.error(f'can only download kline: 1m or 1h, current: {tf}')
            return
        for pair in pairs:
            pair = await download_to_db(exchange, pair, tf, start_ms, end_ms)
            logger.warning(f'{pair}/{tf} down {tr_text} complete')
    else:
        data_dir = config['data_dir']
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        for pair in pairs:
            for tf in timeframes:
                await download_to_file(exchange, pair, tf, start_ms, end_ms, data_dir)
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
    _list_key = 'miner_symbols'
    '''
    交易对实时数据更新，仅用于实盘。
    '''
    def __init__(self, exg_name, pair: str, since: Optional[int] = None):
        super(LiveMiner, self).__init__(pair, self._on_bar_finish)
        self.exchange = get_exchange(exg_name)
        timeframe, tf_secs = '1m', 60  # 固定1m间隔更新数据
        self.tf_secs = tf_secs
        self.state = PairTFCache(timeframe, tf_secs)
        self.state_sec = PairTFCache('1s', 1)  # 用于从交易归集
        self.check_intv = get_check_interval(tf_secs)  # 3s更新一次
        self.key = f'{exg_name}_{pair}'  # 取消标志，启动时设置为tf_secs，删除时表示取消任务
        self.next_since = int(since) if since else None
        self.auto_prefire = AppConfig.get().get('prefire')

    async def run(self):
        async with AsyncRedis() as redis:
            await redis.sadd(self._list_key, self.key)
            await redis.set(self.key, self.tf_secs)
        while True:
            try:
                async with AsyncRedis() as redis:
                    if await redis.get(self.key) != self.tf_secs:
                        break
                await self._try_update()
                await asyncio.sleep(self.check_intv)
            except Exception:
                logger.exception(f'miner watch pair error {self.pair} tf_secs: {self.tf_secs}')

    def _on_bar_finish(self, pair: str, timeframe: str, bar_row: Tuple):
        from banbot.storage import db
        from banbot.storage import KLine
        with db():
            KLine.insert(self.exchange.name, pair, timeframe, [bar_row])

    async def _try_update(self):
        import ccxt
        try:
            is_from_ohlcv = True
            if self.check_intv > 2:
                # 这里不设置limit，如果外部修改了更新间隔，这里能及时输出期间所有的数据，避免出现delay
                ohlcvs_sec = await self.exchange.fetch_ohlcv(self.pair, '1s', since=self.next_since)
                if not ohlcvs_sec:
                    return
                self.next_since = ohlcvs_sec[-1][0] + 1
                finish_bars = ohlcvs_sec  # 从交易所1s抓取的都是已完成的
            else:
                details = await self.exchange.watch_trades(self.pair)
                details = trades_to_ohlcv(details)
                # 交易按1s维度归集和通知
                ohlcvs_sec = [self.state_sec.wait_bar] if self.state_sec.wait_bar else []
                ohlcvs_sec, _ = build_ohlcvc(details, self.state_sec.tf_secs, ohlcvs=ohlcvs_sec)
                do_fire = self.state.tf_secs <= self.state_sec.tf_secs
                finish_bars = self._on_state_ohlcv(self.state_sec, ohlcvs_sec, False, do_fire)
                is_from_ohlcv = False
            if self.state.tf_secs > self.state_sec.tf_secs:
                # 合并得到数据库周期维度1m的数据
                ohlcvs = [self.state.wait_bar] if self.state.wait_bar else []
                # 和旧的bar_row合并更新，判断是否有完成的bar
                ohlcvs, last_finish = build_ohlcvc(ohlcvs_sec, self.tf_secs, ohlcvs=ohlcvs)
                last_finish = last_finish and is_from_ohlcv
                # 检查是否有完成的bar。写入到数据库
                self._on_state_ohlcv(self.state, ohlcvs, last_finish)
            if finish_bars:
                # 发布1s间隔数据到redis订阅方
                sre_data = orjson.dumps((finish_bars, 1))
                async with AsyncRedis() as redis:
                    await redis.publish(self.key, sre_data)
        except ccxt.NetworkError:
            logger.exception(f'get live data exception: {self.pair} {self.tf_secs}')

    @classmethod
    async def cleanup(cls):
        async with AsyncRedis() as redis:
            old_keys = await redis.smembers(cls._list_key)
            old_keys.append(cls._list_key)
            del_num = await redis.delete(*old_keys)
            logger.info(f'stop {del_num} live miners')


class SpiderJob:
    def __init__(self, action: str, *args, **kwargs):
        self.action = action
        self.args = args
        self.kwargs = kwargs

    def dumps(self) -> bytes:
        data = dict(action=self.action, args=self.args, kwargs=self.kwargs)
        return orjson.dumps(data)


async def start_spider():
    await LiveMiner.cleanup()
    spider = LiveSpider()
    logger.info('run data spider...')
    await spider.run_listeners()


class LiveSpider:
    _key = 'spider'
    _ready = False

    '''
    实时数据爬虫；仅用于实盘。负责：实时K线、订单簿等公共数据监听
    历史数据下载请直接调用对应方法，效率更高。
    '''
    def __init__(self):
        self.redis = AsyncRedis()  # 需要持续监听redis，单独占用一个连接
        self.conn = self.redis.pubsub()

    async def run_listeners(self):
        await self.conn.subscribe(self._key)
        asyncio.create_task(self._heartbeat())
        LiveSpider._ready = True
        async for msg in self.conn.listen():
            if msg['type'] != 'message':
                continue
            logger.debug('spider receive msg: %s', msg)
            kwargs = orjson.loads(msg['data'])
            asyncio.create_task(self._run_job(kwargs))

    async def _run_job(self, params: dict):
        action, args, kwargs = params['action'], params['args'], params['kwargs']
        try:
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
            logger.warning(f'ohlcv {key} is already watching')
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
                    asyncio.create_task(start_spider())
                    while not cls._ready:
                        await asyncio.sleep(0.01)
        for job in job_list:
            await redis.publish(cls._key, job.dumps())

    @classmethod
    async def cleanup(cls):
        await LiveMiner.cleanup()
