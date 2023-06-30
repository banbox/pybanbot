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
from banbot.data.tools import *
from banbot.data.wacther import *
from banbot.exchange.crypto_exchange import get_exchange
from banbot.util.redis_helper import AsyncRedis


def get_check_interval(tf_secs: int) -> float:
    '''
    根据监听的交易对和时间帧。计算最小检查间隔。
    <60s的通过WebSocket获取数据，检查更新间隔可以比较小。
    1m及以上的通过API的秒级接口获取数据，3s更新一次
    :param tf_secs:
    :return:
    '''
    if tf_secs <= 3:
        check_intv = 0.5
    elif tf_secs <= 10:
        check_intv = tf_secs * 0.35
    elif tf_secs <= 60:
        check_intv = tf_secs * 0.2
    elif tf_secs <= 300:
        check_intv = tf_secs * 0.15
    elif tf_secs <= 900:
        check_intv = tf_secs * 0.1
    elif tf_secs <= 3600:
        check_intv = tf_secs * 0.07
    else:
        # 超过1小时维度的，10分钟刷新一次
        check_intv = 600
    return check_intv


async def down_pairs_by_config(config: Config):
    '''
    根据配置文件和解析的命令行参数，下载交易对数据（到数据库或文件）
    此方法由命令行调用。
    '''
    from banbot.storage.klines import KLine, db
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
        sess = db.session
        fts = [ExSymbol.exchange == exchange.name, ExSymbol.symbol.in_(set(pairs)),
               ExSymbol.market == exchange.market_type]
        exs_list = sess.query(ExSymbol).filter(*fts).all()
        for exs in exs_list:
            await download_to_db(exchange, exs, tf, start_ms, end_ms)
            logger.warning(f'{exs}/{tf} down {tr_text} complete')
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


class TradesWatcher(Watcher):
    '''
    针对1m以下维度的K线数据监听
    '''
    def __init__(self, exg_name: str, pair: str):
        super(TradesWatcher, self).__init__()
        self.exchange = get_exchange(exg_name)
        self.pair = pair

    async def try_update(self):
        logger.warning(f'watch trades: {self.pair}')
        details = await self.exchange.watch_trades(self.pair)
        details = trades_to_ohlcv(details)
        # 交易按小维度归集和通知；减少传输数据大小；
        ohlcvs_sml = [self.state_sml.wait_bar] if self.state_sml.wait_bar else []
        ohlcvs_sml, _ = build_ohlcvc(details, self.state_sml.tf_secs, ohlcvs=ohlcvs_sml)
        do_fire = self.state.tf_secs <= self.state_sml.tf_secs
        # 未完成数据存储在wait_bar中
        finish_bars = self._on_state_ohlcv(self.pair, self.state_sml, ohlcvs_sml, False, do_fire)


class MinerJob(PairTFCache):
    def __init__(self, pair: str, save_tf: str, check_intv: float, since: Optional[int] = None):
        '''
        K线抓取任务。
        :param pair:
        :param save_tf: 保存到数据库的周期维度，必须是1m或1h，这个<=实际分析用到的维度
        :param check_intv: 检查更新间隔。秒。需要根据实际使用维度计算。此值可能更新
        :param since: 抓取的开始时间，未提供默认从当前时间所属bar开始抓取
        '''
        assert save_tf in {'1m', '1h'}, f'MinerJob save_tf must be 1m/1h, given: {save_tf}'
        tf_secs = tf_to_secs(save_tf)
        super(MinerJob, self).__init__(save_tf, tf_secs)
        self.pair: str = pair
        self.check_intv = check_intv
        self.fetch_tf = '1s' if self.check_intv < 60 else '1m'
        self.fetch_tfsecs = tf_to_secs(self.fetch_tf)
        self.since = int(since) if since else int(time.time() // tf_secs * 1000)
        self.next_run = self.since / 1000

    @classmethod
    def get_tf_intv(cls, timeframe: str) -> Tuple[str, float]:
        '''
        从需要使用的分析维度，计算应该保存的维度和更新间隔。
        '''
        cur_tfsecs = tf_to_secs(timeframe)
        save_tf = '1h' if cur_tfsecs >= 3600 else '1m'
        check_intv = get_check_interval(cur_tfsecs)
        return save_tf, check_intv


class LiveMiner(Watcher):
    loop_intv = 0.5  # 没有任务时，睡眠间隔
    '''
    交易对实时数据更新，仅用于实盘。仅针对1m及以上维度。
    一个LiveMiner对应一个交易所。处理此交易所下所有数据的监听
    '''
    def __init__(self, exg_name: str, market: str):
        super(LiveMiner, self).__init__(self._on_bar_finish)
        self.exchange = get_exchange(exg_name, market)
        self.auto_prefire = AppConfig.get().get('prefire')
        self.jobs: Dict[str, MinerJob] = dict()

    def sub_pair(self, pair: str, timeframe: str, since: int = None):
        save_tf, check_intv = MinerJob.get_tf_intv(timeframe)
        if self.exchange.market_type == 'future':
            # 期货市场最低维度是1m
            check_intv = max(check_intv, 60.)
        job = self.jobs.get(pair)
        if job and job.timeframe == save_tf:
            fmt_args = [self.exchange.name, pair, job.check_intv, check_intv]
            if job.check_intv <= check_intv:
                logger.debug('miner %s/%s  %.1f, skip: %.1f', *fmt_args)
            else:
                job.check_intv = check_intv
                logger.debug('miner %s/%s check_intv %.1f -> %.1f', *fmt_args)
        else:
            job = MinerJob(pair, save_tf, check_intv, since)
            # 将since改为所属bar的开始，避免第一个bar数据不完整
            tf_msecs = tf_to_secs(timeframe) * 1000
            job.since = job.since // tf_msecs * tf_msecs
            self.jobs[pair] = job
            fmt_args = [self.exchange.name, pair, check_intv, job.fetch_tf, since]
            logger.debug('miner sub %s/%s check_intv %.1f, fetch_tf: %s, since: %d', *fmt_args)

    async def run(self):
        while True:
            try:
                if not self.jobs:
                    await asyncio.sleep(1)
                    continue
                batch_jobs = [v for k, v in self.jobs.items() if v.next_run <= time.time()]
                if not batch_jobs:
                    await asyncio.sleep(self.loop_intv)
                    continue
                batch_jobs = sorted(batch_jobs, key=lambda j: j.next_run)[:MAX_CONC_OHLCV]
                tasks = [self._try_update(j) for j in batch_jobs]
                await asyncio.gather(*tasks)
            except Exception:
                logger.exception(f'miner error {self.exchange.name}')

    def _on_bar_finish(self, pair: str, timeframe: str, bar_row: Tuple):
        from banbot.storage import KLine
        sid = ExSymbol.get_id(self.exchange.name, pair, self.exchange.market_type)
        KLine.insert(sid, timeframe, [bar_row])

    async def _try_update(self, job: MinerJob):
        import ccxt
        from banbot.storage import db
        try:
            next_time = time.time() + job.check_intv
            next_bar = next_time // job.tf_secs * job.tf_secs
            job.next_run = next_bar + job.check_intv * 0.07
            if job.wait_bar and next_bar > job.wait_bar[0] // 1000:
                # 当下次轮询会有新的完成数据时，尽可能在第一时间更新
                job.next_run = next_bar
            # 这里不设置limit，如果外部修改了更新间隔，这里能及时输出期间所有的数据，避免出现delay
            ohlcvs_sml = await self.exchange.fetch_ohlcv(job.pair, job.fetch_tf, since=job.since)
            if not ohlcvs_sml:
                job.next_run -= job.check_intv * 0.9
                return
            job.since = ohlcvs_sml[-1][0] + job.fetch_tfsecs * 1000
            if job.tf_secs > job.fetch_tfsecs:
                # 合并得到保存到数据库周期维度的数据
                old_ohlcvs = [job.wait_bar] if job.wait_bar else []
                # 和旧的bar_row合并更新，判断是否有完成的bar
                ohlcvs, last_finish = build_ohlcvc(ohlcvs_sml, job.tf_secs, ohlcvs=old_ohlcvs)
            else:
                ohlcvs, last_finish = ohlcvs_sml, True
            # 检查是否有完成的bar。写入到数据库
            with db():
                self._on_state_ohlcv(job.pair, job, ohlcvs, last_finish)
            # 发布小间隔数据到redis订阅方
            sre_data = orjson.dumps((ohlcvs_sml, job.fetch_tfsecs))
            async with AsyncRedis() as redis:
                pub_key = f'{self.exchange.name}_{self.exchange.market_type}_{job.pair}'
                await redis.publish(pub_key, sre_data)
        except (ccxt.NetworkError, ccxt.BadRequest):
            logger.exception(f'get live data exception: {job.pair} {job.fetch_tf} {job.tf_secs} {job.since}')


class SpiderJob:
    def __init__(self, action: str, *args, **kwargs):
        self.action = action
        self.args = args
        self.kwargs = kwargs

    def dumps(self) -> bytes:
        data = dict(action=self.action, args=self.args, kwargs=self.kwargs)
        return orjson.dumps(data)


async def start_spider():
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
        self.miners: Dict[str, LiveMiner] = dict()

    async def run_listeners(self):
        await self.conn.subscribe(self._key)
        asyncio.create_task(self._heartbeat())
        LiveSpider._ready = True
        async for msg in self.conn.listen():
            if msg['type'] != 'message':
                continue
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

    async def watch_ohlcv(self, exg_name: str, pair: str, market: str, timeframe: str, since: Optional[int] = None):
        cache_key = f'{exg_name}:{market}'
        miner = self.miners.get(cache_key)
        if not miner:
            miner = LiveMiner(exg_name, market)
            self.miners[cache_key] = miner
            asyncio.create_task(miner.run())
        miner.sub_pair(pair, timeframe, since)

    async def unwatch_pairs(self, exg_name: str, market: str, pairs: List[str]):
        cache_key = f'{exg_name}:{market}'
        miner = self.miners.get(cache_key)
        if not miner or not pairs:
            return
        for p in pairs:
            if p not in miner.jobs:
                continue
            del miner.jobs[p]

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

