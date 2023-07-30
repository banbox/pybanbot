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
from banbot.storage import KLine, DisContiError


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
    from banbot.data.toolbox import fill_holes
    await fill_holes()
    pairs = config['pairs']
    timerange = config['timerange']
    start_ms = round(timerange.startts * 1000)
    end_ms = round(timerange.stopts * 1000)
    cur_ms = btime.utcstamp()
    end_ms = min(cur_ms, end_ms) if end_ms else cur_ms
    exchange = get_exchange()
    timeframes = config['timeframes']
    tr_text = btime.to_datestr(start_ms) + ' - ' + btime.to_datestr(end_ms)
    if config['medium'] == 'db':
        tf = timeframes[0]
        if len(timeframes) > 1:
            logger.error('only one timeframe should be given to download into db')
            return
        if tf not in KLine.down_tfs:
            logger.error(f'can only download kline: {KLine.down_tfs}, current: {tf}')
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


class MinerJob(PairTFCache):
    def __init__(self, pair: str, save_tf: str, check_intv: float, since: Optional[int] = None):
        '''
        K线抓取任务。
        :param pair:
        :param save_tf: 保存到数据库的周期维度，必须是1m或1h，这个<=实际分析用到的维度
        :param check_intv: 检查更新间隔。秒。需要根据实际使用维度计算。此值可能更新
        :param since: 抓取的开始时间，未提供默认从当前时间所属bar开始抓取
        '''
        assert save_tf in KLine.down_tfs, f'MinerJob save_tf must in {KLine.down_tfs}, given: {save_tf}'
        tf_secs = tf_to_secs(save_tf)
        super(MinerJob, self).__init__(save_tf, tf_secs)
        self.pair: str = pair
        self.check_intv = check_intv
        self.fetch_tf = '1s' if self.check_intv < 60 else '1m'
        self.fetch_tfsecs = tf_to_secs(self.fetch_tf)
        self.since = int(since) if since else int(btime.utctime() // tf_secs * 1000)
        self.next_run = self.since / 1000

    @classmethod
    def get_tf_intv(cls, timeframe: str) -> Tuple[str, float]:
        '''
        从需要使用的分析维度，计算应该保存的维度和更新间隔。
        '''
        cur_tfsecs = tf_to_secs(timeframe)
        save_tf = KLine.get_down_tf(timeframe)
        check_intv = get_check_interval(cur_tfsecs)
        return save_tf, check_intv


def get_finish_ohlcvs(job: PairTFCache, ohlcvs: List[Tuple], last_finish: bool) -> List[Tuple]:
    if not ohlcvs:
        return ohlcvs
    job.wait_bar = None
    if not last_finish:
        job.wait_bar = ohlcvs[-1]
        ohlcvs = ohlcvs[:-1]
    return ohlcvs


class WebsocketWatcher:
    def __init__(self, exg_name: str, market: str, pair: str):
        self.exchange = get_exchange(exg_name, market)
        self.pair = pair
        self.sid = 0
        self.first_insert = True  # 第一次保存前，有遗漏，从接口抓取
        self.running = True

    async def run(self):
        while self.running:
            try:
                await self.try_update()
            except ccxt.NetworkError as e:
                await asyncio.sleep(0.3)
                logger.error(f'watch trades net fail: {e}')
            except Exception:
                logger.exception('watch trades fail')

    async def try_update(self):
        pass

    async def save_init(self, ohlcv: List[Tuple], save_tf: str, skip_first: bool):
        exs = ExSymbol.get(self.exchange.name, self.exchange.market_type, self.pair)
        self.sid = exs.id
        self.first_insert = False
        tf_msecs = tf_to_secs(save_tf) * 1000
        if skip_first:
            fetch_end_ms = ohlcv[0][0] + tf_msecs  # 第一个插入的bar时间戳，这个是不全的，需要跳过
            ohlcv = ohlcv[1:]
        else:
            fetch_end_ms = ohlcv[0][0]
        start_ms, end_ms = KLine.query_range(self.sid, save_tf)
        if not end_ms or fetch_end_ms <= end_ms:
            # 新的币无历史数据、或当前bar和已插入数据连续，直接插入后续新bar即可
            KLine.insert(self.sid, save_tf, ohlcv)
            return

        async def fetch_and_save():
            try_count = 0
            logger.info(f'start first fetch {self.pair} {end_ms}-{fetch_end_ms}')
            while True:
                try_count += 1
                ins_num = await download_to_db(self.exchange, exs, save_tf, end_ms, fetch_end_ms)
                save_bars = KLine.query(exs, save_tf, end_ms, fetch_end_ms)
                last_ms = save_bars[-1][0] if save_bars else None
                if last_ms and last_ms + tf_msecs == fetch_end_ms:
                    break
                elif try_count > 5:
                    logger.error(f'fetch ohlcv fail {exs} {save_tf} {end_ms}-{fetch_end_ms}')
                    break
                else:
                    # 如果未成功获取最新的bar，等待3s重试（1m刚结束时请求ohlcv可能取不到）
                    logger.info(f'query first fail, ins: {ins_num}, last: {last_ms}, wait 3... {self.pair}')
                    await asyncio.sleep(3)
            KLine.insert(self.sid, save_tf, ohlcv)
            logger.info(f'first fetch ok {self.pair} {end_ms}-{fetch_end_ms}')
        # 异步执行获取和存储，避免影响秒级更新
        asyncio.create_task(fetch_and_save())

    async def save_ohlcvs(self, ohlcv: List[Tuple], save_tf: str, skip_first: bool):
        if self.first_insert:
            await self.save_init(ohlcv, save_tf, skip_first)
        else:
            try:
                KLine.insert(self.sid, save_tf, ohlcv)
            except DisContiError as e:
                logger.warning(f"Kline DisConti {e}, try fill...")
                await self.save_init(ohlcv, save_tf, False)


class TradesWatcher(WebsocketWatcher):
    '''
    websocket实时交易数据监听，归集得到秒级ohlcv
    监听交易流，从交易流实时归集为s级ohlcv
    归集得到的第一个bar无效。
    每天190个期货币种会有100根针：价格突然超出正常范围，成交量未放大。
    故推荐直接监听ohlcv
    '''
    def __init__(self, exg_name: str, market: str, pair: str):
        super(TradesWatcher, self).__init__(exg_name, market, pair)
        self.state_sec = PairTFCache('1s', 1)  # 用于实时归集通知
        self.state_save = PairTFCache('1m', 60)  # 用于数据库更新

    async def try_update(self):
        details = await self.exchange.watch_trades(self.pair)
        details = trades_to_ohlcv(details)
        # 交易按小维度归集和通知；减少传输数据大小；
        ohlcvs_sml = [self.state_sec.wait_bar] if self.state_sec.wait_bar else []
        ohlcvs_sml, _ = build_ohlcvc(details, self.state_sec.tf_secs, ohlcvs=ohlcvs_sml, with_count=False)
        if not ohlcvs_sml:
            return
        self.state_sec.wait_bar = ohlcvs_sml[-1]
        ohlcvs_sml = ohlcvs_sml[:-1]  # 完成的秒级ohlcv
        if not ohlcvs_sml:
            return
        sre_data = orjson.dumps((ohlcvs_sml, self.state_sec.tf_secs))
        async with AsyncRedis() as redis:
            pub_key = f'ohlcv_{self.exchange.name}_{self.exchange.market_type}_{self.pair}'
            await redis.publish(pub_key, sre_data)
        # 更新1m级别bar，写入数据库
        ohlcv_old = [self.state_save.wait_bar] if self.state_save.wait_bar else []
        ohlcvs_save, is_finish = build_ohlcvc(ohlcvs_sml, self.state_save.tf_secs, ohlcvs=ohlcv_old)
        ohlcvs_save = get_finish_ohlcvs(self.state_save, ohlcvs_save, is_finish)
        if not ohlcvs_save:
            return
        logger.info(f'ws ohlcv: {self.pair} {ohlcvs_save}')
        from banbot.storage import db
        with db():
            await self.save_ohlcvs(ohlcvs_save, self.state_save.timeframe, True)


class OhlcvWatcher(WebsocketWatcher):
    '''
    监听trades交易归集得到ohlcv的方式，币安每天190个期货币种会有100根针：
    价格突然超出正常范围，成交量未放大。故使用监听ohlcv方式

    目前仅支持插入期货。
    '''
    def __init__(self, exg_name: str, market: str, pair: str):
        super(OhlcvWatcher, self).__init__(exg_name, market, pair)
        ws_tf = '1m' if market == 'future' else '1s'
        self.state_ws = PairTFCache(ws_tf, tf_to_secs(ws_tf))  # 用于实时归集通知
        self.notify_ts = 0.  # 记录上次通知时间戳，用于ws限流
        self.pbar = None  # 记录上一个bar用于判断是否完成

    async def try_update(self):
        ohlcvs_sml = await self.exchange.watch_ohlcv(self.pair, self.state_ws.timeframe)
        if not ohlcvs_sml:
            return
        cur_ts = btime.utctime()
        if cur_ts - self.notify_ts >= 0.9:
            self.notify_ts = cur_ts
            sre_data = orjson.dumps((ohlcvs_sml, self.state_ws.tf_secs))
            async with AsyncRedis() as redis:
                pub_key = f'ohlcv_{self.exchange.name}_{self.exchange.market_type}_{self.pair}'
                await redis.publish(pub_key, sre_data)
        finish_bars: list = ohlcvs_sml[:-1]
        cur_bar = ohlcvs_sml[-1]
        if not self.pbar:
            self.pbar = cur_bar
        elif cur_bar[0] > self.pbar[0]:
            if not finish_bars or self.pbar[0] > finish_bars[0][0]:
                finish_bars.append(self.pbar)
            self.pbar = cur_bar
        if finish_bars:
            from banbot.storage import db
            with db():
                await self.save_ohlcvs(finish_bars, self.state_ws.timeframe, False)


class LiveMiner:
    loop_intv = 0.5  # 没有任务时，睡眠间隔
    '''
    交易对实时数据更新，仅用于实盘。仅针对1m及以上维度。
    一个LiveMiner对应一个交易所的一个市场。处理此交易所市场下所有数据的监听
    '''
    def __init__(self, exg_name: str, market: str):
        self.exchange = get_exchange(exg_name, market)
        self.auto_prefire = AppConfig.get().get('prefire')
        self.jobs: Dict[str, MinerJob] = dict()
        self.socks: Dict[str, OhlcvWatcher] = dict()

    def sub_pair(self, pair: str, timeframe: str, since: int = None):
        tf_msecs = tf_to_secs(timeframe) * 1000
        if tf_msecs <= 60000:
            # 1m及以下周期的ohlcv，通过websoc获取
            if pair in self.socks:
                return
            self.socks[pair] = OhlcvWatcher(self.exchange.name, self.exchange.market_type, pair)
            asyncio.create_task(self.socks[pair].run())
            return
        save_tf, check_intv = MinerJob.get_tf_intv(timeframe)
        if self.exchange.market_type == 'future':
            # 期货市场最低维度是1m
            check_intv = max(check_intv, 60.)
        job = self.jobs.get(pair)
        if job and job.timeframe == save_tf:
            fmt_args = [self.exchange.name, pair, job.check_intv, check_intv]
            if job.check_intv <= check_intv:
                logger.info('miner %s/%s  %.1f, skip: %.1f', *fmt_args)
            else:
                job.check_intv = check_intv
                logger.info('miner %s/%s check_intv %.1f -> %.1f', *fmt_args)
        else:
            job = MinerJob(pair, save_tf, check_intv, since)
            # 将since改为所属bar的开始，避免第一个bar数据不完整
            job.since = job.since // tf_msecs * tf_msecs
            self.jobs[pair] = job
            fmt_args = [self.exchange.name, pair, check_intv, job.fetch_tf, since]
            logger.info('miner sub %s/%s check_intv %.1f, fetch_tf: %s, since: %d', *fmt_args)

    async def run(self):
        while True:
            try:
                if not self.jobs:
                    await asyncio.sleep(1)
                    continue
                cur_time = btime.utctime()
                batch_jobs = [v for k, v in self.jobs.items() if v.next_run <= cur_time]
                if not batch_jobs:
                    await asyncio.sleep(self.loop_intv)
                    continue
                batch_jobs = sorted(batch_jobs, key=lambda j: j.next_run)[:MAX_CONC_OHLCV]
                items = [f'{j.pair}:{j.since}' for j in batch_jobs]
                logger.info(f'update pairs: {"  ".join(items)}')
                tasks = [self._try_update(j) for j in batch_jobs]
                await asyncio.gather(*tasks)
                logger.info("batch jobs complete")
            except Exception:
                logger.exception(f'miner error {self.exchange.name}')

    async def _try_update(self, job: MinerJob):
        from banbot.util.common import MeasureTime
        import ccxt
        from banbot.storage import db
        measure = MeasureTime()
        do_print = True
        try:
            job.next_run += job.check_intv
            # 这里不设置limit，如果外部修改了更新间隔，这里能及时输出期间所有的数据，避免出现delay
            measure.start_for(f'fetch:{job.pair}')
            ohlcvs_sml = await self.exchange.fetch_ohlcv(job.pair, job.fetch_tf, since=job.since)
            if not ohlcvs_sml:
                if do_print:
                    measure.print_all()
                return
            measure.start_for('build_ohlcv')
            job.since = ohlcvs_sml[-1][0] + job.fetch_tfsecs * 1000
            if job.tf_secs > job.fetch_tfsecs:
                # 合并得到保存到数据库周期维度的数据
                old_ohlcvs = [job.wait_bar] if job.wait_bar else []
                # 和旧的bar_row合并更新，判断是否有完成的bar
                ohlcvs, last_finish = build_ohlcvc(ohlcvs_sml, job.tf_secs, ohlcvs=old_ohlcvs)
            else:
                ohlcvs, last_finish = ohlcvs_sml, True
            # 检查是否有完成的bar。写入到数据库
            measure.start_for(f'write_db:{len(ohlcvs)}')
            with db():
                sid = ExSymbol.get_id(self.exchange.name, self.exchange.market_type, job.pair)
                ohlcvs = get_finish_ohlcvs(job, ohlcvs, last_finish)
                KLine.insert(sid, job.timeframe, ohlcvs)
            # 发布小间隔数据到redis订阅方
            measure.start_for('send_pub')
            sre_data = orjson.dumps((ohlcvs_sml, job.fetch_tfsecs))
            async with AsyncRedis() as redis:
                pub_key = f'ohlcv_{self.exchange.name}_{self.exchange.market_type}_{job.pair}'
                await redis.publish(pub_key, sre_data)
            if do_print:
                measure.print_all()
        except (ccxt.NetworkError, ccxt.BadRequest):
            logger.exception(f'get live data exception: {job.pair} {job.fetch_tf} {job.tf_secs} {job.since}')
        except Exception:
            logger.exception(f'spider exception: {job.pair} {job.fetch_tf} {job.tf_secs} {job.since}')


class SpiderJob:
    def __init__(self, action: str, *args, **kwargs):
        self.action = action
        self.args = args
        self.kwargs = kwargs

    def dumps(self) -> bytes:
        data = dict(action=self.action, args=self.args, kwargs=self.kwargs)
        return orjson.dumps(data)


class LiveSpider(RedisChannel):
    _key = 'spider'
    _job_key = 'spiderjobs'

    '''
    实时数据爬虫；仅用于实盘。负责：实时K线、订单簿等公共数据监听
    历史数据下载请直接调用对应方法，效率更高。
    '''
    def __init__(self):
        super(LiveSpider, self).__init__()
        self.miners: Dict[str, LiveMiner] = dict()

        def call_self(msg_key, msg_data):
            asyncio.create_task(self._run_job(msg_data))
        self.listeners.append((self._key, call_self))

    async def _run_job(self, params: dict):
        action, args, kwargs = params['action'], params['args'], params['kwargs']
        try:
            from banbot.storage import db
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

    async def watch_ohlcv(self, exg_name: str, pair: str, market: str, timeframe: str, since: Optional[int] = None):
        cache_key = f'{exg_name}:{market}'
        miner = self.miners.get(cache_key)
        if not miner:
            miner = LiveMiner(exg_name, market)
            self.miners[cache_key] = miner
            asyncio.create_task(miner.run())
            logger.info(f'start miner for {exg_name}.{market}')
        if pair not in miner.jobs and not since:
            # 新监听的币，且未指定开始时间戳，则初始化获取时间戳
            since = await self._init_symbol(exg_name, market, pair, timeframe)
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

    async def _init_symbol(self, exg_name: str, market_type: str, symbol: str, timeframe: str) -> int:
        '''
        初始化币对，如果前面遗漏蜡烛太多，下载必要的数据
        '''
        prefetch = 1000
        exchange = get_exchange(exg_name, market_type)
        min_save_tf, min_tf_secs = '1m', 60
        save_tf, save_tfsec = min_save_tf, min_tf_secs
        tf_secs = tf_to_secs(timeframe)
        if tf_secs > min_tf_secs:
            if timeframe not in KLine.down_tfs:
                raise ValueError(f'{timeframe} not allowed : {symbol}, {exg_name} {market_type}')
            save_tf, save_tfsec = timeframe, tf_secs
        tf_msecs = tf_secs * 1000
        cur_ms = btime.utcstamp() // tf_msecs * tf_msecs
        start_ms = (cur_ms - tf_msecs * prefetch) // tf_msecs * tf_msecs
        exs = ExSymbol.get(exg_name, market_type, symbol)
        _, end_ms = KLine.query_range(exs.id, save_tf)
        if not end_ms or (cur_ms - end_ms) // tf_msecs > 30:
            # 当缺失数据超过30个时，才执行批量下载
            await download_to_db(exchange, exs, save_tf, start_ms, cur_ms)
            end_ms = cur_ms
        return end_ms

    async def _init_exg_market(self, exg_name: str, market_type: str):
        '''
        从redis获取给定交易所、给定市场下，需要抓取的币对
        '''
        key = f'{self._job_key}_{exg_name}_{market_type}'
        pairs: dict = await self.redis.hgetall(key)
        if not pairs:
            return
        logger.info(f'init {exg_name}.{market_type} with {len(pairs)} symbols: {pairs}')
        prev_tip, completes = time.monotonic(), []
        i = -1
        for symbol, timeframe in pairs.items():
            i += 1
            if time.monotonic() - prev_tip > 5:
                logger.info(f'[{i}/{len(pairs)}] complete {len(completes)}: {completes}')
                prev_tip, completes = time.monotonic(), []
            await self.watch_ohlcv(exg_name, symbol, market_type, timeframe)
            completes.append(symbol)
        logger.info(f'ALL {len(pairs)} symbols listening for {exg_name}.{market_type}')

    async def init_pairs(self):
        '''
        从redis获取需要抓取的比对，检查上次时间，如果缺失蜡烛太多，则进行预下载
        # 检查需要监听的交易对历史数据，先批量下载缺失的数据，然后再监听，逐个处理。
        # 这里只需要指定下载前面1K个，如果中间缺失过多，会自动全部下载
        '''
        space_keys = await self.redis.list_keys(self._job_key + '*')
        exg_markets = [k.split('_')[1:] for k in space_keys]
        exg_markets = sorted(exg_markets, key=lambda x: x[0])
        from itertools import groupby
        gps = groupby(exg_markets, key=lambda x: x[0])
        for key, gp in gps:
            gp_data = list(gp)
            for exg_name, market in gp_data:
                await self._init_exg_market(exg_name, market)

    @classmethod
    async def run_spider(cls):
        from banbot.data.toolbox import sync_timeframes, purge_kline_un
        redis = AsyncRedis()
        if await redis.get(cls._key):
            return
        async with redis.lock(cls._key, acquire_timeout=5, lock_timeout=4):
            if await redis.get(cls._key):
                return
            from banbot.storage import db
            spider = LiveSpider()
            asyncio.create_task(spider._heartbeat())
        # 注册策略信号计算事件处理
        from banbot.worker.sig_sync import reg_redis_event
        await reg_redis_event()

        with db():
            logger.info('[spider] sync timeframe ranges ...')
            sync_timeframes()
            purge_kline_un()
            logger.info('[spider] init exchange, markets...')
            await spider.init_pairs()
        while True:
            try:
                logger.info('[spider] wait job ...')
                await spider.conn.subscribe(spider._key)
                await spider.run()
            except Exception:
                logger.exception('spider listen fail, rebooting...')
            await asyncio.sleep(1)
            logger.info('try restart spider...')

    @classmethod
    async def send(cls, *job_list: SpiderJob):
        '''
        发送命令到爬虫。不等待执行完成；发送成功即返回。
        '''
        redis = AsyncRedis()
        if not await redis.get(cls._key):
            logger.error(f'spider not started, {len(job_list)} job cached')
            return
        for job in job_list:
            await redis.publish(cls._key, job.dumps())


async def run_spider_forever(args: dict):
    '''
    此函数仅用于从命令行启动
    '''
    from banbot.worker.top_change import TopChange
    from banbot.worker.sig_sync import run_tdsig_updater
    logger.info('start top change update timer...')
    await TopChange.start()
    asyncio.create_task(run_tdsig_updater())
    await LiveSpider.run_spider()
    # await TopChange.clear()
