#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : spider.py
# Author: anyongjin
# Date  : 2023/4/25
import os.path

from asyncio import Queue

from banbot.data.tools import *
from banbot.data.wacther import *
from banbot.exchange.crypto_exchange import get_exchange
from banbot.storage import KLine, DisContiError
from banbot.util.banio import ServerIO, BanConn
from banbot.util.tf_utils import *
from banbot.data.cache import BanCache
from banbot.storage import dba, reset_ctx


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
    from banbot.storage.klines import KLine, dba, select
    from banbot.data.toolbox import fill_holes
    await fill_holes()
    pairs = config.get('pairs')
    timerange = config['timerange']
    start_ms = round(timerange.startts * 1000)
    end_ms = round(timerange.stopts * 1000)
    cur_ms = btime.utcstamp()
    end_ms = min(cur_ms, end_ms) if end_ms else cur_ms
    exchange = get_exchange(with_credits=False)
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
        sess = dba.session
        # 加载最新币列表
        await exchange.load_markets()
        all_symbols = list(exchange.get_cur_markets().keys())
        await ExSymbol.ensures(exchange.name, exchange.market_type, all_symbols)
        fts = [ExSymbol.exchange == exchange.name, ExSymbol.market == exchange.market_type]
        if pairs:
            fts.append(ExSymbol.symbol.in_(set(pairs)))
        exs_list: Iterable[ExSymbol] = await sess.scalars(select(ExSymbol).where(*fts))
        symbols = [exs.symbol for exs in exs_list]
        logger.info(f'start download for {len(symbols)} symbols')
        await fast_bulk_ohlcv(exchange, symbols, tf, start_ms, end_ms)
    else:
        data_dir = config['data_dir']
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        for pair in pairs:
            for tf in timeframes:
                await download_to_file(exchange, pair, tf, start_ms, end_ms, data_dir)
                logger.warning(f'{pair}/{tf} down {tr_text} complete')
    await exchange.close()


async def run_down_pairs(args: Dict[str, Any]):
    '''
    解析命令行参数并下载交易对数据
    '''
    config = AppConfig.get()

    async with dba():
        await down_pairs_by_config(config)


"""
*************************** 下面是爬虫部分 ***********************
"""

write_q = Queue(3000)  # 写入数据库的队列，逐个执行，避免写入并发过大出现异常
init_sid: Set[int] = set()  # 不在此集合的，是初次写入，有遗漏，需要抓取缺失的bar


async def _save_init(sid: int, ohlcv: List[Tuple], save_tf: str, skip_first: bool):
    exs = ExSymbol.get_by_id(sid)
    init_sid.add(sid)
    tf_msecs = tf_to_secs(save_tf) * 1000
    if skip_first:
        fetch_end_ms = ohlcv[0][0] + tf_msecs  # 第一个插入的bar时间戳，这个是不全的，需要跳过
        ohlcv = ohlcv[1:]
    else:
        fetch_end_ms = ohlcv[0][0]
    start_ms, end_ms = await KLine.query_range(sid, save_tf)
    if not end_ms or fetch_end_ms <= end_ms:
        # 新的币无历史数据、或当前bar和已插入数据连续，直接插入后续新bar即可
        await KLine.insert(sid, save_tf, ohlcv)
        return

    try_count = 0
    logger.info(f'start first fetch {exs.symbol} {end_ms}-{fetch_end_ms}')
    exchange = get_exchange(exs.exchange, exs.market, with_credits=False)
    while True:
        try_count += 1
        ins_num = await download_to_db(exchange, exs, save_tf, end_ms, fetch_end_ms)
        save_bars = await KLine.query(exs, save_tf, end_ms, fetch_end_ms)
        last_ms = save_bars[-1][0] if save_bars else None
        if last_ms and last_ms + tf_msecs == fetch_end_ms:
            break
        elif try_count > 5:
            logger.error(f'fetch ohlcv fail {exs} {save_tf} {end_ms}-{fetch_end_ms}')
            break
        else:
            # 如果未成功获取最新的bar，等待3s重试（1m刚结束时请求ohlcv可能取不到）
            logger.info(f'query first fail, ins: {ins_num}, last: {last_ms}, wait 3... {exs.symbol}')
            await asyncio.sleep(3)
    await KLine.insert(sid, save_tf, ohlcv)
    logger.info(f'first fetch ok {exs.symbol} {end_ms}-{fetch_end_ms}')


async def consume_db_queue():
    reset_ctx()
    while True:
        try:
            sid, ohlcv, save_tf, skip_first = await write_q.get()
            async with dba():
                if sid not in init_sid:
                    await _save_init(sid, ohlcv, save_tf, skip_first)
                else:
                    try:
                        await KLine.insert(sid, save_tf, ohlcv)
                    except DisContiError as e:
                        logger.warning(f"Kline DisConti {e}, try fill...")
                        await _save_init(sid, ohlcv, save_tf, False)
            write_q.task_done()
        except Exception:
            logger.exception("consume spider write_q error")


def run_consumers(num: int):
    for n in range(num):
        asyncio.create_task(consume_db_queue())


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
        self.since = int(since) if since else align_tfsecs(btime.utctime(), tf_secs) * 1000
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


async def run_price_watch(spider: 'LiveSpider', exchange: CryptoExchange):
    if exchange.market_type != 'future':
        logger.info(f'run_price_watch not support market: {exchange.market_type}, exit')
        return
    send_key = f'update_price_{exchange.name}.{exchange.market_type}'
    while BotGlobal.state == BotState.RUNNING:
        try:
            price_list = await exchange.watch_mark_prices()
            await spider.broadcast(send_key, price_list)
        except ccxt.NetworkError as e:
            logger.error(f'run_price_watch net error: {e}')
            continue
        except Exception:
            logger.exception(f'run_price_watch error')
            continue
    logger.info('run_price_watch stopped')


class WebsocketWatcher:

    def __init__(self, exg_name: str, market: str, pair: str):
        self.exchange = get_exchange(exg_name, market, with_credits=False)
        self.pair = pair
        self.sid = 0
        self.running = True

    async def run(self):
        while self.running:
            try:
                await self.try_update()
            except ccxt.NetworkError as e:
                await asyncio.sleep(0.3)
                tag = f'{self.exchange.name}/{self.exchange.market_type}/{self.pair}'
                logger.error(f'watch {tag} trades net fail: {e}')
            except (ConnectionResetError, ConnectionAbortedError):
                # 连接被重置，需要重新初始化交易所
                logger.warning('exg conn reseted, reconnect to exchange')
                await self.exchange.reconnect()
            except Exception:
                tag = f'{self.exchange.name}/{self.exchange.market_type}/{self.pair}'
                logger.exception(f'watch {tag} trades fail')

    async def try_update(self):
        pass

    def get_sid(self) -> int:
        if self.sid == 0:
            self.sid = ExSymbol.get_id(self.exchange.name, self.exchange.market_type, self.pair)
        return self.sid


class TradesWatcher(WebsocketWatcher):
    '''
    websocket实时交易数据监听，归集得到秒级ohlcv
    监听交易流，从交易流实时归集为s级ohlcv
    归集得到的第一个bar无效。
    每天190个期货币种会有100根针：价格突然超出正常范围，成交量未放大。
    故推荐直接监听ohlcv
    '''
    def __init__(self, spider: 'LiveSpider', exg_name: str, market: str, pair: str):
        self.spider = spider
        super(TradesWatcher, self).__init__(exg_name, market, pair)
        self.state_sec = PairTFCache('1s', 1)  # 用于实时归集通知
        self.state_save = PairTFCache('1m', 60)  # 用于数据库更新

    async def try_update(self):
        details = await self.exchange.watch_trades(self.pair)
        details = trades_to_ohlcv(details)
        # 交易按小维度归集和通知；减少传输数据大小；
        ohlcvs_sml = [self.state_sec.wait_bar] if self.state_sec.wait_bar else []
        pub_tf_secs = self.state_sec.tf_secs
        ohlcvs_sml, _ = build_ohlcvc(details, pub_tf_secs, ohlcvs=ohlcvs_sml, with_count=False)
        if not ohlcvs_sml:
            return
        self.state_sec.wait_bar = ohlcvs_sml[-1]
        ohlcvs_sml = ohlcvs_sml[:-1]  # 完成的秒级ohlcv
        if not ohlcvs_sml:
            return
        # 发送实时的未完成数据
        pub_key = f'uohlcv_{self.exchange.name}_{self.exchange.market_type}_{self.pair}'
        await self.spider.broadcast(pub_key, (ohlcvs_sml, pub_tf_secs, pub_tf_secs))
        # 更新1m级别bar，写入数据库
        ohlcv_old = [self.state_save.wait_bar] if self.state_save.wait_bar else []
        ohlcvs_save, is_finish = build_ohlcvc(ohlcvs_sml, pub_tf_secs, ohlcvs=ohlcv_old)
        ohlcvs_save = get_finish_ohlcvs(self.state_save, ohlcvs_save, is_finish)
        if not ohlcvs_save:
            return
        # 发送已完成数据
        pub_key = f'ohlcv_{self.exchange.name}_{self.exchange.market_type}_{self.pair}'
        await self.spider.broadcast(pub_key, (ohlcvs_save, self.state_save.tf_secs))
        logger.info(f'ws ohlcv: {self.pair} {ohlcvs_save}')
        write_q.put_nowait((self.get_sid(), ohlcvs_save, self.state_save.timeframe, True))


class OhlcvWatcher(WebsocketWatcher):
    '''
    监听trades交易归集得到ohlcv的方式，币安每天190个期货币种会有100根针：
    价格突然超出正常范围，成交量未放大。故使用监听ohlcv方式

    目前仅支持插入期货。
    '''
    def __init__(self, spider: 'LiveSpider', exg_name: str, market: str, pair: str):
        self.spider = spider
        super(OhlcvWatcher, self).__init__(exg_name, market, pair)
        ws_tf = '1m' if market == 'future' else '1s'
        self.state_ws = PairTFCache(ws_tf, tf_to_secs(ws_tf))  # 用于实时归集通知
        self.notify_ts = 0.  # 记录上次通知时间戳，用于ws限流
        self.pbar = None  # 记录上一个bar用于判断是否完成

    async def try_update(self):
        try:
            ohlcvs_sml = await self.exchange.watch_ohlcv(self.pair, self.state_ws.timeframe)
        except ccxt.BadSymbol:
            logger.error(f'{self.pair} not in {self.exchange.name}.{self.exchange.market_type}, stop watch...')
            self.running = False
            return
        if not ohlcvs_sml:
            return
        cur_ts = btime.utctime()
        if cur_ts - self.notify_ts >= 0.9:
            self.notify_ts = cur_ts
            # 发送实时的未完成数据
            pub_key = f'uohlcv_{self.exchange.name}_{self.exchange.market_type}_{self.pair}'
            await self.spider.broadcast(pub_key, (ohlcvs_sml, self.state_ws.tf_secs, 1))
        finish_bars: list = ohlcvs_sml[:-1]
        cur_bar = ohlcvs_sml[-1]
        if not self.pbar:
            self.pbar = cur_bar
        elif cur_bar[0] > self.pbar[0]:
            if not finish_bars or self.pbar[0] > finish_bars[0][0]:
                finish_bars.append(self.pbar)
            self.pbar = cur_bar
        if finish_bars:
            logger.debug('watch ohlcv: %s %s %s', self.pair, self.state_ws.timeframe, finish_bars)
            # 发送已完成数据
            pub_key = f'ohlcv_{self.exchange.name}_{self.exchange.market_type}_{self.pair}'
            await self.spider.broadcast(pub_key, (finish_bars, self.state_ws.tf_secs))
            write_q.put_nowait((self.get_sid(), finish_bars, self.state_ws.timeframe, False))


class LiveMiner:
    loop_intv = 0.5  # 没有任务时，睡眠间隔
    '''
    交易对实时数据更新，仅用于实盘。仅针对1m及以上维度。
    一个LiveMiner对应一个交易所的一个市场。处理此交易所市场下所有数据的监听
    '''
    def __init__(self, spider: 'LiveSpider', exg_name: str, market: str, timeframe: str = '1m'):
        self.spider = spider
        self.exchange = get_exchange(exg_name, market, with_credits=False)
        self.auto_prefire = AppConfig.get().get('prefire')
        self.jobs: Dict[str, MinerJob] = dict()
        self.socks: Dict[str, OhlcvWatcher] = dict()
        self.timeframe = timeframe
        self.tf_msecs = tf_to_secs(self.timeframe) * 1000
        self.ws_pairs = []
        self.ws_started = False

    async def init(self):
        await self.exchange.load_markets()

    def start_watch_price(self):
        if self.exchange.market_type == 'future':
            asyncio.create_task(run_price_watch(self.spider, self.exchange))

    async def sub_pairs(self, pairs: List[str], jtype: str):
        async with ClientIO.lock('edit_pairs'):
            async with dba():
                await ExSymbol.ensures(self.exchange.name, self.exchange.market_type, pairs)
        if jtype == 'ws':
            self.sub_ws_pairs(pairs)
        else:
            for pair in pairs:
                await self.sub_kline(pair)

    def sub_ws_pairs(self, pairs: List[str]):
        cur_pairs = set(self.ws_pairs)
        cur_pairs.update(pairs)
        self.ws_pairs = list(cur_pairs)
        if not self.ws_started:
            self.run_ws_loop()

    def run_ws_loop(self):
        if not self.ws_pairs or self.ws_started:
            return
        self.ws_started = True
        logger.info(f'start watch trades and odbooks for {len(self.ws_pairs)} pairs')

        async def watch_trades():
            while BotGlobal.state == BotState.RUNNING:
                try:
                    trades = await self.exchange.watch_trades_for_symbols(self.ws_pairs)
                    if not trades:
                        continue
                    pair = trades[0]['symbol']
                    pub_key = f'trade_{self.exchange.name}_{self.exchange.market_type}_{pair}'
                    await self.spider.broadcast(pub_key, trades)
                except ccxt.NetworkError as e:
                    logger.error(f'watch_books net error: {e}')
                except Exception:
                    logger.exception(f'watch_books error')
            logger.info('watch_trades stopped.')
            self.ws_started = False

        async def watch_books():
            while BotGlobal.state == BotState.RUNNING:
                # 读取订单簿快照并保存
                try:
                    books = await self.exchange.watch_order_book_for_symbols(self.ws_pairs)
                    if books:
                        pair = books['symbol']
                        pub_key = f'book_{self.exchange.name}_{self.exchange.market_type}_{pair}'
                        data = dict(**books)
                        data['asks'] = list(data['asks'])
                        data['bids'] = list(data['bids'])
                        await self.spider.broadcast(pub_key, data)
                except ccxt.NetworkError as e:
                    logger.error(f'watch_books net error: {e}')
                except Exception:
                    logger.exception(f'watch_books error')
            logger.info('watch_books stopped.')
        asyncio.create_task(watch_trades())
        asyncio.create_task(watch_books())

    async def sub_kline(self, pair: str):
        if self.tf_msecs <= 60000:
            # 1m及以下周期的ohlcv，通过websoc获取
            if pair in self.socks:
                return
            logger.info(f'start ws for {pair}/{self.timeframe}')
            self.socks[pair] = OhlcvWatcher(self.spider, self.exchange.name, self.exchange.market_type, pair)
            asyncio.create_task(self.socks[pair].run())
            return
        if pair in self.jobs:
            return
        logger.info(f'loop fetch for {pair}/{self.timeframe}')
        save_tf, check_intv = MinerJob.get_tf_intv(self.timeframe)
        if self.exchange.market_type == 'future':
            # 期货市场最低维度是1m
            check_intv = max(check_intv, 60.)
        since = await self._init_symbol(pair)
        job = MinerJob(pair, save_tf, check_intv, since)
        # 将since改为所属bar的开始，避免第一个bar数据不完整
        job.since = job.since // self.tf_msecs * self.tf_msecs
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
                logger.debug('update pairs: %s', items)
                tasks = [self._try_update(j) for j in batch_jobs]
                await asyncio.gather(*tasks)
                logger.debug("batch jobs complete")
            except Exception:
                logger.exception(f'miner error {self.exchange.name}')

    async def _try_update(self, job: MinerJob):
        from banbot.util.common import MeasureTime
        import ccxt
        measure = MeasureTime()
        do_print = False
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
            sid = ExSymbol.get_id(self.exchange.name, self.exchange.market_type, job.pair)
            ohlcvs = get_finish_ohlcvs(job, ohlcvs, last_finish)
            if ohlcvs:
                write_q.put_nowait((sid, ohlcvs, job.timeframe, False))
            # 发布小间隔数据到订阅方
            measure.start_for('send_pub')
            pub_key = f'ohlcv_{self.exchange.name}_{self.exchange.market_type}_{job.pair}'
            await self.spider.broadcast(pub_key, (ohlcvs_sml, job.fetch_tfsecs))
            if do_print:
                measure.print_all()
        except (ccxt.NetworkError, ccxt.BadRequest):
            logger.exception(f'get live data exception: {job.pair} {job.fetch_tf} {job.tf_secs} {job.since}')
        except Exception:
            logger.exception(f'spider exception: {job.pair} {job.fetch_tf} {job.tf_secs} {job.since}')

    async def _init_symbol(self, symbol: str) -> int:
        '''
        初始化币对，如果前面遗漏蜡烛太多，下载必要的数据
        '''
        prefetch = 1000
        save_tf, tf_secs = '1m', 60
        tf_msecs = tf_to_secs(save_tf) * 1000
        cur_ms = align_tfmsecs(btime.utcstamp(), tf_msecs)
        start_ms = align_tfmsecs(cur_ms - tf_msecs * prefetch, tf_msecs)
        exs = ExSymbol.get(self.exchange.name, self.exchange.market_type, symbol)
        _, end_ms = await KLine.query_range(exs.id, save_tf)
        if not end_ms or (cur_ms - end_ms) // tf_msecs > 30:
            # 当缺失数据超过30个时，才执行批量下载
            await download_to_db(self.exchange, exs, save_tf, start_ms, cur_ms)
            end_ms = cur_ms
        return end_ms


class LiveSpider(ServerIO):
    '''
    实时数据爬虫；仅用于实盘。负责：实时K线、订单簿等公共数据监听
    历史数据下载请直接调用对应方法，效率更高。
    '''
    obj: Optional['LiveSpider'] = None

    def __init__(self):
        config = AppConfig.get()
        super(LiveSpider, self).__init__(config.get('spider_addr'), 'spider')
        self.miners: Dict[str, LiveMiner] = dict()
        LiveSpider.obj = self

    def get_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> BanConn:
        conn = BanConn(reader, writer, reconnect=False)
        conn.listens.update(
            watch_pairs=self.watch_pairs,
            unwatch_pairs=self.unwatch_pairs,
            get_cache=self.get_cache,
            run_watch_jobs=self.run_watch_jobs,
            watch_top_chg=self.watch_top_chg
        )
        return conn

    async def _get_miner(self, exg_name, market):
        cache_key = f'{exg_name}:{market}'
        miner = self.miners.get(cache_key)
        if not miner:
            miner = LiveMiner(self, exg_name, market)
            self.miners[cache_key] = miner
            await miner.init()
            asyncio.create_task(miner.run())
            logger.info(f'start miner for {exg_name}.{market}')
        return miner

    async def watch_pairs(self, data):
        exg_name, market, jtype, pairs = data
        miner = await self._get_miner(exg_name, market)
        await miner.sub_pairs(pairs, jtype)

    async def unwatch_pairs(self, data):
        exg_name, market, pairs = data
        cache_key = f'{exg_name}:{market}'
        miner = self.miners.get(cache_key)
        if not miner or not pairs:
            return
        for p in pairs:
            if p not in miner.jobs:
                continue
            del miner.jobs[p]

    async def watch_price(self, data):
        exg_name, market = data
        miner = await self._get_miner(exg_name, market)
        miner.start_watch_price()

    def run_watch_jobs(self, data):
        from banbot.worker.watch_job import run_watch_jobs
        asyncio.create_task(run_watch_jobs())

    async def watch_top_chg(self, data):
        from banbot.worker.top_change import TopChange
        await TopChange.start()

    def get_cache(self, data):
        return BanCache.get(data)

    @classmethod
    async def run_spider(cls):
        from banbot.data.toolbox import sync_timeframes, purge_kline_un
        spider = LiveSpider()

        async with dba():
            logger.info('[spider] sync timeframe ranges ...')
            await sync_timeframes()
            await purge_kline_un()
            await ExSymbol.init()
        while True:
            try:
                logger.info('[spider] wait job ...')
                await spider.run_forever()
            except Exception:
                logger.exception('spider listen fail, rebooting...')
            await asyncio.sleep(1)
            logger.info('try restart spider...')

    @classmethod
    async def run_timer_reset(cls):
        """每隔24H重置ccxt交易所，不然websocket订阅会有问题"""
        from croniter import croniter
        # 在每天的00:11重置更新
        loop = croniter('11 0 * * *')
        while BotGlobal.state == BotState.RUNNING:
            wait_ts = loop.next() - btime.time()
            await asyncio.sleep(wait_ts)
            if not cls.obj:
                continue
            for key, miner in cls.obj.miners.items():
                logger.info(f'restart exchange: {key}')
                try:
                    await miner.exchange.reconnect()
                except Exception:
                    logger.exception(f'restart {key} fail')


async def run_spider_forever(args: dict = None):
    '''
    此函数仅用于从命令行启动
    '''
    BotGlobal.state = BotState.RUNNING
    asyncio.create_task(LiveSpider.run_timer_reset())
    run_consumers(5)
    await LiveSpider.run_spider()


def run_spider_prc():
    '''
    此方法仅用于子进程中启动爬虫
    '''
    asyncio.run(run_spider_forever())
