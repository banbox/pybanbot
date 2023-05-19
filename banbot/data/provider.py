#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : live_provider.py
# Author: anyongjin
# Date  : 2023/3/28

from tqdm import tqdm

from banbot.data.feeder import *
from banbot.storage import *
from banbot.util.common import logger


class DataProvider:
    feed_cls = DataFeeder

    def __init__(self, config: Config, callback: Callable):
        self.config = config
        self.exg_name = self.config['exchange']['name']
        self._init_args = dict()
        self.holders: List[DataFeeder] = []

        def handler(*args, **kwargs):
            try:
                callback(*args, **kwargs)
            except Exception:
                logger.exception('LiveData Callback Exception %s %s', args, kwargs)
        self._callback = handler

    def unsub_pairs(self, pairs: List[str]):
        old_map = {h.pair: h for h in self.holders}
        removed = []
        for p in pairs:
            hold = old_map.get(p)
            if hold:
                self.holders.remove(hold)
                removed.append(hold)
        return removed

    def sub_pairs(self, pairs: Dict[str, Dict[str, int]]):
        '''
        从数据提供者添加新的交易对订阅。
        返回最小周期变化的交易对(新增/旧对新周期)、预热任务
        '''
        old_map = {h.pair: h for h in self.holders}
        new_pairs, sub_holds, warm_jobs = [], [], []
        for pair, tf_warms in pairs.items():
            hold: DataFeeder = old_map.get(pair)
            if not hold:
                new_pairs.append((pair, tf_warms))
                continue
            old_min_tf = hold.states[0].timeframe
            new_tfs = hold.sub_tflist(*tf_warms.keys())
            cur_min_tf = hold.states[0].timeframe
            if old_min_tf != cur_min_tf:
                sub_holds.append(hold)
            if new_tfs:
                warm_tf = {tf: tf_warms[tf] for tf in new_tfs}
                warm_jobs.append((hold, warm_tf))
        for p, tf_warms in new_pairs:
            hold = self.feed_cls(p, tf_warms, self._callback, **self._init_args)
            self.holders.append(hold)
            sub_holds.append(hold)
            warm_jobs.append((hold, tf_warms))
        return sub_holds, warm_jobs

    def get_latest_ohlcv(self, pair: str):
        for hold in self.holders:
            if hold.pair == pair:
                return hold.states[0].latest
        raise ValueError(f'unknown pair to get price: {pair}')


class HistDataProvider(DataProvider):
    feed_cls = HistDataFeeder

    def __init__(self, config: Config, callback: Callable):
        super(HistDataProvider, self).__init__(config, callback)
        self.tr: TimeRange = self.config.get('timerange')
        self.holders: List[HistDataFeeder] = []
        self.pbar = None
        self.ptime = 0
        self.plast = 0

    def _update_bar(self):
        if self.pbar is None:
            total_p = sum(h.total_len for h in self.holders) / 100
            self.pbar = tqdm(total=round(total_p))
            self.ptime = time.monotonic()
        elif time.monotonic() - self.ptime > 0.3:
            pg_num = round(sum(h.row_id for h in self.holders) / 100)
            self.pbar.update(pg_num - self.plast)
            self.ptime = time.monotonic()
            self.plast = pg_num

    async def down_data(self):
        timeframes = {s.timeframe for hold in self.holders for s in hold.states}
        jobs = KLine.pause_compress(list(timeframes))
        for hold in self.holders:
            await hold.down_if_need()
        KLine.restore_compress(jobs)

    def loop_main(self):
        try:
            while BotGlobal.state == BotState.RUNNING:
                feeder = sorted(self.holders, key=lambda x: x.next_at)[0]
                bar_time = feeder.next_at
                if bar_time >= time.time() * 1000:
                    break
                btime.cur_timestamp = bar_time / 1000
                feeder()
                self._update_bar()
            self.pbar.close()
        except Exception:
            logger.exception('loop data main fail')


class FileDataProvider(HistDataProvider):
    feed_cls = FileDataFeeder

    def __init__(self, config: Config, callback: Callable):
        super(FileDataProvider, self).__init__(config, callback)
        exg_name = config['exchange']['name']
        self.data_dir = os.path.join(config['data_dir'], exg_name)
        self._init_args = dict(
            auto_prefire=config.get('prefire'),
            data_dir=self.data_dir,
            timerange=config.get('timerange')
        )


class DBDataProvider(HistDataProvider):
    feed_cls = DBDataFeeder

    def __init__(self, config: Config, callback: Callable):
        super(DBDataProvider, self).__init__(config, callback)
        self._init_args = dict(
            auto_prefire=config.get('prefire'),
            timerange=config.get('timerange')
        )


class LiveDataProvider(DataProvider):
    feed_cls = LiveDataFeader
    _obj: Optional['LiveDataProvider'] = None

    def __init__(self, config: Config, callback: Callable):
        super(LiveDataProvider, self).__init__(config, callback)
        from banbot.util.redis_helper import AsyncRedis
        self.redis = AsyncRedis()
        self.conn = self.redis.pubsub()
        LiveDataProvider._obj = self

    async def sub_warm_pairs(self, pairs: Dict[str, Dict[str, int]]):
        new_holds, warm_jobs = super(LiveDataProvider, self).sub_pairs(pairs)
        if not new_holds and not warm_jobs:
            return
        from banbot.data.spider import LiveSpider, SpiderJob
        job_list, since_map = [], dict()
        if warm_jobs:
            tf_symbols: Dict[str, List[str]] = dict()
            tf_arg_list: Dict[str, List[dict]] = dict()
            cur_mtime = int(btime.time() * 1000)
            for hold, tf_warms in warm_jobs:
                for tf, warm_num in tf_warms.items():
                    tf_msecs = tf_to_secs(tf) * 1000
                    end_ms = cur_mtime // tf_msecs * tf_msecs
                    start_ms = end_ms - tf_msecs * warm_num
                    if tf not in tf_symbols:
                        tf_symbols[tf] = []
                    tf_symbols[tf].append(hold.pair)
                    if tf not in tf_arg_list:
                        tf_arg_list[tf] = []
                    tf_arg_list[tf].append(dict(start_ms=start_ms, end_ms=end_ms))

            hold_map = {h.pair: h for h, _ in warm_jobs}

            def ohlcv_cb(data, pair, timeframe, **kwargs):
                since_ms = hold_map[pair].warm_tfs({timeframe: data})
                since_map[f'{pair}/{timeframe}'] = since_ms

            exg = get_exchange(self.exg_name)
            for tf, pairs in tf_symbols.items():
                arg_list = tf_arg_list[tf]
                await bulk_ohlcv_do(exg, pairs, tf, arg_list, ohlcv_cb)
        if new_holds:
            for hold in new_holds:
                sub_tf = hold.states[0].timeframe
                since = since_map.get(f'{hold.pair}/{sub_tf}', 0)
                args = [self.exg_name, hold.pair, sub_tf, since]
                job_list.append(SpiderJob('watch_ohlcv', *args))
                await self.conn.subscribe(f'{self.exg_name}_{hold.pair}')
        # 发送消息给爬虫，实时抓取数据
        await LiveSpider.send(*job_list)

    async def unsub_pairs(self, pairs: List[str]):
        '''
        取消订阅交易对数据
        '''
        from banbot.data.spider import LiveSpider, SpiderJob
        removed = super(LiveDataProvider, self).unsub_pairs(pairs)
        if not removed:
            return
        for hold in removed:
            await self.conn.unsubscribe(f'{self.exg_name}_{hold.pair}')
        await LiveSpider.send(SpiderJob('unwatch_pairs', self.exg_name, pairs))

    @classmethod
    def _on_ohlcv_msg(cls, msg: dict):
        exg_name, pair = msg['channel'].decode().split('_')
        # logger.debug('get ohlcv msg: %s', msg)
        if exg_name != cls._obj.exg_name:
            logger.error(f'receive exg not match: {exg_name}, cur: {cls._obj.exg_name}')
            return
        ohlc_arr, fetch_tfsecs = orjson.loads(msg['data'])
        hold = next((sta for sta in cls._obj.holders if sta.pair == pair), None)
        if not hold:
            logger.error(f'receive pair ohlcv not found: {pair}')
            return
        hold.on_new_data(ohlc_arr, fetch_tfsecs)

    @classmethod
    async def watch_ohlcvs(cls):
        assert cls._obj, '`LiveDataProvider` is not initialized yet!'
        logger.info('start watching ohlcvs from spider...')
        async for msg in cls._obj.conn.listen():
            if msg['type'] != 'message':
                continue
            try:
                cls._on_ohlcv_msg(msg)
            except Exception:
                logger.exception(f'handle live ohlcv listen error: {msg}')
