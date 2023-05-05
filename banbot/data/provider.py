#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : live_provider.py
# Author: anyongjin
# Date  : 2023/3/28
import six
import time

from banbot.data.feeder import *
from banbot.config import *
from banbot.util.common import logger
from banbot.storage.common import *
from tqdm import tqdm


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

    def sub_pairs(self, pairs: Dict[Tuple[str, int], Set[str]]):
        old_map = {h.pair: h for h in self.holders}
        new_pairs = []
        for (pair, warm_secs), tf in pairs.items():
            tf_list = [tf] if isinstance(tf, six.string_types) else tf
            hold: DataFeeder = old_map.get(pair)
            if not hold:
                new_pairs.append((pair, warm_secs, tf_list))
                continue
            new_tfs = hold.sub_tflist(*tf_list)
            if new_tfs:
                hold.warm_tfs(warm_secs, *new_tfs)
        new_holds = [self.feed_cls(p, warm_secs, tf_list, self._callback, **self._init_args)
                     for p, warm_secs, tf_list in new_pairs]
        self.holders.extend(new_holds)
        return new_holds

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

    def loop_main(self):
        try:
            while BotGlobal.state == BotState.RUNNING:
                feeder = sorted(self.holders, key=lambda x: x.next_at)[0]
                bar_time = feeder.next_at
                if bar_time >= time.time() * 1000:
                    break
                btime.cur_timestamp = bar_time
                feeder()
                self._update_bar()
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

    def sub_pairs(self, pairs: Dict[Tuple[str, int], Set[str]]):
        new_holds = super(LiveDataProvider, self).sub_pairs(pairs)
        if not new_holds:
            return
        from banbot.data.spider import LiveSpider, SpiderJob
        job_list = []
        for hold in new_holds:
            tfs = [st.timeframe for st in hold.states]
            since_ms = hold.warm_tfs(hold.warm_secs, *tfs)
            job_list.append(SpiderJob('watch_ohlcv', self.exg_name, hold.pair, since_ms))
            self.conn.subscribe(hold.pair)
        # 发送消息给爬虫，实时抓取数据
        LiveSpider.send(*job_list)

    @classmethod
    def _on_ohlcv_msg(cls, msg: dict):
        exg_name, pair = msg['channel'].split('_')
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
        async for msg in cls._obj.conn.listen():
            if msg['type'] != 'message':
                continue
            try:
                cls._on_ohlcv_msg(msg)
            except Exception:
                logger.exception(f'handle live ohlcv listen error: {msg}')

