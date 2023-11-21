#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : live_provider.py
# Author: anyongjin
# Date  : 2023/3/28
import time
from banbot.data.feeder import *
from banbot.storage import *
from banbot.util.common import logger
from banbot.util.misc import LazyTqdm


class DataProvider:
    feed_cls = KLineFeeder

    def __init__(self, config: Config, callback: Callable):
        self.config = config
        self.exg_name = self.config['exchange']['name']
        self.market = self.config['market_type']
        self._init_args = dict()
        self.holders: List[KLineFeeder] = []

        async def handler(*args, **kwargs):
            try:
                await callback(*args, **kwargs)
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
            hold: KLineFeeder = old_map.get(pair)
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
        cur_price = MarketPrice.get(pair)
        return [btime.utcstamp(), cur_price, cur_price, cur_price, cur_price, 1]


class HistDataProvider(DataProvider):
    feed_cls = HistKLineFeeder

    def __init__(self, config: Config, callback: Callable):
        super(HistDataProvider, self).__init__(config, callback)
        self.tr: TimeRange = self.config.get('timerange')
        self.holders: List[HistKLineFeeder] = []
        self.pbar = None
        self.ptime = 0
        self.plast = 0

    def _update_bar(self):
        if self.pbar is None:
            total_p = sum(h.total_len for h in self.holders) / 100
            self.pbar = LazyTqdm(total=round(total_p))
            self.ptime = time.monotonic()
        elif time.monotonic() - self.ptime > 0.3:
            pg_num = round(sum(h.row_id for h in self.holders) / 100)
            self.pbar.update(pg_num - self.plast)
            self.ptime = time.monotonic()
            self.plast = pg_num

    async def down_data(self):
        for hold in self.holders:
            await hold.down_if_need()

    async def loop_main(self):
        try:
            while BotGlobal.state == BotState.RUNNING:
                items = []
                for hold in self.holders:
                    items.append((hold, await hold.get_next_at()))
                feeder, bar_time = sorted(items, key=lambda x: x[1])[0]
                if bar_time >= btime.utcstamp():
                    break
                btime.cur_timestamp = bar_time / 1000
                await feeder.invoke()
                self._update_bar()
            if self.pbar:
                self.pbar.close()
        except Exception:
            logger.exception('loop data main fail')


class FileDataProvider(HistDataProvider):
    feed_cls = FileKLineFeeder

    def __init__(self, config: Config, callback: Callable):
        super(FileDataProvider, self).__init__(config, callback)
        exg_name = config['exchange']['name']
        exgs = config.get('exg_data_map') or dict()
        sub_name = f'{exg_name}_{self.market}'
        if sub_name in exgs:
            sub_name = exgs[sub_name]
        self.data_dir = os.path.join(config['data_dir'], sub_name)
        self._init_args = dict(
            auto_prefire=config.get('prefire'),
            data_dir=self.data_dir,
            timerange=config.get('timerange')
        )


class DBDataProvider(HistDataProvider):
    feed_cls = DBKLineFeeder

    def __init__(self, config: Config, callback: Callable):
        super(DBDataProvider, self).__init__(config, callback)
        self._init_args = dict(
            auto_prefire=config.get('prefire'),
            timerange=config.get('timerange')
        )


class LiveDataProvider(DataProvider, KlineLiveConsumer):
    '''
    这是用于交易机器人的实时数据提供器。
    会根据需要预热一部分数据。
    '''
    feed_cls = LiveKLineFeader
    _obj: Optional['LiveDataProvider'] = None

    def __init__(self, config: Config, callback: Callable):
        DataProvider.__init__(self, config, callback)
        KlineLiveConsumer.__init__(self)
        LiveDataProvider._obj = self

    async def sub_warm_pairs(self, pairs: Dict[str, Dict[str, int]]):
        from banbot.data.tools import bulk_ohlcv_do
        new_holds, warm_jobs = super(LiveDataProvider, self).sub_pairs(pairs)
        if not new_holds and not warm_jobs:
            return
        since_map = dict()
        if warm_jobs:
            tf_symbols: Dict[str, List[str]] = dict()
            tf_arg_list: Dict[str, List[dict]] = dict()
            cur_mtime = int(btime.time() * 1000)
            for hold, tf_warms in warm_jobs:
                for tf, warm_num in tf_warms.items():
                    tf_msecs = tf_to_secs(tf) * 1000
                    end_ms = align_tfmsecs(cur_mtime, tf_msecs)
                    start_ms = end_ms - tf_msecs * warm_num
                    if tf not in tf_symbols:
                        tf_symbols[tf] = []
                    tf_symbols[tf].append(hold.pair)
                    if tf not in tf_arg_list:
                        tf_arg_list[tf] = []
                    tf_arg_list[tf].append(dict(start_ms=start_ms, end_ms=end_ms))

            hold_map = {h.pair: h for h, _ in warm_jobs}

            async def ohlcv_cb(data, exs: ExSymbol, timeframe: str, **kwargs):
                since_ms = await hold_map[exs.symbol].warm_tfs({timeframe: data})
                since_map[f'{exs.symbol}/{timeframe}'] = since_ms

            exg = get_exchange(self.exg_name)
            for tf, pairs in tf_symbols.items():
                if tf_to_secs(tf) < 60:
                    # 秒级K线不存储，无法预热
                    continue
                arg_list = tf_arg_list[tf]
                await bulk_ohlcv_do(exg, pairs, tf, arg_list, ohlcv_cb)
        market = self.config['market_type']
        if new_holds:
            watch_jobs = []
            for hold in new_holds:
                sub_tf = hold.states[0].timeframe
                since = since_map.get(f'{hold.pair}/{sub_tf}', 0)
                watch_jobs.append(WatchParam(hold.pair, sub_tf, since))
            # 发送消息给爬虫，实时抓取数据
            await self.watch_jobs(self.exg_name, market, 'ohlcv', watch_jobs)
            if BotGlobal.book_pairs:
                jobs = [WatchParam(h.pair, h.states[0].timeframe) for h in new_holds]
                await self.watch_jobs(self.exg_name, market, 'book', jobs)

    async def unsub_pairs(self, pairs: List[str]):
        '''
        取消订阅交易对数据
        '''
        removed = super(LiveDataProvider, self).unsub_pairs(pairs)
        if not removed:
            return
        market = self.config['market_type']
        pairs = [hold.pair for hold in removed]
        await self.unwatch_jobs(self.exg_name, market, 'ohlcv', pairs)

    async def loop_main(self):
        await self.run_forever('bot')

    @classmethod
    async def _on_ohlcv_msg(cls, exg_name: str, market: str, pair: str, ohlc_arr: List[Tuple],
                            fetch_tfsecs: int, update_tfsecs: float):
        if exg_name != cls._obj.exg_name or cls._obj.market != market:
            logger.warning(f'receive exg not match: {exg_name}, cur: {cls._obj.exg_name}')
            return
        hold = next((sta for sta in cls._obj.holders if sta.pair == pair), None)
        if not hold:
            return
        if update_tfsecs >= fetch_tfsecs:
            # 收到的必定已完成
            await hold.on_new_data(ohlc_arr, fetch_tfsecs)
            return
        # 更新频率低于bar周期，收到的可能未完成
        done_arr = ohlc_arr[:-1]
        if hold.wait_bar and hold.wait_bar[0] < ohlc_arr[0][0]:
            # 新bar出现，认为wait_bar完成
            done_arr.insert(0, hold.wait_bar)
        if len(done_arr):
            await hold.on_new_data(done_arr, fetch_tfsecs)
        last_bar = ohlc_arr[-1]
        if update_tfsecs <= 5 and hold.states[0].tf_secs >= 60:
            # 更新很快，需要的周期相对较长，则要求出现下一个bar时认为完成（走上面逻辑）
            hold.wait_bar = last_bar
            return
        # 更新频率相对不高，或占需要的周期比率较大，近似完成认为完成
        # 这里来的蜡烛和缓存的属于统一周期
        bar_end_ts = hold.wait_bar[0] // 1000 + fetch_tfsecs
        end_lack = bar_end_ts - btime.utctime()
        if end_lack < update_tfsecs * 0.5:
            # 缺少的时间不足更新间隔的一半，认为完成。
            await hold.on_new_data([last_bar], fetch_tfsecs)
            hold.wait_bar = None
            return
        hold.wait_bar = last_bar

