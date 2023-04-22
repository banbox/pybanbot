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
from banbot.util.misc import run_async
from banbot.exchange.crypto_exchange import CryptoExchange
from banbot.storage.common import *
from asyncio import gather
from tqdm import tqdm


class DataProvider:
    feed_cls = PairDataFeeder

    def __init__(self, config: Config, exchange: CryptoExchange):
        self.config = config
        self.exchange = exchange
        self._init_args = dict()
        self.holders: List[PairDataFeeder] = []
        self.min_interval = 1
        self.producer_pairs: Dict[str, List[str]] = {}

    def sub_pairs(self, pairs: Dict[str, Union[str, Iterable[str]]]):
        old_map = {h.pair: h for h in self.holders}
        new_pairs = []
        for p, tf in pairs.items():
            tf_list = [tf] if isinstance(tf, six.string_types) else tf
            hold = old_map.get(p)
            if not hold:
                new_pairs.append((p, tf_list))
                continue
            hold.sub_tflist(*tf_list)
        for p, tf_list in new_pairs:
            self.holders.append(self.feed_cls(p, tf_list, **self._init_args))
        if self.holders:
            self.min_interval = min(hold.min_interval for hold in self.holders)

    def get_latest_ohlcv(self, pair: str):
        for hold in self.holders:
            if hold.pair == pair:
                return hold.states[0].bar_row
        raise ValueError(f'unknown pair to get price: {pair}')

    def _set_warmup(self, count: int) -> int:
        '''
        设置数据源的预热数量，返回需要回顾的秒数。用于LIVE、DRY_RUN
        :param count:
        :return:
        '''
        back_secs = 0
        for hold in self.holders:
            hold.warmup_num = count
            back_secs = max(back_secs, hold.states[-1].tf_secs * count)
        return back_secs

    @classmethod
    def _wrap_callback(cls, callback: Callable):
        def handler(*args, **kwargs):
            try:
                run_async(callback, *args, **kwargs)
            except Exception:
                logger.exception('LiveData Callback Exception %s %s', args, kwargs)
        return handler

    def _set_callback(self, callback: Callable):
        wrap_callback = self._wrap_callback(callback)
        for hold in self.holders:
            hold.callback = wrap_callback

    @classmethod
    def create_holders(cls, pairlist: List[Tuple[str, str]], **kwargs) -> List[PairDataFeeder]:
        pair_groups: Dict[str, Set[str]] = dict()
        for pair, timeframe in pairlist:
            if pair not in pair_groups:
                pair_groups[pair] = set()
            pair_groups[pair].add(timeframe)
        return [cls.feed_cls(pair, list(tf_set), **kwargs) for pair, tf_set in pair_groups.items()]

    async def process(self):
        await gather(*[hold.try_fetch() for hold in self.holders])
        update_num = 0
        for hold in self.holders:
            if hold.try_update():
                update_num += 1
        return update_num

    async def loop_main(self, warmup_num: int):
        # 先调用一次数据加载，确保预热阶段、数据加载等应用完成
        last_end = time.monotonic()
        try:
            await self._first_call(warmup_num)
            while BotGlobal.state == BotState.RUNNING:
                left_sleep = self.min_interval
                is_realtime = btime.run_mode in TRADING_MODES
                if is_realtime:
                    left_sleep -= time.monotonic() - last_end
                if left_sleep > 0:
                    await asyncio.sleep(left_sleep)
                    if not is_realtime:
                        btime.cur_timestamp += left_sleep
                last_end = time.monotonic()
                try:
                    if await self.process():
                        self._loop_update()
                except EOFError:
                    logger.warning("data loop complete")
                    BotGlobal.state = BotState.STOPPED
                    break
        except Exception:
            logger.exception('loop data main fail')

    def _loop_update(self):
        pass

    def _reset_state_times(self):
        for hold in self.holders:
            for state in hold.states:
                state.last_check = None

    async def _first_call(self, warmup_num: int):
        back_secs = self._set_warmup(warmup_num)
        ts_inited = True
        if btime.run_mode in TRADING_MODES:
            btime.cur_timestamp = time.time() - back_secs
            btime.run_mode = RunMode.OTHER
        else:
            timerange: TimeRange = self.config.get('timerange')
            if timerange and timerange.startts:
                timerange.startts -= back_secs
                btime.cur_timestamp = timerange.startts
            else:
                ts_inited = False
        await self.process()
        if not ts_inited:
            # 计算第一个bar的实际对应的系统时间戳
            ts_val = 0
            for hold in self.holders:
                state = hold.states[0]
                ts_val = max(ts_val, state.bar_row[0] / 1000 + state.tf_secs - hold.prefire_secs)
            btime.cur_timestamp = ts_val
            self._reset_state_times()

    async def fetch_ohlcv(self, pair: str, timeframe: str, ts_from: float, ts_to: Optional[float] = None):
        '''
        获取给定交易对，给定时间维度，给定范围的K线数据。
        此接口不做缓存，如需缓存调用方自行处理。
        实时模式从交易所获取。
        非实时模式先尝试从本地读取，不存在时从交易所获取。
        :param pair:
        :param timeframe:
        :param ts_from:
        :param ts_to:
        :return:
        '''
        tf_secs = timeframe_to_seconds(timeframe)
        if not ts_to:
            ts_to = btime.time()
        ohlcv, tranges = [], [(round(ts_from * 1000), round(ts_to * 1000))]
        if btime.run_mode not in TRADING_MODES:
            exg_name = self.config['exchange']['name']
            data_dir = os.path.join(self.config['data_dir'], exg_name)
            ohlcv = LocalPairDataFeeder.load_data(data_dir, pair, timeframe, ts_from, ts_to).tolist()
            if ohlcv:
                old_rg = tranges[0]
                tranges = []
                if ohlcv[0][0] - tf_secs > old_rg[0]:
                    tranges.append((old_rg[0], ohlcv[0][0]))
                if ohlcv[-1][0] + tf_secs < old_rg[-1]:
                    tranges.append((ohlcv[-1][0] + 1, old_rg[-1]))
        for rg in tranges:
            limit = round((rg[1] - rg[0]) // 1000 / tf_secs)
            ohlc_arr = await self.exchange.fetch_ohlcv_plus(pair, timeframe, rg[0], limit)
            if not len(ohlc_arr):
                continue
            if not ohlcv:
                ohlcv = ohlc_arr
            elif ohlc_arr[0][0] < ohlcv[0][0]:
                assert ohlc_arr[-1][0] < ohlcv[0][0]
                ohlcv[:0] = ohlc_arr
            else:
                assert ohlc_arr[0][0] > ohlcv[-1][0]
                ohlcv.extend(ohlc_arr)
        return ohlcv

    def get_producer_pairs(self, producer_name: str = 'default') -> List[str]:
        return self.producer_pairs.get(producer_name, []).copy()


class LocalDataProvider(DataProvider):
    feed_cls = LocalPairDataFeeder

    def __init__(self, config: Config, exchange: CryptoExchange, callback: Callable):
        super(LocalDataProvider, self).__init__(config, exchange)
        exg_name = config['exchange']['name']
        self.data_dir = os.path.join(config['data_dir'], exg_name)
        self._init_args = dict(
            auto_prefire=config.get('prefire'),
            data_dir=self.data_dir,
            timerange=config.get('timerange')
        )
        self._set_callback(callback)
        self.pbar = None
        self.ptime = 0
        self.plast = 0

    def _loop_update(self):
        if self.pbar is None:
            total_p = sum(h.total_len for h in self.holders) / 100
            self.pbar = tqdm(total=round(total_p))
            self.ptime = time.monotonic()
        elif time.monotonic() - self.ptime > 0.3:
            pg_num = round(sum(h.row_id for h in self.holders) / 100)
            self.pbar.update(pg_num - self.plast)
            self.ptime = time.monotonic()
            self.plast = pg_num


class LiveDataProvider(DataProvider):
    feed_cls = LivePairDataFeader

    def __init__(self, config: Config, exchange: CryptoExchange, callback: Callable):
        super(LiveDataProvider, self).__init__(config, exchange)
        self._init_args = dict(
            auto_prefire=config.get('prefire'),
            exchange=exchange
        )
        self._set_callback(callback)

