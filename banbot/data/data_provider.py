#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : live_provider.py
# Author: anyongjin
# Date  : 2023/3/28
from banbot.data.feeder import *
from banbot.config import *
from banbot.util.common import logger
from banbot.util.misc import run_async
from banbot.exchange.crypto_exchange import CryptoExchange
from banbot.storage.common import *
from asyncio import Lock, gather


class DataProvider:
    _cb_lock = Lock()

    def __init__(self, config: Config):
        self.config = config
        self.pairlist: List[Tuple[str, str]] = config.get('pairlist')
        self.holders: List[PairDataFeeder] = []
        self.min_interval = 1

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
        async def handler(*args, **kwargs):
            async with cls._cb_lock:
                try:
                    await run_async(callback, *args, **kwargs)
                except Exception:
                    logger.exception('LiveData Callback Exception %s %s', args, kwargs)
        return handler

    def _set_callback(self, callback: Callable):
        wrap_callback = self._wrap_callback(callback)
        for hold in self.holders:
            hold.callback = wrap_callback

    @staticmethod
    def create_holders(feed_cls, pairlist: List[Tuple[str, str]], **kwargs) -> List[PairDataFeeder]:
        pair_groups: Dict[str, Set[str]] = dict()
        for pair, timeframe in pairlist:
            if pair not in pair_groups:
                pair_groups[pair] = set()
            pair_groups[pair].add(timeframe)
        return [feed_cls(pair, list(tf_set), **kwargs) for pair, tf_set in pair_groups.items()]

    async def process(self):
        await gather(*[hold.try_fetch() for hold in self.holders])
        for hold in self.holders:
            await hold.try_update()

    async def loop_main(self, warmup_num: int):
        # 先调用一次数据加载，确保预热阶段、数据加载等应用完成
        last_end = time.monotonic()
        try:
            await self._first_call(warmup_num)
            while BotGlobal.state == BotState.RUNNING:
                left_sleep = self.min_interval - (time.monotonic() - last_end)
                if left_sleep > 0:
                    await asyncio.sleep(left_sleep)
                last_end = time.monotonic()
                try:
                    await self.process()
                except EOFError:
                    break
        except Exception:
            logger.exception('loop data main fail')

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
                ts_val = max(ts_val, state.bar_row[0] + state.tf_secs - hold.prefire_secs)
            btime.cur_timestamp = ts_val


class LocalDataProvider(DataProvider):

    def __init__(self, config: Config, callback: Callable):
        super(LocalDataProvider, self).__init__(config)
        exg_name = config['exchange']['name']
        kwargs = dict(
            auto_prefire=config.get('prefire'),
            data_dir=os.path.join(config['data_dir'], exg_name),
            timerange=config.get('timerange')
        )
        self.holders = DataProvider.create_holders(LocalPairDataFeeder, self.pairlist, **kwargs)
        self._set_callback(callback)
        self.min_interval = min(hold.min_interval for hold in self.holders)


class LiveDataProvider(DataProvider):

    def __init__(self, config: Config, exchange: CryptoExchange, callback: Callable):
        super(LiveDataProvider, self).__init__(config)
        self.exchange = exchange
        kwargs = dict(
            auto_prefire=config.get('prefire'),
            exchange=exchange
        )
        self.holders = DataProvider.create_holders(LivePairDataFeader, self.pairlist, **kwargs)
        self._set_callback(callback)
        self.min_interval = min(hold.min_interval for hold in self.holders)

