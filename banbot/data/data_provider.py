#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : live_provider.py
# Author: anyongjin
# Date  : 2023/3/28
from banbot.data.feeder import *
from banbot.config import *
from banbot.util.common import logger
from banbot.exchange.crypto_exchange import CryptoExchange
from asyncio import Lock, gather


class DataProvider:
    _cb_lock = Lock()

    def __init__(self, config: Config):
        self.pairlist: List[Tuple[str, str]] = config.get('pairlist')
        self.holders: List[PairDataFeeder] = []

    @classmethod
    def _wrap_callback(cls, callback: Callable):
        async def handler(*args, **kwargs):
            async with cls._cb_lock:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(*args, **kwargs)
                    else:
                        callback(*args, **kwargs)
                except Exception:
                    logger.exception(f'LiveData Callback Exception {args} {kwargs}')
        return handler

    def set_callback(self, callback: Callable):
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
        tasks = [hold.try_fetch() for hold in self.holders]
        await gather(*tasks)
        for hold in self.holders:
            await hold.try_update()


class LocalDataProvider(DataProvider):

    def __init__(self, config: Config):
        super(LocalDataProvider, self).__init__(config)
        exg_name = config['exchange']['name']
        kwargs = dict(
            auto_prefire=config.get('prefire'),
            data_dir=os.path.join(config['data_dir'], exg_name)
        )
        self.holders = DataProvider.create_holders(LocalPairDataFeeder, self.pairlist, **kwargs)
        self.min_interval = min(hold.min_interval for hold in self.holders)


class LiveDataProvider(DataProvider):

    def __init__(self, config: Config, exchange: CryptoExchange):
        super(LiveDataProvider, self).__init__(config)
        self.exchange = exchange
        kwargs = dict(
            auto_prefire=config.get('prefire'),
            exchange=exchange
        )
        self.holders = DataProvider.create_holders(LivePairDataFeader, self.pairlist, **kwargs)
        self.min_interval = min(hold.min_interval for hold in self.holders)

