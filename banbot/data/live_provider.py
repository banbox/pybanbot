#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : live_provider.py
# Author: anyongjin
# Date  : 2023/3/28
import inspect

from banbot.data.base import *
from banbot.config import *
from banbot.util.common import logger
from banbot.exchange.crypto_exchange import CryptoExchange
from asyncio import Lock, gather


class LiveDataProvider(DataProvider):
    _cb_lock = Lock()

    def __init__(self, config: Config, exchange: CryptoExchange):
        super(LiveDataProvider, self).__init__()
        self.exchange = exchange
        self.pairlist: List[Tuple[str, str]] = config.get('pairlist')
        self.holders = PairDataHolder.create_holders(self.pairlist, config.get('prefire'))
        self.min_interval = min(hold.min_interval for hold in self.holders)

    @classmethod
    def _wrap_callback(cls, callback: Callable):
        async def handler(*args, **kwargs):
            async with cls._cb_lock:
                try:
                    if inspect.iscoroutinefunction(callback):
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

    async def process(self):
        tasks = [hold.update(self.exchange) for hold in self.holders]
        await gather(*tasks)

