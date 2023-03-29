#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : live_provider.py
# Author: anyongjin
# Date  : 2023/3/28
from banbot.data.base import *
from banbot.config import *
from asyncio import Lock, gather


class LiveDataProvider(DataProvider):
    _cb_lock = Lock()

    def __init__(self, config: dict):
        super(LiveDataProvider, self).__init__()
        self.pairlist: List[Tuple[str, str]] = cfg.get('pairlist')
        self.holders = PairDataHolder.create_holders(self.pairlist, cfg.get('prefire'))
        need_ws = any(hold.is_ws for hold in self.holders)
        self.exchange, self.exg_ws = get_exchange(config, with_ws=need_ws)
        self.min_interval = min(hold.min_interval for hold in self.holders)

    async def init(self):
        markets = await self.exchange.load_markets()
        logger.info(f'{len(markets)} markets loaded for {self.exchange.name}')
        if self.exg_ws:
            markets = await self.exg_ws.load_markets()
            logger.info(f'{len(markets)} markets loaded for {self.exg_ws.name} WebSocket')
        from banbot.exchange.exchange_utils import init_longvars
        await init_longvars(self.exchange, self.pairlist)
        logger.info('init longvars complete')

    @classmethod
    def _wrap_callback(cls, callback: Callable):
        async def handler(*args, **kwargs):
            async with cls._cb_lock:
                try:
                    await callback(*args, **kwargs)
                except Exception:
                    logger.exception(f'LiveData Callback Exception {args} {kwargs}')
        return handler

    def set_callback(self, callback: Callable):
        wrap_callback = self._wrap_callback(callback)
        for hold in self.holders:
            hold.callback = wrap_callback

    async def process(self):
        tasks = [hold.update(self.exchange, self.exg_ws) for hold in self.holders]
        await gather(*tasks)

