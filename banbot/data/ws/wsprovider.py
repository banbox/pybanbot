#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : rtprovider.py
# Author: anyongjin
# Date  : 2023/11/4
import asyncio

from banbot.config import Config
from typing import *

from banbot.config.consts import BotState
from banbot.util.common import logger
from banbot.data.ws.feeds import get_local_wsexg
from banbot.storage import BotGlobal


def init_ws_exg(config: Config):
    from banbot.exchange.crypto_exchange import init_exchange, apply_exg_proxy
    if BotGlobal.live_mode:
        raise NotImplementedError()
    else:
        exg_name, exg_cls = get_local_wsexg(config)
        apply_exg_proxy(exg_name, config)
        return init_exchange(exg_name, exg_cls, config, asyncio_loop=asyncio.get_running_loop())[0]


class WSProvider:
    """实时数据提供器"""
    obj: ClassVar['WSProvider'] = None

    def __init__(self, config: Config, callback: Callable):
        WSProvider.obj = self
        self.config = config
        self.exg_name = config['exchange']['name']
        self.market = config['market_type']
        self.exg = init_ws_exg(config)
        self.pairs: Set[str] = set()
        self.odbooks: Dict[str, dict] = dict()
        '各个交易对的订单簿，可供外部随时调用获取'

        async def handler(*args, **kwargs):
            try:
                await callback(*args, **kwargs)
            except Exception:
                logger.exception('RTData Callback Exception %s %s', args, kwargs)
        self._callback = handler

    def sub_pairs(self, pairs: Iterable[str]):
        for p in pairs:
            self.pairs.add(p)

    async def loop_main(self):
        await self.exg.load_markets()
        BotGlobal.state = BotState.RUNNING
        if hasattr(self.exg, 'run_loop'):
            asyncio.create_task(self.exg.run_loop())
        asyncio.create_task(self._watch_books())
        while BotGlobal.state == BotState.RUNNING:
            trades = await self.exg.watch_trades_for_symbols(self.pairs)
            if not trades:
                continue
            await self._callback(trades[0]['symbol'], trades)
            await asyncio.sleep(0)
        logger.info('WSProvider.loop_main finished.')

    async def _watch_books(self):
        while BotGlobal.state == BotState.RUNNING:
            # 读取订单簿快照并保存
            books = await self.exg.watch_order_book_for_symbols(self.pairs)
            self.odbooks[books['symbol']] = books
            await asyncio.sleep(0)
        logger.info('WSProvider._watch_books stopped.')


async def _run_test():
    from banbot.config import AppConfig
    from banbot.util import btime
    btime.run_mode = btime.RunMode.BACKTEST
    config = AppConfig.get()
    holder = WSProvider(config, lambda x: x)
    holder.sub_pairs(['ETC/USDT:USDT', 'ADA/USDT:USDT'])
    await holder.loop_main()


if __name__ == '__main__':
    asyncio.run(_run_test())
    # _read_test()
