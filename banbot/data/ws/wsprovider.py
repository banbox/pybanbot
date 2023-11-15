#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : rtprovider.py
# Author: anyongjin
# Date  : 2023/11/4
from banbot.data.ws.feeds import *
from banbot.data.feeder import *
from banbot.data.provider import DataProvider
from banbot.data.tools import trades_to_ohlcv


def build_ws_bar(trades: List[dict]) -> Tuple[int, float, float, float, float, float]:
    """从websocket获得的交易构建bar"""
    from banbot.config.consts import secs_year
    details = trades_to_ohlcv(trades)
    ohlcvs_sml, _ = build_ohlcvc(details, secs_year, with_count=False)
    stamp_ms = trades[-1]['timestamp']
    bar = ohlcvs_sml[-1]
    return stamp_ms, bar[1], bar[2], bar[3], bar[4], bar[5]


class LocalWSProvider:
    """实时数据提供器"""

    def __init__(self, config: Config, callback: Callable):
        from banbot.exchange.crypto_exchange import init_exchange, apply_exg_proxy
        self.config = config
        self.exg_name = config['exchange']['name']
        self.market = config['market_type']
        self.pairs: Set[str] = set()
        exg_name, exg_cls = get_local_wsexg(config)
        apply_exg_proxy(exg_name, config)
        self.exg = init_exchange(exg_name, exg_cls, config, asyncio_loop=asyncio.get_running_loop())[0]
        self._callback = callback
        self._trade_q = asyncio.Queue(1000)
        self.exg.emit_q = self._trade_q
        self.pair_bars: Dict[str, tuple] = dict()

    async def loop_main(self):
        from banbot.util.misc import run_async
        from banbot.util import btime
        await self.exg.load_markets()
        BotGlobal.state = BotState.RUNNING
        asyncio.create_task(self.exg.run_loop())
        self.exg.sub_pairs(self.pairs)
        while True:
            dtype, pair, data = await self._trade_q.get()
            try:
                if dtype == 'book':
                    if data:
                        BotCache.odbooks[pair] = data
                elif dtype == 'trade':
                    if data:
                        btime.cur_timestamp = data[-1]['timestamp'] / 1000
                        self.pair_bars[pair] = build_ws_bar(data)
                        await run_async(self._callback, pair, data)
                else:
                    logger.error(f'not support dtype: {dtype}')
            except Exception:
                logger.exception('RTData Callback Exception')
            self._trade_q.task_done()
            if BotGlobal.state == BotState.STOPPED and not self._trade_q.qsize():
                break
        logger.info('WSProvider.loop_main finished.')

    def sub_pairs(self, pairs: Iterable[str]):
        for p in pairs:
            self.pairs.add(p)

    def get_latest_ohlcv(self, pair: str):
        return self.pair_bars.get(pair)


class LiveWSProvider(DataProvider, KlineLiveConsumer):
    """实盘ws数据提供器"""
    feed_cls = LiveWSFeeder
    _obj: Optional['LiveWSProvider'] = None

    def __init__(self, config: Config, callback: Callable):
        DataProvider.__init__(self, config, callback)
        KlineLiveConsumer.__init__(self)
        LiveWSProvider._obj = self
        self.holders: List[LiveWSFeeder] = []

    async def sub_warm_pairs(self, pairs: Dict[str, Dict[str, int]]):
        in_tfs = {t for _, l in pairs.items() for t in l}
        if not (len(in_tfs) == 1 and 'ws' in in_tfs):
            raise ValueError(f'only timeframe: ws is allowed in LiveWSProvider, current: {in_tfs}')
        new_holds, _ = super(LiveWSProvider, self).sub_pairs(pairs)
        if not new_holds:
            return
        market = self.config['market_type']
        watch_jobs = [WatchParam(hold.pair, 'ws', 0) for hold in new_holds]
        # 发送消息给爬虫，实时抓取数据
        await self.watch_klines(self.exg_name, market, *watch_jobs)

    async def unsub_pairs(self, pairs: List[str]):
        removed = super(LiveWSProvider, self).unsub_pairs(pairs)
        if not removed:
            return
        market = self.config['market_type']
        pairs = [hold.pair for hold in removed]
        await self.unwatch_klines(self.exg_name, market, pairs, is_ws=True)

    async def loop_main(self):
        await self.run_forever('bot')

    @classmethod
    async def _on_trades(cls, exg_name: str, market: str, pair: str, trades: List[dict]):
        if exg_name != cls._obj.exg_name or cls._obj.market != market:
            logger.warning(f'receive exg not match: {exg_name}, cur: {cls._obj.exg_name}')
            return
        hold = next((sta for sta in cls._obj.holders if sta.pair == pair), None)
        if not hold:
            return
        print(f'live ws provider _on_trades: {exg_name} {market} {pair}')
        hold.wait_bar = build_ws_bar(trades)
        hold.states[0].latest = hold.wait_bar
        await hold.on_new_data(trades)


async def _run_test():
    from banbot.config import AppConfig
    from banbot.util import btime
    btime.run_mode = btime.RunMode.BACKTEST
    config = AppConfig.get()
    holder = LocalWSProvider(config, lambda x: x)
    holder.sub_pairs(['ETC/USDT:USDT', 'ADA/USDT:USDT'])
    await holder.loop_main()


if __name__ == '__main__':
    asyncio.run(_run_test())
    # _read_test()
