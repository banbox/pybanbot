#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : whack_mole.py
# Author: anyongjin
# Date  : 2023/2/28
import asyncio

from banbot.main.itrader import *
from banbot.exchange.exchange_utils import *
from banbot.data.live_provider import LiveDataProvider
from banbot.util.misc import *


class LiveTrader(Trader):
    '''
    实盘交易
    '''

    def __init__(self):
        super(LiveTrader, self).__init__()
        self.data_hold = LiveDataProvider(cfg)
        self.strategy_map: Dict[str, BaseStrategy] = dict()
        self.data_hold.set_callback(self._make_invoke())

    def _make_invoke(self):
        async def invoke_pair(pair, timeframe, row: np.ndarray):
            set_context(f'{pair}_{timeframe}')
            logger.info(f'ohlc: {pair} {timeframe} {row}')
            await self.on_data_feed(row)
        return invoke_pair

    async def on_data_feed(self, row: np.ndarray):
        pass

    async def run(self, make_strategy):
        for pair, timeframe in self.pairlist:
            symbol = f'{pair}_{timeframe}'
            set_context(symbol)
            self.strategy_map[symbol] = make_strategy()
        await self.data_hold.init()
        logger.info(f'data update interval: {self.data_hold.min_interval}')
        while True:
            start_time = time.time()
            next_check = start_time + self.data_hold.min_interval
            await self.data_hold.process()
            sleep_time = next_check - time.time()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)


if __name__ == '__main__':
    def cmake_strategy():
        from banbot.strategy.macd_cross import MACDCross
        stg = MACDCross()
        # from banbot.strategy.mean_rev import MeanRev
        # stg = MeanRev()
        # from banbot.strategy.classic.trend_model_sys import TrendModelSys
        # strategy = TrendModelSys()
        # if hasattr(strategy, 'debug_ids'):
        #     debug_idx = int(np.where(data['date'] == '2023-02-22 00:15:09')[0][0])
        #     strategy.debug_ids.add(debug_idx)
        return stg
    trader = LiveTrader()
    call_async(trader.run, cmake_strategy)
