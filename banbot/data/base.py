#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : base.py
# Author: anyongjin
# Date  : 2023/3/28
import asyncio
import time
from banbot.exchange.exchange_utils import *


class PairTFCache:
    def __init__(self, timeframe: str, tf_decs: int):
        self.timeframe = timeframe
        self.tf_secs = tf_decs
        self.last_check = None  # 上次检查更新的时间戳
        self.check_interval = get_check_interval(self.tf_secs)
        self.bar_row = None

    def need_check(self) -> bool:
        return not self.last_check or self.last_check + self.check_interval <= time.time()


class PairDataHolder:
    '''
    用于记录交易对的数据更新状态
    '''
    def __init__(self, pair: str, timeframes: List[Tuple[str, int]], auto_prefire=False):
        self.pair = pair
        self.states = [PairTFCache(tf, tf_sec) for tf, tf_sec in timeframes]
        self.min_interval = min(s.check_interval for s in self.states)
        # 超过3s检查一次的（1分钟及以上维度），通过API获取；否则通过Websocket获取
        self.is_ws = self.min_interval <= 2
        self.auto_prefire = auto_prefire
        self.callback = None

    async def update(self, exchange: ccxt.Exchange, exg_ws: ccxtpro.Exchange):
        '''
        对当前交易对检查是否需要从交易所拉取数据
        :param exchange:
        :param exg_ws:
        :return:
        '''
        assert self.callback, '`callback` is not set!'
        # 第一个最小粒度，从api或ws更新行情数据
        state = self.states[0]
        if not state.need_check():
            return
        if not self.is_ws:
            details = await exchange.fetch_ohlcv(self.pair, '1s', limit=state.check_interval * 2)
        else:
            details = await exg_ws.watch_trades(self.pair)
        ohlcvs = [state.bar_row] if state.bar_row else []
        prefire = 0.1 if self.auto_prefire else 0
        ohlcvs = build_ohlcvc(details, state.tf_secs, prefire, ohlcvs=ohlcvs)
        state.last_check = time.time()
        if not state.bar_row or ohlcvs[-1][0] == state.bar_row[0]:
            state.bar_row = ohlcvs[-1]
        elif ohlcvs[0][0] == state.bar_row[0]:
            state.bar_row = ohlcvs[0]
        if ohlcvs[-1][0] > state.bar_row[0]:
            await self.callback(self.pair, state.timeframe, state.bar_row)
            state.bar_row = ohlcvs[-1]
        else:
            # 当前蜡烛未完成，后续更粗粒度也不会完成，直接退出
            return
        # 对于第2个及后续的粗粒度。从第一个得到的OHLC更新
        for state in self.states[1:]:
            cur_ohlcvs = [state.bar_row] if state.bar_row else []
            prefire = 0.05 if self.auto_prefire else 0
            cur_ohlcvs = build_ohlcvc(ohlcvs, state.tf_secs, prefire, ohlcvs=cur_ohlcvs)
            state.last_check = time.time()
            if state.bar_row and cur_ohlcvs[-1][0] > state.bar_row[0]:
                await self.callback(self.pair, state.timeframe, state.bar_row)
            state.bar_row = cur_ohlcvs[-1]

    @staticmethod
    def create_holders(pairlist: List[Tuple[str, str]], prefire: bool = False) -> List['PairDataHolder']:
        pair_map: Dict[str, Dict[str, int]] = dict()
        for pair, timeframe in pairlist:
            if pair not in pair_map:
                pair_map[pair] = dict()
            pair_map[pair][timeframe] = timeframe_to_seconds(timeframe)
        result = []
        for pair, tf_map in pair_map.items():
            tf_list = list(sorted(tf_map.items(), key=lambda x: x[1]))
            for _, tf_secs in tf_list[1:]:
                err_msg = f'{_} of {pair} must be integer multiples of the first ({tf_list[0][0]})'
                assert tf_secs % tf_list[0][1] == 0, err_msg
            result.append(PairDataHolder(pair, tf_list, prefire))
        return result


class DataProvider:
    def __init__(self):
        pass
