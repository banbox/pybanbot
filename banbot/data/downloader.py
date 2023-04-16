#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : downloader.py
# Author: anyongjin
# Date  : 2023/4/15
import os.path
import time

from tqdm import tqdm

import pandas as pd

from banbot.exchange.crypto_exchange import *


class Downloader:
    def __init__(self, config: dict):
        self.config = config
        self.pairs = config['pairs']
        self.exchange = CryptoExchange(config)
        self.timerange = config['timerange']
        self.timeframes = config['timeframes']
        self.data_dir = config['data_dir']

    async def _fetch(self, pair: str, timeframe: str, start_ts: int, end_ts: int):
        '''
        按给定时间段下载交易对的K线数据。
        :param pair:
        :param timeframe:
        :param start_ts: 毫秒时间戳（含）
        :param end_ts: 毫秒时间戳（不含）
        :return:
        '''
        start_text, end_text = btime.to_datestr(start_ts), btime.to_datestr(end_ts)
        logger.info(f'fetch ohlcv {pair}/{timeframe} {start_text} - {end_text}')
        tf_msecs = timeframe_to_seconds(timeframe) * 1000
        batch_size = 1000
        since, total_tr = start_ts, end_ts - start_ts
        result = []
        pbar = tqdm()
        while since + tf_msecs < end_ts:
            data = await self.exchange.fetch_ohlcv(pair, timeframe, since=since, limit=batch_size)
            if not len(data):
                break
            result.extend(data)
            cur_end = round(data[-1][0])
            pbar.update(round((cur_end - since) * 100 / total_tr, 2))
            since = cur_end + 1
        end_pos = len(result) - 1
        while end_pos >= 0 and result[end_pos][0] >= end_ts:
            end_pos -= 1
        return result[:end_pos + 1]

    async def _download(self, pair: str, timeframe: str):
        fname = pair.replace('/', '_') + '-' + timeframe + '.feather'
        data_path = os.path.join(self.data_dir, fname)
        start, end = self.timerange.startts * 1000, self.timerange.stopts * 1000
        end = min(round(time.time() * 1000), end)
        columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        df_list = []
        if os.path.isfile(data_path):
            df = pd.read_feather(data_path)
            if df.date.dtype != 'int64':
                df['date'] = df['date'].apply(lambda x: int(x.timestamp() * 1000))
            old_start = df.iloc[0]['date']
            df_list.append(df)
            if start < old_start:
                predata = await self._fetch(pair, timeframe, start, old_start)
                if predata:
                    df_list.insert(0, pd.DataFrame(predata, columns=columns))
            start = round(df.iloc[-1]['date']) + 1
            if start >= end:
                return
        newdata = await self._fetch(pair, timeframe, start, end)
        if newdata:
            df_list.append(pd.DataFrame(newdata, columns=columns))
        df = pd.concat(df_list).reset_index(drop=True)
        df.to_feather(data_path, compression='lz4')

    async def run(self):
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        for tf in self.timeframes:
            for pair in self.pairs:
                await self._download(pair, tf)
                logger.warning(f'{pair}/{tf} down {self.timerange} complete')
        await self.exchange.close()
