#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : spider.py
# Author: anyongjin
# Date  : 2023/4/25
import asyncio
import os.path
import time
from tqdm import tqdm
import pandas as pd
from typing import Dict, Any, List
from banbot.config.appconfig import AppConfig, Config
from banbot.util import btime
from banbot.util.common import logger
from banbot.util.misc import parallel_jobs
from banbot.exchange.crypto_exchange import CryptoExchange
from banbot.exchange.exchange_utils import timeframe_to_seconds


class Spider:
    def __init__(self, config: Config):
        self.config = config
        self.exchange = CryptoExchange(config)

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

    async def _download_to_file(self, pair: str, timeframe: str, start_ms: int, end_ms: int, out_dir: str):
        fname = pair.replace('/', '_') + '-' + timeframe + '.feather'
        data_path = os.path.join(out_dir, fname)
        columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        df_list = []
        if os.path.isfile(data_path):
            df = pd.read_feather(data_path)
            if df.date.dtype != 'int64':
                df['date'] = df['date'].apply(lambda x: int(x.timestamp() * 1000))
            old_start = df.iloc[0]['date']
            df_list.append(df)
            if start_ms < old_start:
                predata = await self._fetch(pair, timeframe, start_ms, old_start)
                if predata:
                    df_list.insert(0, pd.DataFrame(predata, columns=columns))
            start_ms = round(df.iloc[-1]['date']) + 1
            if start_ms >= end_ms:
                return
        newdata = await self._fetch(pair, timeframe, start_ms, end_ms)
        if newdata:
            df_list.append(pd.DataFrame(newdata, columns=columns))
        df = pd.concat(df_list).reset_index(drop=True)
        df.to_feather(data_path, compression='lz4')

    async def _down_pairs2file(self, pairs: List[str], start_ms: int, end_ms: int):
        data_dir = self.config['data_dir']
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        timeframes = self.config['timeframes']
        tr_text = btime.to_datestr(start_ms) + ' - ' + btime.to_datestr(end_ms)
        args_list = [(pair, tf, start_ms, end_ms, data_dir) for pair in pairs for tf in timeframes]
        for job in parallel_jobs(self._down_pairs2file, args_list):
            pair, tf = (await job)['args'][:2]
            logger.warning(f'{pair}/{tf} down {tr_text} complete')

    async def _download_to_db(self, pair: str, start_ms: int, end_ms: int):
        timeframe = '1m'
        from banbot.data.models import KLine
        old_start, old_end = KLine.query_range(pair)
        if old_start and old_end > old_start:
            if start_ms < old_start:
                # 直接抓取start_ms - old_start的数据，避免出现空洞；可能end_ms>old_end，还需要下载后续数据
                predata = await self._fetch(pair, timeframe, start_ms, old_start)
                KLine.insert(pair, predata)
                start_ms = old_end + 1
            elif end_ms > old_end:
                # 直接抓取old_end - end_ms的数据，避免出现空洞；前面没有需要再下次的数据了。可直接退出
                predata = await self._fetch(pair, timeframe, old_end + 1, end_ms)
                KLine.insert(pair, predata)
                return
            else:
                # 要下载的数据全部存在，直接退出
                return
        newdata = await self._fetch(pair, timeframe, start_ms, end_ms)
        KLine.insert(pair, newdata)

    async def _down_pairs2db(self, pairs: List[str], start_ms: int, end_ms: int):
        tr_text = btime.to_datestr(start_ms) + ' - ' + btime.to_datestr(end_ms)
        args_list = [(pair, start_ms, end_ms) for pair in pairs]
        for job in parallel_jobs(self._download_to_db, args_list):
            pair = (await job)['args'][0]
            logger.warning(f'{pair} down {tr_text} complete')

    async def down_pairs(self):
        pairs = self.config['pairs']
        timerange = self.config['timerange']
        start_ms = round(timerange.startts * 1000)
        end_ms = round(timerange.stopts * 1000)
        cur_ms = round(time.time() * 1000)
        end_ms = min(cur_ms, end_ms) if end_ms else cur_ms
        if self.config['medium'] == 'db':
            await self._down_pairs2db(pairs, start_ms, end_ms)
        else:
            await self._down_pairs2file(pairs, start_ms, end_ms)
        await self.exchange.close()


def run_down_pairs(args: Dict[str, Any]):
    config = AppConfig.init_by_args(args)
    main = Spider(config)
    asyncio.run(main.down_pairs())

