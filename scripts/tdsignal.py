#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : tdsignal.py
# Author: anyongjin
# Date  : 2023/7/1
import asyncio
import random

import pandas as pd
import six

from banbot.storage.symbols import get_symbol_market
from banbot.storage import *
from banbot.util import btime
from banbot.exchange.crypto_exchange import secs_to_tf
from banbot.util.common import logger


async def load_file_signal(xls_path: str):
    '''
    将阿涛从TradingView导出的策略交易信号，导入到banbot数据库
    '''
    df = pd.read_excel(xls_path, header=0)
    action_map = dict(LONG='buy', SHORT='sell')
    df['action'] = df['action'].map(action_map)
    print(df.head())
    sess = db.session
    for rid, row in df.iterrows():
        action = action_map.get(row['action'], row['action'])
        interval = row['interval']
        if isinstance(interval, six.integer_types):
            tf_msecs = interval * 60 * 1000
        elif isinstance(interval, six.string_types):
            interval = interval.lower()
            if interval.endswith('d'):
                tf_msecs = int(interval[:-1]) * 24 * 3600 * 1000
            else:
                raise ValueError(f'not support interval: {interval}')
        else:
            raise ValueError(f'not support interval: {interval}')
        timeframe = secs_to_tf(tf_msecs // 1000)
        if not timeframe:
            if isinstance(interval, six.string_types):
                timeframe = interval
            else:
                raise ValueError(f'not support interval: {interval}')
        symbol, market = get_symbol_market(row['ticker'])
        exs = ExSymbol.get(row['exchange'], symbol, market)
        create_time = row['time'].to_pydatetime()
        create_ts = btime.to_utcstamp(create_time, True, cut_int=True)
        logger.info(f'insert: {timeframe} {create_ts}')
        sess.add(TdSignal(
            symbol_id=exs.id,
            timeframe=timeframe,
            action=action,
            create_at=create_ts,
            bar_ms=(create_ts // tf_msecs) * tf_msecs,
            price=row['open']
        ))
        if random.random() < 0.02:
            sess.commit()
    sess.commit()
    return df


async def load_signals(*xls_paths: str):
    for path in xls_paths:
        await load_file_signal(path)


if __name__ == '__main__':
    from banbot.storage.base import init_db, db
    from banbot.config import AppConfig
    par_xls_dir = r'E:/Data/temp/'
    AppConfig.init_by_args(dict(config=[r'E:\trade\banbot\banbot\config\config.json']))
    init_db()
    with db():
        fnames = ['btc_5m.xlsx', 'btc_1h.xlsx', 'btc_4h.xlsx', 'btc_15m.xlsx', 'btc_1d.xlsx', 'btc_3d.xlsx']
        path_list = [par_xls_dir + n for n in fnames]
        asyncio.run(load_signals(*path_list))
