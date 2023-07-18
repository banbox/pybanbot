#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : tdsignal.py
# Author: anyongjin
# Date  : 2023/7/1
import asyncio
import random

import pandas as pd
import six
import datetime

from banbot.storage.symbols import get_symbol_market
from banbot.storage import *
from banbot.util import btime
from banbot.exchange.crypto_exchange import secs_to_tf
from banbot.util.common import logger


async def load_file_signal(xls_path: str, timezone_off: int = 0):
    '''
    将阿涛从TradingView导出的策略交易信号，导入到banbot数据库
    '''
    df = pd.read_excel(xls_path, header=0)
    action_map = dict(LONG='buy', SHORT='sell')
    df['action'] = df['action'].map(action_map)
    # print(df.head())
    sess = db.session
    off_msecs = timezone_off * 3600000
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
        sig_time = row['time']
        if hasattr(sig_time, 'to_pydatetime'):
            create_time = sig_time.to_pydatetime()
        elif isinstance(sig_time, six.string_types):
            time_part_len = len(sig_time.split(':'))
            if time_part_len == 1:
                fmt = '%Y-%m-%d'
            elif time_part_len == 2:
                fmt = '%Y-%m-%d %H:%M'
            elif time_part_len == 3:
                fmt = '%Y-%m-%d %H:%M:%S'
            else:
                raise ValueError(f'unsupport time fmt: {sig_time}')
            create_time = datetime.datetime.strptime(sig_time, fmt)
        else:
            raise ValueError(f'unsupport time: {type(sig_time)}, {sig_time}')
        create_ts = btime.to_utcstamp(create_time, True, cut_int=True) + off_msecs
        if rid % 500 == 0 and rid:
            logger.info(f'[{rid}/{len(df)}] insert: {timeframe} {create_ts}')
        sess.add(TdSignal(
            symbol_id=exs.id,
            timeframe=timeframe,
            action=action,
            create_at=create_ts,
            bar_ms=(create_ts // tf_msecs) * tf_msecs,
            price=row['open']
        ))
    sess.commit()
    return df


async def load_signals(timezone_off: int, *xls_paths: str):
    logger.warning(f'所有时间将被视为UTC+{timezone_off}H')
    sta_path = r'E:/Data/SignalData.txt'
    try:
        handled = set(open(sta_path, 'r', encoding='utf-8').read().strip().splitlines())
    except FileNotFoundError:
        handled = set()
    state = open(sta_path, 'a', encoding='utf-8')
    for path in xls_paths:
        if path in handled:
            print(f'skip: {path}')
            continue
        print(f'processing: {path}')
        await load_file_signal(path, timezone_off)
        state.write(f'{path}\n')
        state.flush()
    state.close()


def load_signals_from_dir():
    par_xls_dir = r'E:/Data/SignalData/'
    fnames = os.listdir(par_xls_dir)
    with db():
        path_list = [par_xls_dir + n for n in fnames]
        asyncio.run(load_signals(8, *path_list))


def fix_symbol_wrong():
    '''
    针对TV的BTCUSDT.p未做兼容，只兼容了BTCUSDT.P；导致导入信号创建了冗余的交易对。
    信号中symbol_id对应错误。
    此脚本筛选symbol中所有以"-p"结尾的，找到对应的symbol_id，进行tdsignal更新替换。
    '''
    from banbot.storage.base import sa
    with db():
        sess = db.session
        all_symbols = list(sess.query(ExSymbol).all())
        symbol_map = {f'{s.exchange}:{s.market}:{s.symbol}': s for s in all_symbols}
        for i, s in enumerate(all_symbols):
            if i < 300:
                continue
            if not s.symbol.endswith('-p'):
                continue
            sbl_name = s.symbol[:-2]
            true_row = symbol_map.get(f'{s.exchange}:{s.market}:{sbl_name}')
            if not true_row:
                print(f'no true symbol for: {s}')
                continue
            res = sess.execute(sa.text(f'update tdsignal set symbol_id={true_row.id} where symbol_id={s.id}'))
            print(f'[{i}]change signal id {s.id} > {true_row.id}, num: {res.rowcount}')
            sess.commit()
        


if __name__ == '__main__':
    import os
    from banbot.storage.base import init_db, db
    from banbot.config import AppConfig
    AppConfig.init_by_args()
    init_db()
    load_signals_from_dir()
    
