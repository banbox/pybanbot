#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : exchange_utils.py
# Author: anyongjin
# Date  : 2023/3/25
import math
import os

# import ccxt
import time

import ccxt.async_support as ccxt
import ccxt.pro as ccxtpro
import numpy as np

from banbot.util.common import logger
from banbot.config.consts import *
from typing import *


def get_exchange(cfg: dict, with_ws=False) -> Tuple[ccxt.Exchange, Optional[ccxtpro.Exchange]]:
    exg_cfg = cfg['exchange']
    exg_class = getattr(ccxt, exg_cfg['name'])
    run_env = cfg["env"]
    credit = exg_cfg[f'credit_{run_env}']
    has_proxy = bool(exg_cfg.get('proxies'))
    if has_proxy:
        os.environ['HTTP_PROXY'] = exg_cfg['proxies']['http']
        os.environ['HTTPS_PROXY'] = exg_cfg['proxies']['https']
        os.environ['WS_PROXY'] = exg_cfg['proxies']['http']
        os.environ['WSS_PROXY'] = exg_cfg['proxies']['http']
        logger.warning(f"[PROXY] {exg_cfg['proxies']}")
    exchange: ccxt.Exchange = exg_class(dict(
        apiKey=credit['api_key'],
        secret=credit['api_secret'],
        trust_env=has_proxy
    ))
    if run_env == 'test':
        exchange.set_sandbox_mode(True)
        logger.warning('running in TEST mode!!!')
    if has_proxy:
        exchange.proxies = exg_cfg['proxies']
        exchange.aiohttp_proxy = exg_cfg['proxies']['http']
    if not with_ws:
        return exchange, None
    exg_class = getattr(ccxtpro, exg_cfg['name'])
    exg_pro = exg_class(dict(
        newUpdates=True,
        apiKey=credit['api_key'],
        secret=credit['api_secret'],
        aiohttp_trust_env=has_proxy
    ))
    if run_env == 'test':
        exg_pro.set_sandbox_mode(True)
    if has_proxy:
        exg_pro.aiohttp_proxy = exg_cfg['proxies']['http']
    return exchange, exg_pro


def max_sub_timeframe(timeframes: List[str], current: str, force_sub=False) -> Tuple[str, int]:
    '''
    返回交易所支持的最大子时间帧
    :param timeframes: 交易所支持的所有时间帧  exchange.timeframes.keys()
    :param current: 当前要求的时间帧
    :param force_sub: 是否强制使用更细粒度的时间帧，即使当前时间帧支持
    :return:
    '''
    tf_secs = timeframe_to_seconds(current)
    pairs = [(tf, timeframe_to_seconds(tf)) for tf in timeframes]
    pairs = sorted(pairs, key=lambda x: x[1])
    all_tf, all_tf_secs = list(zip(*pairs))
    rev_tf_secs = all_tf_secs[::-1]
    for i in range(len(rev_tf_secs)):
        if force_sub and tf_secs == rev_tf_secs[i]:
            continue
        if tf_secs % rev_tf_secs[i] == 0:
            return all_tf[len(all_tf_secs) - i - 1], rev_tf_secs[i]


async def fetch_ohlcv(exchange: ccxt.Exchange, pair: str, timeframe: str, since=None, limit=None,
                      force_sub=False, min_last_ratio=0.799):
    '''
    某些时间维度不能直接从交易所得到，需要从更细粒度计算。如3s的要从1s计算。2m的要从1m的计算
    :param exchange:
    :param pair:
    :param timeframe:
    :param since:
    :param limit:
    :param force_sub: 是否强制使用更细粒度的时间帧，即使当前时间帧支持
    :param min_last_ratio: 最后一个蜡烛的最低完成度
    :return:
    '''
    if (not force_sub or timeframe == '1s') and timeframe in exchange.timeframes:
        return await exchange.fetch_ohlcv(pair, timeframe, since=since, limit=limit)
    sub_tf, sub_tf_secs = max_sub_timeframe(exchange.timeframes.keys(), timeframe, force_sub)
    cur_tf_secs = timeframe_to_seconds(timeframe)
    if not limit:
        sub_arr = await exchange.fetch_ohlcv(pair, sub_tf, since=since)
        ohlc_arr = build_ohlcvc(sub_arr, cur_tf_secs)
        if sub_arr[-1][0] / 1000 / cur_tf_secs % 1 < min_last_ratio:
            ohlc_arr = ohlc_arr[:-1]
        return ohlc_arr
    fetch_num = limit * round(cur_tf_secs / sub_tf_secs)
    count = 0
    if not since:
        cur_time = int(math.floor(time.time() / cur_tf_secs) * cur_tf_secs)
        since = (cur_time - fetch_num * sub_tf_secs) * 1000
    result = []
    while count < fetch_num:
        sub_arr = await exchange.fetch_ohlcv(pair, sub_tf, since=since, limit=MAX_FETCH_NUM)
        if not sub_arr:
            break
        result.extend(sub_arr)
        count += len(sub_arr)
        since = sub_arr[-1][0] + 1
    ohlc_arr = build_ohlcvc(result[-fetch_num:], cur_tf_secs)
    if result[-1][0] / 1000 / cur_tf_secs % 1 < min_last_ratio:
        ohlc_arr = ohlc_arr[:-1]
    return ohlc_arr


async def _init_longvars(exchange: ccxt.Exchange, pair: str, timeframe: str):
    from banbot.bar_driven.tainds import LongVar, StaSMA, StaNATR, bar_num
    ochl_data = await fetch_ohlcv(exchange, pair, timeframe, limit=900)
    bar_arr = np.array(ochl_data)[:, 1:]
    LongVar.get(LongVar.price_range)
    LongVar.get(LongVar.vol_avg)
    LongVar.get(LongVar.bar_len)
    LongVar.get(LongVar.sub_malong)
    LongVar.get(LongVar.atr_low)
    malong = StaSMA(120)
    natr = StaNATR()
    for i in range(bar_arr.shape[0]):
        bar_num.set(i + 1)
        malong(bar_arr[i, 3])
        natr(bar_arr[:i + 1, :])
        LongVar.update(bar_arr)


async def init_longvars(exchange: ccxt.Exchange, pairlist: List[Tuple[str, str]]):
    from banbot.bar_driven.tainds import set_context
    for pair, timeframe in pairlist:
        set_context(f'{pair}_{timeframe}')
        await _init_longvars(exchange, pair, timeframe)


def timeframe_to_seconds(timeframe: str) -> int:
    """
    Translates the timeframe interval value written in the human readable
    form ('1m', '5m', '1h', '1d', '1w', etc.) to the number
    of seconds for one timeframe interval.
    """
    return ccxt.Exchange.parse_timeframe(timeframe)


def get_check_interval(timeframe_secs: int) -> float:
    '''
    根据监听的交易对和时间帧。计算最小检查间隔。
    <60s的通过WebSocket获取数据，检查更新间隔可以比较小。
    1m及以上的通过API的秒级接口获取数据，3s更新一次
    :param timeframe_secs:
    :return:
    '''
    if timeframe_secs <= 3:
        check_interval = 0.2
    elif timeframe_secs <= 10:
        check_interval = 0.5
    elif timeframe_secs < 60:
        check_interval = 1
    else:
        return 3
    return check_interval


def trade2ohlc(trade: dict) -> Tuple[int, float, float, float, float, float, int]:
    price = trade['price']
    return trade['timestamp'], price, price, price, price, trade['amount'], 1


def build_ohlcvc(details, tf_secs: int, prefire: float = 0., since=None, ohlcvs=None):
    '''
    从交易或子OHLC数组中，构建或更新更粗粒度OHLC数组。
    :param details: 可以是交易列表或子OHLC列表。[dict] or [[t,o,h,l,c,v,cnt]]
    :param tf_secs: 指定要构建的时间粒度，单位：秒
    :param prefire: 是否提前触发构建完成；用于在特定信号时早于其他交易者提早发出信号
    :param since:
    :param ohlcvs: 已有的待更新数组
    :return:
    '''
    ms = tf_secs * 1000
    off_ms = round(ms * prefire)
    ohlcvs = ohlcvs or []
    (timestamp, copen, high, low, close, volume, count) = (0, 1, 2, 3, 4, 5, 6)
    for detail in details:
        row = list(trade2ohlc(detail)) if isinstance(detail, dict) else list(detail)
        # 按给定粒度重新格式化时间戳
        row[timestamp] = int(math.floor((row[timestamp] + off_ms) / ms) * ms)
        if since and row[timestamp] < since:
            continue
        if not ohlcvs or (row[timestamp] >= ohlcvs[-1][timestamp] + ms):
            # moved to a new timeframe -> create a new candle from opening trade
            ohlcvs.append(row)
        else:
            prow = ohlcvs[-1]
            # still processing the same timeframe -> update opening trade
            prow[high] = max(prow[high], row[high])
            prow[low] = min(prow[low], row[low])
            prow[close] = row[close]
            prow[volume] += row[volume]
            if len(row) > count:
                prow[count] += row[count]
    return ohlcvs
