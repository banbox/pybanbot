#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : monitor.py
# Author: anyongjin
# Date  : 2023/8/17
'''
定期监控所有标的基本行情
满足特定条件发送rpc通知或添加到机器人处理。
'''
from banbot.util.common import logger
from banbot.storage import KLine, ExSymbol
from banbot.util.redis_helper import AsyncRedis
from banbot.util import btime
from banbot.exchange.crypto_exchange import tf_to_secs
from banbot.compute.ctx import ccol
from banbot.rpc.rpc_manager import Notify, RPCMessageType
from banbot.config.appconfig import AppConfig

up_rate = 0.03
down_rate = -0.03


async def _check_symbol(rpc: Notify, redis: AsyncRedis, exg_name: str, market: str, symbol: str, timeframe: str):
    time_ms = btime.utcstamp()
    dt_key = f'eye_{btime.to_datestr(time_ms, fmt="%Y%m%d")}'
    up_key = f'{exg_name}_{market}_{symbol}_u'
    dn_key = f'{exg_name}_{market}_{symbol}_d'
    up_reached = await redis.sismember(dt_key, up_key)
    dn_reached = await redis.sismember(dt_key, dn_key)
    if up_reached and dn_reached:
        # 今日已提醒过
        return
    exs = ExSymbol.get(exg_name, market, symbol)
    tf_msecs = tf_to_secs(timeframe) * 1000
    cur_start_ms = time_ms // tf_msecs * tf_msecs - tf_msecs
    cur_bars = KLine.query(exs, timeframe, cur_start_ms, time_ms)
    if not cur_bars:
        logger.warning(f'[market monitor] no bars found for {exs} {timeframe}')
        return
    cur_price = cur_bars[-1][ccol]
    day_msecs = tf_to_secs('1d') * 1000
    prev_end_ms = time_ms // day_msecs * day_msecs
    prev_bars = KLine.query(exs, timeframe, prev_end_ms - tf_msecs * 10, prev_end_ms)
    if not prev_bars:
        return
    prev_price = prev_bars[-1][ccol]
    price_chg = cur_price / prev_price - 1
    if not up_reached and price_chg >= up_rate:
        await redis.sadd(dt_key, up_key)
    elif not dn_reached and price_chg <= down_rate:
        await redis.sadd(dt_key, dn_key)
    else:
        return
    await rpc.send_msg(dict(
        type=RPCMessageType.MARKET_TIP,
        exchange=exg_name,
        market=market,
        symbol=symbol,
        price=cur_price,
        prev_price=prev_price,
        rate_pct=round(price_chg * 100, 2)
    ))


async def run_market_monitor():
    '''
    运行策略信号更新任务。
    此任务应和爬虫在同一个进程。以便读取到爬虫Kline发出的异步事件。
    '''
    config = AppConfig.get()
    webhook = config.get('webhook')
    if not bool(webhook and webhook.get(RPCMessageType.MARKET_TIP)):
        return
    from banbot.storage.base import init_db, db
    init_db()
    logger.info(f'start market change monitor')
    rpc = Notify(config)
    while True:
        exg_name, market, symbol, timeframe = await KLine.wait_bars('*', '*', '*', '5m')
        try:
            with db():
                async with AsyncRedis() as redis:
                    await _check_symbol(rpc, redis, exg_name, market, symbol, timeframe)
        except Exception:
            logger.exception(f'err while check market: {[exg_name, market, symbol]}')
