#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : sig_sync.py
# Author: anyongjin
# Date  : 2023/7/29
'''
策略信号同步到tdsignal
'''
import asyncio
import time

from banbot.storage import *
from banbot.exchange.exchange_utils import *
from banbot.strategy.resolver import get_strategy, BaseStrategy
from banbot.compute.ctx import *
from banbot.compute.sta_inds import tcol
from banbot.compute.tools import append_new_bar
from banbot.config import AppConfig
from banbot.util.common import logger

_stgy_map: Dict[str, BaseStrategy] = dict()


def ensure_sig_update():
    pass


def _get_stgy_inst(exs: ExSymbol, timeframe: str, stg_cls: Type[BaseStrategy]) -> Tuple[BaseStrategy, int]:
    global _stgy_map
    strategy = stg_cls.__name__
    stg_like = f'{strategy}:%'
    stg_ver = f'{strategy}:{stg_cls.version}'
    ctx_key = f'{exs.exchange}_{exs.market}_{exs.symbol}_{timeframe}'
    # 获取策略实例，每个symbol+tf+strategy对应一个策略实例
    cache_key = f'{ctx_key}_{strategy}'
    if cache_key not in _stgy_map:
        config = AppConfig.get()
        stg = get_strategy(strategy)(config)
        _stgy_map[cache_key] = stg
    else:
        stg = _stgy_map[cache_key]
    # 查找策略当前版本已计算进度，如有旧版本则清空进度
    from banbot.util.redis_helper import SyncRedis
    redis = SyncRedis()
    stg_ver_key = f'stgy_version'
    stg_key = f'sig_{strategy}'
    old_ver = redis.hget(stg_ver_key, strategy)
    if str(old_ver) != str(stg_cls.version):
        # 版本不同，删除旧的缓存进度
        redis.delete(stg_key)
        redis.hset(stg_ver_key, strategy, stg_cls.version)
        # 删除旧版本策略的所有信号
        sess = db.session
        fts = [TdSignal.strategy.like(stg_like), TdSignal.strategy != stg_ver]
        del_num = sess.query(TdSignal).filter(*fts).delete(synchronize_session=False)
        sess.commit()
        logger.warning(f'del {del_num} old version signals for {strategy}')
    val = redis.hget(stg_key, ctx_key)
    return stg, val


def calc_stgy_sigs(exs: ExSymbol, timeframe: str, stg_cls: Type[BaseStrategy], end_ms: int):
    '''
    计算某个币，在某个周期下，使用指定策略，产生的信号，并记录到tdsignal
    策略始终从最开始的数据计算。所以只传入截止时间即可。
    '''
    strategy: str = stg_cls.__name__
    # 获取策略上次计算进度
    stg, last_end = _get_stgy_inst(exs, timeframe, stg_cls)
    stg_ver = f'{strategy}:{stg_cls.version}'
    # 初始化开始位置
    if not last_end:
        last_end, _ = KLine.query_range(exs.id, timeframe)
    pair_tf = f'{exs.exchange}/{exs.market}/{exs.symbol}/{timeframe}'
    td_signals, calc_start = [], None
    time_start = time.monotonic()
    with TempContext(pair_tf):
        # 获取期间的蜡烛数据
        fetch_start = last_end
        tf_msecs = tf_to_secs(timeframe) * 1000
        if bar_num.get() < stg_cls.warmup_num:
            # 尚未预热完成，预取数据进行预热
            fetch_start -= tf_msecs * (stg_cls.warmup_num - bar_num.get())
        ohlcvs = KLine.query(exs, timeframe, fetch_start, end_ms)
        if not ohlcvs:
            return
        bar_end_ms = int(ohlcvs[0][0])
        create_args = dict(symbol_id=exs.id, timeframe=timeframe, strategy=stg_ver)
        for i in range(len(ohlcvs)):
            ohlcv_arr = append_new_bar(ohlcvs[i], tf_msecs // 1000)
            stg.state = dict()
            stg.bar_signals = dict()
            stg.on_bar(ohlcv_arr)
            bar_end_ms = int(ohlcv_arr[-1, tcol]) + tf_msecs
            if bar_end_ms < last_end or bar_num.get() < stg_cls.warmup_num:
                continue
            elif not calc_start:
                calc_start = int(ohlcv_arr[-1, tcol])
            for tag, price in stg.bar_signals.items():
                td_signals.append(TdSignal(**create_args, action=tag, price=price, bar_ms=bar_end_ms,
                                           create_at=bar_end_ms))
    time_cost = time.monotonic() - time_start
    if time_cost > 1:
        logger.info(f'calc {len(ohlcvs)} bars cost: {time_cost:.3f} s')
    sess = db.session
    del_num = 0
    if calc_start:
        # 删除此策略时间段内已计算信号
        fts = [TdSignal.symbol_id == exs.id, TdSignal.timeframe == timeframe, TdSignal.strategy == stg_ver,
               TdSignal.bar_ms >= calc_start, TdSignal.bar_ms < bar_end_ms]
        del_num = sess.query(TdSignal).filter(*fts).delete(synchronize_session=False)
    sess.add_all(td_signals)
    logger.warning(f'replace {del_num} with {len(td_signals)} signals for {exs.id}/{timeframe} {calc_start}-{bar_end_ms}')
    sess.commit()
    # 记录策略的计算最新位置
    from banbot.util.redis_helper import SyncRedis
    redis = SyncRedis()
    stg_key = f'sig_{strategy}'
    ctx_key = f'{exs.exchange}_{exs.market}_{exs.symbol}_{timeframe}'
    redis.hset(stg_key, ctx_key, bar_end_ms)


async def run_tdsig_updater():
    from banbot.storage.base import init_db
    init_db()
    config = AppConfig.get()
    strategy = config.get('signal_strategy')
    if not strategy:
        logger.warning('tdsignal updater skip, no `signal_strategy` in config..')
        return
    stgy_cls = get_strategy(strategy)
    if not stgy_cls:
        logger.warning(f'tdsignal updater skip, `{strategy}` not exist')
        return
    logger.info(f'start tdsignal updater: {strategy}, version: {stgy_cls.version}, warmup: {stgy_cls.warmup_num}')
    while True:
        exg_name, market, symbol, timeframe = await KLine.wait_bars('*', '*', '*', '*')
        if tf_to_secs(timeframe) < 300:
            # 跳过5分钟以下维度
            continue
        try:
            with db():
                exs = ExSymbol.get(exg_name, symbol, market)
                calc_stgy_sigs(exs, timeframe, stgy_cls, btime.utcstamp())
        except Exception:
            logger.exception(f'calc stgy sigs error: {exg_name} {market} {symbol} {timeframe}')

