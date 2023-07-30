#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : sig_sync.py
# Author: anyongjin
# Date  : 2023/7/29
'''
策略信号同步到tdsignal
'''
import time

from banbot.storage import *
from banbot.exchange.exchange_utils import *
from banbot.strategy.resolver import get_strategy, BaseStrategy
from banbot.compute.ctx import *
from banbot.compute.sta_inds import tcol
from banbot.compute.tools import append_new_bar
from banbot.config import AppConfig
from banbot.util.redis_helper import SyncRedis
from banbot.util.common import logger

_stgy_map: Dict[str, BaseStrategy] = dict()


async def ensure_sig_update(exs: ExSymbol, timeframe: str, stgy_ver: str, end_ms: int):
    '''
    确认信号已更新到指定位置。
    此方法由web进程调用。
    '''
    strategy, version = stgy_ver.split(':')
    redis = SyncRedis()
    # 检查缓存策略是否和当前指定策略一致，如不一致则无法计算，直接退出
    stg_ver_key = f'stgy_version'
    stg_key = f'sig_{strategy}'
    ctx_key = f'{exs.exchange}_{exs.market}_{exs.symbol}_{timeframe}'
    calc_ver = redis.hget(stg_ver_key, strategy)
    if calc_ver:
        if str(calc_ver) != version:
            return False, None
        val = redis.hget(stg_key, ctx_key)
        if val and val >= end_ms:
            # 计算位置超过当前时间戳位置，退出
            return True, None
    stg_cls = get_strategy(strategy)
    if not stg_cls or str(stg_cls.version) != version:
        # 加载策略版本不一致，退出
        return False, None
    from banbot.data.wacther import RedisChannel
    time_start = time.monotonic()
    ret_data = await RedisChannel.call_remote(f'calcsig', f'{ctx_key}_{end_ms}')
    cost_ts = time.monotonic() - time_start
    logger.info(f'cost {cost_ts} s for {ctx_key}, return: {ret_data}')
    return True, stg_cls


def _get_last_end(ctx_key: str, stg_cls: Type[BaseStrategy]) -> Tuple[BaseStrategy, int]:
    global _stgy_map
    strategy = stg_cls.__name__
    # 查找策略当前版本已计算进度，如有旧版本则清空进度
    redis = SyncRedis()
    stg_ver_key = f'stgy_version'
    stg_key = f'sig_{strategy}'
    old_ver = redis.hget(stg_ver_key, strategy)
    if str(old_ver) != str(stg_cls.version):
        # 版本不同，删除旧的缓存进度
        redis.delete(stg_key)
        redis.hset(stg_ver_key, strategy, stg_cls.version)
        # 将当前使用的策略版本写入到redis
        redis.set('cur_sig_stgy', f'{strategy}:{stg_cls.version}')
    val = redis.hget(stg_key, ctx_key)
    return val


def _get_stgy_obj(stg_cls: Type[BaseStrategy]):
    '''
    获取策略实例，每个symbol+tf+strategy对应一个策略实例
    必须在context上下文中调用
    '''
    strategy = stg_cls.__name__
    cache_key = f'{symbol_tf.get()}_{strategy}'
    if cache_key not in _stgy_map:
        config = AppConfig.get()
        stg = stg_cls(config)
        _stgy_map[cache_key] = stg
    else:
        stg = _stgy_map[cache_key]
    return stg


def del_old_signals(stg_cls: Type[BaseStrategy]):
    strategy = stg_cls.__name__
    stg_like = f'{strategy}:%'
    stg_ver = f'{strategy}:{stg_cls.version}'
    # 删除旧版本策略的所有信号
    sess = db.session
    fts = [TdSignal.strategy.like(stg_like), TdSignal.strategy != stg_ver]
    del_num = sess.query(TdSignal).filter(*fts).delete(synchronize_session=False)
    sess.commit()
    logger.warning(f'del {del_num} old version signals for {strategy}')


def calc_stgy_sigs(exs: ExSymbol, timeframe: str, stg_cls: Type[BaseStrategy], end_ms: int):
    '''
    计算某个币，在某个周期下，使用指定策略，产生的信号，并记录到tdsignal
    策略始终从最开始的数据计算。所以只传入截止时间即可。
    此方法应始终在爬虫进程调用，防止多个进程调用时，策略状态不连续，导致计算错误
    '''
    strategy: str = stg_cls.__name__
    tf_msecs = tf_to_secs(timeframe) * 1000
    # 获取策略上次计算进度
    ctx_key = f'{exs.exchange}_{exs.market}_{exs.symbol}_{timeframe}'
    last_end = _get_last_end(ctx_key, stg_cls)
    if end_ms // tf_msecs * tf_msecs == last_end:
        # 如果没有新数据，直接退出
        return
    stg_ver = f'{strategy}:{stg_cls.version}'
    # 初始化开始位置
    if not last_end:
        last_end, _ = KLine.query_range(exs.id, timeframe)
    td_signals, calc_start = [], None
    valid_num = 0
    time_start = time.monotonic()
    with TempContext(ctx_key):
        # 获取期间的蜡烛数据
        stg = _get_stgy_obj(stg_cls)
        fetch_start = last_end
        warm_lack = stg_cls.warmup_num - stg.calc_num
        if warm_lack > 0:
            # 尚未预热完成，预取数据进行预热
            fetch_start -= tf_msecs * warm_lack
        ohlcvs = KLine.query(exs, timeframe, fetch_start, end_ms)
        if not ohlcvs:
            return
        bar_end_ms = int(ohlcvs[0][0])
        # 检查是否需要重置上下文
        if bar_time.get()[1] >= bar_end_ms:
            reset_context(ctx_key)
            config = AppConfig.get()
            stg = stg_cls(config)
            cache_key = f'{exs.exchange}_{exs.market}_{exs.symbol}_{timeframe}_{strategy}'
            _stgy_map[cache_key] = stg
            logger.warning(f'reset compute context {cache_key}')
        create_args = dict(symbol_id=exs.id, timeframe=timeframe, strategy=stg_ver)
        for i in range(len(ohlcvs)):
            ohlcv_arr = append_new_bar(ohlcvs[i], tf_msecs // 1000)
            stg.on_bar(ohlcv_arr)
            bar_end_ms = int(ohlcv_arr[-1, tcol]) + tf_msecs
            if bar_end_ms < last_end or stg.calc_num < stg_cls.warmup_num:
                continue
            elif not calc_start:
                calc_start = int(ohlcv_arr[-1, tcol])
            valid_num += 1
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
        sess.commit()
    sess.add_all(td_signals)
    date_rg = f'{btime.to_datestr(calc_start)}-{btime.to_datestr(bar_end_ms)}'
    logger.warning(f'replace {del_num} with {len(td_signals)} signals for {valid_num} bars {exs.id}/{timeframe} {date_rg}')
    sess.commit()
    # 记录策略的计算最新位置
    redis = SyncRedis()
    stg_key = f'sig_{strategy}'
    ctx_key = f'{exs.exchange}_{exs.market}_{exs.symbol}_{timeframe}'
    redis.hset(stg_key, ctx_key, bar_end_ms)


def _get_def_strategy():
    config = AppConfig.get()
    strategy = config.get('signal_strategy')
    if not strategy:
        logger.warning('tdsignal updater skip, no `signal_strategy` in config..')
        return None, None
    stgy_cls = get_strategy(strategy)
    if not stgy_cls:
        logger.warning(f'tdsignal updater skip, `{strategy}` not exist')
        return None, None
    return strategy, stgy_cls


def calc_td_sig_args(exg_name: str, market: str, symbol: str, timeframe: str, end_ms: int, stgy_cls: Type[BaseStrategy]):
    try:
        with db():
            exs = ExSymbol.get(exg_name, market, symbol)
            calc_stgy_sigs(exs, timeframe, stgy_cls, end_ms)
    except Exception:
        logger.exception(f'calc stgy sigs error: {exg_name} {market} {symbol} {timeframe}')


async def reg_redis_event():
    from banbot.data.wacther import RedisChannel
    strategy, stgy_cls = _get_def_strategy()

    def calc_tmp_sig(msg_key: str, msg_data):
        exg_name_, market_, symbol_, timeframe_, end_ms = msg_data.split('_')
        calc_td_sig_args(exg_name_, market_, symbol_, timeframe_, int(end_ms), stgy_cls)
        return f'{msg_key}_{msg_data}', 'ok'

    await RedisChannel.subscribe('calcsig', calc_tmp_sig)


async def run_tdsig_updater():
    '''
    运行策略信号更新任务。
    此任务应和爬虫在同一个进程。以便读取到爬虫Kline发出的异步事件。
    '''
    from banbot.storage.base import init_db
    init_db()
    strategy, stgy_cls = _get_def_strategy()
    with SyncRedis() as redis:
        redis.set('cur_sig_stgy', f'{strategy}:{stgy_cls.version}')
    logger.info(f'start tdsignal updater: {strategy}, version: {stgy_cls.version}, warmup: {stgy_cls.warmup_num}')
    while True:
        exg_name, market, symbol, timeframe = await KLine.wait_bars('*', '*', '*', '*')
        if tf_to_secs(timeframe) < 300:
            # 跳过5分钟以下维度
            continue
        calc_td_sig_args(exg_name, market, symbol, timeframe, btime.utcstamp(), stgy_cls)

