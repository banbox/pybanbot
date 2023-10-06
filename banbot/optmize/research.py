#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : research.py
# Author: anyongjin
# Date  : 2023/7/27
from banbot.storage import *
from banbot.exchange.exchange_utils import *
from banbot.strategy.resolver import get_strategy
from banbot.compute.ctx import *
from banbot.compute.sta_inds import tcol, ccol
from banbot.compute.tools import append_new_bar
from banbot.config import AppConfig
from banbot.util.common import logger


async def calc_strategy_sigs(exs: ExSymbol, timeframe: str, strategy: str, start_ms: int, end_ms: int):
    sess = dba.session
    fts = [TdSignal.symbol_id == exs.id, TdSignal.timeframe == timeframe, TdSignal.strategy == strategy,
           TdSignal.bar_ms >= start_ms, TdSignal.bar_ms < end_ms]
    exc_res = await sess.execute(delete(TdSignal).where(*fts).execution_options(synchronize_session=False))
    logger.info(f'delete old: {exc_res.rowcount}')
    await sess.commit()
    tf_msecs = tf_to_secs(timeframe) * 1000
    # 多往前取200个，做预热
    fetch_start = start_ms - tf_msecs * 200
    ohlcvs = await KLine.query(exs, timeframe, fetch_start, end_ms)
    range_str = f'{ohlcvs[0][0]} {ohlcvs[-1][0]}' if ohlcvs else 'empty'
    logger.info(f'get {len(ohlcvs)} bars range: {range_str}')
    config = AppConfig.get()
    pair_tf = f'{exs.exchange}_{exs.market}_{exs.symbol}_{timeframe}'
    with TempContext(pair_tf):
        td_signals = []
        stg = get_strategy(strategy)(config)
        await Overlay.delete(0, exs.id, timeframe, fetch_start, end_ms)
        create_args = dict(symbol_id=exs.id, timeframe=timeframe, strategy=strategy)
        for i in range(len(ohlcvs)):
            ohlcv_arr = append_new_bar(ohlcvs[i], tf_msecs // 1000)
            stg.on_bar(ohlcv_arr)
            bar_ms = ohlcv_arr[-1, tcol] + tf_msecs
            if bar_ms < start_ms:
                continue
            for tag, price in stg.bar_signals.items():
                td_signals.append(TdSignal(**create_args, action=tag, bar_ms=bar_ms, create_at=bar_ms, price=price))
        stg.on_bot_stop()
        sess.add_all(td_signals)
        logger.info(f'insert {len(td_signals)} signals')
    await sess.commit()


async def _test_strategy():
    sess = dba.session
    fts = [ExSymbol.exchange == 'binance', ExSymbol.market == 'future', ExSymbol.symbol == 'BTC/USDT:USDT']
    stmt = select(ExSymbol).where(*fts).limit(1)
    exs: ExSymbol = (await sess.scalars(stmt)).first()
    start_ms, end_ms = 1690866160016, btime.utcstamp()
    logger.info(f'start test symbol: {exs}')
    await calc_strategy_sigs(exs, '15m', 'DigoChain', start_ms, end_ms)


if __name__ == '__main__':
    AppConfig.init_by_args()
    async with dba():
        _test_strategy()
