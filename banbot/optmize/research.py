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


def calc_strategy_sigs(exs: ExSymbol, timeframe: str, strategy: str, start_ms: int, end_ms: int):
    sess = db.session
    fts = [TdSignal.symbol_id == exs.id, TdSignal.timeframe == timeframe, TdSignal.strategy == strategy,
           TdSignal.bar_ms >= start_ms, TdSignal.bar_ms < end_ms]
    del_num = sess.query(TdSignal).filter(*fts).delete(synchronize_session=False)
    logger.info(f'delete old: {del_num}')
    sess.commit()
    tf_msecs = tf_to_secs(timeframe) * 1000
    # 多往前取200个，做预热
    fetch_start = start_ms - tf_msecs * 200
    ohlcvs = KLine.query(exs, timeframe, fetch_start, end_ms)
    range_str = f'{ohlcvs[0][0]} {ohlcvs[-1][0]}' if ohlcvs else 'empty'
    logger.info(f'get {len(ohlcvs)} bars range: {range_str}')
    config = AppConfig.get()
    stg = get_strategy(strategy)(config)
    pair_tf = f'{exs.symbol}/{timeframe}'
    with TempContext(pair_tf):
        td_signals = []
        create_args = dict(symbol_id=exs.id, timeframe=timeframe, strategy=strategy)
        for i in range(len(ohlcvs)):
            ohlcv_arr = append_new_bar(ohlcvs[i], tf_msecs // 1000)
            stg.state = dict()
            stg.bar_signals = dict()
            stg.on_bar(ohlcv_arr)
            bar_ms = ohlcv_arr[-1, tcol] + tf_msecs
            if bar_ms < start_ms:
                continue
            for tag, price in stg.bar_signals.items():
                td_signals.append(TdSignal(**create_args, action=tag, bar_ms=bar_ms, create_at=bar_ms, price=price))
        sess.add_all(td_signals)
        logger.info(f'insert {len(td_signals)} signals')
    sess.commit()


def _test_strategy():
    sess = db.session
    fts = [ExSymbol.exchange == 'binance', ExSymbol.market == 'future', ExSymbol.symbol == 'BCH/USDT:USDT']
    exs: ExSymbol = sess.query(ExSymbol).filter(*fts).first()
    start_ms, end_ms = 1690415400000, btime.utcstamp()
    logger.info(f'start test symbol: {exs}')
    calc_strategy_sigs(exs, '5m', 'DigoChain', start_ms, end_ms)


if __name__ == '__main__':
    AppConfig.init_by_args()
    with db():
        _test_strategy()
