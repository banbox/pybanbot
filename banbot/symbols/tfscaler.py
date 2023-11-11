#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : tfscaler.py
# Author: anyongjin
# Date  : 2023/5/14
from typing import List, Tuple, Dict

from banbot.compute.sta_inds import ocol, hcol, lcol, ccol
from banbot.data.tools import fast_bulk_ohlcv
from banbot.exchange.crypto_exchange import CryptoExchange
from banbot.storage import ExSymbol, BotGlobal
from banbot.util.common import logger


async def calc_symboltf_scales(exg: CryptoExchange, symbols: List[str], back_num: int = 300)\
        -> Dict[str, List[Tuple[str, float]]]:
    from banbot.util.tf_utils import tf_to_secs
    if not BotGlobal.run_tf_secs:
        raise ValueError('`run_timeframes` not set in `StrategyResolver.load_run_jobs`')
    if BotGlobal.ws_mode:
        return {p: [('ws', 1.)] for p in symbols}
    pip_prices = {pair: exg.price_get_one_pip(pair) for pair in symbols}
    res_list = []

    def ohlcv_cb(candles, exs: ExSymbol, timeframe: str, **kwargs):
        tf_secs = tf_to_secs(timeframe)
        if not candles:
            res_list.append((exs.symbol, timeframe, tf_secs, 1))
            return
        kscore = calc_candles_score(exs, candles, pip_prices.get(exs.symbol))
        res_list.append((exs.symbol, timeframe, tf_secs, kscore))

    for cur_tf, _ in BotGlobal.run_tf_secs:
        await fast_bulk_ohlcv(exg, symbols, cur_tf, limit=back_num, callback=ohlcv_cb, allow_lack=0.1)

    from itertools import groupby
    res_list = sorted(res_list, key=lambda x: x[0])
    gp_res = groupby(res_list, key=lambda x: x[0])
    result = dict()
    for symbol, gp in gp_res:
        items = sorted(list(gp), key=lambda x: x[2])
        result[symbol] = list(map(lambda x: (x[1], x[3]), items))
    return result


def calc_candles_score(exs: ExSymbol, candles: List[Tuple], pip_chg: float) -> float:
    '''
    计算K线质量。用于淘汰变动太小，波动不足的交易对；或计算交易对的最佳周期。阈值取0.8较合适
    价格变动：四价相同-1分；bar变动=最小变动单位-1分；70%权重
    平均跳空占比：30%权重

    改进点：目前无法量化横盘频繁密集震动。
    '''
    if not candles:
        return 0
    if not pip_chg:
        logger.warning(f'pip change for {exs} invalid, skip filter')
        return 1
    total_val = fin_score = len(candles)
    jump_rates, prow = [], None
    for i, row in enumerate(candles):
        chg_rate = round((row[hcol] - row[lcol]) / pip_chg, 3)
        if chg_rate == 0. or chg_rate == 1.:
            fin_score -= 1
        elif chg_rate == 2.:
            fin_score -= 0.3
        if prow:
            ner_max_chg = max(prow[hcol], row[hcol]) - min(prow[lcol], row[lcol])
            if not ner_max_chg:
                jump_rates.append(0)
            else:
                jump_rates.append(abs(prow[ccol] - row[ocol]) / ner_max_chg)
        prow = row
    chg_score = fin_score / total_val
    if not jump_rates:
        logger.warning(f'no jump_rates for {exs}, use chg_score for tf_score')
        return chg_score
    # 取平方，扩大分数差距
    jrate_score = pow(1 - sum(jump_rates) / len(jump_rates), 2)
    return round(chg_score * 0.7 + jrate_score * 0.3, 3)

