#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : tfscaler.py
# Author: anyongjin
# Date  : 2023/5/14
from typing import List, Tuple, Dict
from banbot.storage import KLine
from banbot.data.tools import bulk_ohlcv_do
from banbot.exchange.crypto_exchange import CryptoExchange
from banbot.compute.tainds import ocol, hcol, lcol, ccol


async def calc_symboltf_scales(exg: CryptoExchange, symbols: List[str], back_num: int = 300)\
        -> Dict[str, List[Tuple[str, float]]]:
    from banbot.exchange.exchange_utils import tf_to_secs
    agg_list = [agg for agg in KLine.agg_list if agg.secs < 1800]
    pip_prices = {pair: exg.price_get_one_pip(pair) for pair in symbols}
    res_list = []

    def ohlcv_cb(candles, pair, timeframe, **kwargs):
        kscore = calc_candles_score(candles, pip_prices.get(pair))
        tf_secs = tf_to_secs(timeframe)
        res_list.append((pair, timeframe, tf_secs, kscore))

    for agg in agg_list:
        down_args = dict(limit=back_num, allow_lack=0.1)
        await bulk_ohlcv_do(exg, symbols, agg.tf, down_args, ohlcv_cb)

    from itertools import groupby
    res_list = sorted(res_list, key=lambda x: x[0])
    gp_res = groupby(res_list, key=lambda x: x[0])
    result = dict()
    for symbol, gp in gp_res:
        items = sorted(list(gp), key=lambda x: x[2])
        result[symbol] = list(map(lambda x: (x[1], x[3]), items))
    return result


def calc_candles_score(candles: List[Tuple], pip_chg: float) -> float:
    '''
    计算K线质量。用于淘汰变动太小，波动不足的交易对；或计算交易对的最佳周期。阈值取0.8较合适
    价格变动：四价相同-1分；bar变动=最小变动单位-1分；70%权重
    平均跳空占比：30%权重

    改进点：目前无法量化横盘频繁密集震动。
    '''
    if not candles:
        return 0
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
    # 取平方，扩大分数差距
    jrate_score = pow(1 - sum(jump_rates) / len(jump_rates), 2)
    return round(chg_score * 0.7 + jrate_score * 0.3, 3)

