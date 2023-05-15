#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : tfscaler.py
# Author: anyongjin
# Date  : 2023/5/14
from typing import List, Tuple
from banbot.storage import KLine
from banbot.data.tools import auto_fetch_ohlcv
from banbot.exchange.crypto_exchange import CryptoExchange
from banbot.compute.tainds import ocol, hcol, lcol, ccol


async def calc_symboltf_scales(exg: CryptoExchange, symbols: List[str], back_num: int = 300):
    result = dict()
    for symbol in symbols:
        result[symbol] = await calc_tf_scale(exg, symbol, back_num)
    return result


async def calc_tf_scale(exg: CryptoExchange, symbol: str, back_num: int = 300) -> List[Tuple[str, float]]:
    price_pip = exg.price_get_one_pip(symbol)
    result = []
    for agg in KLine.agg_list:
        if agg.secs >= 1800:
            # 只计算30m以下维度
            break
        candles = await auto_fetch_ohlcv(exg, symbol, agg.tf, limit=back_num, allow_lack=0.1)
        kscore = calc_candles_score(candles, price_pip)
        result.append((agg.tf, kscore))
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

