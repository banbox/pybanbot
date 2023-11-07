#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : trade_utils.py
# Author: anyongjin
# Date  : 2023/11/7


def validate_trigger_price(pair: str, short: bool, stoploss: float = None, takeprofit: float = None):
    """验证止损价和止盈价是否有效"""
    if not stoploss and not takeprofit:
        return stoploss, takeprofit
    from banbot.main.addons import MarketPrice
    cur_price = MarketPrice.get(pair)
    pos_flag = -1 if short else 1
    od_tag = 'short' if short else 'long'
    if stoploss:
        if (stoploss - cur_price) * pos_flag >= 0:
            tag = '>' if short else '<'
            raise ValueError(f'[{pair}] stoploss({stoploss}) must {tag} {cur_price} for {od_tag} order')
    if takeprofit:
        if (takeprofit - cur_price) * pos_flag <= 0:
            tag = '<' if short else '>'
            raise ValueError(f'[{pair}] takeprofit({takeprofit}) must {tag} {cur_price} for {od_tag} order')
    return stoploss, takeprofit
