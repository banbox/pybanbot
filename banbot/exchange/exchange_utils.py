#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : exchange_utils.py
# Author: anyongjin
# Date  : 2023/3/25
from typing import *


def text_markets(market_map: Dict[str, Any], min_num: int = 10):
    from tabulate import tabulate
    from itertools import groupby
    headers = ['Quote', 'Count', 'Active', 'Spot', 'Future', 'Margin', 'TakerFee', 'MakerFee']
    records = []
    markets = list(market_map.values())
    markets = sorted(markets, key=lambda x: x['quote'])
    for key, group in groupby(markets, key=lambda x: x['quote']):
        glist = list(group)
        if len(glist) < min_num:
            continue
        active = len([m for m in glist if m.get('active', True)])
        spot = len([m for m in glist if m.get('spot')])
        future = len([m for m in glist if m.get('future')])
        margin = len([m for m in glist if m.get('margin')])
        taker_gps = [f"{tk}/{len(list(tg))}" for tk, tg in groupby(glist, key=lambda x: x['taker'])]
        taker_text = '  '.join(taker_gps)
        maker_gps = [f"{tk}/{len(list(tg))}" for tk, tg in groupby(glist, key=lambda x: x['maker'])]
        maker_text = '  '.join(maker_gps)
        records.append((
            key, len(glist), active, spot, future, margin, taker_text, maker_text
        ))
    records = sorted(records, key=lambda x: x[1], reverse=True)
    return tabulate(records, headers, 'orgtbl')
