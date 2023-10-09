#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : types.py
# Author: anyongjin
# Date  : 2023/4/18
from typing import Dict, Optional, List
from dataclasses import dataclass
from typing_extensions import TypedDict

__all__ = ['Ticker', 'Tickers', 'LeverageTier', 'LeverageTiers', 'TradeModeType']


class Ticker(TypedDict):
    symbol: str
    ask: Optional[float]
    askVolume: Optional[float]
    bid: Optional[float]
    bidVolume: Optional[float]
    last: Optional[float]
    quoteVolume: Optional[float]
    baseVolume: Optional[float]
    # Several more - only listing required.


Tickers = Dict[str, Ticker]


@dataclass
class LeverageTier:
    tier: float
    currency: str
    minNotional: float
    maxNotional: float
    maintenanceMarginRate: float
    maxLeverage: float


class LeverageTiers:
    def __init__(self, raw_list: List[Dict]):
        self.tiers: List[LeverageTier] = []
        for item in raw_list:
            if 'info' in item:
                item.pop('info')
            self.tiers.append(LeverageTier(**item))
        self.max_leverage = 0
        if self.tiers:
            self.max_leverage = int(max([t.maxLeverage for t in self.tiers]))
        self.leverage = 0


class TradeModeType(TypedDict):
    trading_mode: str
    margin_mode: str

