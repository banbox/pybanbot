#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : types.py
# Author: anyongjin
# Date  : 2023/4/18
from typing import Dict, Optional, TypedDict


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
