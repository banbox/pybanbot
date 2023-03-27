#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : exchange_utils.py
# Author: anyongjin
# Date  : 2023/3/25
import ccxt


def timeframe_to_seconds(timeframe: str) -> int:
    """
    Translates the timeframe interval value written in the human readable
    form ('1m', '5m', '1h', '1d', '1w', etc.) to the number
    of seconds for one timeframe interval.
    """
    return ccxt.Exchange.parse_timeframe(timeframe)
