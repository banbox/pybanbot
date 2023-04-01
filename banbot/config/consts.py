#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : consts.py
# Author: anyongjin
# Date  : 2023/3/27
from enum import Enum
from typing import Dict, Any
MIN_STAKE_AMOUNT = 10
MAX_FETCH_NUM = 1000  # 单次请求最大返回数量


class RunMode(Enum):
    """
    Bot running mode (backtest, hyperopt, ...)
    can be "live", "dry-run", "backtest", "edge", "hyperopt".
    """
    LIVE = "live"
    DRY_RUN = "dry_run"
    BACKTEST = "backtest"
    HYPEROPT = "hyperopt"
    PLOT = "plot"
    WEBSERVER = "webserver"
    OTHER = "other"


TRADING_MODES = [RunMode.LIVE, RunMode.DRY_RUN]
OPTIMIZE_MODES = [RunMode.BACKTEST, RunMode.HYPEROPT]

Config = Dict[str, Any]
