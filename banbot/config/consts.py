#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : consts.py
# Author: anyongjin
# Date  : 2023/3/27
from enum import Enum
from typing import Dict, Any
MIN_STAKE_AMOUNT = 50
MAX_FETCH_NUM = 1000  # 单次请求最大返回数量
MAX_CONC_OHLCV = 10  # 最大并发下载K线数量，过大会导致卡死

Config = Dict[str, Any]
NATIVE_TFS = ['1s', '1m', '3m', '5m', '15m', '30m', '1h', '4h', '8h', '1d']
DATETIME_PRINT_FORMAT = '%Y-%m-%d %H:%M:%S'


class RunMode(Enum):
    """
    Bot running mode (backtest, hyperopt, ...)
    can be "live", "dry-run", "backtest", "edge", "hyperopt".
    """
    PROD = "prod"
    DRY_RUN = "dry_run"
    BACKTEST = "backtest"
    HYPEROPT = "hyperopt"
    PLOT = "plot"
    WEBSERVER = "webserver"
    OTHER = "other"


LIVE_MODES = [RunMode.PROD, RunMode.DRY_RUN]
OPTIMIZE_MODES = [RunMode.BACKTEST, RunMode.HYPEROPT]
NORDER_MODES = [RunMode.PLOT, RunMode.WEBSERVER, RunMode.OTHER]


class BotState(Enum):
    """
    Bot application states
    """
    RUNNING = 1
    STOPPED = 2
    RELOAD_CONFIG = 3

    def __str__(self):
        return self.name.lower()


