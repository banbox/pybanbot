#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : global.py
# Author: anyongjin
# Date  : 2023/4/1
from typing import *

from banbot.config.consts import BotState


class _BotStateMeta(type):
    @property
    def is_future(cls: Type['BotGlobal']):
        return cls.market_type == 'future'

    @property
    def live_mode(cls):
        from banbot.util import btime
        return btime.run_mode in btime.LIVE_MODES


class BotGlobal(metaclass=_BotStateMeta):
    '''
    当前交易机器人的全局状态信息。
    一个机器人进程，只允许运行一个交易所的一个市场。
    '''
    state = BotState.STOPPED

    bot_name: str = 'noname'
    '当前机器人的名称'

    start_at: int = 0
    '启动时间，13位时间戳'

    stg_hash: Optional[str] = None
    '''策略+版本号的哈希值；如果和上次相同说明策略没有变化，可使用一个任务'''

    is_wramup = False
    '当前是否处于预热状态'

    run_tf_secs: List[Tuple[str, int]] = []
    '本次运行指定的时间周期'

    exg_name: str = 'binance'
    '当前运行的交易所'

    market_type: str = 'spot'
    '当前运行的市场:spot/future'

    stg_symbol_tfs: List[Tuple[str, str, str]] = []
    '策略、标的、周期'

    pairs: Set[str] = set()
    '当前交易的标的'

    pairtf_stgs: Dict[str, List] = dict()
    '{pair}_{timeframe}: [stg1, stg2]'
