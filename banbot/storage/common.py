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
    def ws_mode(cls: Type['BotGlobal']):
        return any(p for p in cls.run_tf_secs if p[1] < 60)

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

    is_warmup = False
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

    info_pairtfs: Dict[str, List] = dict()
    '策略额外订阅K线 {pair}_{timeframe}: [stg1, stg2]'

    bot_loop = None
    '异步循环，用于rpc中调用exchange的方法'

    forbid_pairs: Set[str] = set()
    '禁止交易的币种'

    book_pairs: Set[str] = set()
    '监听交易对的币种'

    @classmethod
    def get_jobs(cls, pairs: Iterable[str]) -> List[Tuple[str, str, str]]:
        pair_set = set(pairs)
        return [j for j in cls.stg_symbol_tfs if j[1] in pair_set]

    @classmethod
    def remove_jobs(cls, jobs: List[Tuple[str, str, str]]):
        from banbot.compute.ctx import del_context
        for j in jobs:
            if j in cls.stg_symbol_tfs:
                cls.stg_symbol_tfs.remove(j)
                del_context(f'{cls.exg_name}_{cls.market_type}_{j[1]}_{j[2]}')
            pairtf = f'{j[1]}_{j[2]}'
            if pairtf in cls.pairtf_stgs:
                stgs = cls.pairtf_stgs[pairtf]
                cls.pairtf_stgs[pairtf] = [stg for stg in stgs if stg.name != j[0]]
        cls.pairs = {j[1] for j in cls.stg_symbol_tfs}
