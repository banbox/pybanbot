#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : bix.py
# Author: anyongjin
# Date  : 2023/11/9
from typing import *
from banbot.storage.orders import InOutOrder, InOutStatus


class BotCache:
    data: Dict[str, Any] = dict()
    '通用缓存数据'

    open_ods: Dict[int, InOutOrder] = dict()
    '全部打开的订单'

    updod_at = 0
    '上次刷新open_ods的时间戳，10位'

    pair_copied_at: Dict[str, Tuple[int, int]] = dict()
    '[symbol: (int, int)]记录所有标的从爬虫获取到K线的最新时间，以及等待间隔，用于判断是否有长期未收到的'

    tf_pair_hits: Dict[str, Dict[str, int]] = dict()
    '一段时间内各周期各币种的bar数量，用于定时输出'

    last_bar_ms = 0
    '上次收到bar的结束时间，13位时间戳'

    odbooks: Dict[str, Dict] = dict()
    '缓存所有从爬虫收到的订单簿'

    @classmethod
    def set_pair_ts(cls, pair: str, latest_ms: int, wait_ms: int):
        cls.pair_copied_at[pair] = latest_ms, wait_ms
        cls.last_bar_ms = max(cls.last_bar_ms, latest_ms)

    @classmethod
    async def run_bar_waiter(cls):
        from banbot.storage.common import BotGlobal, BotState
        from banbot.util import btime
        from banbot.rpc import Notify, NotifyType
        from banbot.util.common import logger
        from banbot.util.misc import Sleeper
        minute_ms = 60000
        while BotGlobal.state == BotState.RUNNING:
            await Sleeper.sleep(60)
            cur_time_ms = btime.time_ms()
            try:
                items = list(cls.pair_copied_at.items())
                fails = []
                for pair, wait in items:
                    if wait[0] + wait[1] * 2 >= cur_time_ms:
                        continue
                    timeout_min = round((cur_time_ms - wait[0]) / minute_ms)
                    fails.append(f'{pair}: {timeout_min}')
                if fails:
                    fail_text = ', '.join(fails)
                    Notify.send(type=NotifyType.EXCEPTION, status=f'监听爬虫K线超时：{fail_text}')
            except Exception:
                logger.error('run_bar_waiter error')

    @classmethod
    async def run_bar_summary(cls):
        from banbot.storage.common import BotGlobal, BotState
        from banbot.util import btime
        from croniter import croniter
        from itertools import groupby
        from banbot.util.common import logger
        from banbot.util.misc import Sleeper
        # 在每5分钟偏移1，然后加30s时执行，即01:30 06:30 11:30 16:30...
        loop = croniter('1-59/5 * * * * 30')
        while BotGlobal.state == BotState.RUNNING:
            wait_ts = loop.next() - btime.time()
            await Sleeper.sleep(wait_ts)
            shot = dict()
            for k in set(cls.tf_pair_hits.keys()):
                data = cls.tf_pair_hits[k]
                cls.tf_pair_hits[k] = dict()
                items = sorted(list(data.items()), key=lambda x: x[1])
                groups = groupby(items, key=lambda x: x[1])
                for cnt, gp in groups:
                    shot[f'{k}_{cnt}'] = ', '.join([it[0] for it in gp])
            if shot:
                content = '\n'.join(f'[{k}] {v}' for k, v in shot.items())
                logger.info(f'receive bars in 5 mins:\n{content}')
        logger.info('run_bar_summary done')

    @classmethod
    def save_open_ods(cls, ods: List[InOutOrder]):
        for od in ods:
            if od.status < InOutStatus.FullExit:
                cls.open_ods[od.id] = od.clone()
            elif od.id in cls.open_ods:
                del cls.open_ods[od.id]

    @classmethod
    def open_keys(cls):
        return {od.key for _, od in BotCache.open_ods.items()}

    @classmethod
    def print_chgs(cls, old_keys: Set[str], tag: str):
        """显示open_ods前后的列表变化，仅用于调试"""
        new_keys = cls.open_keys()
        adds = new_keys - old_keys
        dels = old_keys - new_keys
        if adds or dels:
            print(f'open_ods update {tag}: {len(old_keys)} -> {len(new_keys)}, {adds} {dels}')
