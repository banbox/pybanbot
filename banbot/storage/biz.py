#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : bix.py
# Author: anyongjin
# Date  : 2023/11/9
import asyncio
from typing import *
from banbot.storage.orders import InOutOrder


class BotCache:
    open_ods: Dict[int, InOutOrder] = dict()
    '全部打开的订单'

    updod_at = 0
    '上次刷新open_ods的时间戳，13位'

    pair_copied_at: Dict[str, Tuple[int, int]] = dict()
    '[symbol: (int, int)]记录所有标的从爬虫获取到K线的最新时间，以及等待下一次收到的时间，用于判断是否有长期未收到的'

    last_bar_ms = 0
    '上次收到bar的结束时间，13位时间戳'

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
        minute_ms = 60000
        while BotGlobal.state == BotState.RUNNING:
            await asyncio.sleep(60)
            cur_time_ms = btime.time_ms()
            try:
                items = list(cls.pair_copied_at.items())
                fails = []
                for pair, wait in items:
                    if wait[0] + wait[1] * 2 >= cur_time_ms:
                        continue
                    timeout_min = round((cur_time_ms - wait[0] - wait[1]) / minute_ms)
                    fails.append(f'{pair}: {timeout_min}')
                if fails:
                    fail_text = ', '.join(fails)
                    Notify.send(type=NotifyType.EXCEPTION, status=f'监听爬虫K线超时：{fail_text}')
            except Exception:
                logger.error('run_bar_waiter error')