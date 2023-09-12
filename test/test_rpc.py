#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_rpc.py
# Author: anyongjin
# Date  : 2023/9/12
import asyncio
from banbot.config import AppConfig
from banbot.rpc import Notify, NotifyType


async def _do_send():
    AppConfig.init_by_args()
    Notify.send(
        type=NotifyType.STRATEGY_MSG,
        msg='test msg'
    )
    await asyncio.sleep(3)


def test_send():
    asyncio.run(_do_send())


if __name__ == '__main__':
    test_send()
