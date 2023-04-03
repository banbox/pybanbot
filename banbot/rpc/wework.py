#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : wework.py
# Author: anyongjin
# Date  : 2023/4/1
import asyncio
from banbot.rpc.rpc import *
from corpwechatbot.app import AppMsgSender
from functools import partial


class WeWork(Webhook):
    def __init__(self, rpc: RPC, config: Config):
        self._config = config
        self.rpc = rpc
        self.api = AppMsgSender(**config['wework'])
        self._retries = 1
        self._retry_delay = 0.1

    async def _do_send_msg(self, payload: dict):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(self.api.send_text, **payload))
