#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : wework.py
# Author: anyongjin
# Date  : 2023/4/1
from functools import partial

from corpwechatbot.app import AppMsgSender

from banbot.rpc.rpc import *


class WeWork(Webhook):
    def __init__(self, rpc: RPC, config: Config):
        super(WeWork, self).__init__(rpc, config)
        import logging
        self._config = config
        self.rpc = rpc
        self.api = AppMsgSender(**config['wework'], log_level=logging.WARNING)

    async def _do_send_msg(self, payload: dict):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(self.api.send_text, **payload))
