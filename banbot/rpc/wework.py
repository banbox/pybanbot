#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : wework.py
# Author: anyongjin
# Date  : 2023/4/1
from banbot.rpc.rpc import *
from corpwechatbot.app import AppMsgSender


class WeWork(Webhook):
    def __init__(self, rpc: RPC, config: Config):
        self._config = config
        self.rpc = rpc
        self.api = AppMsgSender(**config['wework'])
        self._retries = 1
        self._retry_delay = 0.1

    def _do_send_msg(self, payload: dict):
        self.api.send_text(**payload)
