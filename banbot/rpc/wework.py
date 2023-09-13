#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : wework.py
# Author: anyongjin
# Date  : 2023/4/1
from corpwechatbot.app import AppMsgSender

from banbot.rpc.webhook import *


class WeWork(Webhook):
    def __init__(self, config: Config, item: dict):
        super(WeWork, self).__init__(config, item)
        import logging
        arg_keys = ['corpid', 'agentid', 'corpsecret']
        arg_dic = {k: v for k, v in item.items() if k in arg_keys}
        self.api = AppMsgSender(**arg_dic, log_level=logging.WARNING)

    async def _do_send_msg(self, payload: dict):
        self.api.send_text(**payload)
