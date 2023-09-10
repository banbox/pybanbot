#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : telegram_.py
# Author: anyongjin
# Date  : 2023/9/10
from telegram import Bot
from banbot.rpc.webhook import *


class Telegram(Webhook):

    def __init__(self, config: Config, item: dict):
        super(Telegram, self).__init__(config, item)
        token = item.get('token')
        if not token:
            raise ValueError('token is required for telegram channel')
        self.channel_id = item.get('channel')
        if not self.channel_id:
            raise ValueError('channel is required for telegram channel')
        self.bot = Bot(token)

    async def _do_send_msg(self, payload: dict):
        text = payload['content']
        await self.bot.send_message(self.channel_id, text=text)
