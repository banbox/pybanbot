#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : line_.py
# Author: anyongjin
# Date  : 2023/9/12
from banbot.rpc.webhook import *
from banbot.util.net_utils import get_http_sess, parse_http_rsp


class Line(Webhook):
    line_host = 'https://api.line.me'
    push_path = '/v2/bot/message/push'

    def __init__(self, config: Config, item: dict):
        super(Line, self).__init__(config, item)
        token = item.get('token')
        if not token:
            raise ValueError('token is required for line channel')
        self.headers = {'Authorization': f'Bearer {token}'}
        targetIds = item.get('targets')
        self.targets = list(targetIds) if isinstance(targetIds, (list, tuple, set)) else [targetIds]
        if not self.targets:
            raise ValueError('targets is required for line channel')

    async def _do_send_msg(self, payload: dict):
        text = payload['content']
        sess = await get_http_sess(self.line_host)
        for to_id in self.targets:
            data = dict(
                to=to_id,
                messages=[dict(type='text', text=text)]
            )
            rsp = await sess.post(self.push_path, json=data, headers=self.headers)
            if rsp.status != 200:
                logger.error(f'send line msg error[{rsp.status}]: {parse_http_rsp(rsp)}')

