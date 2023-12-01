#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : line_.py
# Author: anyongjin
# Date  : 2023/9/12
from banbot.rpc.webhook import *
from banbot.util.net_utils import get_http_sess, parse_http_rsp
from banbot.util.misc import Sleeper


class Line(Webhook):
    line_host = 'https://api.line.me'
    push_path = '/v2/bot/message/push'
    batch_size = 3

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

    async def _do_send_msg(self, msg_list: List[dict]) -> int:
        merge_text = '\n\n'.join([msg['content'] for msg in msg_list])
        sess = await get_http_sess(self.line_host)
        next_ts = self.next_send_ts
        for to_id in self.targets:
            wait_secs = next_ts - time.time()
            if wait_secs > 0:
                await Sleeper.sleep(wait_secs)
            data = dict(
                to=to_id,
                messages=[dict(type='text', text=merge_text)]
            )
            rsp = await sess.post(self.push_path, json=data, headers=self.headers)
            if rsp.status != 200:
                logger.error(f'send line msg error[{rsp.status}]: {parse_http_rsp(rsp)}')
            next_ts = time.time() + 0.15  # line 每秒最多10个请求
        self.next_send_ts = next_ts
        return len(msg_list)

