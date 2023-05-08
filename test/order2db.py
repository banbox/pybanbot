#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : order2db.py
# Author: anyongjin
# Date  : 2023/5/6
'''
将保存到json文件的订单信息转存到数据库。
针对早期的实盘订单数据。
'''
import os
import orjson
from banbot.storage import *
from typing import *
from banbot.util import btime
from banbot.util.common import logger
data_dir = r'E:\trade\ban_data\live'
file_vers = [
    # UTC 2023-04-11 10:36之后
    ('orders.tokyo7.json', 1),
    ('orders.tokyo8.json', 1),
    # UTC 2023-04-14 06:44
    ('orders.tokyo9.json', 2),
    # UTC 2023-04-15 09:12
    ('orders.tokyo10.json', 3),
    ('orders.tokyo11.json', 3),
    ('orders.tokyo12.json', 3),
    ('orders.tokyo13.json', 3),
]

task: Optional[BotTask] = None
task_ver = None


def update_task(item: dict, ver: int):
    global task, task_ver
    if not task or ver != task_ver:
        sess = db.session
        task = BotTask(mode='live', start_at=btime.to_datetime(item['enter_timestamp']))
        sess.add(task)
        sess.flush()
        task_ver = ver
    item['task_id'] = task.id
    ext_at = item.get('exit_timestamp') or (item['enter_timestamp'] + 1)
    task.stop_at = btime.to_datetime(ext_at)


def convert():
    sess = db.session
    for fname, ver in file_vers:
        fpath = os.path.join(data_dir, fname)
        with open(fpath, 'rb') as fin:
            data = orjson.loads(fin.read())
        all_ods = data['open_ods'] + data['his_ods']
        all_ods = sorted(all_ods, key=lambda x: x['enter_timestamp'])
        logger.info(f'handle: {fname}, ver: {ver}, num: {len(all_ods)}')
        for item in all_ods:
            update_task(item, ver)
            item['strategy'] = item['key'].split('_')[-1]
            item['stg_ver'] = ver
            item['enter_create_at'] = item['enter_timestamp'] * 1000
            item['enter_update_at'] = item['enter_timestamp'] * 1000
            item['enter_at'] = item['enter_create_at']
            ext_at = item.get('exit_timestamp') or (item['enter_timestamp'] + 1)
            item['exit_create_at'] = ext_at * 1000
            item['exit_update_at'] = ext_at * 1000
            item['exit_at'] = item['exit_create_at']
            od = InOutOrder(**item)
            sess.add(od)
            sess.flush()
            od.enter.inout_id = od.id
            sess.add(od.enter)
            if od.exit:
                od.exit.inout_id = od.id
                sess.add(od.exit)
            sess.commit()
        sess.commit()


if __name__ == '__main__':
    from banbot.storage.base import init_db
    from banbot.config import AppConfig
    AppConfig.init_by_args(dict(config=[r'E:\trade\banbot\banbot\config\config.json']))
    init_db()
    with db():
        convert()
