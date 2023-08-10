#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : sched.py
# Author: anyongjin
# Date  : 2023/7/31
'''
这是定时任务调度。
适合：
每隔一段时间执行一个任务
推迟一定时间执行某个函数
'''
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.redis import RedisJobStore
from apscheduler.schedulers.base import STATE_STOPPED  # noqa
global _sched
_sched = None


def get_sched():
    global _sched
    if not _sched:
        import random
        from banbot.storage.base import init_db
        from banbot.util.redis_helper import get_conn_args
        from banbot.util import btime
        init_db()
        conn_args = get_conn_args()
        task_id = random.randrange(10000, 99999)
        conn_args['jobs_key'] = f'apched.job{task_id}'
        conn_args['run_times_key'] = f'apched.run_time{task_id}'
        stores = dict(default=RedisJobStore(**conn_args))
        sche_args = dict(jobstores=stores, timezone=btime.sys_timezone())
        _sched = BackgroundScheduler(**sche_args)
    return _sched


def start_scheduler():
    '''
    启动定时轮训，爬虫，机器人，web进程都可使用，不同进程可注册不同的任务。
    默认注册的：异常通知
    '''
    sched = get_sched()
    # sched.add_job(send_timeout_tips, 'interval', seconds=5)
    sched.start()
