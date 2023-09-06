#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : exc_notify.py
# Author: anyongjin
# Date  : 2023/7/31
import re
import asyncio
import datetime
import sys

from banbot.config import Config
_re_err_stack = re.compile(r'^\s+File\s*"([^"]+)"')


def allow_exc_notify(config: Config):
    '''
    是否启用了异常通知
    '''
    webhook = config.get('webhook')
    return bool(webhook and webhook.get('exception'))


def try_send_exc_notify(cache_key: str, content: str):
    '''
    尝试发送异常通知给管理员。
    如未限流，则立刻发送。如被限流，则记录数量
    '''
    from banbot.util.redis_helper import SyncRedis
    redis = SyncRedis()
    # 对异常消息进行限流
    wait_secs = redis.ttl(cache_key)
    wait_num = redis.get(cache_key, 0) + 1
    if wait_secs > 1:
        # 处于限流时间内，更新数量和最后的消息
        redis.set(cache_key, wait_num, wait_secs)
        redis.set(f'{cache_key}_text', content, wait_secs)
        return
    redis.set(cache_key, max(wait_num, 1), 5)
    # 1s后发送通知，避免此处直接发耗时较多。
    send_exc_notify_after(1, cache_key, content)


def do_send_exc_notify(key: str, detail: str, num: int = 1):
    '''
    发送异常通知给管理员：微信
    不要直接调用此方法。应调用带限流的try_send_exc_notify
    '''
    from banbot.rpc.notify_mgr import Notify, NotifyType
    if not Notify.instance:
        from banbot.config import AppConfig
        if AppConfig.obj:
            Notify(AppConfig.get())
        else:
            print('no appconfig load, send exc notify skip')
            return
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # 这里是在单独线程里，没有自动设置异步循环，需要手动设置
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    content = f'EXC:{key}，数量:{num}\n{del_third_traces(detail)}'
    loop.run_until_complete(Notify.instance.send_msg(dict(
        type=NotifyType.EXCEPTION,
        status=content,
    )))


def send_exc_notify_after(secs: int, key: str, desp: str):
    '''
    在指定时间后，检查是否有待发送的异常，有则发送到管理员微信通知
    :param secs:
    :param key:
    :param desp:
    :return:
    '''
    from banbot.worker.sched import get_sched, STATE_STOPPED
    sched = get_sched()
    start_at = datetime.datetime.now() + datetime.timedelta(seconds=secs)
    args = [key, desp]
    sched.add_job(_send_exc_notify, 'date', run_date=start_at, args=args)
    if sched.state == STATE_STOPPED:
        sched.start()


def _send_exc_notify(key: str, desp: str):
    from banbot.util.redis_helper import SyncRedis
    redis = SyncRedis()
    wait_num = redis.get(key)
    if not wait_num:
        return
    # 180s后允许发送下一条日志
    redis.set(key, 0, 180)
    detail = redis.get(f'{key}_text') or desp
    do_send_exc_notify(key, detail, wait_num)
    # 启动下一次定时发送（如果期间没有数据，则不会发送）
    send_exc_notify_after(170, key, detail)


def del_third_traces(content: str) -> str:
    lines = content.split('\n')
    result = []
    skip_gt_than = sys.maxsize  # 跳过行首空格大于此数值的所有行。
    for i, line in enumerate(lines):
        tab_len = len(line) - len(line.lstrip())
        if tab_len > skip_gt_than:
            continue
        skip_gt_than = sys.maxsize
        mat = _re_err_stack.search(line)
        if mat:
            path = mat.group(1).lower()
            if path.find('/ban') < 0:
                skip_gt_than = tab_len
                continue
        result.append(line)
    return '\n'.join(result)
