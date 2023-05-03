#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_redis_pub.py
# Author: anyongjin
# Date  : 2023/5/1
import asyncio
from banbot.config import AppConfig
from banbot.util.redis_helper import MyRedis, PubSub
from banbot.util.common import logger
STOPWORD = "STOP"


async def handle_msgs(conn: PubSub, callback):
    logger.info('in run_job')

    async for msg in conn.listen():
        if msg['type'] != 'message':
            continue
        # logger.info(f"(Reader) Message Received: {msg}")
        if msg["data"].decode() == STOPWORD:
            logger.info("(Reader) STOP")
            break
        callback(msg["data"])

    logger.info('run complete')


async def test():
    redis = MyRedis()
    conn = redis.pubsub()
    await conn.subscribe('channel1', 'channel2')

    def callback(msg):
        logger.info(f'Receive message: {msg}')

    logger.info('send:zero')
    await redis.publish('channel1', 'zero')

    asyncio.create_task(handle_msgs(conn, callback))

    logger.info('send:first')
    await redis.publish('channel1', 'first')
    await asyncio.sleep(1)

    logger.info('send:hello')
    await redis.publish('channel1', 'hello')
    logger.info('send:world')
    await redis.publish('channel2', 'world')
    logger.info('pub finish')
    await redis.publish('channel1', STOPWORD)
    await asyncio.sleep(1)
    logger.info('test finish')


AppConfig.init_by_args(dict(config=[r'E:\trade\banbot\banbot\config\config.json']))
asyncio.run(test())

