#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_redis_pub.py
# Author: anyongjin
# Date  : 2023/5/1
import asyncio
import datetime

from banbot.config import AppConfig
from banbot.util.redis_helper import AsyncRedis, PubSub
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
        tip_msg = f'{msg["channel"]}: {msg["data"]}'
        callback(tip_msg)

    logger.info('run complete')


async def run_consumer():
    redis = AsyncRedis()
    conn = redis.pubsub()
    await conn.subscribe('fake')

    asyncio.create_task(handle_msgs(conn, logger.info))
    count = 1
    while True:
        count += 1
        if count == 5:
            await conn.subscribe('channel1', 'channel2')
        await asyncio.sleep(1)


async def run_producer():
    redis = AsyncRedis()
    while True:
        await asyncio.sleep(1)
        msg = str(datetime.datetime.now())
        await redis.publish('channel1', msg)
        await redis.publish('channel2', msg)
        await redis.publish('channel3', msg)
        await redis.publish('channel4', msg)


async def test():
    redis = AsyncRedis()

    logger.info('send:zero')
    await redis.publish('channel1', 'zero')

    logger.info('send:first')
    await redis.publish('channel1', 'first')
    await asyncio.sleep(1)

    logger.info('send:world')
    await redis.publish('channel2', 'world')

    await asyncio.sleep(1)
    await redis.publish('channel3', 'this is sub after listen 1')
    await redis.publish('channel4', 'this is sub after listen 2')

    logger.info('pub finish')
    # await redis.publish('channel1', STOPWORD)
    await asyncio.sleep(1)
    logger.info('test finish')


AppConfig.init_by_args()
asyncio.run(run_consumer())

