#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : redis_helper.py
# Author: anyongjin
# Date  : 2023/4/25
import re
import time
import asyncio
import redis
import redis.asyncio as aioredis
from redis.asyncio.client import PubSub as AsyncPubSub
from redis.client import PubSub
from banbot.util.common import Instance
from banbot.config.appconfig import AppConfig

reg_non_key = re.compile(r'[^a-zA-Z0-9_]+')

Expire_time_week = 604800
Expire_time_day = 86400
Expire_time_hour = 3600
Expire_time_minute = 60


def _get_redis_pool(is_async: bool, **kwargs):

    def create_redis():
        redis_url = AppConfig.get()['redis_url']
        module = aioredis if is_async else redis
        return module.ConnectionPool.from_url(redis_url, **kwargs)

    return Instance.getobj(f'redis_{is_async}', create_redis)


def get_native_redis(is_async: bool, decode_rsp=False, connect_timeout=10):
    module = aioredis if is_async else redis
    return module.Redis(connection_pool=_get_redis_pool(is_async, decode_responses=decode_rsp),
                        socket_connect_timeout=connect_timeout, socket_timeout=15)


def build_serializer(json_serializer=None, raise_error=True):
    from banbot.util.misc import json_dumps, safe_json_dumps
    if not json_serializer:
        json_serializer = json_dumps if raise_error else safe_json_dumps

    def any_serializer(val):
        if val is None:
            return None
        type_name = type(val).__name__
        if type_name == 'str':
            return f'str|{val}'
        elif type_name in {'bytes', 'bytearray'}:
            import base64
            base_val = base64.b64encode(val).decode('ascii')
            return f'{type_name}|{base_val}'
        elif type_name in {'int', 'float', 'double', 'bool', 'Decimal'}:
            return f'{type_name}|{val}'
        else:
            return f'{type_name}|{json_serializer(val)}'
    return any_serializer


def build_deserializer(json_deserializer=None, raise_on_error=True):
    import orjson as json
    import six
    if not json_deserializer:
        json_deserializer = json.loads

    def any_deserializer(encoded):
        if encoded is None:
            return None
        if isinstance(encoded, (bytes, bytearray)):
            encoded = encoded.decode('utf-8')
        if not isinstance(encoded, six.string_types):
            return encoded
        type_idx = encoded.find('|')
        if type_idx < 0:
            if raise_on_error:
                from banbot.util.common import logger
                logger.warning(f'ignored invalid serialized object:{encoded}')
                return None
            return encoded
        type_name = encoded[:type_idx]
        encoded = encoded[type_idx + 1:]
        if type_name == 'str':
            return encoded
        elif type_name in {'bytes', 'bytearray'}:
            import base64
            return base64.b64decode(encoded)
        elif type_name in {'int', 'float', 'double'}:
            import builtins
            cur_type = getattr(builtins, type_name)
            return cur_type(encoded)
        elif type_name == 'bool':
            return encoded.lower() == 'true'
        elif type_name == 'Decimal':
            import decimal
            return decimal.Decimal(encoded)
        else:
            return json_deserializer(encoded)
    return any_deserializer


def_serializer = build_serializer()
def_deserializer = build_deserializer()


class LockError(Exception):
    pass


class RedisLock:
    def __init__(self, client: redis.Redis, name: str, acquire_timeout=3, lock_timeout=2,
                 force=False, with_conn=False):
        '''
        :param with_conn: 退出时是否一并关闭redis连接，默认否。
        '''
        self._redis = client
        self._key = f'lock:{name}'
        self.acquire_timeout = acquire_timeout
        self.lock_timeout = lock_timeout
        self.force = force
        self.lock_by = None
        self._with_conn = with_conn

    def __enter__(self):
        import uuid
        identifier = str(uuid.uuid4())

        if self.force:
            if self._redis.set(self._key, identifier, ex=self.lock_timeout):
                self.lock_by = identifier
                return self
        else:
            end = time.monotonic() + self.acquire_timeout
            while time.monotonic() < end:
                # 如果不存在这个锁则加锁并设置过期时间，避免死锁
                if self._redis.set(self._key, identifier, ex=self.lock_timeout, nx=True):
                    self.lock_by = identifier
                    return self
                time.sleep(0.01)
        raise LockError('acquire redis lock timeout')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.lock_by:
            return 0
        if self.lock_by != self._redis.get(self._key):
            return 0
        ret_val = self._redis.delete(self._key)
        self.lock_by = None
        if self._with_conn:
            self._redis.close()
        return ret_val


class AsyncRedisLock:
    def __init__(self, client: aioredis.Redis, name: str, acquire_timeout=3, lock_timeout=2,
                 force=False, with_conn=False):
        '''
        :param with_conn: 退出时是否一并关闭redis连接，默认否。
        '''
        self._redis = client
        self._key = f'lock:{name}'
        self.acquire_timeout = acquire_timeout
        self.lock_timeout = lock_timeout
        self.force = force
        self.lock_by = None
        self._with_conn = with_conn

    async def __aenter__(self):
        import uuid
        identifier = str(uuid.uuid4())

        if self.force:
            if await self._redis.set(self._key, identifier, ex=self.lock_timeout):
                self.lock_by = identifier
                return self
        else:
            end = time.monotonic() + self.acquire_timeout
            while time.monotonic() < end:
                # 如果不存在这个锁则加锁并设置过期时间，避免死锁
                if await self._redis.set(self._key, identifier, ex=self.lock_timeout, nx=True):
                    self.lock_by = identifier
                    return self
                await asyncio.sleep(0.01)
        raise LockError('acquire redis lock timeout')

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self.lock_by:
            return 0
        if self.lock_by != await self._redis.get(self._key):
            return 0
        ret_val = await self._redis.delete(self._key)
        self.lock_by = None
        if self._with_conn:
            await self._redis.close()
        return ret_val


class SyncRedis:
    '''
    经过包装的Redis客户端。可自动序列化/反序列化数据。
    如果在循环中长期使用，请在sleep前调用del释放连接到连接池
    '''
    def __init__(self, serializer=None, deserializer=None, connect_timeout=10):
        self.redis = get_native_redis(False, connect_timeout=connect_timeout)
        self.serializer = serializer or def_serializer
        self.deserializer = deserializer or def_deserializer

    def get_key(self, key: str, max_len=100):
        if not key:
            return key
        if key == 'random':
            import uuid
            token = str(uuid.uuid4())
            while self.redis.exists(token):
                token = str(uuid.uuid4())
            key = token
        safe_key = reg_non_key.sub('_', key)
        if safe_key == '_':
            return None
        return safe_key[:max_len]

    def set(self, key, val='1', expire_time=None):
        '''
        设置redis的键值，过期时间可选
        :param key:
        :param val:
        :param expire_time: 过期的秒数
        :return:
        '''
        key = self.get_key(key)
        if not key:
            return False
        self.redis.set(key, self.serializer(val), ex=expire_time)
        return True

    def get(self, key: str, default_val=None):
        '''
        从redis获取某个值
        :param key:
        :param default_val: 当缓存为None时，默认返回值
        :return:
        '''
        key = self.get_key(key)
        if not key:
            return default_val
        val = self.redis.get(key)
        decode_val = self.deserializer(val)
        if decode_val is None:
            return default_val
        return decode_val

    def ttl(self, key: str):
        key = self.get_key(key)
        if not key:
            return -1
        return self.redis.ttl(key)

    def delete(self, *keys):
        if not keys:
            return 0
        clean_keys = [self.get_key(k) for k in keys]
        clean_keys = [k for k in clean_keys if k]
        if not clean_keys:
            return 0
        return self.redis.delete(*clean_keys)

    def lock(self, lock_name, acquire_timeout=3, lock_timeout=2, force=False, with_conn=False) -> RedisLock:
        '''
        获取基于 Redis 实现的分布式锁对象。针对返回对象应使用with加锁关锁
        :param lock_name: 锁的名称
        :param acquire_timeout: 获取锁的超时时间，默认 3 秒
        :param lock_timeout: 锁的超时时间，默认 2 秒
        :param force: 超时未获取到锁是否强制加锁
        :param with_conn: 退出时是否一并关闭redis连接，默认否
        :return:
        '''
        return RedisLock(self.redis, lock_name, acquire_timeout, lock_timeout, force, with_conn)

    def pubsub(self) -> PubSub:
        '''
        返回一个redis连接，用于监听或取消监听某个消息
        '''
        return self.redis.pubsub()

    def publish(self, channel: str, msg) -> int:
        return self.redis.publish(channel, msg)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.redis.close()


class AsyncRedis:
    '''
    经过包装的Redis客户端。可自动序列化/反序列化数据。
    如果在循环中长期使用，请在sleep前调用del释放连接到连接池
    '''
    def __init__(self, serializer=None, deserializer=None, connect_timeout=10):
        self.redis = get_native_redis(True, connect_timeout=connect_timeout)
        self.serializer = serializer or def_serializer
        self.deserializer = deserializer or def_deserializer

    async def get_key(self, key: str, max_len=100):
        if not key:
            return key
        if key == 'random':
            import uuid
            token = str(uuid.uuid4())
            while await self.redis.exists(token):
                token = str(uuid.uuid4())
            key = token
        safe_key = reg_non_key.sub('_', key)
        if safe_key == '_':
            return None
        return safe_key[:max_len]

    async def set(self, key, val='1', expire_time=None):
        '''
        设置redis的键值，过期时间可选
        :param key:
        :param val:
        :param expire_time: 过期的秒数
        :return:
        '''
        key = await self.get_key(key)
        if not key:
            return False
        await self.redis.set(key, self.serializer(val), ex=expire_time)
        return True

    async def get(self, key: str, default_val=None):
        '''
        从redis获取某个值
        :param key:
        :param default_val: 当缓存为None时，默认返回值
        :return:
        '''
        key = await self.get_key(key)
        if not key:
            return default_val
        val = await self.redis.get(key)
        decode_val = self.deserializer(val)
        if decode_val is None:
            return default_val
        return decode_val

    async def ttl(self, key: str):
        key = await self.get_key(key)
        if not key:
            return -1
        return await self.redis.ttl(key)

    async def delete(self, *keys):
        if not keys:
            return 0
        clean_keys = [await self.get_key(k) for k in keys]
        clean_keys = [k for k in clean_keys if k]
        if not clean_keys:
            return 0
        return await self.redis.delete(*clean_keys)

    def lock(self, lock_name, acquire_timeout=3, lock_timeout=2, force=False, with_conn=False) -> AsyncRedisLock:
        '''
        获取基于 Redis 实现的分布式锁对象。针对返回对象应使用with加锁关锁
        :param lock_name: 锁的名称
        :param acquire_timeout: 获取锁的超时时间，默认 3 秒
        :param lock_timeout: 锁的超时时间，默认 2 秒
        :param force: 超时未获取到锁是否强制加锁
        :param with_conn: 退出时是否一并关闭redis连接，默认否
        :return:
        '''
        return AsyncRedisLock(self.redis, lock_name, acquire_timeout, lock_timeout, force, with_conn)

    def pubsub(self) -> AsyncPubSub:
        '''
        返回一个redis连接，用于监听或取消监听某个消息
        '''
        return self.redis.pubsub()

    async def publish(self, channel: str, msg) -> int:
        return await self.redis.publish(channel, msg)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.redis.close()

