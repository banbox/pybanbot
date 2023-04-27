#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : redis_helper.py
# Author: anyongjin
# Date  : 2023/4/25
import re
from redis import StrictRedis
from banbot.util.common import Instance
from banbot.config.appconfig import AppConfig
from typing import Callable

reg_non_key = re.compile(r'[^a-zA-Z0-9_]+')

Expire_time_week = 604800
Expire_time_day = 86400
Expire_time_hour = 3600
Expire_time_minute = 60


def _get_redis_pool(**kwargs):
    redis_url = AppConfig.get()['redis_url']

    def create_redis():
        from redis import BlockingConnectionPool
        redis_pool = BlockingConnectionPool.from_url(redis_url, **kwargs)
        return redis_pool

    return Instance.getobj(f'redis', create_redis)


def get_native_redis(decode_rsp=True, connect_timeout=10):
    return StrictRedis(connection_pool=_get_redis_pool(decode_responses=decode_rsp),
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


class RedisLock:
    def __init__(self, redis: 'MyRedis', lock_name: str, acquire_timeout=3, lock_timeout=2, force=False):
        self.myredis = redis
        self.lock_name = lock_name
        self.acquire_timeout = acquire_timeout
        self.lock_timeout = lock_timeout
        self.force = force
        self.lock_val = None

    def has_lock(self):
        return self.lock_val

    def __enter__(self):
        import uuid, time
        identifier = str(uuid.uuid4())
        lockname = f'lock:{self.lock_name}'
        end = time.time() + self.acquire_timeout

        while time.time() < end:
            # 如果不存在这个锁则加锁并设置过期时间，避免死锁
            if self.myredis.redis.set(lockname, identifier, ex=self.lock_timeout, nx=True):
                self.lock_val = identifier
                return self
            time.sleep(0.01)

        if self.force and self.myredis.redis.set(lockname, identifier, ex=self.lock_timeout):
            self.lock_val = identifier
            return self

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ret_val = 0
        if self.lock_val:
            unlock_script = """
                    if redis.call("get",KEYS[1]) == ARGV[1] then
                        return redis.call("del",KEYS[1])
                    else
                        return 0
                    end
                    """
            lockname = f'lock:{self.lock_name}'
            unlock = self.myredis.redis.register_script(unlock_script)
            ret_val = unlock(keys=[lockname], args=[self.lock_val])
            self.lock_val = None
        return ret_val


def_serializer = build_serializer()
def_deserializer = build_deserializer()


class MyRedis:
    def __init__(self, serializer=None, deserializer=None, connect_timeout=10):
        self.redis = get_native_redis(connect_timeout=connect_timeout)
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

    def lock(self, lock_name, acquire_timeout=3, lock_timeout=2, force=False) -> RedisLock:
        '''
        获取基于 Redis 实现的分布式锁对象。针对返回对象应使用with加锁关锁
        :param lock_name: 锁的名称
        :param acquire_timeout: 获取锁的超时时间，默认 3 秒
        :param lock_timeout: 锁的超时时间，默认 2 秒
        :param force: 超时未获取到锁是否强制加锁
        :return:
        '''
        return RedisLock(self, lock_name, acquire_timeout, lock_timeout, force)

    def subscribe(self, **kwargs):
        p = self.redis.pubsub()
        p.subscribe(**kwargs)


def get_redis(serializer=None, deserializer=None, connect_timeout=10) -> MyRedis:
    return MyRedis(serializer=serializer, deserializer=deserializer, connect_timeout=connect_timeout)
