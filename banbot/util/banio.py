#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : banio.py
# Author: anyongjin
# Date  : 2023/9/20
import asyncio
import marshal
import random
import contextlib
from typing import Optional, Any, List, Tuple, Dict, Callable, ClassVar
from banbot.util.common import logger
from asyncio import Future
from banbot.util import btime
from banbot.storage import BotGlobal


line_end = b'<\0>'


class BanConn:
    '''
    socket连接，用于Ban不同进程之间通信。
    [服务器端]可从此类继承，实现成员函数处理不同的消息。
    '''
    def __init__(self, reader: Optional[asyncio.StreamReader] = None,
                 writer: Optional[asyncio.StreamWriter] = None, reconnect: bool = True):
        self.reader: Optional[asyncio.StreamReader] = reader
        self.writer: Optional[asyncio.StreamWriter] = writer
        self.tags = set()
        '此连接的标签：用于服务器端订阅列表'
        self.remote = None
        if writer:
            self.remote = writer.get_extra_info('peername')
        self.listens: Dict[str, Callable] = dict()
        self.reconnect = reconnect

    async def connect(self):
        raise NotImplementedError

    async def write_msg(self, msg_type: str, data: Any):
        name = self.remote
        logger.debug('%s write: %s %s', name, msg_type, data)
        dump_data = marshal.dumps((msg_type, data))
        while True:
            try:
                return await self.write(dump_data)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
                err_type = type(e).__name__
                if not self.reconnect:
                    logger.error(f'write {name} {err_type}, skip..')
                    return
                logger.error(f'write {name} {err_type}, sleep 3s and retry...')
                await asyncio.sleep(3)

    async def write(self, data: bytes):
        if not self.writer:
            await self.connect()
        self.writer.write(data)
        self.writer.write(line_end)
        await self.writer.drain()

    async def read(self) -> bytes:
        if not self.reader:
            await self.connect()
        data = await self.reader.readuntil(line_end)
        return data[:-len(line_end)]

    async def read_msg(self) -> Tuple[str, Any]:
        if not self.reader:
            await self.connect()
        data = await self.reader.readuntil(line_end)
        msg_type, msg_data = marshal.loads(data[:-len(line_end)])
        return msg_type, msg_data

    def subscribe(self, data):
        if isinstance(data, (list, tuple, set)):
            self.tags.update(data)
        else:
            self.tags.add(data)

    def unsubscribe(self, data):
        if isinstance(data, (list, tuple, set)):
            self.tags.difference_update(data)
        else:
            if data in self.tags:
                self.tags.remove(data)

    async def close(self):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
            self.writer = None
            self.reader = None

    def get_handle_func(self, action: str):
        if hasattr(self, action):
            return True, getattr(self, action)
        for key, func in self.listens.items():
            if action.startswith(key):
                return key == action, func

    async def run_forever(self, name: str):
        '''
        监听连接发送的信息并处理。
        根据消息的action：
            调用对应成员函数处理；直接传入msg_data
            或从listens中找对应的处理函数，如果精确匹配，传入msg_data，否则传入action, msg_data
        服务器端和客户端都会调用此方法
        '''
        from banbot.util.misc import run_async
        try:
            if not self.reader:
                await self.connect()
            while True:
                data = await self.reader.readuntil(line_end)
                data = data[:-len(line_end)]
                # logger.debug(f'%s receive %s', name, data)
                des_data = marshal.loads(data)
                if not des_data or not hasattr(des_data, '__len__') or len(des_data) != 2:
                    logger.warning(f'{name} invalid msg: {data}')
                    continue
                action, act_data = des_data
                if not action:
                    logger.warning(f'{name} invalid msg: {data}')
                    continue
                handle_res = self.get_handle_func(action)
                if not handle_res:
                    logger.info(f'{name} unhandle msg: {action}, {act_data}')
                    continue
                is_exact, handle_fn = handle_res
                call_args = [act_data] if is_exact else [action, act_data]
                try:
                    await run_async(handle_fn, *call_args)
                except Exception:
                    logger.exception(f'{name} handle msg err: {self.remote}: {data}, {call_args}')
        except (asyncio.IncompleteReadError, ConnectionResetError) as e:
            self.reader = None
            self.writer = None
            err_type = type(e).__name__
            if not self.reconnect:
                # 不尝试重新连接，退出
                logger.error(f'remote {name} {err_type}, remove conn')
                return
            logger.error(f'read {name} {err_type}, sleep 3s and retry...')
            await asyncio.sleep(3)
            await self.run_forever(name)
        except Exception:
            logger.exception(f'{name} handle remote msg error')


class ServerIO:
    '''
    Socket服务器端，监听端口，接受客户端连接，处理消息并发送响应。
    '''
    obj: ClassVar['ServerIO'] = None

    def __init__(self, addr: str, name: str):
        host, port = addr.split(':')
        self.host = host
        self.port = int(port)
        self.name = name or 'banserver'
        self.server: Optional[asyncio.Server] = None
        self.conns: List[BanConn] = []
        self.data: Dict[str, Any] = dict()
        '内存缓存的数据，可供远程端访问设置'
        ServerIO.obj = self

    async def run_forever(self):
        self.server = await asyncio.start_server(self._handle_conn, self.host, self.port)
        logger.info(f'{self.name} serving on {self.host}:{self.port}')

        async with self.server:
            await self.server.serve_forever()

    async def _handle_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        conn = self.get_conn(reader, writer)
        logger.info(f'receive client: {conn.remote}')
        self._wrap_handlers(conn)
        writer.write('ready'.encode())
        writer.write(line_end)
        await writer.drain()
        self.conns.append(conn)
        asyncio.create_task(conn.run_forever(self.name))

    async def broadcast(self, msg_type: str, data: Any):
        dump_data = marshal.dumps((msg_type, data))
        fail_conns = set()
        for conn in self.conns:
            if not conn.writer:
                # 连接已关闭
                fail_conns.add(conn)
                continue
            if msg_type not in conn.tags:
                # logger.info(f'{conn.remote} skip msg: {msg_type}, {conn.tags}')
                continue
            try:
                await conn.write(dump_data)
            except (BrokenPipeError, ConnectionResetError):
                # 连接已断开
                fail_conns.add(conn)
                logger.info(f'conn {conn.remote} disconnected')
            except Exception as e:
                fail_conns.add(conn)
                logger.exception(f'send msg to client fail: {msg_type} {data}: {type(e)}')
        for conn in fail_conns:
            if conn in self.conns:
                self.conns.remove(conn)

    def get_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> BanConn:
        '''
        如果需要自定义逻辑处理消息请求，请重写此方法，替换BanConn的get_handle_func方法
        '''
        return BanConn(reader, writer, reconnect=False)

    def get_val(self, key: str):
        """从服务器端直接获取缓存的值"""
        from banbot.util import btime
        cache_val = self.data.get(key)
        val = None
        if cache_val:
            val, exp_at = cache_val
            if exp_at and exp_at <= btime.utctime():
                val = None
        return val

    def set_val(self, key: str, val, expire_secs: int = None):
        """从服务器端直接设置缓存的值"""
        if val is None:
            # 删除值
            if key in self.data:
                del self.data[key]
            return
        expire_at = None
        if expire_secs:
            from banbot.util import btime
            expire_at = btime.utctime() + expire_secs
        self.data[key] = (val, expire_at)

    def _wrap_handlers(self, conn: BanConn):
        async def _on_get_val(key: str):
            """处理远程端请求数据，读取数据并返回给远程端"""
            logger.info(f'_on_get_val: {key}')
            val = self.get_val(key)
            await conn.write_msg('_on_get_val_res', (key, val))

        def _on_set_val(data):
            logger.info(f'_on_set_val: {data}')
            key, val, expire_secs = data
            self.set_val(key, val, expire_secs)

        conn.listens.update(
            _on_get_val=_on_get_val,
            _on_set_val=_on_set_val
        )


class ClientIO(BanConn):
    '''
    socket客户端，用于连接服务器，发送请求处理并得到响应。也可用于获取服务器缓存数据。
    客户端可从此类继承，实现成员函数，处理服务器主动发送的消息。
    '''
    _obj: ClassVar['ClientIO'] = None

    def __init__(self, server_addr: str):
        super().__init__()
        host, port = server_addr.split(':')
        self.host = host
        self.port = int(port)
        self._waits: Dict[str, Future] = dict()
        '等待触发的异步对象'
        if ClientIO._obj:
            logger.error(f'ClientIO already created: {ClientIO._obj}')
        ClientIO._obj = self

    async def connect(self):
        logger.debug('connecting: %s:%s', self.host, self.port)
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        logger.debug('connected: %s:%s', self.host, self.port)
        self.remote = self.writer.get_extra_info('peername')
        await self.read()
        '收到服务器第一个消息后认为服务器就绪，不管消息是什么'

    async def get_val(self, key: str):
        """从远程端获取指定key的值。"""
        fut = asyncio.get_running_loop().create_future()
        wait_key = f'get_{key}'
        self._waits[wait_key] = fut
        await self.write_msg('_on_get_val', key)
        try:
            return await fut
        finally:
            del self._waits[wait_key]

    def _on_get_val_res(self, data):
        """处理远程端取得数据后回调"""
        key, val = data
        fut = self._waits.get(f'get_{key}')
        if not fut:
            return
        fut.set_result(val)

    async def set_val(self, key: str, val, expire_secs: int = None):
        """保存指定的值到远程端，过期时间可选，不等待返回确认"""
        await self.write_msg('_on_set_val', (key, val, expire_secs))

    @classmethod
    async def get_remote(cls, key: str, catch_err=False):
        """获取服务器端缓存的数据（服务器端也可调用此方法）"""
        if ServerIO.obj:
            return ServerIO.obj.get_val(key)
        if not cls._obj:
            if catch_err:
                return None
            raise ValueError(f'remote not ready, get: {key}')
        return await cls._obj.get_val(key)

    @classmethod
    async def set_remote(cls, key: str, val, expire_secs: int = None, catch_err=False):
        """设置数据缓存到服务器端（服务器端也可调用此方法）"""
        if ServerIO.obj:
            return ServerIO.obj.set_val(key, val, expire_secs)
        if not cls._obj:
            if catch_err:
                return None
            raise ValueError(f'remote not ready, get: {key}')
        await cls._obj.set_val(key, val, expire_secs)

    @classmethod
    async def get_lock(cls, key: str, timeout: int = None) -> int:
        """设置分布式锁，可指定超时时间，不指定默认20分钟"""
        start = btime.time()
        lock_by = await cls.get_remote(key)
        lock_val = random.randrange(1000, 10000000)
        if not lock_by:
            await cls.set_remote(key, lock_val)
            return lock_val
        if not timeout:
            timeout = 1200  # 最大超时时间20分钟
        while btime.time() < start + timeout:
            await asyncio.sleep(0.01)
            lock_by = await cls.get_remote(key)
            if not lock_by:
                await cls.set_remote(key, lock_val)
                return lock_val
        raise TimeoutError(f'wait lock timeout: {key}, cost: {(btime.time() - start):.2f} secs')

    @classmethod
    async def del_lock(cls, key: str, lock_val: int = None):
        """删除分布式锁，提供lock_val时对比是否相同，相同才删除"""
        if lock_val:
            lock_by = await cls.get_remote(key)
            if lock_by != lock_val:
                return
        await cls.set_remote(key, None)

    @classmethod
    @contextlib.asynccontextmanager
    async def lock(cls, key: str, timeout: int = None):
        """请求一个分布式锁；此方法必须用with调用"""
        if not BotGlobal.live_mode:
            yield
            return
        lock_val = await cls.get_lock(key, timeout)
        try:
            yield
        finally:
            await cls.del_lock(key, lock_val)
