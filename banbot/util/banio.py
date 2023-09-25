#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : banio.py
# Author: anyongjin
# Date  : 2023/9/20
import asyncio
import marshal
from typing import Optional, Any, List, Tuple, Dict, Callable
from banbot.util.common import logger


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
        logger.debug('%s write: %s %s', self.remote, msg_type, data)
        dump_data = marshal.dumps((msg_type, data))
        return await self.write(dump_data)

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
        '''
        from banbot.util.misc import run_async
        try:
            if not self.reader:
                await self.connect()
            while True:
                data = await self.reader.readuntil(line_end)
                data = data[:-len(line_end)]
                logger.debug(f'%s receive %s', name, data)
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
                try:
                    call_args = [act_data] if is_exact else [action, act_data]
                    await run_async(handle_fn, *call_args)
                except Exception:
                    logger.exception(f'{name} handle msg err: {self.remote}: {data}')
        except (asyncio.IncompleteReadError, ConnectionResetError) as e:
            self.reader = None
            self.writer = None
            err_type = type(e)
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
    def __init__(self, addr: str, name: str):
        host, port = addr.split(':')
        self.host = host
        self.port = int(port)
        self.name = name or 'banserver'
        self.server: Optional[asyncio.Server] = None
        self.conns: List[BanConn] = []

    async def run_forever(self):
        self.server = await asyncio.start_server(self._handle_conn, self.host, self.port)
        logger.info(f'{self.name} serving on {self.host}:{self.port}')

        async with self.server:
            await self.server.serve_forever()

    async def _handle_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        conn = self.get_conn(reader, writer)
        logger.info(f'receive client: {conn.remote}')
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


class ClientIO(BanConn):
    '''
    socket客户端，用于连接服务器，发送请求处理并得到响应。也可用于获取服务器缓存数据。
    客户端可从此类继承，实现成员函数，处理服务器主动发送的消息。
    '''
    def __init__(self, server_addr: str):
        super().__init__()
        host, port = server_addr.split(':')
        self.host = host
        self.port = int(port)

    async def connect(self):
        logger.debug('connecting: %s:%s', self.host, self.port)
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        logger.debug('connected: %s:%s', self.host, self.port)
        self.remote = self.writer.get_extra_info('peername')
        await self.read()
        '收到服务器第一个消息后认为服务器就绪，不管消息是什么'
