#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : webhook.py
# Author: anyongjin
# Date  : 2023/9/7
import asyncio
from banbot.config import Config
from banbot.storage.common import *
from banbot.util.common import *


class NotifyType:
    STATUS = 'status'
    WARNING = 'warning'
    EXCEPTION = 'exception'
    STARTUP = 'startup'

    ENTRY = 'entry'
    ENTRY_FILL = 'entry_fill'
    ENTRY_CANCEL = 'entry_cancel'

    EXIT = 'exit'
    EXIT_FILL = 'exit_fill'
    EXIT_CANCEL = 'exit_cancel'

    PROTECTION_TRIGGER = 'protection_trigger'
    PROTECTION_TRIGGER_GLOBAL = 'protection_trigger_global'

    STRATEGY_MSG = 'strategy_msg'

    WHITELIST = 'whitelist'
    NEW_CANDLE = 'new_candle'

    MARKET_TIP = 'market_tip'


def map_msg_type(msg_type: str):
    if msg_type in (NotifyType.STATUS, NotifyType.STARTUP):
        return 'status'
    return msg_type


class Webhook:
    """
    消息通知基类
    """
    batch_size = 1  # 每次批量处理消息的数量，对于一次能发送多个消息的渠道可取多个
    next_send_ts = 0  # 对于触发429的渠道，设置下次发送时间，休眠重试

    def __init__(self, config: Config, item: dict) -> None:
        """
        Init the Webhook class, and init the super class RPCHandler
        :param config: Configuration object
        :return: None
        """
        self._name = item['name']
        self._config = config
        self._params = item

        self._url = self._config['webhook'].get('url')
        self._retries = self._config['webhook'].get('retries', 0)
        self._retry_delay = self._config['webhook'].get('retry_delay', 0.1)
        self.keywords = item.get('keywords' ,[])
        if self.keywords:
            self.keywords = [w for w in self.keywords if w]

        self.queue = asyncio.Queue(500)
        self._alive = True

        self.msg_types = set(item.get('msg_types') or [])
        if not self.msg_types:
            logger.error(f'no msg type configured for {item}')

    @property
    def name(self) -> str:
        """ Returns the lowercase name of the implementation """
        return self.__class__.__name__.lower() + '.' + self._name

    async def cleanup(self) -> None:
        """
        Cleanup pending module resources.
        This will do nothing for webhooks, they will simply not be called anymore
        """
        self._alive = False

    def send_msg(self, msg: Dict[str, Any]) -> None:
        """ Send a message to telegram channel """
        try:
            msg_type = msg['type']
            valuedict = self._config['webhook'].get(msg_type)

            if not valuedict:
                logger.info("Message type '%s' not configured for webhooks", msg['type'])
                return

            payload = {key: value.format(**msg) for (key, value) in valuedict.items()}
            logger.debug(f'push rpc msg %s: %s', self.name, payload)

            content: str = payload.get('content')
            if self.keywords and content:
                if not any(w for w in self.keywords if content.find(w) >= 0):
                    return

            self.queue.put_nowait(payload)
        except KeyError as exc:
            logger.exception("Problem calling Webhook. Please check your webhook configuration. "
                             "Exception: %s", exc)

    async def consume_forever(self):
        '''
        消费RPC消息队列。每个渠道单独一个队列。所有队列的消费应在单独一个线程的事件循环中，避免影响主线程的执行。
        '''
        cur_name = self.name
        logger.debug('start consume rpc for %s', cur_name)
        while True:
            try:
                msg_list = []
                wait_secs = self.next_send_ts - time.time()
                if wait_secs > 0:
                    await asyncio.sleep(wait_secs)
                while len(msg_list) < self.batch_size:
                    try:
                        msg_list.append(self.queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                if not msg_list:
                    if not self._alive:
                        logger.info(f'stop consume {cur_name}')
                        break
                    await asyncio.sleep(0.1)
                    continue
                ok_num = await self._send_msg(msg_list)
                logger.debug('send %s done: %s', cur_name, msg_list)
                [self.queue.task_done() for i in range(ok_num)]
            except Exception:
                logger.exception(f'consume rpc {cur_name} error')

    async def _do_send_msg(self, msg_list: List[dict]) -> int:
        '''
        执行发送消息。不带重试。必须按顺序发送，如前面发送失败，后面的不应尝试发送
        '''
        raise NotImplementedError('_do_send_msg not implemented')

    async def _send_msg(self, msg_list: List[dict]) -> int:
        '''
        调用api发送消息。失败则按配置重试
        '''
        sent_num, attempts, total_num = 0, 0, len(msg_list)
        while sent_num < total_num and msg_list and attempts <= self._retries:
            if attempts:
                if self._retry_delay:
                    await asyncio.sleep(self._retry_delay)
                logger.info("Retrying webhook...")

            attempts += 1
            try:
                cur_sent = await self._do_send_msg(msg_list)
                if not isinstance(cur_sent, int):
                    logger.error(f'{self.__class__.__name__}._do_send_msg should return int!')
                    return 0
                sent_num += cur_sent
                msg_list = msg_list[cur_sent:]  # 去除已发送的
            except Exception as exc:
                logger.exception("Could not call webhook url. Exception: %s", exc)
        return sent_num
