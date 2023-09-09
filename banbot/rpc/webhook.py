#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : webhook.py
# Author: anyongjin
# Date  : 2023/9/7
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

    def __init__(self, config: Config, item: dict) -> None:
        """
        Init the Webhook class, and init the super class RPCHandler
        :param rpc: instance of RPC Helper class
        :param config: Configuration object
        :return: None
        """
        self._config = config
        self._params = item

        self._url = self._config['webhook'].get('url')
        self._retries = self._config['webhook'].get('retries', 0)
        self._retry_delay = self._config['webhook'].get('retry_delay', 0.1)

        self.msg_types = set(item.get('msg_types') or [])
        if not self.msg_types:
            logger.error(f'no msg type configured for {item}')

    @property
    def name(self) -> str:
        """ Returns the lowercase name of the implementation """
        return self.__class__.__name__.lower()

    async def cleanup(self) -> None:
        """
        Cleanup pending module resources.
        This will do nothing for webhooks, they will simply not be called anymore
        """
        pass

    async def send_msg(self, msg: Dict[str, Any]) -> None:
        """ Send a message to telegram channel """
        try:
            msg_type = msg['type']
            valuedict = self._config['webhook'].get(msg_type)

            if not valuedict:
                logger.info("Message type '%s' not configured for webhooks", msg['type'])
                return

            payload = {key: value.format(**msg) for (key, value) in valuedict.items()}
            await self._send_msg(payload)
        except KeyError as exc:
            logger.exception("Problem calling Webhook. Please check your webhook configuration. "
                             "Exception: %s", exc)

    async def _do_send_msg(self, payload: dict):
        raise NotImplementedError('_do_send_msg not implemented')

    async def _send_msg(self, payload: dict) -> None:
        """do the actual call to the webhook"""

        success = False
        attempts = 0
        while not success and attempts <= self._retries:
            if attempts:
                if self._retry_delay:
                    await asyncio.sleep(self._retry_delay)
                logger.info("Retrying webhook...")

            attempts += 1

            try:
                await self._do_send_msg(payload)
                success = True

            except Exception as exc:
                logger.warning("Could not call webhook url. Exception: %s", exc)
