#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : rpc.py
# Author: anyongjin
# Date  : 2023/4/1
from abc import abstractmethod
from enum import Enum
from typing import *

from banbot.config import Config
from banbot.storage.common import *
from banbot.util.common import *


class RPCException(Exception):

    def __init__(self, message: str) -> None:
        super().__init__(self)
        self.message = message

    def __str__(self):
        return self.message

    def __json__(self):
        return {
            'msg': self.message
        }


class RPCHandler:

    def __init__(self, rpc: 'RPC', config: Config, item: dict) -> None:
        """
        Initializes RPCHandlers
        :param rpc: instance of RPC Helper class
        :param config: Configuration object
        :return: None
        """
        self._rpc = rpc
        self._config: Config = config
        self._params = item
        self.msg_types = set(item.get('msg_types') or [])
        if not self.msg_types:
            logger.error(f'no msg type configured for {item}')

    @property
    def name(self) -> str:
        """ Returns the lowercase name of the implementation """
        return self.__class__.__name__.lower()

    @abstractmethod
    async def cleanup(self) -> None:
        """ Cleanup pending module resources """

    @abstractmethod
    async def send_msg(self, msg: Dict[str, str]) -> None:
        """ Sends a message to all registered rpc modules """


class RPC:

    def __init__(self, config: Config):
        self._config = config

    def _rpc_start(self) -> Dict[str, str]:
        """ Handler for start """
        if BotGlobal.state == BotState.RUNNING:
            return {'status': 'already running'}

        BotGlobal.state = BotState.RUNNING
        return {'status': 'starting trader ...'}

    def _rpc_stop(self) -> Dict[str, str]:
        """ Handler for stop """
        if BotGlobal.state == BotState.RUNNING:
            BotGlobal.state = BotState.STOPPED
            return {'status': 'stopping trader ...'}

        return {'status': 'already stopped'}


class RPCMessageType:
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
    if msg_type in (RPCMessageType.STATUS, RPCMessageType.STARTUP):
        return 'status'
    return msg_type


class Webhook(RPCHandler):
    """  This class handles all webhook communication """

    def __init__(self, rpc: RPC, config: Config, item: dict) -> None:
        """
        Init the Webhook class, and init the super class RPCHandler
        :param rpc: instance of RPC Helper class
        :param config: Configuration object
        :return: None
        """
        super().__init__(rpc, config, item)

        self._url = self._config['webhook'].get('url')
        self._retries = self._config['webhook'].get('retries', 0)
        self._retry_delay = self._config['webhook'].get('retry_delay', 0.1)

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
