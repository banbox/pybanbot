#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : rpc_manager.py
# Author: anyongjin
# Date  : 2023/4/1
from collections import deque

from banbot.rpc.rpc import *
from banbot.util import btime
from banbot.util.common import Singleton


class RPCManager(metaclass=Singleton):
    instance: 'RPCManager' = None

    def __init__(self, config: Config):
        RPCManager.instance = self
        self.config = config
        self._rpc = RPC(config)
        self.channels: List[RPCHandler] = []
        self.name = config.get('name', '')

        if config.get('wework', {}).get('enabled', False):
            logger.info('start rpc.wework ...')
            try:
                from banbot.rpc.wework import WeWork
                self.channels.append(WeWork(self._rpc, config))
            except Exception:
                logger.exception('init wechat corp fail')

    async def cleanup(self) -> None:
        """ Stops all enabled rpc modules """
        logger.info('Cleaning up rpc modules ...')
        while self.channels:
            mod = self.channels.pop()
            logger.info('Cleaning up rpc.%s ...', mod.name)
            await mod.cleanup()
            del mod

    async def send_msg(self, msg: Dict[str, Any]) -> None:
        """
        Send given message to all registered rpc modules.
        A message consists of one or more key value pairs of strings.
        e.g.:
        {
            'status': 'stopping bot'
        }
        """
        msg['name'] = self.name
        for mod in self.channels:
            logger.debug('Forwarding message to rpc.%s', mod.name)
            try:
                await mod.send_msg(msg)
            except NotImplementedError:
                logger.error("Message type '%s' not implemented by handler %s.", msg['type'], mod.name)
            except Exception:
                logger.exception('Exception occurred within RPC module %s: %s', mod.name, msg)

    async def process_msg_queue(self, queue: deque) -> None:
        """
        Process all messages in the queue.
        """
        while queue:
            msg = queue.popleft()
            logger.info('Sending rpc strategy_msg: %s', msg)
            for mod in self.channels:
                await mod.send_msg({
                    'type': RPCMessageType.STRATEGY_MSG,
                    'msg': msg,
                })

    async def startup_messages(self):
        exg_name = self.config['exchange']['name']
        run_mode = btime.run_mode.value
        stake_amount = self.config['stake_amount']
        await self.send_msg({
            'type': RPCMessageType.STARTUP,
            'status': f'Exg: {exg_name}\nMode: {run_mode}\nStake Amount: {stake_amount}'
        })
