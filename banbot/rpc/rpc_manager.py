#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : rpc_manager.py
# Author: anyongjin
# Date  : 2023/4/1
from banbot.rpc.rpc import *
from collections import deque
from banbot.util import btime


class RPCManager:
    def __init__(self, bot):
        self.config = bot.config
        self._rpc = RPC(bot)
        self.channels: List[RPCHandler] = []
        config = bot.config

        if config.get('wework', {}).get('enabled', False):
            logger.info('start rpc.wework ...')
            from banbot.rpc.wework import WeWork
            self.channels.append(WeWork(self._rpc, config))

    def cleanup(self) -> None:
        """ Stops all enabled rpc modules """
        logger.info('Cleaning up rpc modules ...')
        while self.channels:
            mod = self.channels.pop()
            logger.info('Cleaning up rpc.%s ...', mod.name)
            mod.cleanup()
            del mod

    def send_msg(self, msg: Dict[str, Any]) -> None:
        """
        Send given message to all registered rpc modules.
        A message consists of one or more key value pairs of strings.
        e.g.:
        {
            'status': 'stopping bot'
        }
        """
        for mod in self.channels:
            logger.debug('Forwarding message to rpc.%s', mod.name)
            try:
                mod.send_msg(msg)
            except NotImplementedError:
                logger.error(f"Message type '{msg['type']}' not implemented by handler {mod.name}.")
            except Exception:
                logger.exception('Exception occurred within RPC module %s', mod.name)

    def process_msg_queue(self, queue: deque) -> None:
        """
        Process all messages in the queue.
        """
        while queue:
            msg = queue.popleft()
            logger.info('Sending rpc strategy_msg: %s', msg)
            for mod in self.channels:
                mod.send_msg({
                    'type': RPCMessageType.STRATEGY_MSG,
                    'msg': msg,
                })

    def startup_messages(self):
        exg_name = self.config['exchange']['name']
        run_mode = btime.run_mode
        stake_amount = self.config['stake_amount']
        self.send_msg({
            'type': RPCMessageType.STARTUP,
            'status': f'Exchange: {exg_name}\nRun Mode:{run_mode}\nStake Amount: {stake_amount}'
        })
