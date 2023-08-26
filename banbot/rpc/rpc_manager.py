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
        chl_items = config.get('rpc_channels') or dict()
        for key, item in chl_items.items():
            if not item.get('enabled', False):
                continue
            chl_type = item.get('type')
            try:
                if chl_type == 'wework':
                    from banbot.rpc.wework import WeWork
                    self.channels.append(WeWork(self._rpc, config, item))
                else:
                    logger.error(f'nosupport rpc channel type: {chl_type} for {key}')
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
        msg_type = map_msg_type(msg['type'])
        msg['type'] = msg_type
        for mod in self.channels:
            if msg_type not in mod.msg_types:
                continue
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
            await self.send_msg({
                'type': RPCMessageType.STRATEGY_MSG,
                'msg': msg,
            })

    async def startup_messages(self):
        from banbot.symbols import group_symbols
        exg_name = self.config['exchange']['name']
        market = self.config['market_type']
        run_mode = btime.run_mode.value
        stake_amount = self.config['stake_amount']
        leverage = ''
        if market == 'future':
            leverage = f'\n杠杆：{self.config["leverage"]}'
        groups = group_symbols(BotGlobal.pairs)
        if len(groups) == 1:
            pairs = f"币种：{', '.join(list(groups.values())[0])}"
        else:
            pairs = '币种：'
            for key, items in groups.items():
                pairs += f"\n{key}: {', '.join(items)}"
        await self.send_msg({
            'type': RPCMessageType.STARTUP,
            'status': f'{exg_name}.{market}\n模式: {run_mode}\n单笔金额: {stake_amount}{leverage}\n{pairs}'
        })
