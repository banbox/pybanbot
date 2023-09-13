#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : notify_mgr.py
# Author: anyongjin
# Date  : 2023/4/1
import asyncio

from banbot.rpc.webhook import *
from banbot.util import btime
from banbot.util.common import Singleton


class Notify(metaclass=Singleton):
    instance: 'Notify' = None

    def __init__(self, config: Config):
        from banbot.util.misc import new_async_thread
        Notify.instance = self
        self.config = config
        self.channels: List[Webhook] = []
        self.name = config.get('name', '')
        self._loop = new_async_thread()[0]
        self._init_channels()

    def _init_channels(self):
        chl_items = self.config.get('rpc_channels') or dict()
        for key, item in chl_items.items():
            if not item.get('enabled', False):
                continue
            chl_type = item.get('type')
            try:
                ChlClass = None
                if chl_type == 'wework':
                    from banbot.rpc.wework import WeWork as ChlClass
                elif chl_type == 'telegram':
                    from banbot.rpc.telegram_ import Telegram as ChlClass
                elif chl_type == 'line':
                    from banbot.rpc.line_ import Line as ChlClass
                if ChlClass is None:
                    logger.error(f'nosupport rpc channel type: {chl_type} for {key}')
                else:
                    item['name'] = key
                    chl = ChlClass(self.config, item)
                    # 单独线程运行队列消费任务
                    asyncio.run_coroutine_threadsafe(chl.consume_forever(), self._loop)
                    self.channels.append(chl)
            except Exception:
                logger.exception(f'init rpc.{key}:{chl_type} fail')

    def send_msg(self, **msg) -> None:
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
                mod.send_msg(msg)
            except NotImplementedError:
                logger.error("Message type '%s' not implemented by handler %s.", msg['type'], mod.name)
            except Exception:
                logger.exception('Exception occurred within RPC module %s: %s', mod.name, msg)

    def _startup_msg(self):
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
        self.send_msg(
            type=NotifyType.STARTUP,
            status=f'{exg_name}.{market}\n模式: {run_mode}\n单笔金额: {stake_amount}{leverage}\n{pairs}'
        )

    @classmethod
    def _init_obj(cls):
        if not cls.instance:
            from banbot.config import AppConfig
            Notify(AppConfig.get())
        return True

    @classmethod
    def startup_msg(cls):
        cls._init_obj()
        cls.instance._startup_msg()

    @classmethod
    def send(cls, **msg):
        cls._init_obj()
        cls.instance.send_msg(**msg)

    @classmethod
    async def cleanup(cls) -> None:
        """ Stops all enabled rpc modules """
        if not cls.instance:
            return
        logger.info('Cleaning up rpc modules ...')
        while cls.instance.channels:
            mod = cls.instance.channels.pop()
            logger.info('Cleaning up rpc.%s ...', mod.name)
            await mod.cleanup()
            del mod
