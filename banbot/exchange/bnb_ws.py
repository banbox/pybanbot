#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : bnb_ws.py
# Author: anyongjin
# Date  : 2023/3/30
'''
需要binance-connector 3.0.0rc1以上版本
'''
# from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient as WsStream
# from binance.websocket.spot.websocket_api import SpotWebsocketAPIClient as WsApi
# from banbot.exchange.ws_base import WebsocketClient
# from banbot.util.misc import utime
# from typing import *
#
#
# def get_bnb_client_args(cfg: dict) -> Tuple[dict, dict, dict]:
#     exg_cfg = cfg['exchange']
#     assert exg_cfg['name'].find('binance') >= 0, 'exchange is not binance'
#     is_prod = cfg["env"] == 'prod'
#     client_cfg = exg_cfg[f'credit_{cfg["env"]}']
#     stream_args = dict()
#     if client_cfg.get('stream_url'):
#         stream_args['stream_url'] = client_cfg['stream_url']
#     auth_args = dict(api_key=client_cfg['api_key'], api_secret=client_cfg['api_secret'])
#     rest_args = dict(show_limit_usage=True, show_header=True)
#     if client_cfg.get('base_url'):
#         rest_args['base_url'] = client_cfg['base_url']
#     elif is_prod:
#         prod_hosts = [
#             'https://api1.binance.com',
#             'https://api2.binance.com',
#             'https://api3.binance.com',
#         ]
#         import random
#         rest_args['base_url'] = prod_hosts[random.randrange(0, len(prod_hosts))]
#     return auth_args, rest_args, stream_args
#
#
# class BinanceWS(WebsocketClient):
#     def __init__(self, config: dict):
#         auth_args, rest_args, stream_args = get_bnb_client_args(config)
#         stream_args.update(dict(
#             on_message=self.on_stream_msg,
#             on_error=self.on_error
#         ))
#         self.stream = WsStream(**stream_args)
#         stream_args['on_message'] = self.on_api_msg
#         self.api = WsApi(**auth_args, **stream_args)
#         self.listen_key = None
#         self.listen_key_update = utime(-3600)
#
#     def get_listen_key(self):
#         cur_time = utime(60)
#         if not self.listen_key or self.listen_key_update < cur_time:
#             self.listen_key = self.rest_api.new_listen_key()['data']['listenKey']
#             self.listen_key_update = utime()
#         elif self.listen_key_update < cur_time + 300000:
#             logger.warning('listen key is about expired, renew...')
#             self.rest_api.renew_listen_key(self.listen_key)
#         return self.listen_key
#
#     def on_stream_msg(self, msg_text: str):
#         pass
#
#     def on_api_msg(self, msg: dict):
#         pass
#
#     def on_error(self, err):
#         pass
#
