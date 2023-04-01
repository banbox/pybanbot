#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : __init__.py.py
# Author: anyongjin
# Date  : 2023/3/22
import os
import orjson
from typing import Tuple
from banbot.config.configuration import Configuration, Config


def get_bnb_client_args() -> Tuple[dict, dict, dict]:
    cfg_path = os.path.join(os.path.dirname(__file__), 'config.json')
    cfg: dict = orjson.loads(open(cfg_path, 'rb').read())
    is_prod = cfg["env"] == 'prod'
    client_cfg = cfg['exchange'][f'credit_{cfg["env"]}']
    stream_args = dict()
    if client_cfg.get('stream_url'):
        stream_args['stream_url'] = client_cfg['stream_url']
    auth_args = dict(api_key=client_cfg['api_key'], api_secret=client_cfg['api_secret'])
    rest_args = dict(show_limit_usage=True, show_header=True)
    if client_cfg.get('base_url'):
        rest_args['base_url'] = client_cfg['base_url']
    elif is_prod:
        prod_hosts = [
            'https://api1.binance.com',
            'https://api2.binance.com',
            'https://api3.binance.com',
        ]
        import random
        rest_args['base_url'] = prod_hosts[random.randrange(0, len(prod_hosts))]
    return auth_args, rest_args, stream_args
