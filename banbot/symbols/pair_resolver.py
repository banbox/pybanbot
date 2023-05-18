#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : pair_resolver.py
# Author: anyongjin
# Date  : 2023/4/17
from banbot.core.iresolver import *
from banbot.symbols.pairlist.base import *


class PairResolver(IResolver):
    object_type = PairList
    object_type_str = 'PairList'
    initial_search_path = Path(__file__).parent.joinpath('pairlist').resolve()

    @classmethod
    def load_handlers(cls, exchange, manager, config: Config) -> List[PairList]:
        cls_list = cls.load_object_list(config)
        cls_map = {item.__name__: item for item in cls_list}
        pairlist = config.get('pairlists', [])
        pair_handlers = []
        for handler_cfg in pairlist:
            pair_cls = cls_map.get(handler_cfg['name'])
            if not pair_cls:
                raise RuntimeError(f"unknown pair handler: {handler_cfg['name']}")
            pair_handlers.append(pair_cls(manager, exchange, config, handler_cfg))
        return pair_handlers
