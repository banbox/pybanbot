#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : resolver.py
# Author: anyongjin
# Date  : 2023/3/31
from banbot.strategy.base import BaseStrategy
from banbot.core.iresolver import *
from banbot.util.common import logger


class StrategyResolver(IResolver):

    object_type: Type[Any] = BaseStrategy
    object_type_str: str = 'BaseStrategy'
    user_subdir = 'strategies'
    initial_search_path = Path(__file__).parent.resolve()

    @classmethod
    def load_run_jobs(cls, config: dict, pairlist: List[str]) -> List[Tuple[str, str, List[Type[BaseStrategy]]]]:
        strategy_list = cls.load_object_list(config)
        strategy_map = {item.__name__: item for item in strategy_list}
        logger.info('found strategy: %s', list(strategy_map.keys()))
        result = dict()
        timeframe = '1m'  # 默认周期1m，后期根据K线和策略自动计算
        for policy in config['run_policy']:
            strategy_cls = strategy_map.get(policy['name'])
            if not strategy_cls:
                raise RuntimeError(f'unknown Strategy: {policy["name"]}')
            stg_pairs = set()
            for pair in pairlist:
                if pair in stg_pairs:
                    raise ValueError('one strategy can only handle one timeframe for pair')
                stg_pairs.add(pair)
                key = f'{pair}_{timeframe}'
                if key not in result:
                    result[key] = []
                result[key].append(strategy_cls)
        result_list = []
        for key, slist in result.items():
            pair, timeframe = key.split('_')
            result_list.append((pair, timeframe, slist))
        return result_list
