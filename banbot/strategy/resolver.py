#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : resolver.py
# Author: anyongjin
# Date  : 2023/3/31
from banbot.strategy.base import BaseStrategy
from banbot.core.iresolver import *
from banbot.util.common import logger
from banbot.exchange.exchange_utils import timeframe_to_seconds


class PTFJob:
    pairs: Dict[str, Dict[str, Set[Type[BaseStrategy]]]] = dict()
    pair_stg: Set[Tuple[str, str]] = set()
    pair_warms: Dict[str, int] = dict()

    @classmethod
    def add(cls, pair: str, timeframe: str, stg: Type[BaseStrategy]):
        if (pair, stg.__name__) in cls.pair_stg:
            raise RuntimeError(f'one strategy {stg.__name__} can only handle one timeframe for pair {pair}')
        cls.pair_stg.add((pair, stg.__name__))
        if pair not in cls.pairs:
            cls.pairs[pair] = dict()
        tf_dic = cls.pairs[pair]
        if timeframe not in tf_dic:
            tf_dic[timeframe] = set()
        stg_set = tf_dic[timeframe]
        if stg in stg_set:
            return
        if stg.warmup_num:
            if pair not in cls.pair_warms:
                cls.pair_warms[pair] = 0
            tf_secs = timeframe_to_seconds(timeframe)
            cls.pair_warms[pair] = max(cls.pair_warms[pair], tf_secs * stg.warmup_num)
        stg_set.add(stg)

    @classmethod
    def reset(cls):
        cls.pairs = dict()
        cls.pair_stg = set()
        cls.pair_warms = dict()

    @classmethod
    def tojobs(cls) -> Dict[str, Tuple[int, Dict[str, Set[Type[BaseStrategy]]]]]:
        results: Dict[str, Tuple[int, Dict[str, Set[Type[BaseStrategy]]]]] = dict()
        for pair, tf_dic in cls.pairs.items():
            results[pair] = cls.pair_warms[pair], tf_dic
        return results


class StrategyResolver(IResolver):

    object_type: Type[Any] = BaseStrategy
    object_type_str: str = 'BaseStrategy'
    user_subdir = 'strategies'
    initial_search_path = Path(__file__).parent.resolve()

    @classmethod
    def load_run_jobs(cls, config: dict, pairlist: List[str])\
            -> Dict[str, Tuple[int, Dict[str, Set[Type[BaseStrategy]]]]]:
        PTFJob.reset()
        strategy_list = cls.load_object_list(config)
        strategy_map = {item.__name__: item for item in strategy_list}
        logger.info('found strategy: %s', list(strategy_map.keys()))
        timeframe = '1m'  # 默认周期1m，后期根据K线和策略自动计算
        for policy in config['run_policy']:
            strategy_cls = strategy_map.get(policy['name'])
            if not strategy_cls:
                raise RuntimeError(f'unknown Strategy: {policy["name"]}')
            for pair in pairlist:
                # TODO: 这里可根据策略，自定义此交易对的交易维度
                PTFJob.add(pair, timeframe, strategy_cls)
        return PTFJob.tojobs()
