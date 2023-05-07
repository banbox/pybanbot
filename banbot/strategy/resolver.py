#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : resolver.py
# Author: anyongjin
# Date  : 2023/3/31
from banbot.strategy.base import BaseStrategy
from banbot.core.iresolver import *
from banbot.util.common import logger
from banbot.exchange.exchange_utils import tf_to_secs
from banbot.storage.common import BotGlobal
global strategy_map
strategy_map: Optional[Dict[str, BaseStrategy]] = None


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
            tf_secs = tf_to_secs(timeframe)
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

    @classmethod
    def strategy_hash(cls):
        '''
        计算此次机器人涉及的所有策略及其版本的哈希值。
        用于确保策略修改后创建一个新的任务；策略未修改则使用和上次一样的任务
        '''
        import hashlib
        stg_ver = dict()
        for pair, tf_map in cls.pairs.items():
            for tf, stg_list in tf_map.items():
                for stg in stg_list:
                    stg_ver[stg.__name__] = str(stg.version)
        items = [f'{name}:{ver}' for name, ver in stg_ver.items()]
        stg_text = '|'.join(sorted(items))
        md5 = hashlib.md5()
        md5.update(stg_text.encode())
        return str(md5.hexdigest())


class StrategyResolver(IResolver):

    object_type: Type[Any] = BaseStrategy
    object_type_str: str = 'BaseStrategy'
    user_subdir = 'strategies'
    initial_search_path = Path(__file__).parent.resolve()

    @classmethod
    def load_run_jobs(cls, config: dict, pairlist: List[str])\
            -> Dict[str, Tuple[int, Dict[str, Set[Type[BaseStrategy]]]]]:
        PTFJob.reset()
        timeframe = '1m'  # 默认周期1m，后期根据K线和策略自动计算
        for policy in config['run_policy']:
            strategy_cls = get_strategy(policy['name'])
            if not strategy_cls:
                raise RuntimeError(f'unknown Strategy: {policy["name"]}')
            for pair in pairlist:
                # TODO: 这里可根据策略，自定义此交易对的交易维度
                PTFJob.add(pair, timeframe, strategy_cls)
        # 记录此次任务的策略哈希值。
        BotGlobal.stg_hash = PTFJob.strategy_hash()
        return PTFJob.tojobs()


def get_strategy(name: str) -> Optional[Type[BaseStrategy]]:
    '''
    根据策略名，返回策略的类对象。
    如果未加载，则初始化加载所有可能的策略
    '''
    if not name:
        return None
    global strategy_map
    if strategy_map is None:
        from banbot.config import AppConfig
        config = AppConfig.get()
        strategy_list = StrategyResolver.load_object_list(config)
        strategy_map = {item.__name__: item for item in strategy_list}
        logger.info('found strategy: %s', list(strategy_map.keys()))
    return strategy_map.get(name)
