#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : resolver.py
# Author: anyongjin
# Date  : 2023/3/31
from banbot.core.iresolver import *
from banbot.storage.common import BotGlobal
from banbot.strategy.base import BaseStrategy
from banbot.util.common import logger

global strategy_map
strategy_map: Optional[Dict[str, BaseStrategy]] = None


class PTFJob:
    pairs: Dict[str, Dict[str, Set[Type[BaseStrategy]]]] = dict()
    'pair:timeframe: stg_list'

    pair_stg: Set[Tuple[str, str]] = set()
    'pair+stg：防止一个策略重复处理一个交易对的多个维度'

    ptf_warms: Dict[Tuple[str, str], int] = dict()
    'pair+tf: 预热数量'

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
            wkey = pair, timeframe
            if wkey not in cls.ptf_warms or stg.warmup_num > cls.ptf_warms[wkey]:
                cls.ptf_warms[wkey] = stg.warmup_num
        stg_set.add(stg)

    @classmethod
    def reset(cls):
        cls.pairs = dict()
        cls.pair_stg = set()
        cls.ptf_warms = dict()

    @classmethod
    def tojobs(cls) -> List[Tuple[str, str, int, Set[Type[BaseStrategy]]]]:
        '''
        返回：pair, timeframe, 预热数量, 策略列表
        '''
        results: List[Tuple[str, str, int, Set[Type[BaseStrategy]]]] = []
        for pair, tf_dic in cls.pairs.items():
            for tf, stg_list in tf_dic.items():
                warm_num = cls.ptf_warms.get((pair, tf))
                results.append((pair, tf, warm_num, stg_list))
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
    def load_run_jobs(cls, config: dict, pairlist: List[str],
                      pair_tfscores: Dict[str, List[Tuple[str, float]]] = None)\
            -> List[Tuple[str, str, int, Set[Type[BaseStrategy]]]]:
        '''
        加载策略和交易对，返回对应关系：[(pair, timeframe, 预热数量, 策略列表), ...]
        '''
        PTFJob.reset()
        exg_name = config['exchange']['name']
        allow_filter = True
        if not pairlist:
            pairlist = ['BTC/USDT']
            allow_filter = False
        if not pair_tfscores:
            pair_tfscores = dict()
            for pair in pairlist:
                pair_tfscores[pair] = [('1m', 1.)]
        for policy in config['run_policy']:
            strategy_cls = get_strategy(policy['name'])
            if not strategy_cls:
                raise RuntimeError(f'unknown Strategy: {policy["name"]}')
            if policy.get('max_fee') is not None:
                strategy_cls.max_fee = policy.get('max_fee')
            max_num = policy.get('max_pair') or 999  # 默认一个策略最多999个交易对
            stg_process = 0
            for pair in pairlist:
                timeframe = strategy_cls.pick_timeframe(exg_name, pair, pair_tfscores.get(pair))
                if not timeframe:
                    if allow_filter:
                        continue
                    else:
                        timeframe = '1m'
                stg_process += 1
                PTFJob.add(pair, timeframe, strategy_cls)
                if stg_process >= max_num:
                    break
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
        extra_dirs = config.get('stg_dir')
        strategy_list = StrategyResolver.load_object_list(config, extra_dirs)
        strategy_map = {item.__name__: item for item in strategy_list}
        logger.info('found strategy: %s', list(strategy_map.keys()))
    return strategy_map.get(name)
