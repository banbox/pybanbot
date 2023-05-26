#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : BaseTipper.py
# Author: anyongjin
# Date  : 2023/3/1
import sys

from banbot.strategy.common import *


class BaseStrategy:
    '''
    策略基类。每个交易对，每种时间帧，每个策略，对应一个策略实例。
    可直接在策略的__init__中存储本策略在此交易对和交易维度上的缓存信息。不会和其他交易对冲突。
    '''
    warmup_num = 600
    min_tfscore = 0.8
    skip_exit_on_enter = True
    skip_enter_on_exit = True
    version = 1

    def __init__(self, config: dict):
        self.config = config
        self._cross_up: Dict[str, List[int]] = dict()  # 记录信号向上交叉，由负变正
        self._cross_down: Dict[str, List[int]] = dict()  # 记录信号向下交叉，由正变负
        self._cross_last: Dict[str, float] = dict()  # 记录信号交叉点
        self.state = dict()  # 尽在当前bar生效的临时缓存
        self._state_fn = dict()

    def _calc_state(self, key: str, *args, **kwargs):
        if key not in self.state:
            self.state[key] = self._state_fn[key](*args, **kwargs)
        return self.state[key]

    def _log_cross(self, tag: str, value: float):
        '''
        记录指定标签的每一次交叉。
        '''
        if not np.isfinite(value) or value == 0:
            return
        prev_val = self._cross_last.get(tag)
        self._cross_last[tag] = value
        if prev_val is None:
            return
        if prev_val * value < 0:
            cross_dic = self._cross_up if value > 0 else self._cross_down
            if tag not in cross_dic:
                cross_dic[tag] = []
            cross_dic[tag].append(bar_num.get())
            if len(cross_dic[tag]) > 200:
                cross_dic[tag] = cross_dic[tag][-100:]

    def _cross_dist(self, tag: str, dirt=0):
        '''
        获取指定标签，与上次交叉的距离。如果尚未发生交叉，返回一个极大值
        dirt：0表示不限制；1上穿；-1下穿
        '''
        up_dist, down_dist = sys.maxsize, sys.maxsize
        if dirt >= 0:
            pos = self._cross_up.get(tag)
            if pos:
                up_dist = bar_num.get() - pos[-1]
        if dirt <= 0:
            pos = self._cross_down.get(tag)
            if pos:
                down_dist = bar_num.get() - pos[-1]
        return min(up_dist, down_dist)

    def on_bar(self, arr: np.ndarray):
        '''
        计算指标。用于后续入场出场信号判断使用。
        :param arr:
        :return:
        '''
        pass

    def _update_inds(self, arr: np.ndarray, *args):
        cur_close, cur_row = arr[-1, ccol], arr[-1]
        for ind in args:
            dim_sub = arr.ndim - ind.input_dim
            if dim_sub == 1:
                in_val = cur_row
            elif dim_sub == 0:
                in_val = arr
            elif dim_sub == 2:
                in_val = cur_close
            else:
                raise ValueError(f'unsupport dim sub: {dim_sub} from {type(ind)}')
            ind(in_val)

    def on_entry(self, arr: np.ndarray) -> Optional[str]:
        '''
        时间升序，最近的是最后一个
        :param arr:
        :return:
        '''
        pass

    def custom_cost(self, enter_tag: str) -> float:
        '''
        返回自定义的此次订单金额
        :param enter_tag:
        :return:
        '''
        return self.config.get('stake_amount', 1000)

    def on_exit(self, arr: np.ndarray) -> Optional[str]:
        pass

    def custom_exit(self, arr: np.ndarray, od: InOutOrder) -> Optional[str]:
        return None

    @property
    def name(self):
        return self.__class__.__name__

    @classmethod
    def pick_timeframe(cls, exg_name: str, symbol: str, tfscores: List[Tuple[str, float]]) -> Optional[str]:
        if not tfscores:
            return None
        for tf, score in tfscores:
            if score >= cls.min_tfscore:
                return tf
