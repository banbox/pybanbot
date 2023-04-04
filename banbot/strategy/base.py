#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : BaseTipper.py
# Author: anyongjin
# Date  : 2023/3/1
import numpy as np

from banbot.strategy.common import *


def append_new_bar(row: np.ndarray) -> np.ndarray:
    result = bar_arr.get()
    copen, chigh, clow, close = row[ocol:vcol]
    dust = min(0.00001, max(close, 0.001) * 0.0001)
    max_chg = dust + chigh - clow
    real = abs(close - copen)
    solid_rate = real / max_chg
    hline_rate = (chigh - max(close, copen)) / max_chg
    lline_rate = (min(close, copen) - clow) / max_chg
    bar_num.set(bar_num.get() + 1)
    crow = np.concatenate([row, [max_chg, real, solid_rate, hline_rate, lline_rate]], axis=0)
    exp_crow = np.expand_dims(crow, axis=0)
    if not len(result):
        fea_col_start.set(len(row))
        result = exp_crow
    else:
        result = np.append(result, exp_crow, axis=0)
    bar_arr.set(result)
    bar_end_time.set(btime.to_datestr(row[0] / 1000 + timeframe_secs.get() - 1))
    LongVar.update(result)
    return result


class BaseStrategy:
    warmup_num = 900
    skip_exit_on_enter = True
    skip_enter_on_exit = True

    def __init__(self, config: dict):
        self.config = config
        self.long_sigs: Dict[str, Tuple[int, float]] = dict()  # 记录做多信号。key: [bar_num, score]
        self.short_sigs: Dict[str, Tuple[int, float]] = dict()  # 记录做空信号。key: [bar_num, score]
        self.patterns: List[Dict[str, float]] = []  # 识别出的K线形态
        self.ma_cross: List[float] = []  # 记录重要的均线交叉点
        self.extrems_ma5 = []  # 保存近期极大值极小值[(bar_num, ma5)]，两个极值点差不超过avg_bar_len的会被过滤
        self.extrems_ma20 = []
        self.extrems_ma120 = []
        self.state = dict()
        self._state_fn = dict()

    def _calc_state(self, key: str, *args, **kwargs):
        if key not in self.state:
            self.state[key] = self._state_fn[key](*args, **kwargs)
        return self.state[key]

    def _log_ma_cross(self, ma_a: StaSMA, ma_b: StaSMA):
        if len(ma_a.arr) < 3 or np.isnan(ma_a.arr[-3]):
            return
        b, a = ma_a.arr[-1] - ma_b.arr[-1], ma_a.arr[-3] - ma_b.arr[-3]
        if max(a, b) > 0 and a + b < abs(a) + abs(b):
            crs_id = bar_num.get() - 1
            if not self.ma_cross or crs_id - self.ma_cross[-1] >= 3:
                self.ma_cross.append(crs_id)
                if len(self.ma_cross) > 300:
                    self.ma_cross = self.ma_cross[-100:]

    def _get_sigs(self, tag: str, period: int = 1) -> List[Tuple[str, float, int]]:
        '''
        获取做空或做多信号。按bar_num降序、置信度降序
        :param tag: long/short
        :param period: 1
        :return:
        '''
        min_id = bar_num.get() - period
        if tag == 'long':
            sigs = [(lkey, litem[1], litem[0]) for lkey, litem in self.long_sigs.items() if litem[0] > min_id]
        else:
            sigs = [(lkey, litem[1], litem[0]) for lkey, litem in self.short_sigs.items() if litem[0] > min_id]
        return sorted(sigs, key=lambda x: (x[1], x[2]), reverse=True)

    def on_bar(self, arr: np.ndarray):
        '''
        计算指标。用于后续入场出场信号判断使用。
        :param arr:
        :return:
        '''
        pass

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
