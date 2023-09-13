#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : ctx.py
# Author: anyongjin
# Date  : 2023/5/4
'''
上下文变量的归类：pair+timeframe
跟策略无关，不考虑策略。（即pair+timeframe对应多个策略时，也是同一个上下文环境）
请勿在以下协程方法中使用上下文变量，会创建新的上下文：
get_event_loop
create_task
wait_fir
wait
gather
'''
import sys
from contextvars import Context, ContextVar, copy_context
from typing import *

import numpy as np
import six
import operator

from collections.abc import Sequence

# 所有对上下文变量的数据访问，均应先获取锁，避免协程交互访问时发生混乱
bar_num = ContextVar('bar_num')
symbol_tf = ContextVar('symbol_tf')
bar_arr = ContextVar('bar_arr')
bar_time = ContextVar('bar_time', default=(0, 0))
_symbol_ctx: Dict[str, Context] = dict()
symbol_tf.set('')
# 几个常用列的索引
tcol, ocol, hcol, lcol, ccol, vcol = 0, 1, 2, 3, 4, 5


def _update_context(kwargs):
    for key, val in kwargs:
        key.set(val)


def get_cur_symbol(ctx: Optional[Context] = None):
    from banbot.storage.symbols import ExSymbol
    pair_tf = ctx[symbol_tf] if ctx else symbol_tf.get()
    exg_name, market, symbol, timeframe = pair_tf.split('_')
    exs = ExSymbol.get(exg_name, market, symbol)
    return exs, timeframe


def get_context(pair_tf: str) -> Context:
    '''
    此方法获取一个缓存的上下文环境。仅可用于读取，不可用于写入。
    :param pair_tf:
    :return:
    '''
    if symbol_tf.get() == pair_tf:
        return copy_context()
    return _symbol_ctx[pair_tf]


def set_context(pair_tf: str):
    '''
    设置交易对和时间单元上下文。
    :param pair_tf: binance/future/BTC/USDT/1m
    :return:
    '''
    old_ctx = copy_context()
    if symbol_tf in old_ctx and symbol_tf.get() in _symbol_ctx:
        if symbol_tf.get() == pair_tf:
            # 上下文未变化，直接退出
            return
        # 保存旧的值到旧的上下文
        save_ctx = _symbol_ctx[symbol_tf.get()]
        save_ctx.run(_update_context, old_ctx.items())
    if not pair_tf:
        reset_context('')
        return
    if pair_tf not in _symbol_ctx:
        reset_context(pair_tf)
        _symbol_ctx[pair_tf] = copy_context()
    else:
        # 从新的上下文恢复上次的状态
        _update_context(_symbol_ctx[pair_tf].items())


def reset_context(pair_tf: str):
    symbol_tf.set(pair_tf)
    bar_num.set(0)
    bar_arr.set([])
    bar_time.set((0, 0))
    if pair_tf:
        LongStat.reset(pair_tf)
        MetaStateVar.reset(pair_tf)


class TempContext:
    def __init__(self, pair_tf: str):
        self.pair_tf = pair_tf
        self.is_locked = False
        self.back_symbol = None

    def __enter__(self):
        try:
            cur_val = symbol_tf.get()
        except LookupError:
            cur_val = None
        self.back_symbol = cur_val
        if self.back_symbol != self.pair_tf:
            self.back_symbol = cur_val
            self.is_locked = True
            set_context(self.pair_tf)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_locked:
            return
        set_context(self.back_symbol)
        self.is_locked = False


def avg_in_range(view_arr, start_rate=0.25, end_rate=0.75, is_abs=True) -> float:
    if is_abs:
        view_arr = np.abs(view_arr)
    sort_arr = np.sort(view_arr, axis=None)
    arr_len = len(sort_arr)
    start, end = round(arr_len * start_rate), round(arr_len * end_rate)
    return np.average(sort_arr[start: end])


class LongStat:
    '''
    大周期更新计算的变量。必须在上下文变量中使用。
    '''
    _instances: Dict[str, 'LongStat'] = dict()
    reg_list = []
    '所有大周期变量，都需要在这里注册'

    update_step: int = 200
    '每隔多少个bar计算一次'

    back_num: int = 600
    '每次计算时回顾的bar数量'

    alpha: float = 0.7
    '每次计算时，新值的权重，旧值权重：1-alpha'

    def __init__(self):
        self.val: float = np.nan
        '当前的值'

        self._next_calc: int = self.update_step
        '下次计算的bar'

    @classmethod
    def calc(cls):
        raise NotImplementedError

    @classmethod
    def get(cls, prefix: str = None):
        if not prefix:
            prefix = symbol_tf.get()
        obj = cls._instances.get(f'{prefix}_{cls.__name__}')
        if not obj:
            return np.nan
        return obj.val

    @classmethod
    def init(cls):
        '''
        针对当前上下文变量，初始化所有长变量。仅需调用一次。
        '''
        prefix = symbol_tf.get()
        for fac in cls.reg_list:
            cls._instances[f'{prefix}_{fac.__name__}'] = fac()

    @classmethod
    def update(cls, prefix: str = None, cur_num: int = 0):
        if not prefix:
            prefix = symbol_tf.get()
        if not cur_num:
            cur_num = bar_num.get()
        care_inds = [obj for key, obj in cls._instances.items()
                     if key.startswith(prefix) and cur_num >= obj._next_calc]
        for ind in care_inds:
            new_val = ind.calc()
            if ind.val is None or not np.isfinite(ind.val):
                ind.val = new_val
            else:
                ind.val = ind.val * (1 - ind.alpha) + new_val * ind.alpha
            ind._next_calc = cur_num + ind.update_step

    @classmethod
    def reset(cls, ctx_key: str):
        for key in list(cls._instances):
            if key.startswith(ctx_key):
                del cls._instances[key]


class LongChange(LongStat):
    @classmethod
    def calc(cls):
        return max(Bar.high[:cls.back_num]) - min(Bar.low[:cls.back_num])


class LongVolAvg(LongStat):
    @classmethod
    def calc(cls):
        val_arr = Bar.vol[:cls.back_num]
        return sum(val_arr) / len(val_arr)


class LongBarLen(LongStat):
    @classmethod
    def calc(cls):
        bar_lens = np.array(Bar.close[:cls.back_num]) - np.array(Bar.open[:cls.back_num])
        return avg_in_range(bar_lens)


LongStat.reg_list.extend([LongChange, LongVolAvg, LongBarLen])


class MetaStateVar(type):
    _instances = {}

    def __call__(cls, key: str, *args, **kwargs):
        if not key or not isinstance(key, six.string_types):
            raise ValueError('`key` is required for `SeriesVar`')
        if key not in cls._instances:
            serie_obj = super(MetaStateVar, cls).__call__(key, *args, **kwargs)
            cls._instances[key] = serie_obj
        else:
            serie_obj = cls._instances[key]
        return serie_obj

    @classmethod
    def reset(cls, ctx_key: str):
        all_keys = cls._instances.keys()
        del_keys = {key for key in all_keys if key.startswith(ctx_key)}
        for key in del_keys:
            del cls._instances[key]


class SeriesVar(metaclass=MetaStateVar):
    '''
    序列变量，存储任意类型的序列数据。用于带状态指标的计算和缓存
    '''
    keep_num = 1000

    def __init__(self, key: str):
        '''
        序列变量的初始化，应只用于ohlcv等基础数据
        '''
        self.val: Optional[List[float]] = []
        self.cols: Optional[List[SeriesVar]] = None
        self.key = key
        self.time = 0

    def append(self, row, check_len=True):
        in_time = bar_time.get()[1]
        if in_time <= self.time:
            raise ValueError(f'repeat update on {self.key}')
        self.time = in_time
        if hasattr(row, '__len__'):
            if not self.cols:
                self.cols = []
                for i, v in enumerate(row):
                    self.cols.append(SeriesVar(self.key + f'_{i}'))
            for i, v in enumerate(row):
                col = self.cols[i]
                if col.cached():
                    continue
                col.append(v[0] if isinstance(v, SeriesVar) else v)
        else:
            self.val.append(row)
            if check_len:
                if len(self.val) > self.keep_num * 2:
                    self.val = self.val[-self.keep_num:]
        return self

    def cached(self):
        return bar_time.get()[1] <= self.time

    def __getitem__(self, item):
        if self.cols is not None:
            raise ValueError('merge cols SeriesVar cannot be accessed by index')
        arr_len = len(self.val)
        if isinstance(item, slice):
            # 处理切片语法 obj[start:stop:step]
            # 将索引转为从末尾开始的逆序索引
            if item.start is None:
                stop = arr_len
            elif item.start < 0:
                raise IndexError(f'slice.start for SeriesVar should >= 0, current: {item}')
            else:
                stop = max(0, arr_len - item.start)
            if item.stop is None:
                start = 0
            elif item.stop < 0:
                raise IndexError(f'slice.stop for SeriesVar should >= 0, current: {item}')
            else:
                start = max(0, arr_len - item.stop)
            if item.step is None:
                return self.val[start:stop]
            else:
                return self.val[start:stop:item.step]
        if item >= arr_len or item < -arr_len:
            # 无效索引，返回nan
            return np.nan
        elif item >= 0:
            # 当直接使用0 1 2 3等取值时，反转索引，从末尾返回
            return self.val[arr_len - item - 1]
        else:
            # 负数索引逆序取值，保持不变
            return self.val[item]

    def __len__(self):
        if self.cols is None:
            return len(self.val)
        elif self.cols:
            return len(self.cols[0])
        return 0

    def __str__(self):
        if self.val:
            return f'{self.key}: {self.val[-1]}, len: {len(self.val)}'
        cur_vals = [c[-1] for c in self.cols]
        return f'{self.key}: {cur_vals}, len: {len(self.cols[0])}'

    def __repr__(self):
        if self.val:
            return f'{self.key}: {self.val[-1]}, len: {len(self.val)}'
        cur_vals = [c[-1] for c in self.cols]
        return f'{self.key}: {cur_vals}, len: {len(self.cols[0])}'

    def _apply(self, other, opt_func, fmt: str, is_rev=False) -> 'SeriesVar':
        other_key, other_val = self.key_val(other)
        if is_rev:
            res_key = fmt.format(other_key, self.key)
            res_val = opt_func(other_val, self[-1])
        else:
            res_key = fmt.format(self.key, other_key)
            res_val = opt_func(self[-1], other_val)
        return SeriesVar(res_key).append(res_val)

    def __add__(self, other):
        return self._apply(other, operator.add, '({0}+{1})')

    def __radd__(self, other):
        return self._apply(other, operator.add, '({0}+{1})')

    def __sub__(self, other):
        return self._apply(other, operator.sub, '({0}-{1})')

    def __rsub__(self, other):
        return self._apply(other, operator.sub, '({0}-{1})')

    def __mul__(self, other):
        return self._apply(other, operator.mul, '{0}*{1}')

    def __rmul__(self, other):
        return self._apply(other, operator.mul, '{0}*{1}')

    def __truediv__(self, other):
        return self._apply(other, operator.truediv, '{0}/{1}')

    def __rtruediv__(self, other):
        return self._apply(other, operator.truediv, '{0}/{1}', is_rev=True)

    def __floordiv__(self, other):
        return self._apply(other, operator.floordiv, '{0}//{1}')

    def __rfloordiv__(self, other):
        return self._apply(other, operator.floordiv, '{0}//{1}', is_rev=True)

    def __mod__(self, other):
        return self._apply(other, operator.mod, '({0}%{1})')

    def __rmod__(self, other):
        return self._apply(other, operator.mod, '({0}%{1})', is_rev=True)

    def __pow__(self, other):
        return self._apply(other, operator.pow, 'pow({0},{1})')

    def __rpow__(self, other):
        return self._apply(other, operator.pow, 'pow({0},{1})', is_rev=True)

    def __abs__(self) -> 'SeriesVar':
        return SeriesVar(f'abs({self.key})').append(abs(self[-1]))

    def __eq__(self, other):
        return isinstance(other, SeriesVar) and self.key == other.key and len(other) == len(self)

    @classmethod
    def key_val(cls, obj):
        if isinstance(obj, SeriesVar):
            key, val = obj.key, obj[-1]
        elif isinstance(obj, six.string_types):
            key, val = obj, obj
        elif isinstance(obj, (Sequence, np.ndarray)):
            key, val = obj[-1], obj[-1]
        else:
            key, val = obj, obj
        return key, val


class CrossLog(metaclass=MetaStateVar):
    def __init__(self, key: str):
        self.key = key
        self.prev_valid = None
        self.state = 0
        self.hist = []
        self.xup_num = sys.maxsize
        self.xdn_num = sys.maxsize

    def log(self, cur_diff: float, cur_num: int):
        self.state = 0
        if not self.prev_valid or not np.isfinite(self.prev_valid):
            self.prev_valid = cur_diff
        elif not cur_diff:
            pass
        else:
            factor = self.prev_valid * cur_diff
            if factor < 0:
                self.prev_valid = cur_diff
                self.state = 1 if cur_diff > 0 else -1
                self.hist.append((self.state, cur_num))
                if cur_diff > 0:
                    self.xup_num = cur_num
                else:
                    self.xdn_num = cur_num
        return self.state


def Cross(obj1: Union[SeriesVar, float], obj2: Union[SeriesVar, float]) -> int:
    '''
    计算最近一次交叉的距离。
    返回值：正数上穿，负数下穿，0表示未知或重合；abs(ret) - 1表示交叉点与当前bar的距离
    '''
    if not isinstance(obj1, SeriesVar) and not isinstance(obj2, SeriesVar):
        raise ValueError('one of obj1 or obj2 should be SeriesVar')
    key1, val1 = SeriesVar.key_val(obj1)
    key2, val2 = SeriesVar.key_val(obj2)
    obj: CrossLog = CrossLog(f'{key1}_xup_{key2}')
    cur_num = bar_num.get()
    obj.log(val1 - val2, cur_num)
    if obj.hist:
        item = obj.hist[-1]
        return item[0] * (cur_num - item[1] + 1)
    return 0


class MetaBar(type):

    def __getitem__(self, item):
        cols = [None, 'open', 'high', 'low', 'close', 'vol']
        if isinstance(item, int) and 0 < item < len(cols):
            return getattr(self, cols[item])
        return None


class Bar(metaclass=MetaBar):
    '''
    快速访问蜡烛数据
    '''
    open: SeriesVar
    high: SeriesVar
    low: SeriesVar
    close: SeriesVar
    vol: SeriesVar

