#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : ind_common.py
# Author: anyongjin
# Date  : 2023/3/9
import sys
import inspect

import numpy as np

from banbot.compute.utils import to_snake_case, SingletonArg, logger
from banbot.compute.num_utils import *
from typing import Callable, Dict, List
from contextvars import ContextVar

# 计算指标中间结果的函数注册缓存
dep_handlers: Dict[str, Callable] = dict()  # 重置指标时需要清理
ind_call_map: Dict[str, Callable] = dict()
use_cols = ContextVar('use_cols')  # 涉及的通用依赖列(缓存中间计算结果，提高速度)
use_refs = ContextVar('use_refs')  # 键是指标名，值是引用的列集合
use_id_map = ContextVar('use_id_map')  # 记录依赖列对应的ID
ohcl_arr = ContextVar('ohcl_arr')
use_ind_objs = ContextVar('use_ind_objs')
bar_num = ContextVar('bar_num')
inrows_num = ContextVar('inrows_num')  # 输入数据的最大长度，取决于各个指标对历史数据依赖长度
use_cols.set([])
use_refs.set(dict())
use_id_map.set(dict())
use_ind_objs.set([])
bar_num.set(0)
inrows_num.set(100)


def avg_in_range(view_arr: np.ndarray, start_rate=0.25, end_rate=0.75, is_abs=True) -> float:
    if is_abs:
        view_arr = np.abs(view_arr)
    sort_arr = np.sort(view_arr, axis=None)
    arr_len = len(sort_arr)
    start, end = round(arr_len * start_rate), round(arr_len * end_rate)
    return np.average(sort_arr[start: end])


class LongVar:
    def __init__(self, key: str, roll_len: int, update_step: int, make_calc=None, old_rate=0.7):
        self.name = key
        self.old_rate = old_rate
        self.make_calc_func = None
        if make_calc:
            self.make_calc_func = make_calc
            self.calc_func = 1
        elif self.name == 'long_price_range':
            self.calc_func = lambda view_arr: np.max(view_arr[:, 1]) - np.min(view_arr[:, 2])
        elif self.name == 'long_vol_avg':
            self.calc_func = lambda view_arr: np.average(view_arr[:, 4])
        elif self.name == 'long_bar_avg':
            self.calc_func = lambda arr: avg_in_range(arr[:, 0] - arr[:, 3])
        else:
            raise ValueError(f'unsupport norm type: {self.name}')
        self.val = np.nan
        self.update_step = update_step
        self.roll_len = roll_len
        self.update_at = 0

    def reset(self, init_val: float):
        self.val = init_val
        self.update_at = bar_num.get() + self.update_step

    def ensure_func(self):
        if self.calc_func == 1:
            assert self.make_calc_func, 'make_calc is required'
            self.calc_func = self.make_calc_func()

    def on_bar(self, arr: np.ndarray):
        if arr.shape[0] < self.roll_len:
            return
        num_off = bar_num.get() - self.update_at
        cur_val = self.val
        if np.isnan(cur_val) or num_off == self.update_step:
            new_val = self.calc_func(arr[-self.roll_len:, :])
            if np.isnan(cur_val):
                cur_val = new_val
            else:
                cur_val = cur_val * self.old_rate + new_val * (1 - self.old_rate)
            self.val = cur_val
            self.update_at = bar_num.get()


long_price_range = LongVar('long_price_range', 600, 600)
long_vol_avg = LongVar('long_vol_avg', 600, 600)
long_bar_avg = LongVar('long_bar_avg', 600, 600)


def make_ind_compute(ind: 'BaseInd', val_handle):
    def _safe_compute_ind(arr: np.ndarray):
        if arr.shape[0] == 1:
            arr = ind.calc_first(arr)
            ohcl_arr.set(arr)
            return val_handle(arr[-1, ind.out_idx], arr)
        ohcl_arr.set(arr)
        ind()
        after_arr = ohcl_arr.get()
        return val_handle(after_arr[-1, ind.out_idx], arr)

    return _safe_compute_ind


def get_ind_callable(col: str):
    import re
    cls_key = next((i for i in ind_call_map if re.match(i + r'($|\d)', col)), None)
    if cls_key:
        return ind_call_map[cls_key], cls_key
    return None, None


def _get_dep_func(col: str) -> Callable:
    if col in dep_handlers:
        return dep_handlers[col]
    # 检查是否可从指标计算；然后检查是否是自定义逻辑
    call_func, cls_key = get_ind_callable(col)
    if call_func:
        arg_text = col[len(cls_key):]
        if inspect.isclass(call_func) and issubclass(call_func, BaseInd):
            ind = call_func.get_by_str(cls_key, arg_text)
            dep_handlers[col] = make_ind_compute(ind, lambda x, arr: x)
        elif callable(call_func):
            dep_handlers[col] = call_func(arg_text)
        else:
            raise ValueError(f'unknown function for symbol: {col} in ind_call_map')
    elif col.startswith('c_div'):
        period = int(col[len('c_div'):])
        dep_handlers[col] = lambda a: a[-1, 3] / period
    else:
        raise ValueError(f'unsupport col: {col}')
    return dep_handlers[col]


def _calc_dep_val(arr: np.ndarray, col: str, col_idx: int) -> np.ndarray:
    ohcl_arr.set(arr)
    dep_val = _get_dep_func(col)(arr)
    arr = ohcl_arr.get()
    arr[-1, col_idx] = dep_val
    return arr


class BaseInd(metaclass=SingletonArg):
    '''
    基于单个蜡烛更新计算的指标基类。基于参数的单例模式。
    输入numpy数组：[open, high, low, close, volume, count, long_vol, ...]
    后续列是特征列
    '''
    col_offset = -1  # 缓存列开始的偏移，-1表示未设置

    def __init__(self, roll_len: int = 1, name: str = None):
        '''
        初始化一个指标。
        :param roll_len: 窗口长度
        :param name: 指标名
        '''
        self.roll_len = roll_len
        if not name:
            suffix = str(roll_len) if roll_len > 1 else ''
            name = to_snake_case(self.__class__.__name__) + suffix
        self.name = name
        self.long_vars = []

    @property
    def out_idx(self) -> int:
        use_id_map_val = use_id_map.get()
        if self.name in use_id_map_val:
            return use_id_map_val[self.name]
        raise ValueError(f'{self.name} not added in `use_id_map`')

    def _add_dep(self, *dep_items: str):
        assert BaseInd.col_offset > 0, '`col_offset` is not initialized'
        use_cols_val = use_cols.get()
        use_id_map_val = use_id_map.get()
        use_refs_val = use_refs.get()
        dep_cols = []
        for item in dep_items:
            if isinstance(item, LongVar):
                self.long_vars.append(item)
                continue
            assert item, '`name` is required for add_dep'
            dep_cols.append(item)
            if item not in use_cols_val:
                use_cols_val.append(item)
                use_id_map_val[item] = BaseInd.col_offset + len(use_cols_val) - 1
        if self.name not in use_refs_val:
            use_refs_val[self.name] = set()
        use_refs_val[self.name].update(dep_cols)

    def calc_first(self, arr: np.ndarray) -> np.ndarray:
        '''
        用于计算第一行ohlc的值，因为列会变化。所以需要做一些expand
        :param arr:
        :return:
        '''
        assert len(arr.shape) == 2 and arr.shape[0] == 1, 'arr shape must be [1, n]'
        # 补充缓存列
        pad_col_num = self._get_pad_len(arr.shape[1])
        if pad_col_num > 0:
            arr = append_nan_cols(arr, pad_col_num)
        # 计算依赖列的值
        arr = self._calc_dep(arr)
        # 计算当前指标的值
        ind_val = self._calc_val(arr)
        # 将当前指标值更新到arr中
        crow = arr[-1, :]
        use_id_map_val = use_id_map.get()
        if self.name not in use_id_map_val:
            use_cols_val = use_cols.get()
            use_cols_val.append(self.name)
            use_id_map_val[self.name] = BaseInd.col_offset + len(use_cols_val) - 1
            pad_list = [np.nan] * self._get_pad_len(len(crow) + 1) + [ind_val]
            crow = np.concatenate([crow, pad_list], axis=0)
            return np.expand_dims(crow, axis=0)
        # 如果当前指标被其他指标依赖，则已添加到use_id_map
        crow[self.out_idx] = ind_val
        return arr

    def __call__(self, *args, **kwargs):
        '''
        仅用于计算第二行及以后ohlc的指标值。arr shape: [>1, n]
        :param args:
        :param kwargs:
        :return:
        '''
        arr = ohcl_arr.get()
        arr = self._calc_dep(arr)
        # 将此指标添加到array中
        arr[-1, self.out_idx] = self._calc_val(arr)
        ohcl_arr.set(arr)

    def _get_pad_len(self, cur_len: int):
        use_cols_val = use_cols.get()
        return len(use_cols_val) + BaseInd.col_offset - cur_len

    def _calc_dep(self, arr: np.ndarray) -> np.ndarray:
        dep_cols = use_refs.get().get(self.name)
        if dep_cols:
            # 计算依赖的列的值
            use_id_map_val = use_id_map.get()
            for col in dep_cols:
                col_idx = use_id_map_val[col]
                if not np.isnan(arr[-1, col_idx]):
                    continue
                arr = _calc_dep_val(arr, col, col_idx)
        return arr

    def _calc_val(self, arr: np.ndarray) -> np.float64:
        ind_val = np.nan
        if arr.shape[0] >= self.roll_len:
            ind_val = self._compute(arr[-self.roll_len:, :])
        return ind_val

    def _calc_norm(self, view_arr: np.ndarray):
        raise NotImplementedError('_calc_norm is not implement')

    def _compute(self, arr: np.ndarray) -> np.float64:
        '''
        计算当前bar的此指标的值
        :param arr: 此指标计算所需要的period周期数据
        :return:
        '''
        raise NotImplementedError(f'_compute in {self.name}')

    @classmethod
    def get_by_str(cls, ind_key: str, args: str) -> 'BaseInd':
        call_args = [parse_int(args)] if args else []
        return cls(*call_args)
