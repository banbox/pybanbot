#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : num_utils.py
# Author: anyongjin
# Date  : 2023/3/2
from typing import List

import numpy as np


def cross_zero(arr: np.ndarray, go_type=0) -> List[int]:
    prev_val = arr[0]
    result = []
    for i in range(1, len(arr)):
        val = arr[i]
        if not val or prev_val * val > 0:
            continue
        if prev_val * val < 0:
            if go_type >= 0 and val > 0:
                result.append(i)
            elif go_type <= 0 and val < 0:
                result.append(i)
        prev_val = val
    return result


def cluster_kmeans(arr: np.ndarray, cls_num: int, max_iter=20):
    import warnings
    from sklearn.cluster import KMeans
    warnings.filterwarnings("ignore")
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if len(arr.shape) == 1:
        arr = arr.reshape((-1, 1))
    if len(arr) < cls_num:
        x_groups = list(range(len(arr)))
        org_centers = arr.flatten().tolist()
    else:
        kmeans = KMeans(n_clusters=cls_num, max_iter=max_iter, random_state=0, n_init='auto')
        x_groups = kmeans.fit_predict(arr).tolist()
        org_centers = kmeans.cluster_centers_.reshape(-1).tolist()
    warnings.resetwarnings()
    return x_groups, org_centers


def np_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def append_nan_cols(arr: np.ndarray, num: int) -> np.ndarray:
    if not num:
        return arr
    if len(arr.shape) > 1:
        pad_arr = np.zeros((arr.shape[0], num), dtype=np.float64)
        pad_arr[:] = np.nan
        return np.concatenate([arr, pad_arr], axis=1)
    else:
        pad_arr = np.zeros(num, dtype=np.float64)
        pad_arr[:] = np.nan
        return np.concatenate([arr, pad_arr], axis=0)


def parse_int(text: str, def_val=0):
    try:
        return int(text)
    except Exception:
        return def_val


def to_pytypes(data):
    if hasattr(data, 'dtype'):
        return data.item()
    return data


def arg_valid_id(a):
    '''
    返回第一个非Nan或Inf的索引
    '''
    bool_arr = np.logical_or(np.isnan(a), np.isinf(a))
    return np.argmax(~bool_arr)


def np_rolling(a, window: int, axis=1, pad='nan', calc_fn=None):
    pad_width = [(0, 0)] * a.ndim
    pad_width[axis - 1] = (window - 1, 0)
    if pad == 'same':
        a = np.pad(a, pad_width=pad_width, mode='edge')
    else:
        a = np.pad(a, pad_width=pad_width, constant_values=(np.nan, np.nan))
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    rol_arr = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    if not calc_fn:
        return rol_arr
    return calc_fn(rol_arr, axis=axis)


def max_rolling(a, window, axis=1, pad='nan'):
    return np_rolling(a, window, axis=axis, pad=pad, calc_fn=np.max)


def min_rolling(a, window, axis=1, pad='nan'):
    return np_rolling(a, window, axis=axis, pad=pad, calc_fn=np.min)

