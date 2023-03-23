#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : num_utils.py
# Author: anyongjin
# Date  : 2023/3/2
import numpy as np


def cluster_kmeans(arr: np.ndarray, cls_num: int, max_iter=20):
    from sklearn.cluster import KMeans
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if len(arr.shape) == 1:
        arr = arr.reshape((-1, 1))
    kmeans = KMeans(n_clusters=cls_num, max_iter=max_iter, random_state=0, n_init='auto')
    x_groups = kmeans.fit_predict(arr).tolist()
    org_centers = kmeans.cluster_centers_.reshape(-1).tolist()
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
