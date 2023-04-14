#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : common.py
# Author: anyongjin
# Date  : 2023/4/12
import numpy as np


def assert_arr_equal(arr_a: np.ndarray, arr_b: np.ndarray):
    assert len(arr_a) == len(arr_b), f'lan a != b, {len(arr_a)} != {len(arr_b)}'
    assert arr_a.shape == arr_b.shape, f'shape a != b, {arr_a.shape} != {arr_b.shape}'
    for i in range(len(arr_a)):
        vala, valb = arr_a[i], arr_b[i]
        assert np.isnan(vala) == np.isnan(valb), f'nan: {i} {vala} {valb}'
        if np.isnan(vala):
            continue
        assert abs(vala - valb) / max(abs(vala), abs(valb)) < 0.0001, f'val {i} {vala} != {valb}'
