#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : common.py
# Author: anyongjin
# Date  : 2023/4/12
import pandas as pd

from banbot.compute.tainds import *
candles = [
    [0, 26891.67, 26962.98, 26834.81, 26914.97, 10247.46713],
    [0, 26832.77, 27192, 26629, 26890.67, 53157.73523],
    [0, 27414.81, 27495.7, 26371.01, 26832.77, 61674.70312],
    [0, 27050.24, 27520, 26586.36, 27415.92, 50880.01319],
    [0, 27185.21, 27317.71, 26867.79, 27050.24, 47244.381570000005],
    [0, 26939.51, 27678.39, 26754.78, 27185.22, 51100.45154],
    [0, 26789.12, 27218.49, 26569.08, 26938.48, 26228.06098],
    [0, 26812.5, 27062.94, 26701.41, 26789.44, 24605.496470000002],
    [0, 26991.98, 27111.89, 25822, 26810.74, 51827.44158999999],
    [0, 27625.07, 27656, 26721, 26990.71, 52783.61571000001],
    [0, 27655.26, 28340, 26800, 27625.07, 59128.7589],
    [0, 27701.59, 27853.69, 27378.93, 27655.26, 35068.21305999995],
    [0, 28472.77, 28673.67, 27288.22, 27701.59, 34562.96968999999],
    [0, 28896.29, 29173.46, 28437.5, 28472.77, 25241.63320000002],
    [0, 29532.15, 29850, 28388, 28896.34, 33936.28628],
    [0, 28866.52, 29708.55, 28825, 29533.97, 46223.62572000006],
    [0, 29048.31, 29400, 28690.02, 28865.4, 48028.34416999999],
    [0, 28701.03, 29285.93, 28144.44, 29042.99, 47371.76221000003],
    [0, 27993.04, 28900, 27797.46, 28700.81, 34897.97145999996],
    [0, 29000.36, 29052.53, 27561.02, 27993.04, 26208.83683999998],
    [0, 29253.86, 29972, 28942.1, 29000.32, 33055.20687000001],
    [0, 29339.73, 29468.7, 29060.81, 29254.87, 22012.70613999999],
    [0, 29515.79, 29647.84, 28914.17, 29338.41, 41741.86541999998],
    [0, 28445.86, 29940, 28405.37, 29515.79, 74979.10387000002],
    [0, 28318.7, 30065.88, 27263, 28445, 93720.98509999998],
    [0, 27531.16, 28410.57, 27207.98, 28318.7, 58044.8419400001],
    [0, 27602.9, 28022.36, 27002.66, 27531.17, 56076.372839999996],
    [0, 27824.93, 27828.48, 27350, 27602.9, 31392.63947000003],
    [0, 27271.3, 27897.47, 27160, 27824.92, 31534.241970000057],
    [0, 28253.28, 28387.31, 27143.33, 27270.02, 47614.79214000002],
    [0, 28818.69, 29110, 28001, 28255.22, 53141.693630000074],
    [0, 30387.2, 30418.41, 28600.01, 28815.05, 56079.82421999995],
    [0, 29450.13, 30495.01, 29115.24, 30387.27, 51972.14477999999],
    [0, 30322.06, 30344.14, 29250.11, 29450.16, 44429.02792999996],
    [0, 30324.28, 30574, 30142.48, 30323.15, 30366.779200000004],
    [0, 30490.34, 30625, 30233.93, 30324.29, 26322.614529999963],
    [0, 30407.05, 31073.61, 30000, 30490, 48525.85093000003],
]
ohlcv_arr = np.array(candles)
high_arr, low_arr, close_arr = ohlcv_arr[:, hcol], ohlcv_arr[:, lcol], ohlcv_arr[:, ccol]
high_col, low_col, close_col = pd.Series(high_arr), pd.Series(low_arr), pd.Series(close_arr)


def print_tares(vec_res, sta_res, ta_cres=None, ta_mres=None, mytt_res=None, pta_res=None):
    print('\n' + ' Vector Res '.center(60, '='))
    print(vec_res)
    print('\n' + ' State Res '.center(60, '='))
    print(sta_res)
    if ta_cres is not None:
        print('\n' + ' Ta-lib Classic '.center(60, '='))
        print(ta_cres)
    if ta_mres is not None:
        print('\n' + ' Ta-lib MetaStock '.center(60, '='))
        print(ta_mres)
    if mytt_res is not None:
        print('\n' + ' MyTT '.center(60, '='))
        print(mytt_res)
    if pta_res is not None:
        if hasattr(pta_res, 'to_numpy'):
            pta_res = pta_res.to_numpy()
        print('\n' + ' Pandas-TA '.center(60, '='))
        print(pta_res)


def calc_state_ind(ind: BaseInd, input_arr):
    '''
    计算有状态指标的值，并返回结果。如果是多列，则返回多列
    '''
    if not isinstance(input_arr, np.ndarray):
        input_arr = np.array(input_arr)
    dim_sub = input_arr.ndim - ind.input_dim
    with TempContext('BTC/TUSD/1m'):
        bar_num.set(1)
        result = []
        for i in range(len(input_arr)):
            bar_num.set(bar_num.get() + 1)
            if dim_sub == 1:
                in_val = input_arr[i]
            elif dim_sub == 0:
                in_val = input_arr[:i + 1]
            elif dim_sub == 2:
                in_val = input_arr[i, ccol]
            else:
                raise ValueError(f'unsupport dim sub: {dim_sub}')
            result.append(ind(in_val))
    if len(result) and isinstance(result[0], (np.ndarray, List, Tuple)):
        res_list = list(zip(*result))
        return [np.array(res) for res in res_list]
    else:
        return np.array(result)


def assert_arr_equal(arr_a: np.ndarray, arr_b: np.ndarray):
    assert len(arr_a) == len(arr_b), f'lan a != b, {len(arr_a)} != {len(arr_b)}'
    assert arr_a.shape == arr_b.shape, f'shape a != b, {arr_a.shape} != {arr_b.shape}'
    for i in range(len(arr_a)):
        vala, valb = arr_a[i], arr_b[i]
        assert np.isnan(vala) == np.isnan(valb), f'nan: {i} {vala} {valb}'
        if np.isnan(vala):
            continue
        assert abs(vala - valb) / max(abs(vala), abs(valb)) < 0.0001, f'val {i} {vala} != {valb}'
