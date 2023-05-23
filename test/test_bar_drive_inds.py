#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_bar_drive_inds.py
# Author: anyongjin
# Date  : 2023/3/2

from banbot.compute.sta_inds import *
import pandas as pd
import os


def load_bnb_1s():
    data_dir = r'E:\trade\freqtd_data\user_data\spec_data\bnb1s'
    fname = 'BTCUSDT-1s-2023-02-22-2023-02-26.feather'
    return pd.read_feather(os.path.join(data_dir, fname))


origin_df = load_bnb_1s()[:10000]
arr = origin_df.to_numpy()[:, 1:]
debug_idx = int(np.where(origin_df['date'] == '2023-02-22 01:14:55')[0][0])
print(f'debug idx: {debug_idx}')


def test_inds():
    global arr
    reset_ind_context(arr.shape[1])
    use_inds = [SMA(5), NATR(3), NTRRoll(3)]
    set_use_inds(use_inds)
    result = apply_inds_to_first(arr[0])
    pad_len = result.shape[1] - arr.shape[1]
    arr = append_nan_cols(arr, pad_len)
    for i in range(1, len(arr)):
        result = np.append(result, np.expand_dims(arr[i, :], axis=0), axis=0)
        result = apply_inds(result)
        ind_text = [f'{c.name}: {result[i, c.out_idx]}' for c in use_inds]
        print('   '.join(ind_text))


def test_mean_rev():
    global arr
    from banbot.strategy.mean_rev import MeanRev
    tipper = MeanRev({debug_idx})
    result, ptn = tipper.on_bar(arr[:1])
    tipper.on_entry(result)
    pad_len = tipper.out_dim - arr.shape[1]
    arr = append_nan_cols(arr, pad_len)
    entry_tags = []
    for i in range(1, len(arr)):
        result = np.append(result, np.expand_dims(arr[i, :], axis=0), axis=0)
        result, ptn = tipper.on_bar(result)
        tag = tipper.on_entry(result)
        entry_tags.append(tag)
        if tag:
            logger.warning(f'found entry {tag} at {origin_df.loc[i, "date"]}')


test_mean_rev()
