#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : mldata.py
# Author: anyongjin
# Date  : 2023/2/14
import time

import pandas as pd
import os
import numpy as np
import lightgbm as lgb
import logging
logger = logging.getLogger(__name__)

data_dir = r'E:\trade\freqtd_data\user_data\freqai_data\feaprob'
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)


def process_df(data_df: pd.DataFrame, pos_thres: int = 35, neg_thres: int = -30):
    '''
    将profit列高于40作为1，低于-40作为-1，否则作为0。形成三个类别
    profit是预期利润（单位：万分之）
    :param data_df:
    :param pos_thres:
    :param neg_thres:
    :return:
    '''
    profit = data_df['profit'].to_numpy()
    data_y = np.zeros(profit.shape, dtype=np.int32)
    data_y[profit >= pos_thres] = 1
    data_y[profit <= neg_thres] = 2
    data_df.drop(columns=['profit'], inplace=True)
    data_x = np.nan_to_num(data_df.to_numpy(), posinf=0, neginf=0)
    return data_x, data_y, data_df.columns.tolist()


def _load_train_df(timeframe: str, pos_thres: int = 35, neg_thres: int = -30, pred_off: int = 1):
    c_test_path = os.path.join(data_dir, f'test_{timeframe}.feather')
    c_train_path = os.path.join(data_dir, f'train_{timeframe}.feather')
    max_train_size = 3000000
    if os.path.isfile(c_train_path):
        train_df = pd.read_feather(c_train_path)
        test_df = pd.read_feather(c_test_path)
    else:
        '当数据不存在时，从全部回测数据实时生成并分割为训练&测试数据'
        logger.warning('no train/test data found, generating...')
        from banbot.compute.datatools import load_all_pairs, calc_next_profit
        from banbot.compute.main import MicroWave
        mw = MicroWave(timeframe)
        pairs = load_all_pairs(timeframe)
        pair_datas = []
        last_time = time.monotonic()
        for i, sam_df in enumerate(pairs):
            calc_next_profit(sam_df, pred_off)
            handled = mw.compute_prob(sam_df)
            handled['profit'] = sam_df['profit']
            handled = handled.dropna(thresh=2).reset_index(drop=True)
            good_len = len(np.where(handled['profit'] >= pos_thres)[0])
            norm_ids = np.where((handled['profit'] > neg_thres) & (handled['profit'] < pos_thres))[0]
            # 普通数据保持正例的1.5倍
            del_num = round(len(norm_ids) - good_len * 1.5)
            if del_num > 0:
                del_row_ids = np.random.choice(norm_ids, del_num, replace=False)
                handled.drop(del_row_ids, inplace=True)
            pair_datas.append(handled)
            cur_time = time.monotonic()
            if cur_time - last_time > 5:
                logger.warning(f'process {i+1} pair data ok')
                last_time = cur_time
        merge_df = pd.concat(pair_datas, ignore_index=True, sort=False)
        merge_df = merge_df.sample(frac=1).reset_index(drop=True)
        test_df = merge_df.sample(frac=0.2, random_state=42)
        train_df = merge_df.drop(index=test_df.index, axis=1)
        test_df.reset_index(drop=True).to_feather(c_test_path, compression='lz4')
        train_df.reset_index(drop=True).to_feather(c_train_path, compression='lz4')
    if len(train_df) > max_train_size:
        logger.warning(f'lgb train size over {max_train_size}, cutting...')
        sample_rate = max_train_size / len(train_df)
        train_df = train_df.sample(frac=sample_rate).reset_index(drop=True)[:max_train_size]
        test_df = test_df.sample(frac=sample_rate).reset_index(drop=True)
        test_df.to_feather(c_test_path, compression='lz4')
        train_df.to_feather(c_train_path, compression='lz4')
    return train_df, test_df


def load_data(as_dataset=True, timeframe: str = '5m', pos_thres: int = 40, neg_thres: int = -30, pred_off: int = 1):
    train_df, test_df = _load_train_df(timeframe, pos_thres, neg_thres, pred_off=pred_off)
    logger.warning('load lightGbm data, train: %d, test: %d', len(train_df), len(test_df))
    test_x, test_y, col_names = process_df(test_df, pos_thres, neg_thres)
    train_x, train_y, col_names = process_df(train_df, pos_thres, neg_thres)
    if as_dataset:
        train_data = lgb.Dataset(train_x, train_y)
        test_data = lgb.Dataset(test_x, test_y, reference=train_data)
        return train_data, test_data, col_names
    return train_x, train_y, test_x, test_y, col_names
