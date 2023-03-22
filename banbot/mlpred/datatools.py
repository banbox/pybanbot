#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : datatools.py
# Author: anyongjin
# Date  : 2023/1/29
'''
数据加载、配置读写、聚类，通用DF数据处理
'''
import json
import os
import time

from typing import List, Tuple, Dict, Optional
from pathlib import Path
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
from freqtrade.data import history
from freqtrade.configuration import TimeRange
from banbot.compute.classic_inds import *
from banbot.mlpred.consts import *

data_dir = Path('E:/trade/freqtd_data/user_data/data_210601/binance')
timerange = TimeRange.parse_timerange('20210601-20230101')
cfg_map = dict()
# 计算机最大能处理的行数，这里测试16G内存机器。
max_df_rows = 45000000


def auto_clus(df: DataFrame, inds: List[IndInfo], timeframe: str):
    for ind in inds:
        if ind.clus_type == 'mean':
            new_col, centers = mean_cluster_col(df[ind.name], ind.num, ind.clus_round, ind.min, ind.max)
        else:
            new_col, centers = cluster_col_kmeans(timeframe, df[ind.name], ind.num, ind.clus_round, ind.clus_key)
        df[ind.name] = new_col
        ind.centers = centers
    return df


def load_stg_config(timeframe: str, key: Optional[str] = None):
    cfg_path = os.path.join(os.path.dirname(__file__), f'strategy_{timeframe}.json')
    global cfg_map
    cfg_cache = cfg_map.get(timeframe)
    if cfg_cache is None:
        if not os.path.isfile(cfg_path):
            return None
        import orjson
        file_data = open(cfg_path, 'rb').read()
        if not file_data:
            return None
        cfg_cache = orjson.loads(file_data)
        cfg_map[timeframe] = cfg_cache
    if not key or not cfg_cache:
        return cfg_cache
    return cfg_cache.get(key)


def save_stg_config(timeframe: str, key: str, val):
    cfg_path = os.path.join(os.path.dirname(__file__), f'strategy_{timeframe}.json')
    config = load_stg_config(timeframe) or dict()
    config[key] = val
    import orjson
    with open(cfg_path, 'wb') as fout:
        fout.write(orjson.dumps(config))


def col_range_map(col: Series, case_list: List[Tuple[float, float, int]]) -> np.ndarray:
    '''
    根据范围将某列数据修改为指定的值。范围映射列表应为升序
    :param col:
    :param case_list: [(start, end, index), ...]
    :return: 映射后的numpy数组
    '''
    data_col = col.to_numpy()
    dtype = np.int32 if isinstance(case_list[0][2], int) else np.float32
    new_col = np.zeros(col.shape, dtype=dtype)
    start_fill, end_fill = None, None
    for start_val, end_val, map_val in case_list:
        if start_val is None:
            if start_fill is None:
                new_col[data_col <= end_val] = map_val
                start_fill = end_val
            else:
                raise ValueError(f'duplicate start range found: {col.name} {start_fill} -- {end_val}')
        elif end_val is None:
            if end_fill is None:
                new_col[data_col > start_val] = map_val
                end_fill = start_val
            else:
                raise ValueError(f'duplicate end range found: {col.name} {end_fill} -- {start_val}')
        else:
            new_col[(data_col > start_val) & (data_col <= end_val)] = map_val
    return new_col


def mean_cluster_col(col: Series, cls_num: int, val_handle, min_val: int, max_val: int):
    '''
    对给定列的值进行均匀聚类。可指定聚类数量。（按最大最小值均匀分割聚类）
    :param col:
    :param cls_num:
    :param val_handle:
    :param min_val:
    :param max_val:
    :return:
    '''
    start = time.time()
    cls_step = (max_val - min_val) / cls_num
    cur_center = min_val + cls_step / 2
    case_list, centers = [], []
    start_val = min_val
    for i in range(cls_num):
        end_val = None if i + 1 >= cls_num else start_val + cls_step
        if i == 0:
            start_val = None
        case_list.append((start_val, end_val, len(centers)))
        centers.append(val_handle(cur_center))
        start_val = end_val
        cur_center += cls_step

    new_col = col_range_map(col, case_list)
    cost = time.time() - start
    if cost >= 0.05:
        logger.warning(f'mean cluster col {col.name}, cost: {cost: .2f}, centers: {centers}')
    return new_col, centers


def do_cluster(timeframe: str, col: Series, cls_num: int, val_handle, cache_key: str) -> dict:
    from sklearn.cluster import KMeans
    train_x = col.to_numpy()
    nan_ids = np.where(~np.isfinite(train_x))[0].tolist()
    if nan_ids:
        raise ValueError(f'{len(nan_ids)} nan values found while cluster `{col.name}` : {nan_ids[:10]}')
    org_x = np.sort(train_x, kind='mergesort')
    max_x_num, max_iter = 1000000, 200
    if cls_num <= 5:
        # 聚类数量较小时，数据复杂度不高，可降低采样和迭代次数
        max_x_num, max_iter = 300000, 130
    if len(train_x) > 1.2 * max_x_num:
        train_x = np.random.choice(train_x, max_x_num)
    train_x = train_x.reshape((-1, 1))
    kmeans = KMeans(n_clusters=cls_num, max_iter=max_iter, random_state=0)
    kmeans.fit(train_x)
    org_centers = [v[0] for v in kmeans.cluster_centers_]
    centers = [val_handle(v[0]) for v in kmeans.cluster_centers_]
    if len(set(centers)) < len(centers):
        raise ValueError(f'invalid round func for `{col.name}`: {centers}  --  {org_centers}')
    if any(c for c in centers if isinstance(c, float)):
        raise ValueError(f'`{col.name}` round func should return int type!')
    cur_labels = kmeans.predict(org_x.reshape((-1, 1)))
    diff_idxs = (np.nonzero(cur_labels[1:] - cur_labels[:-1])[0] + 1).tolist()
    map_list, cur_start = [], None
    for idx in diff_idxs:
        cur_end = float((org_x[idx] + org_x[idx - 1]) / 2)
        map_list.append((cur_start, cur_end, int(cur_labels[idx - 1])))
        cur_start = cur_end
    map_list.append((cur_start, None, int(cur_labels[-1])))
    cache_cfg = dict(train_size=len(col), range_map=map_list, centers=centers)
    save_stg_config(timeframe, cache_key, cache_cfg)
    return cache_cfg


def cluster_col_kmeans(timeframe: str, col: Series, cls_num: int, val_handle, clus_key: str = None):
    '''
    对给定列的值进行聚类。可指定聚类数量。
    在调用此方法前应该去除nan值；从样本随机挑选50W个聚类
    如果有已缓存的kmeans模型，且模型足够宽泛，则使用缓存模型，否则训练模型
    :param col: 要聚类的列
    :param cls_num:
    :param val_handle: 聚类中心的处理函数，可用于四舍五入
    :param clus_key: 聚类缓存key，避免同样的数据重复聚类
    :return:
    '''
    start = time.time()
    if not clus_key:
        clus_key = col.name
    cache_key = f'clus_{clus_key}_{cls_num}'
    cache_cfg = load_stg_config(timeframe, cache_key) or dict()
    if cache_cfg and cache_cfg.get('train_size', 0) * 1.1 < len(col):
        # 如果缓存模型数据比当前数据低，则重新训练
        logger.warning(f'kmeans train_size for {clus_key} is too small, skipping...')
        cache_cfg = dict()
    if not cache_cfg:
        logger.warning(f'preform cluster for {clus_key} to {cls_num} groups')
        cache_cfg = do_cluster(timeframe, col, cls_num, val_handle, cache_key)
    pos_list = cache_cfg.get('range_map')
    new_col = col_range_map(col, pos_list)
    cost = time.time() - start
    centers = cache_cfg.get("centers")
    if cost >= 0.05:
        logger.warning(f'kmeans cluster col {col.name}, cost: {cost: .2f}, centers: {centers}')
    return new_col, centers


def calc_next_profit(df: pd.DataFrame, offset: int = 1):
    if 'typical' not in df:
        df['typical'] = qtpylib.typical_price(df)
    # 未来预测偏移，为1或者2比较合适。此时平均窗口选3,；如果预测偏移超过4，窗口才可选5平滑。
    roll_size = 5 if offset > 3 else 3
    future = df['typical'].rolling(roll_size, center=True).mean().shift(-offset)
    # 当前价格不能直接用close，否则随机性太强，匹配的行数太少。
    # 故这里使用SMA平滑。窗口为未来窗口的60%为宜。
    # avg_now = ta.SMA(df['close'], round(roll_size * 0.6))
    df['profit'] = (future / df['close'] - 1) * 10000
    df['profit'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    df['profit'] = df['profit'].round(0).astype(np.int32)
    to_low_precision(df)


def update_df_valid_range(df: pd.DataFrame, cols=None):
    '''
    更新DataFrame的有效行区间
    :param df:
    :param cols: 指定需要检查的列
    :return:
    '''
    if cols is None:
        cols = df.columns.to_list()
    a, b = df[cols].apply(pd.Series.first_valid_index).max(), df[cols].apply(pd.Series.last_valid_index).min()
    df.valid_start = max(a, df.valid_start)
    df.valid_end = min(b, df.valid_end)
    for col in cols:
        bad_ids = df[df[col].isin([np.nan, np.inf, -np.inf])].index.tolist()
        bad_ids = [v for v in bad_ids if df.valid_start <= v <= df.valid_end]
        if bad_ids:
            raise ValueError(f'{len(bad_ids)} nan values found in `{col}` : {bad_ids[:30]}')


def load_pair(pair: str, timeframe: str = '5m') -> pd.DataFrame:
    df = history.load_pair_history(pair, timeframe, data_dir, timerange=timerange)
    if not len(df):
        raise ValueError('no data loaded')
    return df


def to_low_precision(df: DataFrame):
    float64_cols = df.select_dtypes(include='float64').columns
    mapper = {col_name: np.float32 for col_name in float64_cols}
    int64_cols = df.select_dtypes(include='int64').columns
    for col in int64_cols:
        mapper[col] = np.int32
    if mapper:
        map_df = df.astype(mapper)
        chg_names = list(mapper.keys())
        df[chg_names] = map_df[chg_names]


def load_all_pairs(timeframe: str = '5m', min_len=1000, pair_num: int = 0) -> List[pd.DataFrame]:
    cache_path = data_dir / f'all_{timeframe}.feather'
    cache_info_path = data_dir / f'all_{timeframe}.json'
    names = sorted(os.listdir(data_dir))
    names = [n for n in names if n.find(f'-{timeframe}.') > 0]
    df_list, total_rows, merge_offs = [], 0, []
    # read cache
    start = time.time()
    pair_max_row = int(max_df_rows / len(names))
    if cache_path.is_file():
        merge_df = pd.read_feather(cache_path)
        to_low_precision(merge_df)
        merge_offs = json.load(open(cache_info_path, 'r', encoding='utf-8'))
        total_rows = len(merge_df)
        added_names = {info[0] for info in merge_offs}
        new_names = set(names) - added_names
        pair_max_row = int(max_df_rows / (len(merge_offs) + len(new_names)))
        for info in merge_offs:
            df = merge_df[info[1]: info[2]].reset_index(drop=True)
            if pair_max_row < len(df):
                df = df[-pair_max_row:].reset_index(drop=True)
            df.valid_start = 0
            df.valid_end = len(df)
            update_df_valid_range(df)
            df_list.append(df)
        if not new_names:
            load_cost = time.time() - start
            if pair_num:
                df_list = df_list[:pair_num]
            logger.warning(f'{len(df_list)} pairs loaded from cache, total {total_rows} rows, cost: {load_cost:.2f}s')
            return df_list
        logger.warning(f'{len(df_list)} pairs loaded from cache, new {len(new_names)} pairs loading...')
        start = time.time()
        names = sorted(new_names)
    before_num = len(df_list)
    for name in names:
        pair_end = name.find(f'-{timeframe}.')
        data_format = name[name.rfind('.') + 1:]
        df = history.load_pair_history(name[:pair_end], timeframe, data_dir,
                                       timerange=timerange, data_format=data_format)
        if len(df) < min_len:
            # merge_offs.append((name, total_rows, total_rows))
            continue
        if pair_max_row > len(df):
            df = df[-pair_max_row:].reset_index(drop=True)
        to_low_precision(df)
        df.valid_start = 0
        df.valid_end = len(df)
        update_df_valid_range(df)
        off_start = total_rows
        total_rows += len(df)
        merge_offs.append((name, off_start, total_rows))
        df_list.append(df)
    if not df_list:
        raise ValueError(f'no of {len(names)} pairs loaded')
    load_cost = time.time() - start
    new_load = len(df_list) - before_num
    logger.warning(f'{new_load} new pairs loaded, total {total_rows} rows, cost: {load_cost:.2f}s')
    if new_load:
        # write to cache
        merge_df = pd.concat(df_list, ignore_index=True, sort=False)
        merge_df.to_feather(cache_path)
        json.dump(merge_offs, open(cache_info_path, 'w', encoding='utf-8'), ensure_ascii=False)
    if pair_num:
        df_list = df_list[:pair_num]
    return df_list


