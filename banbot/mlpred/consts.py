#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : consts.py
# Author: anyongjin
# Date  : 2023/2/10
import time
from logging import getLogger
from typing import List
from pandas import DataFrame
logger = getLogger(__name__)


class IndInfo:
    def __init__(self, **kwargs):
        self.name: str = kwargs.get('name')
        self.min = kwargs.get('min')
        self.max = kwargs.get('max')
        self.num: int = kwargs.get('num', 0)
        self.clus_key = kwargs.get('clus_key')
        self.clus_type = kwargs.get('clus_type')
        self.clus_round = kwargs.get('clus_round')
        self.centers = []


class TFeature:
    enable = True

    def __init__(self, name: str, full_name: str = None, ind_infos: List[IndInfo] = None):
        self.name: str = name
        self.full_name: str = full_name or name
        self.ind_infos: List[IndInfo] = ind_infos or []
        self.col_names = [c.name for c in self.ind_infos]
        self.add_cols = set()  # 计算指标后冗余的列（包含ind_infos中的）
        self.clus_merge_enums = dict()

    def compute(self, df: DataFrame):
        from banbot.compute.datatools import to_low_precision
        old_cols = set(df.columns.tolist())
        self.do_compute(df)
        to_low_precision(df)
        after_cols = set(df.columns.tolist())
        self.add_cols = after_cols - old_cols - {self.name}

    def do_compute(self, df: DataFrame):
        raise NotImplementedError('Indicator.compute is not implement!')

    def del_useless_cols(self, df: DataFrame, keep_ind_cols: bool = False):
        del_cols = self.add_cols.intersection(df.columns.tolist())
        if keep_ind_cols:
            del_cols -= set(c.name for c in self.ind_infos)
        if not del_cols:
            return
        df.drop(columns=list(del_cols), inplace=True)

    def _load_clus_enums(self):
        if len(self.ind_infos) <= 1 or self.clus_merge_enums:
            return
        enum_list = []
        ind_shape = [len(ind.centers) for ind in self.ind_infos]
        ind_dims = [len(str(len(ind.centers) - 1)) for ind in self.ind_infos]
        ind_dims.insert(0, 0)
        cur_pos = [0] * len(self.ind_infos)
        while True:
            sum_val = cur_pos[0]
            for i in range(1, len(self.ind_infos)):
                sum_val = sum_val * pow(10, ind_dims[i]) + cur_pos[i]
            self.clus_merge_enums[sum_val] = len(enum_list)
            enum_list.append(sum_val)
            for i in range(len(self.ind_infos)):
                cur_pos[i] += 1
                if cur_pos[i] < ind_shape[i]:
                    break
                if i + 1 >= len(self.ind_infos):
                    return
                cur_pos[i] = 0

    def cluster(self, df: DataFrame, keep_ind_cols: bool = False, use_index: bool = False,
                timeframe: str = '5m'):
        from banbot.compute.datatools import auto_clus, to_low_precision
        # 对此特征用到的指标进行聚类，减少复杂度
        auto_clus(df, self.ind_infos, timeframe)
        if len(self.ind_infos) > 1:
            # 如果多个指标，合并为一列，提高效率
            # 使用整型累加特征，避免转字符串，效率提升100倍+
            self._load_clus_enums()
            merge_col, last_dim = None, 0
            for ind in self.ind_infos:
                if merge_col is None:
                    merge_col = df[ind.name]
                else:
                    merge_col = merge_col * pow(10, last_dim) + df[ind.name]
                # 记录存储当前特征的最大位数
                last_dim = len(str(len(ind.centers) - 1))
            if use_index:
                df[self.name] = merge_col.apply(lambda x: self.clus_merge_enums[x])
            else:
                df[self.name] = merge_col
        else:
            df[self.name] = df[self.ind_infos[0].name]
        self.del_useless_cols(df, keep_ind_cols)
        to_low_precision(df)
