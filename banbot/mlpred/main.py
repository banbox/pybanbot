#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : main.py
# Author: anyongjin
# Date  : 2023/2/13
import os.path
from pandas import DataFrame
from banbot.mlpred.features import *
from banbot.mlpred.datatools import *


class MicroWave:
    '''
    K线特征提取器。如果需要更改min_acc,应重新执行fea_selection统计特征。否则胜率是错误的。
    '''
    def __init__(self, timeframe: str, model_dir: Optional[str] = None):
        self.timeframe = timeframe
        self.fea_map, self.fea_groups, self.use_feas = get_use_feas(use_policy)
        self.policy_maps: Dict[str, DataFrame] = dict()
        self._model_path = None
        if model_dir:
            self._model_path = os.path.join(model_dir, f'lgb_{self.timeframe}.txt')
        self.model = None
        self._load_policy_maps()
        self._load_model()
        self.fea_cols = list()

    def _load_policy_maps(self):
        config = load_stg_config(self.timeframe)
        if not config:
            raise ValueError(f'config load fail: {self.timeframe}')
        for policy in use_policy:
            po_name = policy.replace('+', '_')
            prob_map: dict = config.get(po_name)
            if not prob_map:
                raise ValueError(f'no prob_map found for policy: {policy}')
            fea_names = policy.split('+') + [po_name + '_p']
            maprows = []
            for k, v in prob_map.items():
                fea_vals = [int(v) for v in k.split('+')]
                fea_vals.append(v[0])
                maprows.append(dict(zip(fea_names, fea_vals)))
            mapdf = DataFrame(maprows)
            mapdf.set_index(policy.split('+'), inplace=True)
            self.policy_maps[policy] = mapdf

    def _load_model(self):
        if not self._model_path:
            return
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=self._model_path)

    def compute_inds(self, df: DataFrame):
        for fea in self.use_feas:
            fea.compute(df)

    def compute_prob(self, df: DataFrame, use_index: bool = False, keep_cols: Optional[list] = None) -> DataFrame:
        for fea in self.use_feas:
            fea.compute(df)
            fea.cluster(df, keep_ind_cols=True, use_index=use_index)
        fea_cols = []
        for policy in use_policy:
            map_df = self.policy_maps.get(policy)
            df = df.merge(map_df, how='left', left_on=policy.split('+'), right_index=True)
            fea_cols.extend(map_df.columns.tolist())
        if not self.fea_cols:
            self.fea_cols.extend(fea_cols)
        if keep_cols:
            fea_cols.extend(keep_cols)
        return df[fea_cols]

    def predict(self, df: DataFrame, keep_cols: Optional[list] = None):
        if not self.model:
            raise ValueError(f'model for {self.timeframe} not load')
        computed = self.compute_prob(df.copy(), keep_cols=keep_cols)
        features = computed[self.fea_cols]
        valid_rows = np.where(features.count(1) >= 1)[0]
        data_x = np.nan_to_num(features.loc[valid_rows].to_numpy(), posinf=0, neginf=0)
        labels = self.model.predict(data_x).argmax(axis=1)

        max_probs = np.zeros(data_x.shape[0], dtype=np.float32)
        neg_ids = np.where(labels == 2)[0]
        max_probs[neg_ids] = np.max(np.absolute(data_x[neg_ids]), 1)
        pos_ids = np.where(labels == 1)[0]
        max_probs[pos_ids] = np.max(data_x[pos_ids], 1)

        if keep_cols:
            df[keep_cols] = computed[keep_cols]
        df['prob'] = 0
        df.loc[valid_rows, 'prob'] = max_probs
        df['action'] = 0
        df.loc[valid_rows, 'action'] = labels


