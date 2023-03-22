#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : bad_attempts.py
# Author: anyongjin
# Date  : 2023/2/2
from banbot.compute.classic_inds import *
from banbot.mlpred.consts import *
import pandas_ta as pta
from typing import Dict


class CTI(TFeature):
    '''
    相关趋势指标。反应价格震荡趋势。
    计算速度慢，效果表现不佳
    ********  cti|cti_sub1|cti_sub3 pos relevance  **********
      ('95|49|45', 0.7, 10)
      ('10|0|-103', 0.545, 11)

    ********  cti|cti_sub1|cti_sub3 neg relevance  **********
      ('-19|-49|45', 0.588, 17)
      ('57|-49|0', 0.556, 9)
      ('38|-49|45', 0.538, 13)
    :param df_list:
    :return:
    '''
    def __init__(self):
        super().__init__('cti', 'cti_s13', ind_infos=[
            IndInfo(name='cti', num=21, min=-1, max=1, clus_type='mean', clus_round=lambda x: round(x * 100)),
            IndInfo(name='cti_sub1', num=5, clus_round=lambda x: round(x * 100)),
            IndInfo(name='cti_sub3', num=5, clus_round=lambda x: round(x * 100)),
        ])

    def do_compute(self, df: DataFrame):
        if 'cti' not in df:
            df['cti'] = pta.cti(df['close'], length=10)
            df.loc[df['cti'].isin([np.nan, np.inf, -np.inf]), 'cti'] = 0
        df['cti_sub1'] = df['cti'] - df['cti'].shift()
        df['cti_sub3'] = df['cti'] - df['cti'].shift(3)


class MAMA(TFeature):
    '''
    此指标本身就是一个策略，适用于中长期。不适用于短期行情预测。
    ********  fama_sub1|mama_diff pos relevance  **********
      ('-1|-77', 0.955, 22)
      ('-7|-77', 0.857, 14)
      ('3|23', 0.536, 28)

    ********  fama_sub1|mama_diff neg relevance  **********
      ('8|11', 0.526, 57)
      ('-7|-22', 0.482, 622)
    :param df_list:
    :return:
    '''
    def __init__(self):
        super().__init__('mama', 'mama_fama', ind_infos=[
            IndInfo(name='fama_sub1', num=7, clus_round=lambda x: round(x * 1000)),
            IndInfo(name='mama_diff', num=9, clus_round=lambda x: round(x * 1000)),
        ])

    def do_compute(self, df: DataFrame):
        if 'mama' not in df:
            df['hl2'] = (df['high'] + df['low']) / 2.0
            df['mama'], df['fama'] = ta.MAMA(df['hl2'], 0.5, 0.05)
        df['fama_sub1'] = df['fama'] / df['fama'].shift() - 1
        df['mama_diff'] = ((df['mama'] - df['fama']) / df['hl2'])


class LinearReg(TFeature):
    def __init__(self):
        super().__init__('cti', 'cti_s13', ind_infos=[
            IndInfo(name='linreg_7', num=7, clus_round=lambda x: round(x * 10)),
            IndInfo(name='linreg_7_sub2', num=5, clus_round=lambda x: round(x * 10)),
        ])

    def do_compute(self, df: DataFrame):
        lr_period = 7
        df['hh_20'] = ta.MAX(df['high'], lr_period)
        df['ll_20'] = ta.MIN(df['low'], lr_period)
        df['avg_hh_ll_20'] = (df['hh_20'] + df['ll_20']) / 2.0
        df['avg_close_20'] = ta.SMA(df['close'], lr_period)
        df['avg_val_20'] = (df['avg_hh_ll_20'] + df['avg_close_20']) / 2.0
        df['linreg_7'] = ta.LINEARREG(df['close'] - df['avg_val_20'], lr_period, 0)
        df['linreg_7'] = df['linreg_7'].clip(upper=10, lower=-10)
        df['linreg_7_sub2'] = (df['linreg_7'] - df['linreg_7'].shift(2)).clip(upper=10, lower=-10)


class T3AVG(TFeature):
    '''
    T3移动均线，价格反应比其他均线更快，适合与SMA等均线一起用，作为趋势策略。
    '''
    def __init__(self):
        super().__init__('t3', 't3_avg_s', ind_infos=[
            IndInfo(name='t3_avg_sub1', num=5, clus_round=lambda x: round(x * 1000)),
            IndInfo(name='t3_avg_sub_sma', num=5, clus_round=lambda x: round(x * 1000))
        ])

    def do_compute(self, df: DataFrame):
        df['t3_avg'] = t3_average(df)
        df['t3_avg_sub1'] = df['t3_avg'] / df['t3_avg'].shift(1) - 1
        df['sma_10'] = ta.SMA(df['close'], period=9)
        df['t3_avg_sub_sma'] = df['t3_avg'] / df['sma_10'] - 1

