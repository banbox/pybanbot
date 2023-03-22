#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : indicators.py
# Author: anyongjin
# Date  : 2023/1/28
'''
指标，indicator: 指原始指标计算结果，值通常是连续数值。（SMA，BB，CMF）
特征，feature：基于指标提取组合聚类后的，通常是离散值。（sma3，bb20，cmf）
特征组合，feature group：基于特征的组合对，可发现预测涨跌（natr+crsi）

这里是自定义特征，归一化聚类后的。方便统计训练
【特征相关性分析：spearman】
aroon: rmi 0.621,  mfi 0.544,  ewo 0.526
bb20: sma3 0.759,  kdj 0.713,  srsi 0.683
cmf: rmi 0.461,  ewo 0.447,  wr 0.429
crsi: sma3 0.711,  bb20 0.637,  srsi 0.582
ewo: rmi 0.878,  rsi 0.844,  kama 0.706【重复】
kama: rmi 0.726,  ewo 0.706,  wr 0.639
kdj: bb20 0.713,  sma3 0.657,  srsi 0.545
mfi: rmi 0.714,  ewo 0.653,  rsi 0.608
natr: kdj -0.009,  bb20 -0.012,  mfi -0.018
rmi: ewo 0.878,  rsi 0.799,  kama 0.726【重复】
rsi: ewo 0.844,  rmi 0.799,  wr 0.662
sma3: bb20 0.759,  crsi 0.711,  srsi 0.664
srsi: wr 0.798,  bb20 0.683,  sma3 0.664
wr: srsi 0.798,  rmi 0.694,  rsi 0.662


【特征相关性分析：MIC】
aroon: rmi 0.452,  wr 0.355,  bb20 0.347
bb20: rmi 0.573,  wr 0.550,  sma3 0.474
cmf: ewo 0.145,  bb20 0.139,  rmi 0.133
crsi: sma3 0.553,  rsi 0.545,  srsi 0.543
ewo: rmi 0.660,  rsi 0.655,  wr 0.453【重复】
kama: rmi 0.410,  ewo 0.364,  bb20 0.338
kdj: wr 0.512,  bb20 0.413,  srsi 0.406
mfi: rmi 0.332,  bb20 0.316,  aroon 0.314
natr: ewo 0.294,  sma3 0.233,  kama 0.185
rmi: ewo 0.660,  bb20 0.573,  rsi 0.504
rsi: ewo 0.655,  crsi 0.545,  rmi 0.504
sma3: crsi 0.553,  srsi 0.503,  bb20 0.474
srsi: crsi 0.543,  sma3 0.503,  wr 0.449
wr: bb20 0.550,  kdj 0.512,  rmi 0.491


【特征组合的结果相关度】
[bb20]
pos_true: 0.69/sma3_pt  pos_false: 0.46/sma3_pf  neg_true: 0.81/sma3_nt  neg_false: 0.59/sma3_nf
[sma3]
pos_true: 0.97/sma3+rsi_pt  pos_false: 0.95/sma3+rsi_pf  neg_true: 0.98/sma3+rsi_nt  neg_false: 0.96/sma3+rsi_nf
[kama]
pos_true: 0.90/sma3_pt  pos_false: 0.81/sma3_pf  neg_true: 0.69/sma3_nt  neg_false: 0.51/sma3_nf
[natr+crsi]
pos_true: 0.74/sma3_pt  pos_false: 0.64/sma3_pf  neg_true: 0.75/sma3_nt  neg_false: 0.53/sma3_nf
[natr+srsi]
pos_true: 0.72/sma3+rsi_pt  pos_false: 0.51/sma3+rsi_pf  neg_true: 0.75/sma3_nt  neg_false: 0.55/sma3_nf
[natr+kdj]
pos_true: 0.65/sma3+rsi_pt  pos_false: 0.39/sma3+rsi_pf  neg_true: 0.69/sma3_nt  neg_false: 0.46/sma3_nf
[sma3+natr+cmf]
pos_true: 0.83/sma3+rsi_pt  pos_false: 0.78/sma3+rsi_pf  neg_true: 0.91/sma3_nt  neg_false: 0.89/sma3_nf
[sma3+natr+mfi]
pos_true: 0.82/sma3+rsi_pt  pos_false: 0.77/sma3+rsi_pf  neg_true: 0.91/sma3_nt  neg_false: 0.89/sma3_nf
[sma3+natr+wr]
pos_true: 0.94/sma3+rsi_pt  pos_false: 0.92/sma3_pf  neg_true: 0.96/sma3_nt  neg_false: 0.96/sma3_nf
[sma3+rsi]
pos_true: 0.95/sma3_pt  pos_false: 0.93/sma3_pf  neg_true: 1.00/sma3_nt  neg_false: 1.00/sma3_nf
[rsi+aroon]
pos_true: 0.79/sma3_pt  pos_false: 0.58/sma3_pf  neg_true: 0.85/sma3_nt  neg_false: 0.57/sma3_nf
'''
from banbot.compute.classic_inds import *
from banbot.mlpred.consts import *
from sklearn.preprocessing import MinMaxScaler
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.indicators import RMI
from typing import Dict, Tuple


use_policy = [
    'bb20',
    'sma3',
    'kama',
    'natr+crsi',
    'natr+srsi',
    'natr+kdj',
    'sma3+natr+cmf',
    'sma3+natr+mfi',
    'sma3+natr+wr',
    'sma3+rsi',
    'rsi+aroon',
]


def get_fea_map() -> Dict[str, TFeature]:
    import sys, inspect
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    cls_map = dict()
    for (name, cls) in clsmembers:
        if not issubclass(cls, TFeature) or name == 'TFeature':
            continue
        if not cls.enable:
            continue
        instance = cls()
        cls_map[instance.name] = instance
    fea_names = ','.join(sorted(cls_map.keys()))
    logger.warning(f'features loaded: {fea_names}')
    return cls_map


def get_use_feas(groups: List[str]) -> Tuple[Dict[str, TFeature], List[List[TFeature]], List[TFeature]]:
    fea_map: Dict[str, TFeature] = get_fea_map()
    fea_groups, use_feas = [], set()
    for gp in groups:
        gp_fea = []
        for n in gp.split('+'):
            fea = fea_map.get(n)
            if not fea:
                raise ValueError(f'invalid feature: {n}')
            gp_fea.append(fea)
            use_feas.add(fea)
        fea_groups.append(gp_fea)
    use_feas = list(use_feas)
    return fea_map, fea_groups, use_feas


class BB20(TFeature):

    def __init__(self):
        super().__init__('bb20', 'bb20_sub_1_3_dir', [
            IndInfo(name='bb20_sub1', num=5, clus_round=lambda x: round(x * 10)),
            IndInfo(name='bb20_sub3', num=5, clus_round=lambda x: round(x * 10)),
            IndInfo(name='bb20_facwid', num=11, clus_round=lambda x: round(x * 100))
        ])

    def do_compute(self, df: DataFrame):
        if 'typical' not in df:
            df['typical'] = qtpylib.typical_price(df)
        if 'bb20_mid' not in df:
            bb_20_std2 = qtpylib.bollinger_bands(df['typical'], window=20, stds=2)
            df['bb20_mid'] = bb_20_std2['mid']
            df['bb20_low'] = bb_20_std2['lower']
            df['bb20_upp'] = bb_20_std2['upper']
        if 'bb20_dir' not in df:
            df['bb20_width'] = df['bb20_upp'] - df['bb20_low']
            if 'ema_3' not in df:
                df['ema_3'] = ta.EMA(df, timeperiod=3)
            df['bb20_dir'] = (df['ema_3'] - df['bb20_low']) / df['bb20_width'] * 2 - 1
            df['bb20_dir'] = df['bb20_dir'].clip(lower=-1.2, upper=1.2)
            # 有些没有交易的行，bb20_width是0，会导致bb20_dir变成nan
            df.loc[(df['volume'] == 0) | (df['bb20_width'] == 0), 'bb20_dir'] = 0
        bb20_fac = (df['bb20_width'] / df['ema_3']).values.reshape(-1, 1)
        bb20_fac = np.nan_to_num(bb20_fac)
        df['bb20_wid'] = MinMaxScaler().fit_transform(bb20_fac)
        df['bb20_facwid'] = df['bb20_dir'] * df['bb20_wid']  # [-1.2, 1.2]
        df['bb20_sub1'] = df['bb20_dir'] - df['bb20_dir'].shift(1)
        df['bb20_sub3'] = df['bb20_dir'] - df['bb20_dir'].shift(3)


class SMA3(TFeature):
    def __init__(self):
        super().__init__('sma3', 'close_sma3', ind_infos=[
            IndInfo(name='sma3_chg', num=15, clus_round=lambda x: round(x * 3000)),
            IndInfo(name='sma3_chg_s1', num=7, clus_round=lambda x: round(x * 1000)),
        ])

    def do_compute(self, df: DataFrame):
        if 'sma_3' not in df:
            df['sma_3'] = ta.SMA(df, timeperiod=3)
        df['sma3_chg'] = df['close'] / df['sma_3'] - 1
        df['sma3_chg_s1'] = df['sma3_chg'].shift()


class CMF20(TFeature):
    '''
    横盘震荡期间使用，结合NATR较好。
    natr+sma3+cmf
    '''
    def __init__(self):
        super().__init__('cmf', 'cmf20_s3', ind_infos=[
            IndInfo(name='raw_cmf', num=11, clus_round=lambda x: round(x * 100)),
            IndInfo(name='cmf_sub3', num=5, clus_round=lambda x: round(x * 10)),
            # IndInfo(name='r7max_d', num=5, clus_round=lambda x: round(x)),
        ])

    def do_compute(self, df: DataFrame):
        raw_key = 'raw_cmf'
        df[raw_key] = chaikin_money_flow(df, fillna=True)
        df['cmf_sma3'] = ta.SMA(df[raw_key], timeperiod=3)
        df['cmf_sub3'] = df[raw_key] - df[raw_key].shift(3)

        # 超过或低于100内极值点，出现概率较低。暂不作为特征。
        # # cmf|over_100max|cmf_sub1
        # over_100max = df['close'] / df['close'].rolling(100).max().shift(5) - 1
        # over_100max[over_100max < -0.03] = -0.03
        # df['over_100max'] = over_100max

        # 近期极值点比较，没有增加效果，暂不使用
        # r7max = df[raw_key].rolling(7).max().shift(1)
        # r7max[(r7max < 0.02) & (r7max >= 0)] = 0.02
        # r7max[(r7max >= -0.02) & (r7max < 0)] = -0.02
        # df['r7max_d'] = r7max / r7max.shift(8) - 1
        # df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)


class NATR(TFeature):
    '''
    判断震荡、趋势强弱。单独使用无任何效果。
    '''
    def __init__(self):
        super().__init__('natr', 'natr14_s7', ind_infos=[
            IndInfo(name='atr', num=11, clus_round=lambda x: round(x * 100)),
            IndInfo(name='atr_sub7', num=5, clus_round=lambda x: round(x * 100)),
        ])

    def do_compute(self, df: DataFrame):
        if 'atr' not in df:
            df['atr'] = natr(df)
        df['atr_sub7'] = df['atr'] - df['atr'].shift(7)


class CRSI(TFeature):
    '''
    针对RSI的优化，横盘整理期间使用，仅检测做空信号。
    natr+crsi
    '''
    def __init__(self):
        super().__init__('crsi', 'crsi', ind_infos=[
            IndInfo(name='raw_crsi', num=33, min=0, max=100, clus_type='mean', clus_round=lambda x: round(x)),
        ])

    def do_compute(self, df: DataFrame):
        raw_key = 'raw_crsi'
        if raw_key not in df:
            df[raw_key] = ConnorsRSI(df)
        df.loc[(df[raw_key] >= 26) & (df[raw_key] <= 74), raw_key] = 50


class EWORich(TFeature):
    '''
    和其他特征相关性略高，暂不适用
    '''
    enable = False

    def __init__(self):
        super().__init__('ewo', 'ewo_s_1_3', ind_infos=[
            IndInfo(name='raw_ewo', num=11, clus_round=lambda x: round(x * 1000)),
            IndInfo(name='ewo_sub1', num=5, clus_round=lambda x: round(x * 1000)),
            IndInfo(name='ewo_sub3', num=5, clus_round=lambda x: round(x * 1000)),
        ])

    def do_compute(self, df: DataFrame):
        raw_key = 'raw_ewo'
        if raw_key not in df:
            df[raw_key] = ewo(df)
        df['ewo_sub1'] = df[raw_key] - df[raw_key].shift()
        df['ewo_sma3'] = ta.SMA(df[raw_key], period=3)
        df['ewo_sub3'] = df[raw_key] - df['ewo_sma3'].shift(3)


class KAMA(TFeature):
    def __init__(self):
        super().__init__('kama', 'kama_f_s1_10', ind_infos=[
            IndInfo(name='km_fac', num=9, clus_round=lambda x: round(x * 1000)),
            IndInfo(name='km_sub1', num=5, clus_round=lambda x: round(x * 1000)),
            IndInfo(name='km_sub10', num=5, clus_round=lambda x: round(x * 1000)),
        ])

    def do_compute(self, df: DataFrame):
        raw_key = 'raw_kama'
        if raw_key not in df:
            df[raw_key] = ta.KAMA(df['close'], 3)
        if 'kama_10' not in df:
            df['kama_10'] = ta.KAMA(df['close'], 10)
        df['km_fac'] = df[raw_key] / df['kama_10'] - 1
        df['km_sub1'] = df[raw_key] / df[raw_key].shift() - 1
        df['km_sub10'] = df['kama_10'] / df['kama_10'].shift() - 1


class SRSI(TFeature):
    def __init__(self):
        super().__init__('srsi', 'srsi_k_sd', ind_infos=[
            IndInfo(name='srsi_k', num=21, min=0, max=100, clus_type='mean', clus_round=lambda x: round(x)),
            IndInfo(name='srsi_k_sub_d', num=5, clus_round=lambda x: round(x)),
        ])

    def do_compute(self, df: DataFrame):
        if 'srsi_k' not in df:
            stoch = ta.STOCHRSI(df, 14, 14, 3, 3)
            df['srsi_k'] = stoch['fastk']
            df['srsi_d'] = stoch['fastd']
            df['srsi_k_sub_d'] = df['srsi_k'] - df['srsi_d']


class MFI(TFeature):
    def __init__(self):
        super().__init__('mfi', 'mfi_s1', ind_infos=[
            IndInfo(name='raw_mfi', num=15, min=0, max=100, clus_type='mean', clus_round=lambda x: round(x)),
            IndInfo(name='mfi_sub1', num=5, clus_round=lambda x: round(x))
        ])

    def do_compute(self, df: DataFrame):
        raw_key = 'raw_mfi'
        if raw_key not in df:
            df[raw_key] = ta.MFI(df)
        df['mfi_sub1'] = df[raw_key] - df[raw_key].shift()


class WR(TFeature):
    '''
    检测超买超卖。在趋势行情中表现好。结合SMA等趋势指标使用
    '''
    def __init__(self):
        super().__init__('wr', 'wr14', ind_infos=[
            IndInfo(name='raw_wr', num=11, min=-100, max=0, clus_type='mean', clus_round=lambda x: round(x)),
        ])

    def do_compute(self, df: DataFrame):
        raw_key = 'raw_wr'
        df[raw_key] = williams_r(df)
        df[raw_key] = df[raw_key].replace([np.inf, -np.inf, np.nan], 0)


class RMIRich(TFeature):
    '''
    震荡指标，结合SMA/NATR对短期市场的做多和做空均有良好检测效果
    和其他特征重复性略高，且计算略耗时。暂不适用
    '''
    enable = False

    def __init__(self):
        super().__init__('rmi', 'rmi_s1', ind_infos=[
            IndInfo(name='raw_rmi', num=15, min=0, max=100, clus_type='mean', clus_round=lambda x: round(x)),
            IndInfo(name='rmi_sub1', num=5, clus_round=lambda x: round(x))
        ])

    def do_compute(self, df: DataFrame):
        raw_key = 'raw_rmi'
        if raw_key not in df:
            df[raw_key] = RMI(df)
        df['rmi_sub1'] = df[raw_key] - df[raw_key].shift()


class RSI(TFeature):
    def __init__(self):
        super().__init__('rsi', 'rsi_s1', ind_infos=[
            IndInfo(name='rsi_25', num=15, min=0, max=100, clus_type='mean', clus_round=lambda x: round(x)),
            IndInfo(name='rsi_25_sub1', num=5, clus_round=lambda x: round(x))
        ])

    def do_compute(self, df: DataFrame):
        raw_key = 'rsi_25'
        df[raw_key] = ta.RSI(df, timeperiod=25)
        df['rsi_25_sub1'] = df[raw_key] - df[raw_key].shift(1)
        bad_ids = (df[raw_key] >= 34) & (df[raw_key] <= 66)
        df.loc[bad_ids, raw_key] = 50
        df.loc[bad_ids, 'rsi_25_sub1'] = 0


class KDJ(TFeature):
    def __init__(self):
        super().__init__('kdj', 'kdj_s1', ind_infos=[
            IndInfo(name='kdj_k_sub1', num=5, clus_round=lambda x: round(x)),
            IndInfo(name='kdj_j_d', num=5, clus_round=lambda x: round(x)),
            IndInfo(name='slowk', num=13, min=0, max=100, clus_type='mean', clus_round=lambda x: round(x)),
        ])

    def do_compute(self, df: DataFrame):
        df['slowk'], df['slowd'] = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=9, slowk_period=3,
                                            slowk_matype=0, slowd_period=3, slowd_matype=0)
        df['kdj_j_d'] = df['slowk'] - df['slowd']
        df['kdj_k_sub1'] = df['slowk'] - df['slowk'].shift()


class Aroon(TFeature):
    def __init__(self):
        super().__init__('aroon', 'aroon', ind_infos=[
            IndInfo(name='arron_up', num=11, min=0, max=100, clus_type='mean', clus_round=lambda x: round(x)),
            IndInfo(name='aroon_down', num=11, min=0, max=100, clus_type='mean', clus_round=lambda x: round(x)),
        ])

    def do_compute(self, df: DataFrame):
        if 'aroon_down' not in df:
            df['aroon_down'], df['arron_up'] = ta.AROON(df['high'], df['low'], timeperiod=14)

