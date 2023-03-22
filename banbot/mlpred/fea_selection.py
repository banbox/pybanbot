#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : ind_mining.py
# Author: anyongjin
# Date  : 2023/1/28
'''
测试指标组合效果。
指标特征相关性分析
'''
import os
import pickle
import time

import numpy as np

from banbot.mlpred.datatools import *
from banbot.mlpred import features as fts
from banbot.compute.utils import *
import statistics

'''特征发现：指标组合胜率测试'''


def analysis_inds_group(group_list: List[str], pos_thres=35, neg_thres=-30, min_acc: float = 0.58,
                        pred_off=2, show_recall: bool = True, timeframe: str = '5m'):
    '''
    分析指标组合的效果
    :param group_list: ['sma3+cmf', 'cmf+sma_10']
    :param pos_thres: 预计做多阈值
    :param neg_thres: 预计做多阈值
    :param min_acc: 显示的最低置信度
    :param pred_off: 预测未来的偏移
    :param show_recall: 是否显示指标召回相关度分析
    :param timeframe: 时间维度，默认5m
    :return:
    '''
    fea_map, fea_groups, use_feas = fts.get_use_feas(group_list)
    pair_datas = load_all_pairs(timeframe=timeframe)
    merge_df = calc_clus_feas(pair_datas, use_feas, profit_off=pred_off, timeframe=timeframe)
    if not show_recall:
        # 不显示指标召回时，删除冗余列，节省内存
        drop_cols = set(iv.name for f in fea_map.values() for iv in f.ind_infos)
        drop_cols = drop_cols.intersection(merge_df.columns.tolist())
        drop_cols -= set(fea_map.keys())
        merge_df.drop(columns=drop_cols, inplace=True)
    # 分析特征组合的胜率
    merge_df['idx'] = range(len(merge_df))
    profile_args = dict(pos_thres=pos_thres, neg_thres=neg_thres, min_acc=min_acc, show_recall=show_recall,
                        timeframe=timeframe)
    sta_cache = 'policy_stas.pkl'
    cache_file = open(sta_cache, 'wb')
    for gp in fea_groups:
        gp_cache = profile_ind_group(merge_df, gp, **profile_args)
        pickle.dump(gp_cache, cache_file)
    del merge_df
    cache_file.close()
    policy_detail = list(load_pickle_objects(sta_cache))
    # 计算特征组合的结果相关度
    analysis_fea_group_intros(group_list, policy_detail)
    # 删除临时文件
    os.remove(sta_cache)
    bell()


def load_pickle_objects(file_path):
    with open(file_path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def analysis_fea_group_intros(groups: List[str], detail: List[Tuple[set, set, set, set]]):
    '''
    分析各特征组合的结果相关度。
    :param groups:
    :param detail:
    :return:
    '''
    for rid, row in enumerate(detail):
        pos_true, pos_false, neg_true, neg_false = row
        pt_rate, pt_tag = 0, None
        pf_rate, pf_tag = 0, None
        nt_rate, nt_tag = 0, None
        nf_rate, nf_tag = 0, None
        for oid, orow in enumerate(detail):
            if oid == rid:
                continue
            opos_true, opos_false, oneg_true, oneg_false = orow
            oname = groups[oid]
            if pos_true:
                cur_pt_rate = len(pos_true.intersection(opos_true)) / len(pos_true)
                if cur_pt_rate > pt_rate:
                    pt_rate = cur_pt_rate
                    pt_tag = f'{oname}_pt'
                cur_pt_rate = len(pos_true.intersection(oneg_false)) / len(pos_true)
                if cur_pt_rate > pt_rate:
                    pt_rate = cur_pt_rate
                    pt_tag = f'{oname}_nf'
            if pos_false:
                cur_pf_rate = len(pos_false.intersection(oneg_true)) / len(pos_false)
                if cur_pf_rate > pf_rate:
                    pf_rate = cur_pf_rate
                    pf_tag = f'{oname}_nt'
                cur_pf_rate = len(pos_false.intersection(opos_false)) / len(pos_false)
                if cur_pf_rate > pf_rate:
                    pf_rate = cur_pf_rate
                    pf_tag = f'{oname}_pf'
            if neg_true:
                cur_nt_rate = len(neg_true.intersection(opos_false)) / len(neg_true)
                if cur_nt_rate > nt_rate:
                    nt_rate = cur_nt_rate
                    nt_tag = f'{oname}_pf'
                cur_nt_rate = len(neg_true.intersection(oneg_true)) / len(neg_true)
                if cur_nt_rate > nt_rate:
                    nt_rate = cur_nt_rate
                    nt_tag = f'{oname}_nt'
            if neg_false:
                cur_nf_rate = len(neg_false.intersection(oneg_false)) / len(neg_false)
                if cur_nf_rate > nf_rate:
                    nf_rate = cur_nf_rate
                    nf_tag = f'{oname}_nf'
                cur_nf_rate = len(neg_false.intersection(opos_true)) / len(neg_false)
                if cur_nf_rate > nf_rate:
                    nf_rate = cur_nf_rate
                    nf_tag = f'{oname}_pt'
        fea_gp = groups[rid]
        print(f'[{fea_gp}]')
        if pt_tag:
            print(f'pos_true: {pt_rate:.2f}/{pt_tag}', end='  ')
        if pf_tag:
            print(f'pos_false: {pf_rate:.2f}/{pf_tag}', end='  ')
        if nt_tag:
            print(f'neg_true: {nt_rate:.2f}/{nt_tag}', end='  ')
        if nf_tag:
            print(f'neg_false: {nf_rate:.2f}/{nf_tag}', end='  ')
        print('')


def calc_clus_feas(df_list: List[DataFrame], inds: List[TFeature], profit_off: int = 2,
                   keep_ind_cols: bool = True, timeframe: str = '5m') -> DataFrame:
    '''
    计算指标并进行聚类形成特征，方便后续分析统计
    :param df_list:
    :param inds: 要组合的自定义指标
    :param profit_off: 预测未来第n个蜡烛，0则不计算
    :param keep_ind_cols: 是否保留特征的指标列（如果后续要计算指标相关度，这里需要保留）
    :return:
    '''
    merge_list = []
    # 每个pair的数据需要单独计算指标、收益等列，因为会依赖前面n行的数据
    logger.warning(f'calc inds for {len(df_list)} pairs')
    last_time = time.time()
    for i, df in enumerate(df_list):
        if not len(df):
            logger.warning(f'empty data found for {i}')
            continue
        # 计算指标
        try:
            [fea.compute(df) for fea in inds]
        except Exception:
            logger.exception(f'calc inds for pair: {i} error')
            exit(0)
        # 更新有效行范围
        col_names = [v for id in inds for v in id.col_names]
        if profit_off:
            # 计算实际收益率（未来第3个蜡烛）
            calc_next_profit(df, profit_off)
            col_names += ['profit']
        to_low_precision(df)
        update_df_valid_range(df, col_names)
        merge_list.append(df[df.valid_start: df.valid_end][col_names])
        if time.time() - last_time > 5:
            logger.warning(f'process {i} pair complete')
            last_time = time.time()

    # 将所有数据表合并为一个，此后所有操作应是行间无关操作
    merge_df = pd.concat(merge_list, ignore_index=True, sort=False)
    del merge_list
    merge_df.valid_start = 0
    merge_df.valid_end = len(merge_df)
    # 执行聚类，降低搜索空间
    for iv in inds:
        logger.warning(f'cluster for {iv.name}')
        iv.cluster(merge_df, keep_ind_cols, timeframe=timeframe)
    return merge_df


def _print_ind_relevance(rel_stas, gp_name: str, tag: str, min_num=7, skip_mismatch: bool = True,
                         min_acc: float = 0.55):
    pos_stas = sorted(rel_stas, key=lambda x: x[1], reverse=True)
    total_match, total_num = 0, 0
    case_lines = [f'\n********  {gp_name} {tag} relevance  **********']
    for item in pos_stas:
        if item[2] < min_num:
            continue
        if skip_mismatch and item[1] <= min_acc:
            break
        case_lines.append(f'  {item}')
        total_match += round(item[1] * item[2])
        total_num += item[2]
    if len(case_lines) > 1:
        print('\n'.join(case_lines))
    return f'  {tag}:   {total_match} / {total_num}'


def _profile_ind_recall(df: DataFrame, profit_thres: int, feas: List[TFeature], profit: Series):
    tag = 'pos' if profit_thres > 0 else 'neg'
    print(f'**********  ind recall for [{tag}] profit   case_val:  `relance_score` / `proportion`  *************')
    max_end_thres = profit_thres * 9999
    thres_st, thres_end = profit_thres, max_end_thres
    if thres_st > 0:
        match_idx = profit[(profit >= thres_st) & (profit < thres_end)].index
    else:
        match_idx = profit[(profit <= thres_st) & (profit > thres_end)].index
    if not len(match_idx):
        return
    col_names = [n for f in feas for n in f.col_names]
    col_names = list(set(col_names))
    for name in col_names:
        # 计算匹配的行，此指标情况占比
        gp_cnts = df.loc[match_idx].groupby([name]).size().reset_index(name='count')
        gp_cnts = gp_cnts.sort_values(by=[name], ascending=False)
        gp_cnts['count'] = gp_cnts['count'].astype(int)
        # 计算在所有数据中，此指标的情况占比
        all_cnts = df.groupby([name]).size().reset_index(name='count')
        all_cnts = all_cnts.sort_values(by=[name], ascending=False)
        all_cnts['count'] = all_cnts['count'].astype(int)

        all_count_map, all_total = dict(), all_cnts['count'].sum()
        for ri, row in all_cnts.iterrows():
            all_count_map[str(row[name])] = round(row["count"] * 100 / all_total, 2)

        case_rows, gp_total = [], gp_cnts['count'].sum()
        for ri, row in gp_cnts.iterrows():
            case_pct = round(row["count"] * 100 / gp_total, 2)
            if case_pct < 1:
                # 占比少于1%的情况不做分析
                continue
            pct_in_all = all_count_map.get(str(row[name]), 0)
            case_rows.append((row[name], case_pct / pct_in_all, case_pct))

        pct_list = list(zip(*case_rows))[1]
        std_dev = 0
        if len(pct_list) >= 2:
            std_dev = statistics.stdev(pct_list) / statistics.mean(pct_list)
        case_rows = sorted(case_rows, key=lambda x: abs(x[1] - 1), reverse=True)
        # 如果分布和全局分布相差不超过20%，则认为不够突出，不显示
        case_list = [f'{row[0]}: {row[1]:.2f} / {row[2]:.1f}%' for row in case_rows if abs(row[1] - 1) >= 0.2]
        if not case_list:
            continue
        print(f'{name} range [{thres_st} - {thres_end}]:   {len(match_idx)},  stddev: {std_dev:.3f}')
        print('   ' + ('    '.join(case_list)))


def profile_ind_group(df: pd.DataFrame, feas: List[TFeature], pos_thres: int = 35, neg_thres: int = -30,
                      min_num: int = 5, skip_mismatch: bool = True, side: str = 'both', min_acc: float = 0.58,
                      show_recall: bool = True, timeframe: str = '5m'):
    '''
    true列：未来第3个蜡烛的5窗口均值预期收益率
    分析指标在给定股票指标组合的值与胜率、频率的关系。
    :param df:
    :param feas: 指标名组合
    :param pos_thres: 做多（买入）的预期利润阈值，万分之
    :param neg_thres: 做空（卖出）信号阈值。万分之。
    :param min_num: 最小出现次数，少于此次数不显示（过滤次数太少的指标组合）
    :param skip_mismatch: 跳过不匹配的分析统计行
    :param side: both,long,short
    :param min_acc: 最小显示准确率
    :param show_recall: 是否显示召回相关度
    :return:
    '''
    show_name = ' + '.join(['|'.join([f'{ii.name}:{ii.num}' for ii in f.ind_infos]) for f in feas])
    logger.warning(f'start profile ind group {show_name} with {len(df)} candles')
    profit = df['profit']
    pos_stas, neg_stas, save_stas = [], [], dict()
    # 分析此指标组合符合行情时，盈利的概率（准确率）
    gp_names = [f.name for f in feas]
    ind_stas = df.groupby(gp_names, as_index=False)['profit', 'idx'].aggregate(lambda x: list(x))
    logger.warning(f'grouped as {len(ind_stas)} rows, calc profits..')
    all_pos_true, all_pos_false, all_neg_true, all_neg_false = set(), set(), set(), set()
    for rid, row in ind_stas.iterrows():
        cur_reals = np.array([row['profit'], row['idx']]).transpose()
        all_len = len(cur_reals)
        pos_ids = np.where(cur_reals[:, 0] > pos_thres)[0]
        pos_len = len(pos_ids)
        pair_name = "+".join([str(row[f.name]) for f in feas])
        pos_prob = round(pos_len / all_len, 3)
        pos_stas.append((pair_name, pos_prob, all_len))
        if pos_prob >= min_acc and all_len >= min_num:
            save_stas[pair_name] = (pos_prob, all_len)
            pos_true = set(cur_reals[pos_ids, -1].tolist())
            all_pos_true.update(pos_true)
            cur_reals = np.delete(cur_reals, pos_ids, axis=0)
            all_pos_false.update(cur_reals[:, -1].tolist())

        neg_ids = np.where(cur_reals[:, 0] < neg_thres)[0]
        neg_len = len(neg_ids)
        neg_prob = round(neg_len / all_len, 3)
        neg_stas.append((pair_name, neg_prob, all_len))
        if neg_prob >= min_acc and all_len >= min_num:
            save_stas[pair_name] = (-neg_prob, all_len)
            neg_true = set(cur_reals[neg_ids, -1].tolist())
            all_neg_true.update(neg_true)
            cur_reals = np.delete(cur_reals, neg_ids, axis=0)
            all_neg_false.update(cur_reals[:, -1].tolist())

    addi_lines = []
    policy_name = '_'.join(gp_names)
    if side != 'short':
        addi_lines.append(_print_ind_relevance(pos_stas, show_name, 'pos', min_num, skip_mismatch, min_acc))
    if side != 'long':
        addi_lines.append(_print_ind_relevance(neg_stas, show_name, 'neg', min_num, skip_mismatch, min_acc))
    if save_stas:
        save_stg_config(timeframe, policy_name, save_stas)
    print(f'{show_name}  pos_thres:{pos_thres}  neg_thres: {neg_thres}')
    print('\n'.join(addi_lines))
    if show_recall:
        # 分析盈利时，此指标组合的匹配度。（召回率）
        _profile_ind_recall(df, pos_thres, feas, profit)
        _profile_ind_recall(df, neg_thres, feas, profit)
    return all_pos_true, all_pos_false, all_neg_true, all_neg_false


def statistic_pos_neg_ids(df: DataFrame, gp_names: list, save_stas: dict, price_thres: int):
    print(f'statistic_pos_neg_ids: {gp_names}', end=' ')
    start = time.time()
    pos_thres, neg_thres = abs(price_thres), - abs(price_thres)
    # 统计正负样本的ID
    np_arr = df[gp_names + ['profit']].to_numpy(dtype=np.int64)
    # 添加索引列
    idx_col = np.array(range(np_arr.shape[0])).reshape((-1, 1))
    np_arr = np.concatenate([np_arr, idx_col], axis=1)
    pos_list, neg_list = [], []
    from functools import reduce
    from operator import and_
    for pos_key in save_stas:
        case_vals = [int(v) for v in pos_key.split('+')]
        is_pos = save_stas[pos_key][0] > 0
        where_list = [np_arr[:, i] == v for i, v in enumerate(case_vals)]
        match_ids = np.where(reduce(and_, where_list))[0]
        save_list = pos_list if is_pos else neg_list
        save_list.append(match_ids)
    pos_ids = np.concatenate(pos_list)
    neg_ids = np.concatenate(neg_list)
    pos_arr = np_arr[pos_ids]
    pos_true_ids = np.where(pos_arr[:, -2] >= pos_thres)[0]
    pos_true = pos_arr[pos_true_ids, -1].tolist()
    pos_false = np.delete(pos_arr, pos_true_ids, axis=0)[:, -1].tolist()
    neg_arr = np_arr[neg_ids]
    neg_true_ids = np.where(neg_arr[:, -2] <= neg_thres)[0]
    neg_true = neg_arr[neg_true_ids, -1].tolist()
    neg_false = np.delete(neg_arr, neg_true_ids, axis=0)[:, -1].tolist()
    cost = time.time() - start
    print(f'cost: {cost: .2f}s')
    return set(pos_true), set(pos_false), set(neg_true), set(neg_false)


'''特征相关度分析'''


def get_fea_dataframe(pair_num=0, profit_off=3) -> DataFrame:
    fea_map = fts.get_fea_map()
    cache_path = 'all_feas.feather'
    if not os.path.isfile(cache_path):
        pair_datas = load_all_pairs(pair_num=pair_num)
        merge_df = calc_clus_feas(pair_datas, list(fea_map.values()), profit_off=profit_off, keep_ind_cols=False)
        merge_df.to_feather(cache_path)
    else:
        merge_df = pd.read_feather(cache_path)
    return merge_df


def print_corr(df: DataFrame):
    import seaborn as sns
    import matplotlib.pyplot as plt
    for rid, row in df.iterrows():
        col_sims = list(sorted(dict(row).items(), key=lambda x: x[1], reverse=True)[1:4])
        sim_text = ',  '.join([f"{v[0]} {v[1]:.3f}" for v in col_sims])
        print(f'{row.name}: {sim_text}')

    fig, ax = plt.subplots(figsize=(15, 9))
    sns.heatmap(df, ax=ax, annot=True)
    fig.show()


def analysis_fea_correlation():
    '''
    计算特征相关矩阵
    :return:
    '''
    df = get_fea_dataframe(pair_num=50, profit_off=0)
    logger.warning(f'load fea data: {len(df)}')
    corr_res = df.corr(method='spearman')
    logger.warning('print corr for spearman')
    print_corr(corr_res)

    # MIC计算很慢，这里随机采样2W行计算
    max_row = 100000
    df = df.sample(frac=max_row / len(df)).reset_index(drop=True)
    from minepy import MINE
    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    all_cols = df.columns.tolist()
    mic_df = DataFrame(1, index=all_cols, columns=all_cols)
    for rid, col in enumerate(all_cols):
        for cid, other in enumerate(all_cols):
            if cid <= rid:
                continue
            print(f'calc mic corr for {col} -- {other}', end=' ')
            mine.compute_score(df[col], df[other])
            mic_df.loc[col, other] = mine.mic()
            mic_df.loc[other, col] = mine.mic()
            print(mine.mic())
    logger.warning('print corr for MIC')
    print_corr(mic_df)


'''特征胜率组合分析'''


def test_fea_probs():
    from banbot.compute.main import MicroWave
    model_path = r'E:\trade\freqtd_data\user_data\models\feaprob4\lgb.txt'
    mw = MicroWave(model_path)
    pairs = load_all_pairs()
    sam_df = pairs[0]
    # calc_next_profit(sam_df)
    handled = mw.predict(sam_df)
    print(handled.head())


def _profile_profits(arr: np.ndarray, tag: str):
    total_len = len(arr)
    print(f'****  {tag} profit levels, total: {total_len}  ******')
    pct_levels = [50, 70, 80, 85, 90, 93, 95, 97, 99, 99.3, 99.5, 99.7, 99.9]
    pos_ends = [round(total_len * v / 100) for v in pct_levels]
    for i, end_val in enumerate(pos_ends):
        pct_val = pct_levels[i]
        print(f'{pct_val}% < {arr[end_val]}')
    print('')


def group_profit_levels(timeframe: str, profit_off: int):
    '''
    统计分析利润分布情况，用于确定利润阈值。
    :param timeframe:
    :param profit_off:
    :return:
    '''
    pair_datas = load_all_pairs(timeframe=timeframe)
    profit_list = []
    for i, df in enumerate(pair_datas):
        if not len(df):
            logger.warning(f'empty data found for {i}')
            continue
        # 计算实际收益率（未来第3个蜡烛）
        calc_next_profit(df, profit_off)
        profit_list.extend(df['profit'][profit_off * 2:].tolist())
    profit_arr = np.sort(np.array(profit_list))
    _profile_profits(profit_arr[profit_arr >= 0], 'pos')
    neg_arr = np.sort(np.abs(profit_arr[profit_arr < 0]))
    _profile_profits(neg_arr, 'neg')


if __name__ == '__main__':
    # 特征：aroon,bb20,cmf,crsi,kama,kdj,mfi,natr,rsi,sma3,srsi,wr
    # group_profit_levels('15m', 1)

    # 15m维度，预测后一个，和5m结合使用
    analysis_inds_group(fts.use_policy, pos_thres=64, neg_thres=-53, min_acc=0.7, show_recall=False,
                        timeframe='15m', pred_off=1)

    # # 5m维度，后1个召回是后2个的95%，但训练模型正确率提升10%。暂预测后1个
    # analysis_inds_group(fts.use_policy, pos_thres=40, min_acc=0.65, show_recall=False, pred_off=1)

    # # 一分钟维度特征。预测后1分钟的正确率最高
    # analysis_inds_group(fts.use_policy, pos_thres=25, neg_thres=-20, min_acc=0.66, show_recall=False,
    #                     timeframe='1m', pred_off=1)


    # analysis_fea_correlation()
    # test_fea_probs()
