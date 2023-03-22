#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : addons.py
# Author: anyongjin
# Date  : 2023/3/17
import numpy as np
from typing import Tuple
from banbot.compute.patterns import *


def _log_extrem(log_list: list, arr: np.ndarray, col_idx: int, min_dist: float = 0):
    ind_vals = arr[-3:, col_idx]
    mid_ind = ind_vals[-2]
    if min(ind_vals[-3], ind_vals[-1]) <= mid_ind <= max(ind_vals[-3], ind_vals[-1]):
        return
    extype = 1 if mid_ind > ind_vals[-1] else -1
    if log_list:
        pextype, pprice = log_list[-1][2], log_list[-1][1]
        if extype == pextype:
            if extype == 1 and mid_ind > pprice or extype == -1 and mid_ind < pprice:
                log_list.pop(-1)
                log_list.append((bar_num.get() - 1, mid_ind, extype))
            return
        if abs(pprice - mid_ind) < min_dist:
            # 与前一个极值点波动过低，过滤掉
            return
    log_list.append((bar_num.get() - 1, mid_ind, extype))
    if len(log_list) > 150:
        del log_list[:50]


def make_log_extrem(log_ma5, log_ma20, log_ma120, ma5_id, ma20_id, ma120_id):
    def calc_func(arr: np.ndarray):
        '''
        检查当前是否出现极值点并记录。
        :param arr:
        :return:
        '''
        if arr.shape[0] < 3 or np.isnan(long_bar_avg.val):
            return
        if not np.isnan(arr[-3, ma5_id]):
            _log_extrem(log_ma5, arr, ma5_id, long_bar_avg.val * 2)
        if not np.isnan(arr[-3, ma20_id]):
            _log_extrem(log_ma20, arr, ma20_id, long_bar_avg.val)
        if not np.isnan(arr[-3, ma120_id]):
            _log_extrem(log_ma120, arr, ma120_id, long_bar_avg.val * 0.5)
    return calc_func


def make_big_vol_prc(nvol_col_id: int, ntr_rol_id: int, col_num: int):
    def calc_func(arr: np.ndarray) -> float:
        '''
        是否巨量。是成交量均值的15倍，或10倍且5周期最大，或5倍且10周期最大
        :param arr:
        :return: 巨量的分数。5倍是基准，得分为1；2.5倍最低，得分0.5；20倍得分是2
        '''
        nvol_val = arr[-1, nvol_col_id]
        min_nvol, base_nvol = 2.5, 5
        if np.isnan(nvol_val) or nvol_val < min_nvol:
            return 0
        vol_score = nvol_val / base_nvol
        if vol_score > 1:
            vol_score = pow(vol_score, 0.5)
        is_big_vol = nvol_val >= 15 or nvol_val >= 10 and nvol_val >= np.max(arr[-5:-1, nvol_col_id]) or \
                     nvol_val >= min_nvol and nvol_val >= np.max(arr[-10:-1, nvol_col_id])
        if not is_big_vol:
            return 0

        # 判断是否价格剧烈变化
        cur_ntr = arr[-1, ntr_rol_id]
        prev_ntr = np.max(arr[-4:-1, ntr_rol_id])
        ntr_chg = cur_ntr / prev_ntr
        is_price_huge_chg = ntr_chg >= 2 or ntr_chg >= 1.5 and cur_ntr >= 0.3
        if np.isnan(prev_ntr) or not is_price_huge_chg:
            # 当前波动不够剧烈（是前一个2倍，或1.5倍但波动幅度超过历史30%）
            return 0
        prc_score = max(ntr_chg / 2.5, cur_ntr / 0.4)
        copen, chigh, clow, close = arr[-1, :4]
        max_chg, real, solid_rate, hline_rate, lline_rate = arr[-1, col_num - 5: col_num]
        if solid_rate < 0.7:
            # 蜡烛实体部分至少占比70%
            return 0
        prc_score *= solid_rate / 0.8
        end_back = hline_rate if close >= copen else lline_rate
        if end_back >= 0.25:
            # 如果结束时，价格回撤超过25%，则认为不够强势
            return 0
        prc_score *= (1 - end_back) / 0.9
        return prc_score * vol_score
    return calc_func


def norm_score(arr: np.ndarray, ntr_rol_id: int):
    '''
    趋势规范化分数，用于判断处于趋势还是盘整。
    >1.7 是轻微趋势  >2.5 是明显趋势
    :param arr:
    :param ntr_rol_id:
    :return:
    '''
    cur_ntr = arr[-1, ntr_rol_id]
    prev_ntr = np.max(arr[-4:-1, ntr_rol_id])
    return cur_ntr / prev_ntr


def make_calc_shorts(col_num: int, ma5_id: int, ma20_id: int, ma120_id: int, ntr_rol_id: int):
    def calc(arr: np.ndarray, his_ptns: List[Dict[str, float]], huge_score: float) -> List[Tuple[str, float]]:
        copen, chigh, clow, close, vol = arr[-1, :5]
        max_chg, real, solid_rate, hline_rate, lline_rate = arr[-1, col_num - 5: col_num]

        ma5_val, ma20_val, ma120_val = arr[-1, ma5_id], arr[-1, ma20_id], arr[-1, ma120_id]
        ma5_chg = ma5_val - arr[-10, ma5_id] if arr.shape[0] >= 10 else np.nan
        ma5_sub10_val = ma5_chg / long_bar_avg.val / 1.5  # 下跌指数，<0表示下跌，>0表示上涨，>1表示大涨

        result = dict()

        bar_sub = close - max(copen, arr[-2, 3])
        bar_solid_rate = bar_sub / long_bar_avg.val
        if bar_solid_rate < -1 and solid_rate > 0.66:
            # 大阴线
            big_down_score = abs(bar_solid_rate * 1.3) * solid_rate
            if huge_score > 0.1:
                # 成交量和价格剧烈变化
                pass
            elif ma5_sub10_val < -1:
                # 震荡&下跌趋势中--大阴线下跌
                result['big_down_ensure'] = big_down_score

            elif close - ma120_val > long_sub_ma120.val:
                # 高位大阴线下跌
                result['big_down_rev'] = big_down_score

        prv_ptns, cur_ptns = his_ptns[-2:]
        if ma5_sub10_val > 1:
            # 处于牛市中
            bear_doji = {'hammer', 'dragonfly_doji', 'gravestone_doji', 'inv_hammer', 'doji', 'shooting_star'}
            # 看跌吞没，看跌孕育，看跌亲吻，流星，顶十字星
            down_ptns = dict(bear_engulf=0.9, bear_harami=0.8, kiss_down=0.7, shooting_star=0.9, star=0.7)
            if bear_doji.intersection(cur_ptns):
                vol_score = big_vol_score(arr)
                # n_score = norm_score(arr, ntr_rol_id)
                # 10周期内明显上涨，出现锤子，放量，非盘整形态，是吊颈线，要下跌
                if vol_score > 0.1:
                    result['doji_top'] = vol_score

            else:
                down_key = next(iter(set(down_ptns.keys()).intersection(cur_ptns)), None)
                if down_key:
                    result[down_key] = cur_ptns[down_key] * down_ptns[down_key]

        bear_twice = {'bear_engulf', 'dark_cloud_cover'}
        # 黄昏垂星，倾盆大雨，吊人
        bear_ptns = dict(new3_down=1, down2_mid=1, evening_star=1, black_out_down=1, hanging_man=0.8)
        if bear_twice.intersection(cur_ptns):
            # 乌云盖顶形态。较大概率下跌
            score = 0.8 if close > arr[-2, 0] else 1
            if vol < arr[-2, 5] * 0.66 and vol < np.average(arr[-6:-1, 5]):
                # 阴线的成交量需是前一日的2/3或前5日均量的1倍，否则可靠性降低
                score *= 0.7
            result['bear_top_dark'] = score

        else:
            down_key = next(iter(set(bear_ptns.keys()).intersection(cur_ptns)), None)
            if down_key:
                result[down_key] = cur_ptns[down_key] * bear_ptns[down_key]

        # 这里检查前几个周期信号，是否有当前周期能确认的，如有更新信号置信度
        return sorted([(k, v) for k, v in result.items() if v > 0.3], key=lambda x: x[1], reverse=True)
    return calc
