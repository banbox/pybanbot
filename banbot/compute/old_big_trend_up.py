#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : old_big_trend_up.py
# Author: anyongjin
# Date  : 2023/3/16

def _get_exm_maxmin(self, back_len: int, end: int = 0):
    chk_id = len(self.extrems_ma5) - 1
    end_bar_num = bar_num.get()
    if end:
        chk_id = end
        end_bar_num = self.extrems_ma5[chk_id][0]
    high_id, low_id = -1, -1
    high_val, low_val = -1, -1
    while chk_id >= 0:
        exm = self.extrems_ma5[chk_id]
        if exm[2] == 1:
            if high_val < 0 or exm[1] > high_val:
                high_id, high_val = chk_id, exm[1]
        else:
            if low_val < 0 or exm[1] < low_val:
                low_id, low_val = chk_id, exm[1]
        if exm[0] < end_bar_num - back_len:
            break
        chk_id -= 1
    return high_id, low_id, high_val, low_val


def _is_trend(self, arr: np.ndarray, start: int, end: int, dirt: int):
    crs_num = len([num for num in self.ma_cross if start <= num <= end])
    if crs_num > 2:
        return False
    diff = arr[start, self.ma20.out_idx] - arr[end, self.ma20.out_idx]
    diff_type = 1 if diff < 0 else -1
    return diff_type == dirt


def _is_down_trend(self, arr: np.ndarray, period: int):
    '''
    判断前面是否有给定周期的稳定下跌。
    :param period:
    :return:
    '''
    if len(self.extrems_ma20) < 2:
        return False
    exm = self.extrems_ma20[-1]
    cur_ma20 = arr[-1, self.ma20.out_idx]
    if cur_ma20 > exm[1]:
        if exm[2] == 1:
            if len(self.extrems_ma20) < 3:
                return False
            start, end = self.extrems_ma20[-3][0], self.extrems_ma20[-2][0]
        else:
            start, end = self.extrems_ma20[-2][0], self.extrems_ma20[-1][0]
    else:
        start, end = exm[0], bar_num.get()
    if end - start < period * 0.9:
        return False
    return self._is_trend(arr, start - 1, end - 1, -1)


def trend_trace(self, arr: np.ndarray) -> Tuple[Optional[str], float]:
    rsm_width, rsm_len = 3, 3
    if bar_num.get() % rsm_width:
        # 每个重采样窗口计算一次
        return None, 0
    sml_up, back_len = 'sml_up', rsm_width * rsm_len
    old_sta = self.long_sigs.get(sml_up)
    if old_sta and bar_num.get() - old_sta[0] <= back_len or arr.shape[0] < back_len:
        return None, 0
    big_arr = resample_candlesticks(arr, rsm_width, rsm_len)
    if big_arr.shape[0] < rsm_len:
        return None, 0

    if len(np.where(big_arr[:, ocol] >= big_arr[:, ccol])[0]):
        # 三个都必须是阳线
        return None, 0
    max_chg, real, solid_rate, hline_rate, lline_rate = big_arr[-1, self.col_num - 5: self.col_num]
    if solid_rate < 0.66 or real < np.min(big_arr[:, self.col_num - 4]) * 1.2:
        # 最后一个实体>66%，且实体部分不能是最低的
        return None, 0
    avg_solid_rate = np.average(big_arr[:, self.col_num - 3])
    if avg_solid_rate < 0.6:
        # 周期内实体占比均值不低于60%
        return None, 0
    if not self._is_down_trend(arr, round(back_len * 2.5)):
        return None, 0
    score = (avg_solid_rate / 0.8) * (solid_rate / 0.87)
    self.long_sigs[sml_up] = bar_num.get(), score
    return sml_up, score
