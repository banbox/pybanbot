#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : patterns.py
# Author: anyongjin
# Date  : 2023/3/4
import math
import os
import time
import uuid

import pandas as pd
import numpy as np
from typing import List, Dict
from banbot.compute.tainds import *
'''
K线形态参考：
https://github.com/SpiralDevelopment/candlestick-patterns/tree/master/candlestick/patterns
https://github.com/cm45t3r/candlestick/blob/master/src/candlestick.js
https://github.com/stockalgo/stolgo/blob/master/lib/stolgo/candlestick.py
'''

tkey = 'morning_star_doji'

_ptn_req_lens = dict(
    dragonfly_doji=1,
    gravestone_doji=1,
    hammer=1,
    inv_hammer=1,
    doji=1,
    doji_star=2,
    bear_engulf=2,
    bear_harami=2,
    bull_engulf=2,
    bull_harami=2,
    dark_cloud_cover=2,
    piercing=2,
    rain_drop=2,
    rain_drop_doji=2,
    shooting_star=5,
    star=2,
    evening_star=3,
    hanging_man=3,
    morning_star=3,
)


def big_vol_score(arr: np.ndarray, idx: int = -1):
    '''
    判断给定蜡烛是否放量。
    :param arr:
    :param idx:
    :return:
    '''
    cur_vol, prev_vol = arr[idx, vcol], arr[idx - 1, vcol]
    vol_avg_val = LongVar.get(LongVar.vol_avg).val
    avg_score = cur_vol / vol_avg_val / 2.5
    prev_score = cur_vol / prev_vol / 3
    if avg_score > 1 and cur_vol > prev_vol or prev_score > 1:
        return max(avg_score, prev_score) / 1.5
    return 0


def detect_pattern(arr: np.ndarray) -> Dict[str, float]:
    '''
    K线模式形态识别。这里只生成形态信号，使用信号时应根据市场环境和其他指标综合决策。
    :param arr:
    :return: 匹配的形态名称
    '''
    col_start: int = fea_col_start.get()
    candle = arr[-1, :]
    has_p1 = arr.shape[0] > 1
    has_p2 = arr.shape[0] > 2
    c_max_chg, c_real, c_solid_rate, c_hline_rate, c_lline_rate = candle[col_start: col_start + 5]
    copen, chigh, clow, close = candle[:4]
    dust = min(0.00001, close * 0.0001)

    popen, phigh, plow, pclose = 0, 0, 0, 0
    p_max_chg, p_solid_rate, p_hline, p_lline, p_real = 0, 0, 0, 0, 0
    p2open, p2high, p2low, p2close = 0, 0, 0, 0
    p2_max_chg, p2_real, p2_solid_rate = 0, 0, 0
    if has_p1:
        pcandle = arr[-2, :]
        popen, phigh, plow, pclose = pcandle[:4]
        p_max_chg, p_real, p_solid_rate, p_hline_rate, p_lline_rate = pcandle[col_start: col_start + 5]
        if has_p2:
            p2candle = arr[-3, :]
            p2open, p2high, p2low, p2close = p2candle[:4]
            p2_max_chg, p2_real, p2_solid_rate, p2_hline_rate, p2_lline_rate = p2candle[col_start: col_start + 5]

    result = dict()

    if 0.1 > c_solid_rate > c_hline_rate and c_lline_rate > 3 * c_solid_rate:
        # T字，长下影。蜻蜓十字。趋势将反转。等待下一个蜡烛确认
        result['dragonfly_doji'] = 1

    if 0.1 > c_solid_rate > c_lline_rate and c_hline_rate > 3 * c_solid_rate:
        # 倒T，长上影。墓碑十字。看跌趋势。
        result['gravestone_doji'] = 1

    if 0.07 < c_solid_rate < 0.33 and c_hline_rate * 3 < c_solid_rate and c_lline_rate > 0.6:
        # 锤子。下降趋势底部出现。是反转信号。
        result['hammer'] = 1

    if 0.07 < c_solid_rate < 0.33 and c_lline_rate * 3 < c_solid_rate and c_hline_rate > 0.6:
        # 倒锤子。下跌趋势中看涨反转。上涨趋势中看跌反转。
        result['inv_hammer'] = 1

    cur_doji = c_solid_rate < 0.1 and 3 * c_solid_rate < min(c_hline_rate, c_lline_rate) and \
               abs(c_hline_rate - c_lline_rate) < min(c_hline_rate, c_lline_rate)
    if cur_doji:
        # 十字星
        result['doji'] = 1

    if not has_p1:
        return result
    phalf = (popen + pclose) / 2
    if cur_doji and p_solid_rate >= 0.7 and min(copen, close) > pclose > phalf:
        # 前70%阳，见顶十字星
        result['doji_star'] = 1
    bar_len_val = LongVar.get(LongVar.bar_len).val

    def contain_score():
        '''
        吞没或孕育的分数计算。
        :return:
        '''
        if c_real > p_real:
            long_real, long_real_rate = c_real, c_solid_rate
            short_real, short_full = p_real, p_max_chg
        else:
            long_real, long_real_rate = p_real, p_solid_rate
            short_real, short_full = c_real, c_max_chg
        long_score = min(1, long_real / min(max(short_real, dust), bar_len_val) / 2)
        add_score = pow(max(1, long_real * 1.5 / short_full), 0.5) - 1
        _vol_score = min(1.5, arr[-1, vcol] / arr[-2, vcol])
        return (_vol_score * long_score + add_score) * long_real_rate

    def kiss_score():
        '''
        亲吻线的分数计算
        :return:
        '''
        all_len = abs(popen - copen)
        len_score = min(1.5, all_len / bar_len_val / 3)
        len_rate = min(1, c_real / p_real / 1.1)
        solid_rate = min(p_solid_rate, c_solid_rate)
        return len_score * len_rate * solid_rate

    psolid_sml = abs(popen - pclose) / 3
    if min(c_solid_rate, p_solid_rate) >= 0.5:
        if copen >= pclose > popen >= close:
            # 看跌吞没：前阳后阴。后阴线完全包裹住前阳线。
            result['bear_engulf'] = contain_score()

        elif popen <= close < copen <= pclose:
            # 看跌孕育：前阳后阴。前阳线完全包裹后阴线。
            result['bear_harami'] = contain_score()

        elif close >= popen > pclose >= copen:
            # 看涨吞噬：前阴后阳。后阳线完全包裹前阴线。
            result['bull_engulf'] = contain_score()

        elif pclose <= copen < close <= popen:
            # 看涨孕育：前阴后阳。前阴线完全包裹后阳线。
            result['bull_harami'] = contain_score()

        elif max(copen, pclose) >= pclose > phalf > close > popen:
            # 乌云盖顶：前阳后阴。阴线整体高于阳线
            result['dark_cloud_cover'] = 1

        elif popen < pclose < copen and pclose - psolid_sml < close < pclose + psolid_sml < copen and \
                c_real >= p_real * 0.7:
            # 看跌亲吻：前阳后阴。后阴线完全位于阳线上方，意义弱于乌云盖顶
            result['kiss_down'] = kiss_score()

        elif pclose > popen > close and pclose > copen > close and pclose - popen > bar_len_val:
            # 倾盆大雨（分手线），前阳后阴。阴线整体低于阳线。看跌强于乌云盖顶
            # 分数：成交量至少前一个的1/2；第一日应为大阳线；第二日应为大/中阴线；第二根低开和低收的力度
            vol_score = min(1.5, arr[-1, vcol] / arr[-2, vcol])
            plen_score = min(1.2, p_real / bar_len_val / 1.5)
            clen_score = min(1, c_real / bar_len_val)
            # 低开或低收的分数加成
            diff_score = pow(1 + max(pclose - copen, popen - close) / (pclose - popen), 0.5)
            result['black_out_down'] = vol_score * plen_score * clen_score * diff_score

        elif popen > close > phalf > pclose >= plow > copen:
            # 穿孔。前阴后阳。前阴整体略高于后阳线。看涨反转
            result['piercing'] = 1

    else:
        _popen_min = popen if not has_p2 else min(p2close, popen)
        _popen_max = popen if not has_p2 else max(p2close, popen)
        if copen >= pclose > _popen_min >= close:
            # 看跌吞没：前阳后阴。后阴线完全包裹住前阳线。
            # 第一根是高开形成的小阴线、十字星等也有效
            result['bear_engulf'] = contain_score()

        elif max(copen, pclose) >= pclose > phalf > close > _popen_min:
            # 乌云盖顶：前阳后阴。阴线整体高于阳线
            # 第一根高开形成的一字线、十字线、T线、锤头线与大阳线有相同意义。
            # 第二根低开形成的小阴线、十字线、倒T线、倒锤头线与后阴线有相同意义。
            result['dark_cloud_cover'] = 1

    if p_solid_rate >= 0.6 and c_solid_rate <= 0.3 and p_real > c_real * 3:
        # 前大实体，后小实体长影线
        if max(copen, close) < pclose < popen:
            # 下雨。前大阴。后略长影线作为底。看涨反转
            if abs(c_hline_rate - c_lline_rate) < min(c_hline_rate, c_lline_rate) and c_solid_rate < 0.1:
                # 下雨。前阴。后十字线作为底。
                result['rain_drop_doji'] = 1
            else:
                result['rain_drop'] = 1
        elif popen < pclose < min(copen, close):
            # 顶十字星。前阳。后长影线带小柱子，整体高于前阳线。
            result['star'] = 1

        elif popen < pclose < copen and c_hline_rate >= 3 * c_solid_rate and c_lline_rate < c_solid_rate:
            # 流星：看跌，前阳。后倒锤子（阴阳皆可）。必须在上涨趋势中出现。
            result['shooting_star'] = 1

    if not has_p2:
        return result

    def mon_eve_star_score():
        '''
        黄昏垂星/晨星的分数计算
        :return:
        '''
        end_deep_score = abs(close - p2close) / abs(p2open - p2close)
        if close > copen:
            # 最后是阳线，晨星。
            end_lrate = c_lline_rate
        else:
            # 最后是阴线，黄昏垂星
            end_lrate = c_hline_rate
        len_score = min(1, c_real / bar_len_val / 2)
        return (1 - end_lrate) * len_score * end_deep_score

    def jump_out_score():
        '''
        跳空并列分数计算
        :return:
        '''
        dir_score = 1
        if p2open < p2close:
            # 上涨
            pos_score = (max(min(close, copen), min(pclose, popen)) - p2open) / p2_real
            req_dir = 1
        else:
            pos_score = (p2open - min(max(close, copen), max(pclose, popen))) / p2_real
            req_dir = -1
        if p_solid_rate < 0.1 or (pclose - popen) * req_dir < 0:
            dir_score -= 0.1
        if c_solid_rate < 0.1 or (close - copen) * req_dir < 0:
            dir_score -= 0.1
        res_score = pos_score * dir_score
        # 跳空的成交量不能太大，也不能太小
        jump_vol_rate = np.max(arr[-2:, vcol]) / arr[-3, vcol]
        if jump_vol_rate > 1.2:
            res_score /= jump_vol_rate
        elif jump_vol_rate < 1:
            res_score *= jump_vol_rate
        return res_score

    if p2_solid_rate > 0.5 and c_solid_rate > 0.5 and p_real * 3 < min(p2_real, c_real):
        # 中间小实体，左右长实体至少50%
        if min(popen, pclose) > p2close > p2open and close < copen < min(popen, pclose):
            # 黄昏垂星。阳+顶部阴/阳+阴
            result['evening_star'] = mon_eve_star_score()

        elif max(popen, pclose) < p2close < p2open and close > copen > max(popen, pclose):
            # 晨星。阴+底+阳
            result['morning_star'] = mon_eve_star_score()

    elif c_solid_rate < 0.25 and c_lline_rate >= 0.7 and max(phigh, p2high) < copen:
        # 吊人
        result['hanging_man'] = 1

    elif close < pclose < p2close < p2open and close < copen < popen < p2open and \
            min(p2_solid_rate, c_solid_rate) > 0.4:
        # 黑三鸦；连续三个下跌；第一个和最后一个必须是阴线且占比不低于40%
        avg_len = (p2_max_chg + c_max_chg) / 2
        chg_rate1 = (p2close - pclose) / avg_len
        chg_rate2 = (pclose - close) / avg_len
        chg_score = min(1.5, pow((chg_rate1 + chg_rate2) / 2, 0.5))
        len_score = min(1, avg_len * 1.5 / bar_len_val)
        solid_rate = c_real / p2_real * pow(c_solid_rate, 0.7)
        result['new3_down'] = chg_score * len_score * solid_rate

    elif close < p2close < p2open and close < copen < p2open and min(p2_solid_rate, c_solid_rate) > 0.6 and \
            phigh < p2high and plow > clow and min(pclose, popen) > close and max(pclose, popen) < p2open:
        # 两阴夹一阳：第一和最后一个都是阴线，且占比不低于60%，最后一个低于第一个；中间的没超出范围
        vol_score = min(1.3, max(0.7, arr[-1, vcol] * 1.3 / arr[-3, vcol]))
        # 中间的如果是阳线应该量小于第一个
        mid_vol_score = 1 if popen > pclose else min(1.5, arr[-3, vcol] / arr[-2, vcol]) / 1.2
        # 最后一个下降应该略大
        chg_score = pow((p2close - close) / p2_max_chg + 1, 0.5)
        result['down2_mid'] = vol_score * mid_vol_score * chg_score

    elif p2_solid_rate > 0.66 and max(p_real, c_real) * 3 < p2_real and \
            (min(p_solid_rate, c_solid_rate) > 0.4 or max(p_max_chg, c_max_chg) * 2 < p2_real):
        # 第一根大长线，后续两个均实体较小，也没有太长的影线；必须用于明显趋势的行情中
        if min(close, copen, pclose, popen) > p2close - p2_real * 0.2 > p2open:
            # 向上跳空并列，继续上涨（中期信号）
            result['jump_up'] = jump_out_score()
        elif max(close, copen, pclose, popen) < p2close + p2_real * 0.2 < p2open:
            # 向下挑空并列，继续下跌（中期信号）
            result['jump_down'] = jump_out_score()

    return result


def test_pattern_draws(arr: np.ndarray, bar_avg_chg: int):
    from PIL import Image, ImageDraw
    imw, imh = 1600, 1200
    im = Image.new('RGB', (imw, imh))
    dr = ImageDraw.Draw(im)
    itw, ith = 180, 150
    text_size = 16
    rown, coln = math.floor(imw / itw), math.floor(imh / ith)

    def _init_img_draw():
        dr.rectangle((0, 0, imw, imh), fill='white')
        for x in range(0, imw, itw):
            dr.line((x, 0, x, imh), fill='black')
        for y in range(0, imh, ith):
            dr.line((0, y, imw, y), fill='black')

    def _show_img():
        import cv2
        img = np.array(im)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(tkey or 'image', img)
        cv2.waitKey()

    _init_img_draw()
    pos = 0
    max_num = rown * coln

    def _draw_pattern(tags: List[str], varr: np.ndarray):
        back_len = max([_ptn_req_lens[t] for t in tags])
        view_arr = varr[-back_len:, :]
        if view_arr.shape[0] < back_len:
            return
        # 所有价格减去最小值
        view_arr[:, ocol:vcol] -= np.min(view_arr[:, lcol])
        # 所有价格改为像素长度，方便绘制
        view_arr[:, ocol:vcol] *= ith * 0.6 / np.max(view_arr[:, hcol])
        top = ith * math.floor(pos / coln)
        left = itw * (pos % rown)
        top += round(ith * 0.07)
        bar_h = ith * 0.6
        bar_wid = round(bar_h * 0.06)
        bar_left = left + (itw - bar_wid * back_len * 2) / 2
        for di in range(back_len):
            o, c = view_arr[di, ocol], view_arr[di, ccol]
            rmin, rmax = min(o, c), max(o, c)
            fill = 'green' if o <= c else 'red'
            dr.rectangle((bar_left, top + (bar_h - rmax), bar_left + bar_wid, top + (bar_h - rmin)), fill=fill)
            h, l = view_arr[di, hcol], view_arr[di, lcol]
            center_x = bar_left + bar_wid * 0.5
            dr.line((center_x, top + (bar_h - h), center_x, top + (bar_h - l)), fill=fill)
            bar_left += bar_wid * 2
        top += ith * 0.63
        max_chr_len = math.floor(text_size * itw / 100)

        def _draw_line_texts(texts: List[str]):
            nonlocal top
            fin_lines = []
            for text in texts:
                fin_lines.extend([text[i: i+max_chr_len] for i in range(0, len(text), max_chr_len)])
            for line in fin_lines:
                _, _, w, h = dr.textbbox((0, 0), line)
                dr.text((left + (itw - w) / 2, float(top)), line, fill='black')
                top += h * 1.5
        _draw_line_texts([' '.join(tags), str(view_arr[0, 0])[:19]])

    count = 0
    for i in range(arr.shape[0]):
        tags = detect_pattern(arr)
        if not tags or tkey and tkey not in tags:
            continue
        if pos + 1 > max_num:
            pos = 0
            _show_img()
            im = Image.new('RGB', (imw, imh))
            dr = ImageDraw.Draw(im)
            _init_img_draw()
        _draw_pattern(list(tags.keys()), arr)
        pos += 1
        count += 1
        if count >= 180:
            break
    _show_img()


def load_bnb_1s():
    data_dir = r'E:\trade\freqtd_data\user_data\spec_data\bnb1s'
    fname = 'BTCUSDT-1s-2023-02-22-2023-02-26.feather'
    return pd.read_feather(os.path.join(data_dir, fname))


if __name__ == '__main__':
    origin_df = load_bnb_1s()
    og_arr = origin_df[:10000].to_numpy()[:, :8]
    bar_avg_chg600 = np.average(og_arr[:, hcol] - og_arr[:, lcol])
    test_pattern_draws(og_arr, bar_avg_chg600)
