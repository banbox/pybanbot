#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : common.py
# Author: anyongjin
# Date  : 2023/3/27
from banbot.storage.orders import *
from dataclasses import dataclass


@dataclass
class OlayPoint(dict):
    '''
    覆盖物的点信息。
    '''
    timestamp: int
    '毫秒时间戳'
    value: float
    '对应y轴的值'
    dataIndex: int = None
    'bar索引，默认为空'

    def dict(self):
        res = dict(timestamp=self.timestamp, value=self.value)
        if self.dataIndex:
            res['dataIndex'] = self.dataIndex
        return res


def trail_info(elp_num: int, enter_price: float, is_short):
    '''
    返回相对入场价格的损失等信息
    :return 盈利价格差，入场后最大盈利价差，当前回撤率
    '''
    if not elp_num:
        return 0, 0, 0
    flag = -1 if is_short else 1
    # 全局最高价，这里测试使用收盘价计算效果更好
    close_arr = Bar.close[:elp_num]
    exm_price = min(close_arr) if is_short else max(close_arr)
    cur_close = Bar.close[0]
    # loss_val = cur_close - max(enter_price, max_price)
    # 这里应该用相对入场价的损失，刚入场波动较大，否则很容易被中途甩出
    gain_chg = (cur_close - enter_price) * flag
    dust = min(0.00001, cur_close * 0.0001)
    max_gain_price = max(dust, abs(exm_price - enter_price))
    back_rate = abs(exm_price - cur_close) / max_gain_price
    return gain_chg, max_gain_price, back_rate


def trail_stop_loss_core(elp_num: int, max_gain_price: float, gain_chg: float, back_rate: float,
                         loss_thres: List[float], odlens: List[int] = None, back_rates: List[float] = None):
    if odlens:
        flen, slen, mlen, llen = odlens
    else:
        flen, slen, mlen, llen = 3, 5, 10, 15
    pf_n2, pf_n1, pf_1, pf_2, pf_3 = loss_thres
    if elp_num <= flen and gain_chg < pf_n2:
        return 'ls_b'
    if flen < elp_num <= slen and gain_chg < pf_n1:
        return 'ls_s'
    if not back_rates:
        back_rates = 0.47, 0.28, 0.18
    if back_rate >= back_rates[0] and (slen < elp_num <= mlen or max_gain_price > pf_1):
        return 'back.5'
    elif back_rate >= back_rates[1] and (mlen < elp_num <= llen or max_gain_price > pf_2):
        return 'back.3'
    elif back_rate >= back_rates[2] and (llen < elp_num or max_gain_price > pf_3):
        return 'back.2'


def trail_stop_loss(enter_price: float, elp_num: int, is_short: bool, loss_thres: List[float],
                    odlens: List[int] = None, back_rates: List[float] = None) -> Optional[dict]:
    '''
    跟踪止损。
    3周期内，价格跌破bar_len出场
    3-5周期内收益为0出场；
    5-10周期内（或价格突破bar_len的1.5倍）跌破最高收益的50%出场
    10-15周期内（或价格突破bar_len的2倍）跌破最高收益的30%出场
    >15周期内（或价格突破bar_len的3.6倍）跌破最高收益的20%出场
    :param enter_price: 订单对象
    :param elp_num:
    :param is_short: 是否是做空单
    :param odlens: 周期分割点，4个，分割出5个区间，默认：3,5,10,15
    :param loss_thres: 5个区间的价格止损倍数（基于bar_len），默认：-1, 0, 1.5, 2, 3.6
    :param back_rates: 回撤比率。默认：0.47, 0.28, 0.18
    :return:
    '''
    gain_chg, max_gain_price, back_rate = trail_info(elp_num, enter_price, is_short)
    exit_tag = trail_stop_loss_core(elp_num, max_gain_price, gain_chg, back_rate, loss_thres, odlens, back_rates)
    return dict(tag=exit_tag) if exit_tag else None
