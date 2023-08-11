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


def trail_info(arr: np.ndarray, elp_num: int, enter_price: float):
    # 全局最高价，这里测试使用收盘价计算效果更好
    max_price = np.max(arr[-elp_num:, ccol])
    cur_close = arr[-1, ccol]
    # loss_val = cur_close - max(enter_price, max_price)
    # 这里应该用相对入场价的损失，刚入场波动较大，否则很容易被中途甩出
    loss_val = cur_close - enter_price
    dust = min(0.00001, cur_close * 0.0001)
    max_updiff = max(dust, max_price - enter_price)
    back_rate = (max_price - cur_close) / max_updiff
    return loss_val, max_updiff, back_rate


def trail_stop_loss_core(elp_num: int, max_up: float, loss_val: float, back_rate: float,
                         loss_thres: List[float], odlens: List[int] = None, back_rates: List[float] = None):
    if odlens:
        flen, slen, mlen, llen = odlens
    else:
        flen, slen, mlen, llen = 3, 5, 10, 15
    pf_n2, pf_n1, pf_1, pf_2, pf_3 = loss_thres
    if elp_num <= flen and loss_val < pf_n2:
        return 'ls_b'
    if flen < elp_num <= slen and loss_val < pf_n1:
        return 'ls_s'
    if not back_rates:
        back_rates = 0.47, 0.28, 0.18
    if back_rate >= back_rates[0] and (slen < elp_num <= mlen or max_up > pf_1):
        return 'back.5'
    elif back_rate >= back_rates[1] and (mlen < elp_num <= llen or max_up > pf_2):
        return 'back.3'
    elif back_rate >= back_rates[2] and (llen < elp_num or max_up > pf_3):
        return 'back.2'


def trail_stop_loss(arr: np.ndarray, enter_price: float, elp_num: int, loss_thres: List[float],
                    odlens: List[int] = None, back_rates: List[float] = None) -> Optional[dict]:
    '''
    跟踪止损。
    3周期内，价格跌破bar_len出场
    3-5周期内收益为0出场；
    5-10周期内（或价格突破bar_len的1.5倍）跌破最高收益的50%出场
    10-15周期内（或价格突破bar_len的2倍）跌破最高收益的30%出场
    >15周期内（或价格突破bar_len的3.6倍）跌破最高收益的20%出场
    :param arr: K线数据
    :param enter_price: 订单对象
    :param elp_num:
    :param odlens: 周期分割点，4个，分割出5个区间，默认：3,5,10,15
    :param loss_thres: 5个区间的价格止损倍数（基于bar_len），默认：-1, 0, 1.5, 2, 3.6
    :param back_rates: 回撤比率。默认：0.47, 0.28, 0.18
    :return:
    '''
    loss_val, max_up, back_rate = trail_info(arr, elp_num, enter_price)
    exit_tag = trail_stop_loss_core(elp_num, max_up, loss_val, back_rate, loss_thres, odlens, back_rates)
    return dict(tag=exit_tag) if exit_tag else None
