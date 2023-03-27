#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : common.py
# Author: anyongjin
# Date  : 2023/3/27
from banbot.bar_driven.tainds import *
from banbot.persistence.trades import *


def trail_stop_loss(arr: np.ndarray, od: Order, odlens: List[int] = None, loss_thres: List[float] = None,
                    back_rates: List[float] = None) -> Optional[str]:
    '''
    跟踪止损。
    3周期内，价格跌破bar_len出场
    3-5周期内收益为0出场；
    5-10周期内跌破最高收益的50%出场（价格突破bar_len的1.5倍时也执行此项）
    10-15周期内跌破最高收益的30%出场（价格突破bar_len的2倍时也执行此项）
    >15周期内跌破最高收益的20%出场（价格突破bar_len的3.6倍时也执行此项）
    :param arr: K线数据
    :param od: 订单对象
    :param odlens: 周期分割点，4个，分割出5个区间，默认：3,5,10,15
    :param loss_thres: 5个区间的价格止损倍数（基于bar_len），默认：-1, 0, 1.5, 2, 3.6
    :param back_rates: 回撤比率。默认：0.47, 0.28, 0.18
    :return:
    '''
    if bar_num.get() == bar_state.get()['last_enter']:
        return
    elp_num = bar_num.get() - od.enter_at
    max_price = np.max(arr[-elp_num:, 3])
    cur_close = arr[-1, 3]
    max_loss = min(cur_close - od.price, cur_close - max_price)
    bar_len = LongVar.get(LongVar.bar_len).val
    flen, slen, mlen, llen = odlens if odlens else 3, 5, 10, 15
    pf_n2, pf_n1, pf_1, pf_2, pf_3 = loss_thres if loss_thres else -1, -0, 1.5, 2, 3.6
    if elp_num <= flen and max_loss < bar_len * pf_n2:
        return 'sm_ls'
    if flen < elp_num <= slen and max_loss < bar_len * pf_n1:
        return 'loss6'
    if not back_rates:
        back_rates = 0.47, 0.28, 0.18
    dust = min(0.00001, cur_close * 0.0001)
    max_change = max(dust, max_price - od.price)
    back_rate = (max_price - cur_close) / max_change
    if back_rate >= back_rates[0] and (slen < elp_num <= mlen or max_change > bar_len * pf_1):
        return 'back.5'
    elif back_rate >= back_rates[1] and (mlen < elp_num <= llen or max_change > bar_len * pf_2):
        return 'back.3'
    elif back_rate >= back_rates[2] and (llen < elp_num or max_change > bar_len * pf_3):
        return 'back.2'