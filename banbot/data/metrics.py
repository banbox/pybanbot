#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : metrics.py
# Author: anyongjin
# Date  : 2023/9/7
from typing import *
import numpy as np


def calc_expectancy(profits: List[float]) -> Tuple[float, float]:
    '''
    计算期望盈利。返回：[单笔期望盈利， 风险回报率]
    '''
    if not profits:
        return 0, 0
    win_profits = [val for val in profits if val >= 0]
    loss_profits = [val for val in profits if val < 0]
    profit_sum = sum(win_profits)
    loss_sum = abs(sum(loss_profits))
    win_num, loss_num = len(win_profits), len(loss_profits)

    avg_win = profit_sum / win_num if win_num else 0
    avg_loss = loss_sum / loss_num if loss_num else 0
    win_rate = win_num / len(profits)
    loss_rate = loss_num / len(profits)

    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    expectancy_ratio = 0
    if avg_loss > 0:
        risk_reward_ratio = avg_win / avg_loss
        expectancy_ratio = ((1 + risk_reward_ratio) * win_rate) - 1

    return expectancy, expectancy_ratio


def _calc_drawdowns(profits: List[float], init_balance: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    计算每个周期的回撤。传入的利润一般应先按日分组
    返回：cumulative, highs, drawdown, drawdown_relative
    '''
    val_arr = np.array(profits)
    cumulative = np.cumsum(val_arr)
    highs = np.maximum.accumulate(cumulative)
    drawdown = cumulative - highs
    if init_balance:
        cumulative_balance = init_balance + cumulative
        max_balance = init_balance + highs
        drawdown_rel = (max_balance - cumulative_balance) / max_balance
    else:
        drawdown_rel = (highs - cumulative) / highs
    return cumulative, highs, drawdown, drawdown_rel


def calc_max_drawdown(profits: List[float], init_balance: float = 0, relative=False):
    '''
    计算最大回撤
    返回：绝对最大回撤、最大值索引、最小值索引、最大值、最小值、相对最大回撤
    '''
    cumulative, highs, drawdown, drawdown_rel = _calc_drawdowns(profits, init_balance)
    try:
        idxmin = np.argmax(drawdown_rel) if relative else np.argmin(drawdown)
    except Exception:
        return 0, -1, -1, 0, 0, 0
    idxmax = np.argmax(highs[:idxmin + 1])
    high_val = cumulative[idxmax]
    low_val = cumulative[idxmin]
    max_drawdown_rel = drawdown_rel[idxmin]

    max_drawdown_val = drawdown[idxmin]
    return max_drawdown_val, idxmax, idxmin, high_val, low_val, max_drawdown_rel
