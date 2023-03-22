#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : reports.py
# Author: anyongjin
# Date  : 2023/3/20
from tabulate import tabulate
from typing import *
import pandas as pd


def text_order_tags(df: pd.DataFrame, tag_type: str):
    headers = ['标签', '数量', '平均利润 %', '总利润 %', '总利润', '总利润 %', '持续时间', '盈亏 胜率']
    headers[0] = '入场信号' if tag_type != 'exit' else '退出信号'
    hd_fmts = ['s', 'd', '.2f', '.2f', '.3f', '.2f', '.1f', 's']
    col_key = f'{tag_type}_tag'
    data = []
    for tag, count in df[col_key].value_counts().items():
        result = df[df[col_key] == tag]
        wins = len(result[result['profit'] > 0])
        draws = len(result[result['profit'] == 0])
        losses = len(result[result['profit'] < 0])
        if wins > 0 and losses == 0:
            win_rate = '100'
        elif wins == 0:
            win_rate = '0'
        else:
            win_rate = f'{100.0 / (wins + draws + losses) * wins:.1f}' if losses > 0 else '100'
        profit_sum = result['profit'].sum()
        profit_tot_pct = profit_sum / result['amount'].sum()
        data.append([
            tag,
            len(result),
            result['profit_rate'].mean() * 100 if len(result) else 0.0,  # profit_mean
            result['profit_rate'].sum() * 100,  # profit_sum
            profit_sum,  # profit_total
            profit_tot_pct,  # 总利润率
            result['duration'].mean() if len(result) else 0,  # duration_avg
            f'{wins:>4}  {draws:>4}  {losses:>4}  {win_rate:>4}'
        ])
    return tabulate(data, headers, 'orgtbl', hd_fmts, stralign='right')


def print_backtest(df: pd.DataFrame):
    table = text_order_tags(df, 'enter')
    if isinstance(table, str) and len(table) > 0:
        print(' 入场信号 '.center(len(table.splitlines()[0]), '='))
        print(table)
    table = text_order_tags(df, 'exit')
    if isinstance(table, str) and len(table) > 0:
        print(' 退出信号 '.center(len(table.splitlines()[0]), '='))
        print(table)
