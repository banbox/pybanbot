#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : reports.py
# Author: anyongjin
# Date  : 2023/3/20
from tabulate import tabulate
from typing import *
import pandas as pd


def _calc_winloss(df: pd.DataFrame):
    wins = len(df[df['profit'] > 0])
    draws = len(df[df['profit'] == 0])
    losses = len(df[df['profit'] < 0])
    if wins > 0 and losses == 0:
        win_rate = '100'
    elif wins == 0:
        win_rate = '0'
    else:
        win_rate = f'{100.0 / (wins + draws + losses) * wins:.1f}' if losses > 0 else '100'
    return f'{wins:>4}  {draws:>4}  {losses:>4}  {win_rate:>4}'


def text_order_tags(df: pd.DataFrame, tag_type: str):
    if not len(df):
        return ''
    headers = ['Tag', 'Count', 'AvgProfit %', 'SumProfit %', 'SumProfit', 'TotProfit %', 'Duration', 'WinDrawLoss']
    headers[0] = 'EnterTag' if tag_type != 'exit' else 'ExitTag'
    hd_fmts = ['s', 'd', '.2f', '.2f', '.3f', '.2f', '.1f', 's']
    col_key = f'{tag_type}_tag'
    data = []
    for tag, count in df[col_key].value_counts().items():
        result = df[df[col_key] == tag]
        profit_sum = result['profit'].sum()
        profit_tot_pct = profit_sum / result['enter_cost'].sum()
        data.append([
            tag,
            len(result),
            result['profit_rate'].mean() * 100 if len(result) else 0.0,  # profit_mean
            result['profit_rate'].sum() * 100,  # profit_sum
            profit_sum,  # profit_total
            profit_tot_pct,  # 总利润率
            result['duration'].mean() if len(result) else 0,  # duration_avg
            _calc_winloss(result)
        ])
    return tabulate(data, headers, 'orgtbl', hd_fmts, stralign='right')


def text_bt_metrics(data: dict):
    headers = ['Metric', 'Value']
    records = [
        ('Backtest From', data['date_from']),
        ('Backtest To', data['date_to']),
        ('Max open trades', data['max_open_orders']),
        ('Total Trade/Bar Num', f"{data['orders_num']}/{data['bar_num']}"),
        ('Starting Balance', data['start_balance']),
        ('Finish Balance', data['final_balance']),
        ('Absolute profit', data['abs_profit']),
        ('Total profit %', data['total_profit_pct']),
        ('Avg profit %o', data['avg_profit_pct']),
        ('Avg. stake amount', data['avg_stake_amount']),
        ('Total trade volume', data['tot_stake_amount']),
        ('Best trade', data['best_trade']),
        ('Worst trade', data['worst_trade']),
        ('Min balance', data['min_balance']),
        ('max_balance', data['max_balance']),
        ('Market change', data['market_change'])
    ]
    return tabulate(records, headers, 'orgtbl')


def _group_counts(df: pd.DataFrame, col: str, show_num=3):
    res = df.groupby(col)[col].count().reset_index(name='num').sort_values('num', ascending=False)[:show_num]
    out_list = []
    for rid, row in res.iterrows():
        out_list.append(f"{row[col]}/{row['num']}")
    return ' '.join(out_list)


def text_order_profits(df: pd.DataFrame) -> str:
    if not len(df):
        return ''
    from banbot.util.num_utils import cluster_kmeans
    headers = ['Profit Range', 'Count', 'AvgProfit %', 'TotProfit', 'Duration', 'EnterTags', 'ExitTags']
    if len(df) > 150:
        cls_num = min(19, round(pow(len(df), 0.5)))
    else:
        cls_num = round(pow(len(df), 0.6))
    row_gps, centers = cluster_kmeans(df['profit_rate'].tolist(), cls_num)
    centers = sorted(zip(range(len(centers)), centers), key=lambda x: x[1], reverse=True)
    records = []
    for gid, center_val in centers:
        row_ids = [i for i, v in enumerate(row_gps) if v == gid]
        gp_df = df.loc[row_ids]
        profit_min, profit_max = gp_df['profit_rate'].min(), gp_df['profit_rate'].max()
        tot_profit = gp_df['profit'].sum()
        profit_rate = tot_profit / gp_df['enter_cost'].sum()
        records.append([
            f'{profit_min * 100:.2f} ~ {profit_max * 100:.2f}%',
            len(gp_df),
            f"{profit_rate * 100:.2f}%",
            f"{tot_profit:.3f}",
            f"{gp_df['duration'].mean():.1f}",
            _group_counts(gp_df, 'enter_tag'),
            _group_counts(gp_df, 'exit_tag'),
        ])
    return tabulate(records, headers, 'orgtbl')


def print_backtest(order_df: pd.DataFrame, result: dict):
    table = text_order_profits(order_df)
    if isinstance(table, str) and len(table) > 0:
        print(' Profit Groups '.center(len(table.splitlines()[0]), '='))
        print(table)
    table = text_order_tags(order_df, 'enter')
    if isinstance(table, str) and len(table) > 0:
        print(' Enter Tag '.center(len(table.splitlines()[0]), '='))
        print(table)
    table = text_order_tags(order_df, 'exit')
    if isinstance(table, str) and len(table) > 0:
        print(' Exit Tag '.center(len(table.splitlines()[0]), '='))
        print(table)
    table = text_bt_metrics(result)
    if isinstance(table, str) and len(table) > 0:
        print(' SUMMARY METRICS '.center(len(table.splitlines()[0]), '='))
        print(table)
