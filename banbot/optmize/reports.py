#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : reports.py
# Author: anyongjin
# Date  : 2023/3/20
from typing import *

import numpy as np
import pandas as pd
from tabulate import tabulate

from banbot.util.common import logger


def _calc_winloss(df: pd.DataFrame):
    wins = len(df[df['profit'] >= 0])
    losses = len(df[df['profit'] < 0])
    if wins > 0 and losses == 0:
        win_rate = '100'
    elif wins == 0:
        win_rate = '0'
    else:
        win_rate = f'{100.0 / (wins + losses) * wins:.0f}' if losses > 0 else '100'
    return f'{wins:>4}  {losses:>4}  {win_rate:>4}%'


def text_pair_groups(df: pd.DataFrame):
    col_key = 'symbol'
    pairs = list(df[col_key].value_counts().items())
    if len(pairs) <= 1:
        # 单币种回测不显示按币种的收益
        return ''
    headers = ['Pair', 'Count', 'AvgProfit %', 'TotProfit %', 'SumProfit', 'Duration', 'WinLossRate']
    hd_fmts = ['s', 'd', '.2f', '.2f', '.3f', '.2f', '.1f', 's']
    data = []
    for key, count in pairs:
        result = df[df[col_key] == key]
        profit_sum = result['profit'].sum()
        profit_tot_pct = profit_sum * 100 * len(result) / result['enter_cost'].sum()
        data.append([
            key,
            len(result),
            result['profit_rate'].mean() * 100 if len(result) else 0.0,  # profit_mean
            profit_tot_pct,
            profit_sum,
            _group_durations(result['duration'].tolist(), 3),
            _calc_winloss(result)
        ])
    return tabulate(data, headers, 'orgtbl', hd_fmts, stralign='right')


def text_order_tags(df: pd.DataFrame, tag_type: str):
    if not len(df):
        return ''
    headers = ['Tag', 'Count', 'AvgProfit %', 'TotProfit %', 'SumProfit', 'Duration', 'WinLossRate']
    headers[0] = 'EnterTag' if tag_type != 'exit' else 'ExitTag'
    hd_fmts = ['s', 'd', '.2f', '.2f', '.3f', '.2f', '.1f', 's']
    col_key = f'{tag_type}_tag'
    data = []
    for tag, count in df[col_key].value_counts().items():
        result = df[df[col_key] == tag]
        profit_sum = result['profit'].sum()
        profit_tot_pct = profit_sum * 100 * len(result) / result['enter_cost'].sum()
        data.append([
            tag,
            len(result),
            result['profit_rate'].mean() * 100 if len(result) else 0.0,  # profit_mean
            profit_tot_pct,
            profit_sum,
            _group_durations(result['duration'].tolist(), 3),
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
        ('Total Investment', data['total_invest']),
        ('Finish Balance', data['final_balance']),
        ('Finish WithDraw', data['final_withdraw']),
        ('Absolute profit', data['abs_profit']),
        ('Total Fee', data['total_fee']),
        ('Total profit %', data['total_profit_pct']),
        ('Avg profit ‰', data['avg_profit_pct']),
        ('Avg. stake amount', data['avg_stake_amount']),
        ('Total trade volume', data['tot_stake_amount']),
        ('Best trade', data['best_trade']),
        ('Worst trade', data['worst_trade']),
        ('Min balance', data['min_balance']),
        ('max_balance', data['max_balance']),
    ]
    return tabulate(records, headers, 'orgtbl')


def _group_counts(df: pd.DataFrame, col: str, show_num=3):
    res = df.groupby(col)[col].count().reset_index(name='num').sort_values('num', ascending=False)[:show_num]
    out_list = []
    for rid, row in res.iterrows():
        out_list.append(f"{row[col]}/{row['num']}")
    return ' '.join(out_list)


def _group_durations(durations: List[int], cls_num: int) -> str:
    from banbot.util.num_utils import cluster_kmeans
    if not durations:
        return '0'
    from io import StringIO
    cls_num = min(len(durations), cls_num)
    row_gps, centers = cluster_kmeans(np.array(durations), cls_num)
    centers = sorted(zip(range(len(centers)), centers), key=lambda x: x[1])
    result = StringIO()
    for gid, center_val in centers:
        row_ids = [i for i, v in enumerate(row_gps) if v == gid]
        if result.tell():
            result.write('  ')
        result.write(str(round(center_val)))
        result.write('/')
        rate = round(len(row_ids) * 20 / len(durations))
        result.write(str(rate))
    return result.getvalue()


def text_profit_groups(df: pd.DataFrame) -> str:
    if not len(df):
        return ''
    from banbot.util.num_utils import cluster_kmeans
    headers = ['Profit Range', 'Count', 'TotProfit %', 'TotProfit', 'Duration', 'EnterTags', 'ExitTags']
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
        profit_rate = tot_profit * len(gp_df) / gp_df['enter_cost'].sum()
        records.append([
            f'{profit_min * 100:.2f} ~ {profit_max * 100:.2f}%',
            len(gp_df),
            f"{profit_rate * 100:.2f}%",
            f"{tot_profit:.3f}",
            _group_durations(gp_df['duration'].tolist(), 3),
            _group_counts(gp_df, 'enter_tag'),
            _group_counts(gp_df, 'exit_tag'),
        ])
    return tabulate(records, headers, 'orgtbl')


def text_day_profits(df: pd.DataFrame):
    from banbot.util import btime
    headers = ['Date', 'Count', 'TotProfit %', 'TotProfit', 'WinLossRate', 'Duration', 'EnterTags', 'ExitTags']
    records = []
    df['date'] = df['enter_create_at'].apply(lambda x: btime.to_datestr(x, fmt='%Y-%m-%d'))

    def profile_record(result: pd.DataFrame, idx_val: str):
        profit_sum = result['profit'].sum()
        profit_tot_pct = profit_sum * 100 / result['enter_cost'].mean()
        records.append([
            idx_val,
            len(result),
            f'{profit_tot_pct:.2f}',  # 总利润率（相对于总成本）
            f'{profit_sum:.3f}',  # profit_total
            _calc_winloss(result),
            _group_durations(result['duration'].tolist(), 3),
            _group_counts(result, 'enter_tag'),
            _group_counts(result, 'exit_tag'),
        ])
    for date_val, count in df['date'].value_counts().sort_index().items():
        profile_record(df[df['date'] == date_val], date_val)
    profile_record(df, 'total')
    return tabulate(records, headers, 'orgtbl')


async def get_order_df() -> pd.DataFrame:
    from banbot.storage import InOutOrder
    his_orders = await InOutOrder.his_orders()
    data_list = [od.dict(flat_sub=True) for od in his_orders]
    return pd.DataFrame(data_list)


async def print_backtest(result: dict):
    order_df = await get_order_df()
    out = []
    if len(order_df):
        table = text_pair_groups(order_df)
        if isinstance(table, str) and len(table) > 0:
            out.append(' Pair Profits '.center(len(table.splitlines()[0]), '='))
            out.append(table)
        table = text_day_profits(order_df)
        if isinstance(table, str) and len(table) > 0:
            out.append(' Day Profits '.center(len(table.splitlines()[0]), '='))
            out.append(table)
        table = text_profit_groups(order_df)
        if isinstance(table, str) and len(table) > 0:
            out.append(' Profit Groups '.center(len(table.splitlines()[0]), '='))
            out.append(table)
        table = text_order_tags(order_df, 'enter')
        if isinstance(table, str) and len(table) > 0:
            out.append(' Enter Tag '.center(len(table.splitlines()[0]), '='))
            out.append(table)
        table = text_order_tags(order_df, 'exit')
        if isinstance(table, str) and len(table) > 0:
            out.append(' Exit Tag '.center(len(table.splitlines()[0]), '='))
            out.append(table)
    else:
        logger.error('NO Order Found !')
    table = text_bt_metrics(result)
    if isinstance(table, str) and len(table) > 0:
        out.append(' SUMMARY METRICS '.center(len(table.splitlines()[0]), '='))
        out.append(table)
    if out:
        out_text = "\n".join(out)
        logger.info(f'BackTest Reports:\n{out_text}')
