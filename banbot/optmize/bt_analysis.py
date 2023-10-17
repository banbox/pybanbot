#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : bt_analysis.py
# Author: anyongjin
# Date  : 2023/3/22
'''
分析回测结果，优化策略
'''
import datetime
import os.path

import orjson
import pandas as pd

from banbot.storage.orders import *


async def dump_orders(task_id: int, out_dir: str):
    '''
    将任务订单输出到CSV文件方便对比
    '''
    if not task_id or task_id < 0:
        return
    from banbot.storage.orders import get_db_orders
    from banbot.util import btime
    iorders = await get_db_orders(task_id=task_id)
    iorders = sorted(iorders, key=lambda x: x.enter_at)
    result = []
    for iod in iorders:
        weight = iod.get_info('weight')
        cost_rate = iod.get_info('cost_rate')
        wallet_left = iod.get_info('wallet_left')
        item = dict(
            sid=iod.sid,
            symbol=iod.symbol,
            timeframe=iod.timeframe,
            direction='short' if iod.short else 'long',
            weight=weight,
            cost_rate=cost_rate,
            leverage=iod.leverage,
            wallet_left=wallet_left,
            enter_at=btime.to_datestr(iod.enter_at),
            enter_tag=iod.enter_tag,
        )
        enter_cost = 0
        in_fee, out_fee, profit_val = iod.fee_profit()
        if iod.enter and iod.enter.price and iod.enter.amount:
            enter_cost = iod.enter.price * iod.enter.amount
            if iod.leverage and iod.leverage > 1:
                enter_cost /= iod.leverage
            item.update(dict(
                enter_price=round(iod.enter.price, 6),
                enter_amount=iod.enter.amount,
                enter_cost=enter_cost,
                enter_fee=round(in_fee, 5)
            ))
        item.update(dict(
            exit_at=btime.to_datestr(iod.exit_at),
            exit_tag=iod.exit_tag
        ))
        if iod.exit:
            item.update(dict(
                exit_price=round(iod.exit.price, 6),
                exit_amount=iod.exit.amount,
                exit_got=enter_cost + iod.profit,
                exit_fee=round(out_fee, 5)
            ))
        item.update(dict(profit_rate=iod.profit_rate, profit=profit_val))
        result.append(item)
    df = pd.DataFrame(result)
    out_path = os.path.join(out_dir, 'orders.csv')
    df.to_csv(out_path, sep=',')


def dump_graph(assets: Dict[str, List[Tuple[datetime.datetime, float]]], out_dir: str, max_num: int = 600):
    '''
    输出总资产曲线
    '''
    import plotly.graph_objects as go
    import plotly.express as px
    plot_data = []
    colors = px.colors.qualitative.Plotly
    idx = -1
    for key, data in assets.items():
        idx += 1
        if len(data) > max_num * 2:
            step = round(len(data) / max_num)
            last = data[-1]
            data = [r for i, r in enumerate(data[:-1]) if i % step == 0]
            data.append(last)
        x_dates, y_values = list(zip(*data))
        color = colors[idx % len(colors)]
        plot_data.append(go.Scatter(x=x_dates, y=y_values, line=dict(color=color), name=key))
    fig = go.Figure(
        data=plot_data,
        layout=dict(title=dict(text='实时资产/利润/余额/提现'))
    )
    out_path = os.path.join(out_dir, 'assets.html')
    fig.write_html(out_path)


class BTAnalysis:
    def __init__(self, **kwargs):
        self.result = kwargs

    async def save(self, save_dir: str):
        task_id = self.result['task_id']
        task_dir = os.path.join(save_dir, f'task_{task_id}')
        if not os.path.isdir(task_dir):
            os.mkdir(task_dir)
        dump_path = os.path.join(task_dir, 'result.json')
        with open(dump_path, 'wb') as fout:
            fout.write(orjson.dumps(self.result))
        # 保存订单记录到CSV
        await dump_orders(task_id, task_dir)
        # 保存总资产曲线
        dump_graph(self.result['graph_data'], task_dir)

    @staticmethod
    def load(save_dir: str) -> 'BTAnalysis':
        dump_path = os.path.join(save_dir, 'backtest.json')
        with open(dump_path, 'rb') as fout:
            data: dict = orjson.loads(fout.read())
            return BTAnalysis(**data)


