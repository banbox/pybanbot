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

import aiofiles as aiof
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
    iorders = await get_db_orders(task_id)
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
        if iod.enter and iod.enter.price and iod.enter.amount:
            enter_cost = iod.enter.price * iod.enter.amount
            if iod.leverage and iod.leverage > 1:
                enter_cost /= iod.leverage
            item.update(dict(
                enter_price=iod.enter.price,
                enter_amount=iod.enter.amount,
                enter_cost=enter_cost
            ))
        item.update(dict(
            exit_at=btime.to_datestr(iod.exit_at),
            exit_tag=iod.exit_tag
        ))
        if iod.exit:
            item.update(dict(
                exit_price=iod.exit.price,
                exit_amount=iod.exit.amount,
                exit_got=enter_cost + iod.profit
            ))
        item.update(dict(profit_rate=iod.profit_rate, profit=iod.profit))
        result.append(item)
    df = pd.DataFrame(result)
    out_path = os.path.join(out_dir, 'orders.csv')
    df.to_csv(out_path, sep=',')


def dump_hist_assets(assets: List[Tuple[datetime.datetime, float]], out_dir: str, max_num: int = 600):
    '''
    输出总资产曲线
    '''
    if len(assets) > max_num * 2:
        step = round(len(assets) / max_num)
        assets = [r for i, r in enumerate(assets) if i % step == 0]
    x_dates, y_values = list(zip(*assets))
    import plotly.graph_objects as go
    fig = go.Figure(
        data=[go.Scatter(x=x_dates, y=y_values, line=dict(color='blue'))],
        layout=dict(title=dict(text='总资产曲线'))
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
        async with aiof.open(dump_path, 'wb') as fout:
            await fout.write(orjson.dumps(self.result))
        # 保存订单记录到CSV
        await dump_orders(task_id, task_dir)
        # 保存总资产曲线
        dump_hist_assets(self.result['bar_assets'], task_dir)

    @staticmethod
    async def load(save_dir: str) -> 'BTAnalysis':
        dump_path = os.path.join(save_dir, 'backtest.json')
        async with aiof.open(dump_path, 'rb') as fout:
            data: dict = orjson.loads(await fout.read())
            return BTAnalysis(**data)


