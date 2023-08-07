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


def order_plot_ind(task_id: int, symbol: str, in_color=None, out_color=None, line_color=None) -> dict:
    from banbot.storage import db, InOutOrder
    from banbot.exchange.exchange_utils import tf_to_secs
    from itertools import groupby
    io_where = [InOutOrder.task_id == task_id, InOutOrder.symbol == symbol]
    orders: List[InOutOrder] = db.session.query(InOutOrder).filter(*io_where).all()
    ex_where = [Order.task_id == task_id, Order.symbol == symbol]
    exods: List[Order] = db.session.query(Order).filter(*ex_where).all()
    exods = sorted(exods, key=lambda x: x.inout_id)
    exod_gps = groupby(exods, key=lambda x: x.inout_id)
    exgp_dic = dict()
    for key, gp in exod_gps:
        exgp_dic[key] = list(gp)
    enter_at, exit_at, enter_tag, exit_tag, enter_price, exit_price = [], [], [], [], [], []
    for od in orders:
        if not od.exit_at:
            continue
        tf_ms = tf_to_secs(od.timeframe) * 1000
        # enter_at和exit_at都是发出信号时的13位时间戳。这里近似取整，确保和bar的13位时间戳一致，方便显示到K线图上。
        # 这里实际对应的应是下一个bar，因为基于上一个bar完成后才计算和发出信号。
        enter_at.append(round(od.enter_at / tf_ms) * tf_ms)
        exit_at.append(round(od.exit_at / tf_ms) * tf_ms)
        enter_tag.append(od.enter_tag)
        exit_tag.append(od.exit_tag)
        ex_list = exgp_dic[od.id]
        ex_enter = next((o for o in ex_list if o.enter), None)
        enter_price.append(ex_enter.price)
        ex_exit = next((o for o in ex_list if not o.enter), None)
        exit_price.append(ex_exit.price if ex_exit else ex_enter.price)
    return dict(
        col='open',
        type='order',
        enter_at=pd.to_datetime(enter_at, utc=True, unit='ms'),
        exit_at=pd.to_datetime(exit_at, utc=True, unit='ms'),
        enter_tag=enter_tag,
        exit_tag=exit_tag,
        enter_price=enter_price,
        exit_price=exit_price,
        enter_color=in_color,
        exit_color=out_color,
        line_color=line_color
    )


def dump_orders(task_id: int, out_dir: str):
    '''
    将任务订单输出到CSV文件方便对比
    '''
    if not task_id or task_id < 0:
        return
    from banbot.storage.orders import get_db_orders
    from banbot.util import btime
    iorders = get_db_orders(task_id)
    iorders = sorted(iorders, key=lambda x: x.enter_at)
    result = []
    for iod in iorders:
        weight = iod.get_info('weight')
        cost_rate = iod.get_info('cost_rate')
        item = dict(
            sid=iod.sid,
            symbol=iod.symbol,
            timeframe=iod.timeframe,
            lock_key=iod.lock_key,
            direction='short' if iod.short else 'long',
            weight=weight,
            cost_rate=cost_rate,
            enter_at=btime.to_datestr(iod.enter_at),
            enter_tag=iod.enter_tag,
        )
        total_fee = 0
        if iod.enter:
            item.update(dict(
                enter_price=iod.enter.price,
                enter_amount=iod.enter.amount,
                enter_cost=iod.enter.price * iod.enter.amount
            ))
            total_fee = iod.enter.fee
        item.update(dict(
            exit_at=btime.to_datestr(iod.exit_at),
            exit_tag=iod.exit_tag
        ))
        if iod.exit:
            total_fee += iod.exit.fee
            item.update(dict(
                exit_price=iod.exit.price,
                exit_amount=iod.exit.amount,
                exit_got=iod.exit.price * (1 - total_fee) * iod.exit.amount
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
        dump_orders(task_id, task_dir)
        # 保存总资产曲线
        dump_hist_assets(self.result['bar_assets'], task_dir)

    @staticmethod
    async def load(save_dir: str) -> 'BTAnalysis':
        dump_path = os.path.join(save_dir, 'backtest.json')
        async with aiof.open(dump_path, 'rb') as fout:
            data: dict = orjson.loads(await fout.read())
            return BTAnalysis(**data)

    def load_df(self, start_ms: int = None, stop_ms: int = None):
        from banbot.storage import KLine, ExSymbol
        from banbot.data import KCols
        pair, timeframe = self.result['pair'], self.result['timeframe']
        if not start_ms:
            start_ms = self.result['ts_from']
        if not stop_ms:
            stop_ms = self.result['ts_to']
        exs = ExSymbol.get('binance', 'spot', pair)
        candles = KLine.query(exs, timeframe, start_ms, stop_ms + 1)
        df = pd.DataFrame(candles, columns=KCols)
        df['date'] = df['date'].astype(np.int64)
        return df

    def to_plot(self, symbol: str) -> List[dict]:
        from banbot.storage import db
        with db():
            result = [order_plot_ind(self.result['task_id'], symbol)]
        if self.result.get('enters'):
            enter_ind = self.result['enters']
            enter_ind.update(type='mark', symbol='triangle-up-dot')
            result.append(enter_ind)
        return result
