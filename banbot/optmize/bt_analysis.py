#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : bt_analysis.py
# Author: anyongjin
# Date  : 2023/3/22
'''
分析回测结果，优化策略
'''
import os.path
import aiofiles as aiof
import orjson
import pandas as pd

from banbot.storage.orders import *


def order_plot_ind(task_id: int, in_color=None, out_color=None, line_color=None) -> dict:
    from banbot.storage import db, InOutOrder
    from banbot.exchange.exchange_utils import tf_to_secs
    from itertools import groupby
    orders: List[InOutOrder] = db.session.query(InOutOrder).filter(InOutOrder.task_id == task_id).all()
    exods: List[Order] = db.session.query(Order).filter(Order.task_id == task_id).all()
    exods = sorted(exods, key=lambda x: x.inout_id)
    exod_gps = groupby(exods, key=lambda x: x.inout_id)
    exgp_dic = dict()
    for key, gp in exod_gps:
        exgp_dic[key] = list(gp)
    enter_at, exit_at, enter_tag, exit_tag, enter_price, exit_price = [], [], [], [], [], []
    for od in orders:
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


class BTAnalysis:
    def __init__(self, **kwargs):
        self.result = kwargs

    async def save(self, save_dir: str):
        dump_path = os.path.join(save_dir, 'backtest.json')
        async with aiof.open(dump_path, 'wb') as fout:
            await fout.write(orjson.dumps(self.result))

    @staticmethod
    async def load(save_dir: str) -> 'BTAnalysis':
        dump_path = os.path.join(save_dir, 'backtest.json')
        async with aiof.open(dump_path, 'rb') as fout:
            data: dict = orjson.loads(await fout.read())
            return BTAnalysis(**data)

    def load_df(self, start_ms: int = None, stop_ms: int = None):
        from banbot.storage import KLine
        from banbot.data import KCols
        pair, timeframe = self.result['pair'], self.result['timeframe']
        if not start_ms:
            start_ms = self.result['ts_from']
        if not stop_ms:
            stop_ms = self.result['ts_to']
        candles = KLine.query('binance', pair, timeframe, start_ms, stop_ms + 1)
        df = pd.DataFrame(candles, columns=KCols)
        df['date'] = df['date'].astype(np.int64)
        return df

    def to_plot(self) -> List[dict]:
        from banbot.storage import db
        with db():
            result = [order_plot_ind(self.result['task_id'])]
        if self.result.get('enters'):
            enter_ind = self.result['enters']
            enter_ind.update(type='mark', symbol='triangle-up-dot')
            result.append(enter_ind)
        return result
