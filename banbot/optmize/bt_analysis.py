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
from banbot.storage.orders import *


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
            data['orders'] = [InOutOrder(**od) for od in data['orders']]
            return BTAnalysis(**data)

    def to_plot(self) -> List[dict]:
        from banbot.storage import db, InOutOrder
        task_id = self.result['task_id']
        with db():
            orders = db.session.query(InOutOrder).filter(InOutOrder.task_id == task_id).all()
        result = [dict(
            col='open',
            type='order',
            enter_id=[od.enter_at for od in orders],
            exit_id=[od.exit_at for od in orders],
            enter_tag=[od.enter_tag for od in orders],
            exit_tag=[od.exit_tag for od in orders],
            enter_price=[od.enter.price for od in orders],
            exit_price=[(od.exit.price if od.exit else od.enter.price) for od in orders],
        )]
        if self.result.get('enters'):
            enter_ind = self.result['enters']
            enter_ind.update(type='mark', symbol='triangle-up-dot')
            result.append(enter_ind)
        return result
