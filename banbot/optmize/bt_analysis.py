#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : bt_analysis.py
# Author: anyongjin
# Date  : 2023/3/22
'''
分析回测结果，优化策略
'''
import os.path

import orjson
from typing import *
from banbot.persistence.trades import *
from pandas import DataFrame


class BTAnalysis:
    def __init__(self, orders: List[Order], **kwargs):
        self.orders = orders
        self.result = kwargs

    def save(self, save_dir: str):
        self.result['orders'] = [od.to_dict() for od in self.orders]
        dump_path = os.path.join(save_dir, 'backtest.json')
        with open(dump_path, 'wb') as fout:
            fout.write(orjson.dumps(self.result))

    @staticmethod
    def load(save_dir: str) -> 'BTAnalysis':
        dump_path = os.path.join(save_dir, 'backtest.json')
        with open(dump_path, 'rb') as fout:
            data: dict = orjson.loads(fout.read())
            data['orders'] = [Order(**od) for od in data['orders']]
            return BTAnalysis(**data)

    def to_dataframe(self) -> DataFrame:
        from banbot.optmize.backtest import BackTest
        return BackTest.load_data()

    def to_plot(self) -> List[dict]:
        enter_id = [od.enter_at - 1 for od in self.orders]
        exit_id = [od.exit_at - 1 for od in self.orders]
        enter_tag = [od.enter_tag for od in self.orders]
        exit_tag = [od.exit_tag for od in self.orders]
        return [dict(
            col='open',
            type='order',
            enter_id=enter_id,
            exit_id=exit_id,
            enter_tag=enter_tag,
            exit_tag=exit_tag
        )]
