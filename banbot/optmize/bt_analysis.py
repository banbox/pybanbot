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
    def __init__(self, orders: List[Order], data_path: str, start_pos: int = 0, end_pos: int = 0):
        self.orders = orders
        self.data_path = data_path
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.df: Optional[DataFrame] = None

    def save(self, save_dir: str):
        data = dict(
            orders=[od.to_dict() for od in self.orders],
            data_path=self.data_path,
            start_pos=self.start_pos,
            end_pos=self.end_pos
        )
        dump_path = os.path.join(save_dir, 'backtest.json')
        with open(dump_path, 'wb') as fout:
            fout.write(orjson.dumps(data))

    @staticmethod
    def load(save_dir: str) -> 'BTAnalysis':
        dump_path = os.path.join(save_dir, 'backtest.json')
        with open(dump_path, 'rb') as fout:
            data: dict = orjson.loads(fout.read())
            data['orders'] = [Order(**od) for od in data['orders']]
            return BTAnalysis(**data)
