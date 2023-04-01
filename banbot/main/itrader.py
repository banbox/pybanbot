#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : itrader.py
# Author: anyongjin
# Date  : 2023/3/17
from __future__ import annotations

from banbot.strategy.base import *
from banbot.config import *
from banbot.storage.od_manager import *
from banbot.strategy.resolver import load_run_jobs


class Trader:
    def __init__(self):
        self.pairlist: List[Tuple[str, str]] = cfg.get('pairlist')
        self.wallets: WalletsLocal = None
        self.order_hold: OrderManager = None
        self.symbol_stgs: Dict[str, List[BaseStrategy]] = dict()

    def _load_strategies(self):
        run_jobs = load_run_jobs(cfg)
        for pair, timeframe, cls_list in run_jobs:
            symbol = f'{pair}/{timeframe}'
            set_context(symbol)
            self.symbol_stgs[symbol] = [cls(cfg) for cls in cls_list]

    def on_data_feed(self, row: np.ndarray):
        strategy_list = self.symbol_stgs[symbol_tf.get()]
        state = pair_state.get()
        pair, base_s, quote_s, timeframe = get_cur_symbol()
        pair_arr = append_new_bar(row)
        self._bar_callback()
        self.order_hold.update_by_bar(pair_arr)
        for strategy in strategy_list:
            stg_name = strategy.__class__.__name__
            strategy.state = dict()
            strategy.on_bar(pair_arr)
            # 调用策略生成入场和出场信号
            entry_tag = strategy.on_entry(pair_arr)
            exit_tag = strategy.on_exit(pair_arr)
            if entry_tag and not exit_tag:
                state['last_enter'] = bar_num.get()
                cost = strategy.custom_cost(entry_tag)
                price = pair_arr[-1][ccol]
                self.order_hold.enter_order(stg_name, pair, entry_tag, cost, price)
            elif exit_tag:
                self.order_hold.exit_open_orders(stg_name, pair, exit_tag)
            self.order_hold.check_custom_exits(pair_arr, strategy)

    def _bar_callback(self):
        pass

    def run(self):
        raise NotImplementedError('`run` is not implemented')

