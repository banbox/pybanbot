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
    def __init__(self, config: Config):
        self.config = config
        self.pairlist: List[Tuple[str, str]] = config.get('pairlist')
        self.wallets: WalletsLocal = None
        self.order_hold: OrderManager = None
        self.symbol_stgs: Dict[str, List[BaseStrategy]] = dict()

    def _load_strategies(self):
        run_jobs = load_run_jobs(self.config)
        for pair, timeframe, cls_list in run_jobs:
            symbol = f'{pair}/{timeframe}'
            set_context(symbol)
            self.symbol_stgs[symbol] = [cls(self.config) for cls in cls_list]

    def on_data_feed(self, row: np.ndarray):
        strategy_list = self.symbol_stgs[symbol_tf.get()]
        pair, base_s, quote_s, timeframe = get_cur_symbol()
        pair_arr = append_new_bar(row)
        self._bar_callback()
        self.order_hold.update_by_bar(pair_arr)
        start_time = time.time()
        for strategy in strategy_list:
            stg_name = strategy.__class__.__name__
            strategy.state = dict()
            strategy.on_bar(pair_arr)
            # 调用策略生成入场和出场信号
            entry_tag = strategy.on_entry(pair_arr)
            exit_tag = strategy.on_exit(pair_arr)
            if entry_tag or exit_tag:
                logger.trade_info(f'[{stg_name}] enter: {entry_tag}  exit: {exit_tag}')
            if entry_tag and (not strategy.skip_enter_on_exit or not exit_tag):
                cost = strategy.custom_cost(entry_tag)
                price = pair_arr[-1][ccol]
                self.order_hold.enter_order(stg_name, pair, entry_tag, cost, price)
            if not strategy.skip_exit_on_enter or not entry_tag:
                if exit_tag:
                    self.order_hold.exit_open_orders(stg_name, pair, exit_tag)
                self.order_hold.check_custom_exits(pair_arr, strategy)
        cost = time.time() - start_time
        if cost > 0.05 and btime.run_mode in btime.TRADING_MODES:
            logger.info(f'handle bar {bar_end_time.get()} cost: {cost * 1000:.1f} ms')

    def _bar_callback(self):
        pass

    def run(self):
        raise NotImplementedError('`run` is not implemented')

