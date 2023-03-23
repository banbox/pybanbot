#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : backtest.py
# Author: anyongjin
# Date  : 2023/3/17
import os.path

import numpy as np

from banbot.main.itrader import Trader
from banbot.strategy.mean_rev import *
from banbot.config.config import cfg


class BackTest(Trader):
    def __init__(self, max_num: int = 0):
        super(BackTest, self).__init__(MeanRev)
        self.max_num = max_num
        self.data_dir: str = cfg['data_dir']
        self.out_dir: str = os.path.join(self.data_dir, 'backtest')
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
        self.result = dict()
        self.max_open_orders = 1
        self.min_balance = 0
        self.max_balance = 0

    @staticmethod
    def load_data():
        data_dir = r'E:\trade\freqtd_data\user_data\spec_data\bnb1s'
        fname = 'BTCUSDT-1s-2023-02-22-2023-02-26.feather'
        return pd.read_feather(os.path.join(data_dir, fname))

    def bot_start(self):
        self.min_balance = self.stake_amount
        self.max_balance = self.stake_amount
        self.wallets[self.stake_symbol] = (self.stake_amount, 0)
        self.wallets[self.base_symbol] = 0, 0

    def run(self):
        from banbot.optmize.reports import print_backtest
        from banbot.optmize.bt_analysis import BTAnalysis
        self.bot_start()
        data = self.load_data()
        if self.max_num:
            data = data[:self.max_num]
        self.result['date_from'] = str(data.loc[0, 'date'])
        self.result['date_to'] = str(data.loc[len(data) - 1, 'date'])
        self.result['start_balance'] = self.wallets[self.stake_symbol][0]
        arr = data.to_numpy()[:, 1:]
        for i in range(arr.shape[0]):
            self.cur_time += timedelta(seconds=1)
            self.on_data_feed(arr[i])
        # 关闭未完成订单
        self.force_exit_all()
        self._calc_result_done(arr)
        od_list = [r.to_dict() for r in self.his_orders]
        print_backtest(pd.DataFrame(od_list), self.result)
        BTAnalysis(self.his_orders, **self.result).save(self.out_dir)
        print(f'complete, write to: {self.out_dir}')

    def _calc_result_done(self, arr: np.ndarray):
        self.result['max_open_orders'] = self.max_open_orders
        self.result['bar_num'] = len(arr)
        self.result['orders_num'] = len(self.his_orders)
        fin_balance = self.wallets[self.stake_symbol][0]
        start_balance = self.result['start_balance']
        self.result['final_balance'] = f"{fin_balance:.3f} {self.stake_symbol}"
        abs_profit = fin_balance - start_balance
        self.result['start_balance'] = f"{start_balance:.3f} {self.stake_symbol}"
        self.result['abs_profit'] = f"{abs_profit:.3f} {self.stake_symbol}"
        self.result['total_profit_pct'] = f"{abs_profit / start_balance * 100:.2f}%"
        tot_amount = sum(r.amount * r.price for r in self.his_orders)
        self.result['avg_stake_amount'] = f"{tot_amount / len(self.his_orders):.3f} {self.stake_symbol}"
        self.result['tot_stake_amount'] = f"{tot_amount:.3f} {self.stake_symbol}"
        od_sort = sorted(self.his_orders, key=lambda x: x.profit_rate)
        self.result['best_trade'] = f"{od_sort[-1].symbol} {od_sort[-1].profit_rate * 100:.2f}%"
        self.result['worst_trade'] = f"{od_sort[0].symbol} {od_sort[0].profit_rate * 100:.2f}%"
        self.result['min_balance'] = f'{self.min_balance:.3f} {self.stake_symbol}'
        self.result['max_balance'] = f'{self.max_balance:.3f} {self.stake_symbol}'
        self.result['market_change'] = f"{(arr[-1, 3] / arr[0, 0] - 1) * 100: .2f}%"

    def _new_order(self, tag: str):
        def callback(arr: np.ndarray):
            od: Order = self.open_orders[tag]
            enter_price = arr[-1, 3]
            quote_amount = enter_price * od.amount
            self.update_wallets(**{self.stake_symbol: -quote_amount, self.base_symbol: od.amount})
            od.status = OrderStatus.FullEnter
            od.enter_at = bar_num.get()
            od.filled = od.amount
            od.average = enter_price
            if not od.price:
                od.price = enter_price
            TradeLock.unlock(f'{od.symbol}_{od.enter_tag}', self.cur_time)
            self.max_open_orders = max(self.max_open_orders, len(self.open_orders))

        self._bar_listeners.append((bar_num.get() + 1, callback))

    def _close_order(self, tag: str):
        def calc_fn(arr: np.ndarray):
            od: Order = self.open_orders[tag]
            od.exit_at = bar_num.get()
            od.stop_price = arr[-1, 3]
            od.status = OrderStatus.FullExit
            quote_amount = od.stop_price * od.amount
            self.update_wallets(**{self.stake_symbol: quote_amount, self.base_symbol: -od.amount})
            self._finish_order(tag)
            if not self.open_orders:
                balance = sum(self.wallets[self.stake_symbol])
                self.min_balance = min(self.min_balance, balance)
                self.max_balance = min(self.max_balance, balance)

        self._bar_listeners.append((bar_num.get() + 1, calc_fn))


if __name__ == '__main__':
    bot = BackTest(10000)
    bot.run()
