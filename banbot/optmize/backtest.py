#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : backtest.py
# Author: anyongjin
# Date  : 2023/3/17


from banbot.main.itrader import Trader
from banbot.strategy.mean_rev import *
from banbot.compute.utils import *


class BackTest(Trader):
    def __init__(self, max_num: int = 0):
        super(BackTest, self).__init__(MeanRev)
        self.max_num = max_num

    @staticmethod
    def load_data():
        data_dir = r'E:\trade\freqtd_data\user_data\spec_data\bnb1s'
        fname = 'BTCUSDT-1s-2023-02-22-2023-02-26.feather'
        return pd.read_feather(os.path.join(data_dir, fname))

    def bot_start(self):
        self.wallets[self.stake_symbol] = (self.stake_amount, 0)
        self.wallets[self.base_symbol] = 0, 0

    def run(self):
        self.bot_start()
        data = self.load_data()
        if self.max_num:
            data = data[:self.max_num]
        arr = data.to_numpy()[:, 1:]
        for i in range(arr.shape[0]):
            self.cur_time += timedelta(seconds=1)
            self.on_data_feed(arr[i])
        # 关闭未完成订单
        self.force_exit_all()
        from banbot.optmize.reports import print_backtest
        od_list = [r.to_dict() for r in self.his_orders]
        print_backtest(pd.DataFrame(od_list))
        print('complete')

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

        self._bar_listeners.append((bar_num.get() + 1, calc_fn))


if __name__ == '__main__':
    bot = BackTest(3000)
    bot.run()
