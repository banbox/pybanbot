#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : backtest.py
# Author: anyongjin
# Date  : 2023/3/17
import os.path

from banbot.main.itrader import *
from banbot.config import *
data_dir = r'E:\trade\freqtd_data\user_data\data_recent\binance'


class BackTest(Trader):
    def __init__(self, config: Config, max_num: int = 0):
        btime.run_mode = btime.RunMode.BACKTEST
        super(BackTest, self).__init__(config)
        self.wallets = WalletsLocal()
        exg_name = config['exchange']['name']
        self.order_hold = OrderManager(config, exg_name, self.wallets)
        self.max_num = max_num
        self.data_dir: str = config['data_dir']
        self.out_dir: str = os.path.join(self.data_dir, 'backtest')
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
        self.result = dict()
        self.stake_amount: float = config.get('stake_amount', 1000)
        self._bar_listeners: List[Tuple[int, Callable]] = []
        self.max_open_orders = 1
        self.min_balance = 0
        self.max_balance = 0

    @staticmethod
    def load_data():
        fname = 'BTC_TUSD-1m.feather'
        return pd.read_feather(os.path.join(data_dir, fname))

    def bot_start(self):
        self.min_balance = self.stake_amount
        self.max_balance = self.stake_amount
        _, base_s, quote_s, timeframe = get_cur_symbol()
        self.wallets.set_wallets(**{base_s: 0, quote_s: self.stake_amount})
        self.order_hold.callbacks.append(self.order_callback)
        self._load_strategies()

    def run(self):
        from banbot.optmize.reports import print_backtest
        from banbot.optmize.bt_analysis import BTAnalysis
        set_context(f'BTC/TUSD/1m')
        self.bot_start()
        _, base_s, quote_s, timeframe = get_cur_symbol()
        data = self.load_data()
        if self.max_num:
            data = data[:self.max_num]
        self.result['date_from'] = str(data.loc[0, 'date'])
        self.result['date_to'] = str(data.loc[len(data) - 1, 'date'])
        self.result['start_balance'] = self.wallets.get(quote_s)[0]
        data['date'] = data['date'].apply(lambda x: int(x.timestamp() * 1000))
        arr = data.to_numpy()
        for i in range(arr.shape[0]):
            btime.add_secs(timeframe_secs.get())
            self.on_data_feed(arr[i])
        # 关闭未完成订单
        self.force_exit_all()
        self._calc_result_done(arr)

        his_orders = self.order_hold.his_list
        od_list = [r.to_dict() for r in his_orders]
        print_backtest(pd.DataFrame(od_list), self.result)
        BTAnalysis(his_orders, **self.result).save(self.out_dir)
        print(f'complete, write to: {self.out_dir}')

    def _bar_callback(self):
        self.order_hold.fill_pending_orders(bar_arr.get())

    def order_callback(self, od: InOutOrder, is_enter: bool):
        open_orders = self.order_hold.open_orders
        if is_enter:
            self.max_open_orders = max(self.max_open_orders, len(open_orders))
        elif not open_orders:
            _, base_s, quote_s, timeframe = get_cur_symbol()
            balance = sum(self.wallets.get(quote_s))
            self.min_balance = min(self.min_balance, balance)
            self.max_balance = min(self.max_balance, balance)

    def _calc_result_done(self, arr: np.ndarray):
        pair, base_s, quote_s, timeframe = get_cur_symbol()
        self.result['max_open_orders'] = self.max_open_orders
        self.result['bar_num'] = len(arr)
        his_orders = self.order_hold.his_list
        self.result['orders_num'] = len(his_orders)
        fin_balance = self.wallets.get(quote_s)[0]
        start_balance = self.result['start_balance']
        self.result['final_balance'] = f"{fin_balance:.3f} {quote_s}"
        abs_profit = fin_balance - start_balance
        self.result['start_balance'] = f"{start_balance:.3f} {quote_s}"
        self.result['abs_profit'] = f"{abs_profit:.3f} {quote_s}"
        self.result['total_profit_pct'] = f"{abs_profit / start_balance * 100:.2f}%"
        tot_amount = sum(r.enter.amount * r.enter.price for r in his_orders)
        if his_orders:
            self.result['avg_profit_pct'] = f"{abs_profit / start_balance / len(his_orders) * 1000:.3f}%o"
            self.result['avg_stake_amount'] = f"{tot_amount / len(his_orders):.3f} {quote_s}"
            self.result['tot_stake_amount'] = f"{tot_amount:.3f} {quote_s}"
            od_sort = sorted(his_orders, key=lambda x: x.profit_rate)
            self.result['best_trade'] = f"{od_sort[-1].symbol} {od_sort[-1].profit_rate * 100:.2f}%"
            self.result['worst_trade'] = f"{od_sort[0].symbol} {od_sort[0].profit_rate * 100:.2f}%"
        else:
            self.result['avg_stake_amount'] = f"0 {quote_s}"
            self.result['tot_stake_amount'] = f"0 {quote_s}"
            self.result['best_trade'] = "None"
            self.result['worst_trade'] = "None"
        self.result['min_balance'] = f'{self.min_balance:.3f} {quote_s}'
        self.result['max_balance'] = f'{self.max_balance:.3f} {quote_s}'
        self.result['market_change'] = f"{(arr[-1, ccol] / arr[0, ocol] - 1) * 100: .2f}%"

    def force_exit_all(self):
        self.order_hold.exit_open_orders(None, 'force_exit')
        self.order_hold.fill_pending_orders(bar_arr.get())


if __name__ == '__main__':
    bot = BackTest(10000)
    bot.run()
