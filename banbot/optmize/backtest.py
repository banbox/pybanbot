#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : backtest.py
# Author: anyongjin
# Date  : 2023/3/17
import os.path

from banbot.main.itrader import *
from banbot.data.data_provider import *


class BackTest(Trader):
    def __init__(self, config: Config, max_num: int = 0):
        super(BackTest, self).__init__(config)
        btime.run_mode = btime.RunMode.BACKTEST
        super(BackTest, self).__init__(config)
        self.wallets = WalletsLocal()
        exg_name = config['exchange']['name']
        self.data_hold = LocalDataProvider(config, self._pair_row_callback)
        self.order_hold = OrderManager(config, exg_name, self.wallets, self.data_hold)
        self.max_num = max_num
        self.out_dir: str = os.path.join(config['data_dir'], 'backtest')
        self.result = dict()
        self.stake_amount: float = config.get('stake_amount', 1000)
        self._bar_listeners: List[Tuple[int, Callable]] = []
        self.max_open_orders = 1
        self.min_balance = 0
        self.max_balance = 0
        self.first_data = True
        self.bar_count = 0
        self.open_price = 0
        self.close_price = 0

    async def _pair_row_callback(self, pair, timeframe, row):
        self.bar_count += 1
        set_context(f'{pair}/{timeframe}')
        if self.first_data:
            self.open_price = row[ocol]
            self.result['date_from'] = btime.to_datestr(row[0])
            self.first_data = False
        else:
            self.close_price = row[ccol]
            self.result['date_to'] = btime.to_datestr(row[0])
        await self.on_data_feed(np.array(row))

    def init(self):
        self.min_balance = self.stake_amount
        self.max_balance = self.stake_amount
        for pair, tf in self.pairlist:
            base_s, quote_s = pair.split('/')
            self.wallets.set_wallets(**{base_s: 0, quote_s: self.stake_amount})
        self.result['start_balance'] = self.order_hold.get_legal_value()
        self.order_hold.callbacks.append(self.order_callback)

    async def run(self):
        from banbot.optmize.reports import print_backtest
        from banbot.optmize.bt_analysis import BTAnalysis
        set_context(f'BTC/TUSD/1m')
        self.init()
        cur_time = btime.time()
        await self._loop_tasks([
            [self.data_hold.process, self.data_hold.min_interval, cur_time],
        ])
        # 关闭未完成订单
        await self.cleanup()
        self._calc_result_done()

        his_orders = self.order_hold.his_list
        od_list = [r.to_dict() for r in his_orders]
        print_backtest(pd.DataFrame(od_list), self.result)
        await BTAnalysis(his_orders, **self.result).save(self.out_dir)
        print(f'complete, write to: {self.out_dir}')

    async def _bar_callback(self):
        await self.order_hold.fill_pending_orders(bar_arr.get())

    def order_callback(self, od: InOutOrder, is_enter: bool):
        open_orders = self.order_hold.open_orders
        if is_enter:
            self.max_open_orders = max(self.max_open_orders, len(open_orders))
        elif not open_orders:
            _, base_s, quote_s, timeframe = get_cur_symbol()
            balance = sum(self.wallets.get(quote_s))
            self.min_balance = min(self.min_balance, balance)
            self.max_balance = min(self.max_balance, balance)

    def _calc_result_done(self):
        pair, base_s, quote_s, timeframe = get_cur_symbol()
        self.result['max_open_orders'] = self.max_open_orders
        self.result['bar_num'] = self.bar_count
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
            self.result['avg_profit_pct'] = '0'
            self.result['avg_stake_amount'] = f"0 {quote_s}"
            self.result['tot_stake_amount'] = f"0 {quote_s}"
            self.result['best_trade'] = "None"
            self.result['worst_trade'] = "None"
        self.result['min_balance'] = f'{self.min_balance:.3f} {quote_s}'
        self.result['max_balance'] = f'{self.max_balance:.3f} {quote_s}'
        self.result['market_change'] = f"{(self.close_price / self.open_price - 1) * 100: .2f}%"

    async def cleanup(self):
        await self.order_hold.exit_open_orders('force_exit', 0)
        await self.order_hold.fill_pending_orders(None)


if __name__ == '__main__':
    bot = BackTest(10000)
    bot.run()
