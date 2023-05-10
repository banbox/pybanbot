#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : backtest.py
# Author: anyongjin
# Date  : 2023/3/17
import os.path

from banbot.util.num_utils import to_pytypes
from banbot.main.itrader import *
from banbot.data.provider import *
from banbot.symbols.pair_manager import PairManager
from banbot.storage import *


class BackTest(Trader):
    def __init__(self, config: Config):
        super(BackTest, self).__init__(config)
        self.wallets = WalletsLocal()
        self.exchange = get_exchange()
        self.data_mgr = DBDataProvider(config, self.on_data_feed)
        self.pair_mgr = PairManager(config, self.exchange)
        self.order_mgr = LocalOrderManager(config, self.exchange, self.wallets, self.data_mgr, self.order_callback)
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
        self.enter_list = []

    def on_data_feed(self, pair, timeframe, row):
        self.bar_count += 1
        if self.first_data:
            self.open_price = row[ocol]
            self.result['pair'] = pair
            self.result['timeframe'] = timeframe
            self.result['date_from'] = btime.to_datestr(row[0])
            self.result['ts_from'] = row[0]
            self.first_data = False
        else:
            self.close_price = row[ccol]
            self.result['date_to'] = row[0]
            self.result['ts_to'] = row[0]
        enter_list, exit_list, ext_tags = super(BackTest, self).on_data_feed(pair, timeframe, row)
        if enter_list:
            ctx = get_context(f'{pair}/{timeframe}')
            price = to_pytypes(row[ccol])
            enter_text = ','.join([v[1] for v in enter_list])
            self.enter_list.append((ctx[bar_num], enter_text, price))

    async def init(self):
        await self.exchange.load_markets()
        self.min_balance = self.stake_amount
        self.max_balance = self.stake_amount
        await self.pair_mgr.refresh_pairlist()
        pair_tfs = self._load_strategies(self.pair_mgr.symbols)
        with db():
            await self.data_mgr.sub_pairs(pair_tfs)
        self.result['task_id'] = BotTask.cur_id
        for pair in self.pair_mgr.symbols:
            base_s, quote_s = pair.split('/')
            self.wallets.set_wallets(**{base_s: 0, quote_s: self.stake_amount})
        self.result['start_balance'] = self.order_mgr.get_legal_value()

    async def run(self):
        from banbot.optmize.reports import print_backtest
        from banbot.optmize.bt_analysis import BTAnalysis
        await self.init()
        # 轮训数据
        with db():
            bt_start = time.monotonic()
            self.data_mgr.loop_main()
            bt_cost = time.monotonic() - bt_start
            print('')
            logger.info(f'Complete! cost: {bt_cost:.3f}s, avg: {self.bar_count / bt_cost:.1f} bar/s')
            # 关闭未完成订单
            self.order_mgr.cleanup()
            self._calc_result_done()

            print_backtest(self.result)

        await BTAnalysis(**self.result).save(self.out_dir)
        print(f'complete, write to: {self.out_dir}')

    def order_callback(self, od: InOutOrder, is_enter: bool):
        open_orders = InOutOrder.open_orders()
        if is_enter:
            self.max_open_orders = max(self.max_open_orders, len(open_orders))
        elif not open_orders:
            quote_s = od.symbol.split('/')[1]
            balance = sum(self.wallets.get(quote_s))
            self.min_balance = min(self.min_balance, balance)
            self.max_balance = min(self.max_balance, balance)

    def _calc_result_done(self):
        # 输出入场信号
        if self.enter_list:
            enter_ids, enter_tags, enter_prices = list(zip(*self.enter_list))
        else:
            enter_ids, enter_tags, enter_prices = [], [], []
        self.result['enters'] = dict(ids=enter_ids, tags=enter_tags, valy=enter_prices)
        quote_s = 'TUSD'
        self.result['date_to'] = btime.to_datestr(self.result['date_to'])
        self.result['max_open_orders'] = self.max_open_orders
        self.result['bar_num'] = self.bar_count
        his_orders = InOutOrder.his_orders()
        self.result['orders_num'] = len(his_orders)
        fin_balance = self.wallets.get(quote_s)[0]
        start_balance = self.result['start_balance']
        self.result['final_balance'] = f"{fin_balance:.3f} {quote_s}"
        abs_profit = sum(od.profit for od in his_orders) if his_orders else 0
        self.result['start_balance'] = f"{start_balance:.3f} {quote_s}"
        self.result['abs_profit'] = f"{abs_profit:.3f} {quote_s}"
        self.result['total_profit_pct'] = f"{abs_profit / start_balance * 100:.2f}%"
        if his_orders:
            total_fee = sum((od.enter.fee + od.exit.fee) * od.enter.amount * od.enter.price for od in his_orders)
            self.result['total_fee'] = f"{total_fee:.3f} {quote_s}"
            tot_amount = sum(r.enter.amount * r.enter.price for r in his_orders)
            self.result['avg_profit_pct'] = f"{abs_profit / start_balance / len(his_orders) * 1000:.3f}%o"
            self.result['avg_stake_amount'] = f"{tot_amount / len(his_orders):.3f} {quote_s}"
            self.result['tot_stake_amount'] = f"{tot_amount:.3f} {quote_s}"
            od_sort = sorted(his_orders, key=lambda x: x.profit_rate)
            self.result['best_trade'] = f"{od_sort[-1].symbol} {od_sort[-1].profit_rate * 100:.2f}%"
            self.result['worst_trade'] = f"{od_sort[0].symbol} {od_sort[0].profit_rate * 100:.2f}%"
        else:
            self.result['total_fee'] = f"0 {quote_s}"
            self.result['avg_profit_pct'] = '0'
            self.result['avg_stake_amount'] = f"0 {quote_s}"
            self.result['tot_stake_amount'] = f"0 {quote_s}"
            self.result['best_trade'] = "None"
            self.result['worst_trade'] = "None"
        self.result['min_balance'] = f'{self.min_balance:.3f} {quote_s}'
        self.result['max_balance'] = f'{self.max_balance:.3f} {quote_s}'
        self.result['market_change'] = f"{(self.close_price / self.open_price - 1) * 100: .2f}%"


