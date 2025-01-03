#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : backtest.py
# Author: anyongjin
# Date  : 2023/3/17
import os.path

from banbot.main.itrader import *
from banbot.storage import *
from banbot.symbols.pair_manager import PairManager
show_num = 600


class BackTest(Trader):
    def __init__(self, config: Config):
        super(BackTest, self).__init__(config)
        self.wallets = WalletsLocal(self.exchange)
        self.data_mgr = self._init_data_mgr()
        self.order_mgr = LocalOrderManager(config, self.exchange, self.wallets, self.data_mgr, self.order_callback)
        self.out_dir: str = os.path.join(config['data_dir'], 'backtest')
        self.result = dict()
        self.stake_amount: float = config.get('stake_amount', 1000)
        self.quote_symbols: Set[str] = set(config.get('stake_currency') or [])
        self.draw_balance_over: float = config.get('draw_balance_over', 0)
        self.max_open_orders = 1
        self.min_balance = 0
        self.max_balance = 0
        self.bar_count = 0
        self.graph_data = dict(ava=[], profit=[], real=[], withdraw=[])
        self.last_check_trades = 0
        self.daterange_from = None
        self.daterange_to = None
        self.graph_every = 1

    async def on_data_feed(self, pair, timeframe, row):
        self.bar_count += 1
        try:
            await super(BackTest, self).on_data_feed(pair, timeframe, row)
        except AccountBomb as e:
            self._on_bomb(e, row[0])
            return
        self._log_state(row[0])

    async def on_pair_trades(self, pair: str, trades: List[dict]):
        try:
            await super(BackTest, self).on_pair_trades(pair, trades)
        except AccountBomb as e:
            self._on_bomb(e, trades[-1]['timestamp'])
            return
        if self.last_check_trades + 5000 > btime.time_ms():
            return
        self.last_check_trades = btime.time_ms()
        self._log_state(btime.time_ms())

    def _on_bomb(self, e: AccountBomb, time_ms: int):
        date_str = btime.to_datestr(time_ms)
        if self.config.get('charge_on_bomb'):
            self.reset_wallet(e.coin)
            self.result['total_invest'] += self.wallets.total_legal(self.quote_symbols)
            logger.error(f'wallet {e.coin} BOMB at {date_str}, reset wallet and continue...')
        else:
            BotGlobal.state = BotState.STOPPED
            logger.error(f'wallet {e.coin} BOMB at {date_str}, exit')

    def _log_state(self, time_ms: int):
        if not self.daterange_from:
            self.daterange_from = time_ms
        self.daterange_to = time_ms
        self._log_graph(time_ms)
        # 更新最大最小余额
        quote_legal = self.wallets.total_legal(self.quote_symbols)
        self.min_balance = min(self.min_balance, quote_legal)
        self.max_balance = max(self.max_balance, quote_legal)

    def _init_data_mgr(self):
        if self.is_ws_mode():
            from banbot.data.ws import LocalWSProvider
            return LocalWSProvider(self.config, self.on_pair_trades)
        kline_source = self.config.get('kline_source')
        if kline_source == 'file':
            return FileDataProvider(self.config, self.on_data_feed)
        return DBDataProvider(self.config, self.on_data_feed)

    async def init(self):
        await AppConfig.test_db()
        from banbot.data.toolbox import sync_timeframes
        await self.exchange.load_markets()
        self.min_balance = sys.maxsize
        async with dba():
            # 创建回测任务，记录相关周期
            await self._init_task()
            # 同步K线数据，防止不同周期数据有未更新
            await sync_timeframes()
            all_symbols = list(self.exchange.markets.keys())
            await ExSymbol.ensures(self.exchange.name, self.exchange.market_type, all_symbols)
            await ExSymbol.init()
            await self.pair_mgr.refresh_pairlist()
            if not self.pair_mgr.symbols:
                raise ValueError('no pairs generate from PairManager')
            logger.info(f'backtest for {len(self.pair_mgr.symbols)} symbols: {self.pair_mgr.symbols[:5]}...')
            pair_tfs = self._load_strategies(self.pair_mgr.symbols, self.pair_mgr.pair_tfscores)
            self.data_mgr.sub_pairs(pair_tfs)
        self.result['task_id'] = BotTask.cur_id
        # 初始化钱包余额
        self.reset_wallet()
        self.result['total_invest'] = self.wallets.total_legal(self.quote_symbols)

        def pair_time_gen():
            wait_secs = self.pair_mgr.get_refresh_wait()
            return btime.time() + wait_secs
        if not hasattr(self.data_mgr, 'pairs'):
            self.data_mgr.timers.append((self.refresh_pairs, pair_time_gen))

    def reset_wallet(self, *symbols):
        wallets = self.config.get('wallet_amounts')
        if not wallets:
            raise ValueError('wallet_amounts is required for backtesting...')
        if symbols:
            wallets = {k: v for k, v in wallets.items() if k in symbols}
        self.wallets.set_wallets(**wallets)

    async def run(self):
        from banbot.optmize.reports import print_backtest
        from banbot.optmize.bt_analysis import BTAnalysis
        await self.init()
        BotGlobal.state = BotState.RUNNING
        # 轮训数据
        if hasattr(self.data_mgr, 'down_data'):
            # 按ohlcv回测，下载必要数据
            async with dba():
                await self.data_mgr.down_data()
        async with dba():
            bt_start = time.monotonic()
            await self.data_mgr.loop_main()
            bt_cost = time.monotonic() - bt_start
            logger.info(f'Complete! cost: {bt_cost:.3f}s, avg: {self.bar_count / bt_cost:.1f} bar/s')
            # 关闭未完成订单
            await self.order_mgr.cleanup()
            await self._calc_result_done()

            await print_backtest(self.result)

            await BTAnalysis(**self.result).save(self.out_dir)
        print(f'complete, write to: {self.out_dir}')

    def order_callback(self, od: InOutOrder, is_enter: bool):
        if is_enter:
            self.max_open_orders = max(self.max_open_orders, len(BotCache.open_ods))
        elif self.draw_balance_over:
            quote_legal = self.wallets.ava_legal(self.quote_symbols)
            if self.draw_balance_over < quote_legal:
                self.wallets.withdraw_legal(quote_legal - self.draw_balance_over, self.quote_symbols)

    async def _init_task(self):
        await BotTask.init()
        if BotTask.cur_id > 0:
            from banbot.util.common import set_log_file
            task_dir = os.path.join(self.out_dir, f'task_{BotTask.cur_id}')
            if not os.path.isdir(task_dir):
                os.mkdir(task_dir)
            log_path = os.path.join(task_dir, 'out.log')
            self.config['logfile'] = log_path
            set_log_file(logger, log_path)

    def _log_graph(self, time_ms: int):
        if self.bar_count % self.graph_every:
            return
        spl_step = 5
        if len(self.graph_data['real']) >= show_num * spl_step:
            # 检查数据是否太多，超过采样总数5倍时，进行重采样
            self.graph_every *= spl_step
            keys = ['real', 'ava', 'profit', 'withdraw']
            data = {k: [] for k in keys}
            for i in range(0, len(self.graph_data['real']), spl_step):
                for k in keys:
                    data[k].append(self.graph_data[k][i])
            for k in keys:
                self.graph_data[k] = data[k]
            if self.bar_count % self.graph_every:
                return

        ava_legal = self.wallets.ava_legal()
        total_legal = self.wallets.total_legal(with_upol=True)
        profit_legal = self.wallets.profit_legal()
        draw_legal = self.wallets.get_withdraw_legal()
        cur_date = btime.to_datetime(time_ms)
        self.graph_data['real'].append((cur_date, total_legal))
        self.graph_data['ava'].append((cur_date, ava_legal))
        self.graph_data['profit'].append((cur_date, profit_legal))
        self.graph_data['withdraw'].append((cur_date, draw_legal))

    async def _calc_result_done(self):
        total_invest = self.result['total_invest']
        task = BotTask.obj
        task.set_info(total_invest=total_invest, stake_amount=self.stake_amount)
        self.result['graph_data'] = self.graph_data
        # 更新最大最小余额
        quote_legal = self.wallets.total_legal(self.quote_symbols)
        self.min_balance = min(self.min_balance, quote_legal)
        self.max_balance = max(self.max_balance, quote_legal)
        self.result['final_withdraw'] = self.wallets.get_withdraw_legal(self.quote_symbols)
        quote_s = 'USD'
        timerange = self.config['timerange']
        self.result['date_from'] = btime.to_datestr(self.daterange_from or timerange.startts)
        self.result['date_to'] = btime.to_datestr(self.daterange_to or timerange.stopts)
        self.result['max_open_orders'] = self.max_open_orders
        self.result['bar_num'] = self.bar_count
        his_orders = await InOutOrder.his_orders()
        self.result['orders_num'] = len(his_orders)
        fin_balance = self.wallets.ava_legal(self.quote_symbols)
        self.result['final_balance'] = f"{fin_balance:.3f} {quote_s}"
        abs_profit = sum(od.profit for od in his_orders) if his_orders else 0
        self.result['total_invest'] = f"{total_invest:.3f} {quote_s}"
        self.result['abs_profit'] = f"{abs_profit:.3f} {quote_s}"
        self.result['total_profit_pct'] = f"{abs_profit / total_invest * 100:.2f}%"
        if his_orders:
            total_fee = sum((od.enter.fee + od.exit.fee or 0) for od in his_orders)
            self.result['total_fee'] = f"{total_fee:.3f} {quote_s}"
            tot_amount = sum(r.enter.amount * r.enter.price for r in his_orders
                             if r.enter.amount and r.enter.price)
            self.result['avg_profit_pct'] = f"{abs_profit / total_invest / len(his_orders) * 1000:.3f}‰"
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


