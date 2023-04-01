#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : LeekReaper.py
# Author: anyongjin
# Date  : 2023/2/28

from banbot.bnbscan import *


class LeekReaper(BnbScan):
    '''
    https://github.com/richox/okcoin-leeks-reaper
    '''

    def __init__(self, adjust_pct=0.03):
        self.init_ok = False
        super().__init__()
        self.smo_vol = 0
        self.bid_price = 0
        self.ask_price = 0
        self.his_prices = []
        self.price_dust = 0.01  # 价格对比容差
        self.target_pos = 0.
        self.pos_step = 0.02
        # 最小调整单位是10美元，一次调整3笔订单
        self.adjust_money = max(self.min_notional * 3.1, self.all_assets * adjust_pct)
        self.ask_arr, self.bid_arr = [], []
        self.init_ok = True

    def run_forever(self):
        try:
            while True:
                try:
                    time.sleep(0.4)
                    self.sm_just_to_half()
                except KeyboardInterrupt:
                    logger.error('KeyboardInterrupt received, stopping...')
                    self.close()
                    break
        except Exception:
            logger.exception('trade fail')

    def fmt_odbook(self):
        self.ask_arr, self.bid_arr = [], []
        ask_iter, bid_iter = iter(self.asks), iter(self.bids)
        for i in range(3):
            ask_price = next(ask_iter)
            self.ask_arr.append((float(ask_price), float(self.asks[ask_price])))
            bid_price = next(bid_iter)
            self.bid_arr.append((float(bid_price), float(self.bids[bid_price])))

    def on_data_feed(self, last_secs, tags: List):
        if not self.init_ok:
            return
        his_plen = 15
        last_k = self.klines[0]
        self.smo_vol = self.smo_vol * 0.7 + last_k[4] * 0.3
        self.fmt_odbook()
        self.bid_price = self.bid_arr[0][0] * 0.618 + self.ask_arr[0][0] * 0.382 + self.price_dust
        self.ask_price = self.bid_arr[0][0] * 0.382 + self.ask_arr[0][0] * 0.618 - self.price_dust
        # 0是距离现在最近的，-1是最早的
        self.his_prices = self.klines[:his_plen, ccol].tolist()
        if len(self.his_prices) < his_plen:
            pad_len = his_plen - len(self.his_prices)
            self.his_prices += [self.his_prices[-1]] * pad_len
        self.his_prices.insert(0,
            (self.bid_arr[0][0] + self.ask_arr[0][0]) / 2 * 0.7 +
            (self.bid_arr[1][0] + self.ask_arr[1][0]) / 2 * 0.2 +
            (self.bid_arr[2][0] + self.ask_arr[2][0]) / 2 * 0.1
        )
        # logger.info(f'on data feed: {self.his_prices[:2]} b:{self.bid_price} a:{self.ask_price} {self.smo_vol}')
        # logger.info(f'ask prices: {self.ask_arr}')
        # logger.info(f'bid prices: {self.bid_arr}')
        self.find_entrys()

    def sm_just_to_half(self):
        '''
        仓位平衡策略，理想情况是持仓50%。如果发生偏移（买入卖出了），则间断性提交小单使仓位逐渐回归50%
        减少趋势策略中的反转+大滑点带来的回撤
        :return:
        '''
        min_pos, max_pos = self.target_pos - self.pos_step, self.target_pos + self.pos_step
        if min_pos <= self.position <= max_pos:
            return
        self.fmt_odbook()
        if self.position < self.target_pos:
            od_prices = [self.bid_arr[i][0] for i in range(3)]
            dir_fac = 1
            wallet_money = self.quote_v[0]
        else:
            od_prices = [self.ask_arr[i][0] for i in range(3)]
            dir_fac = -1
            wallet_money = self.base_v[0]
        total_adjust = min(self.adjust_money, wallet_money * 0.99)
        sub_money = total_adjust / len(od_prices)
        total_cost = 0
        for item in od_prices:
            quantity = sub_money / item
            cost = self.submit_order(dir_fac, item, quantity)
            if not cost:
                break
            total_cost += cost
        if total_cost >= self.price_dust:
            logger.info(f'adjust: {dir_fac}, cost: {total_cost:.2f} pos:{self.position:.3f}')

    def find_entrys(self):
        burst_price = self.his_prices[0] * 0.00005
        cur_price = self.his_prices[0]
        if cur_price > max(self.his_prices[1:5]) + burst_price or \
                cur_price > max(self.his_prices[2:5]) + burst_price and cur_price > self.his_prices[1]:
            mtype = 1
            amount = self.quote_v[0] / self.bid_price * 0.99
        elif cur_price < min(self.his_prices[1:5]) - burst_price or \
                cur_price < min(self.his_prices[2:5]) - burst_price and cur_price < self.his_prices[1]:
            mtype = -1
            amount = self.wallets.get(self.base_key, [0])[0]
        else:
            return
        # 确保下单数量不超过历史成交量均值
        amount = min(self.smo_vol, amount)
        # 1. 小成交量的趋势成功率比较低，减小力度
        # 2. 过度频繁交易有害，减小力度
        # 3. 短时价格波动过大，减小力度
        # 4. 盘口价差过大，减少力度
        amount_fac = 1
        down_tags = []
        avg_vol = self.klines[:, vcol].mean()
        if self.smo_vol < avg_vol:
            amount_fac *= self.smo_vol / avg_vol
            down_tags.append('vol_sm')
        if self.tick_num < 10:
            amount_fac *= 0.8
            down_tags.append('tick<10')
        if mtype > 0 and cur_price < max(self.his_prices):
            amount_fac *= 0.9
            down_tags.append('p<max')
        elif mtype < 0 and cur_price < min(self.his_prices):
            amount_fac *= 0.9
            down_tags.append('p<max')
        price_diff = abs(cur_price - self.his_prices[1])
        if price_diff > burst_price * 2:
            amount_fac *= 0.9
            down_tags.append('pdiff>d*2')
        if price_diff > burst_price * 3:
            amount_fac *= 0.9
            down_tags.append('pdiff>d*3')
        if price_diff > burst_price * 4:
            amount_fac *= 0.9
            down_tags.append('pdiff>d*4')
        market_diff = self.ask_arr[0][0] - self.bid_arr[0][0]
        if market_diff > burst_price * 2:
            amount_fac *= 0.9
            down_tags.append('mdiff>d*2')
        if market_diff > burst_price * 3:
            amount_fac *= 0.9
            down_tags.append('mdiff>d*3')
        if market_diff > burst_price * 4:
            amount_fac *= 0.9
            down_tags.append('mdiff>d*4')
        fin_amount = amount * amount_fac
        if fin_amount * cur_price < 11:
            return
        price = self.bid_price if mtype > 0 else self.ask_price
        cost_money = self.submit_order(mtype, price, fin_amount)
        logger.warning(f'new order, cost: {cost_money:.2f}  {amount_fac} {down_tags}')
