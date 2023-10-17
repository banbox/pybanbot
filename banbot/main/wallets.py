#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : wallets.py
# Author: anyongjin
# Date  : 2023/3/29

from typing import *
from dataclasses import dataclass, field

import ccxt

from banbot.config import AppConfig
from banbot.config.consts import MIN_STAKE_AMOUNT
from banbot.exchange.crypto_exchange import CryptoExchange, loop_forever
from banbot.storage import InOutOrder, ExSymbol, split_symbol, ExitTags
from banbot.main.addons import *
from banbot.util import btime
from banbot.util.common import logger
from banbot.types import *


@dataclass
class ItemWallet:
    coin: str
    '币代码，非交易对'

    available: float = 0
    '可用余额'

    pendings: Dict[str, float] = field(default_factory=dict)
    '买入卖出时锁定金额，键可以是订单id'

    frozens: Dict[str, float] = field(default_factory=dict)
    '空单等长期冻结金额，键可以是订单id'

    unrealized_pol: float = 0
    '此币的公共未实现盈亏，合约用到，可抵扣其他订单保证金占用。每个bar重新计算'

    used_upol: float = 0
    '已占用的未实现盈亏（用作其他订单的保证金）'

    withdraw: float = 0
    '从余额提现的，不会用于交易。'

    def total(self, with_upol=False):
        sum_val = self.available
        for k, v in self.pendings.items():
            sum_val += v
        for k, v in self.frozens.items():
            sum_val += v
        if with_upol:
            sum_val += self.unrealized_pol
        return sum_val

    def fiat_value(self, with_upol=False):
        """
        获取此钱包的法币价值
        """
        return self.total(with_upol) * MarketPrice.get(self.coin)

    def set_margin(self, od_key: str, amount: float):
        """
        设置保证金占用。优先从unrealized_pol-used_upol中取。不足时从余额中取。
        超出时释放到余额
        """
        # 提取旧保证金占用值
        old_amt = self.frozens.get(od_key) or 0
        ava_upol = self.unrealized_pol - self.used_upol
        # logger.info(f'set margin: {od_key} {old_amt:.4f} -> {amount:.4f} upol: {ava_upol:.4f} ava: {self.available:.4f}')

        upol_cost = 0
        if ava_upol > 0:
            # 优先使用可用的未实现盈亏余额
            self.used_upol += amount
            if self.used_upol <= self.unrealized_pol:
                # 未实现盈亏足够，无需冻结
                if od_key in self.frozens:
                    del self.frozens[od_key]
                upol_cost = amount
                amount = 0
            else:
                # 未实现盈亏不足，更新还需占用的
                new_amount = self.used_upol - self.unrealized_pol
                upol_cost = amount - new_amount
                amount = new_amount
                self.used_upol = self.unrealized_pol

        add_val = amount - old_amt
        if add_val <= 0:
            # 已有保证金超过要求值，释放到余额
            self.available -= add_val
        else:
            # 已有保证金不足，从余额中扣除
            if self.available < add_val:
                # 余额不足
                self.used_upol -= upol_cost
                err_msg = f'avaiable {self.coin} Insufficient, frozen require: {add_val:.5f}, {od_key}'
                raise LackOfCash(add_val - self.available, err_msg)
            self.available -= add_val
        self.frozens[od_key] = amount

    def set_frozen(self, od_key: str, amount: float, with_avaiable: bool = True):
        """
        设置冻结金额为固定值。可从余额或pending中同步。
        不足则从另一侧取用，超出则添加到另一侧。
        """
        old_amt = self.frozens.get(od_key) or 0
        add_val = amount - old_amt
        if with_avaiable:
            if add_val > 0 and self.available < add_val:
                raise ValueError(f'avaiable {self.coin} Insufficient, frozen require: {add_val:.5f}, {od_key}')
            self.available -= add_val
        else:
            pend_val = self.pendings.get(od_key) or 0
            if add_val > 0 and pend_val < add_val:
                raise ValueError(f'pending {self.coin} Insufficient, frozen require: {add_val:.5f}, {od_key}')
            self.pendings[od_key] = pend_val - add_val
        self.frozens[od_key] = amount

    def reset(self):
        logger.debug('reset wallet %s %s %s %s', self.coin, self.available, self.pendings, self.frozens)
        self.available = 0
        self.unrealized_pol = 0
        self.used_upol = 0
        self.frozens = dict()
        self.pendings = dict()


class WalletsLocal:
    obj: 'WalletsLocal' = None

    def __init__(self, exchange: CryptoExchange):
        WalletsLocal.obj = self
        self.exchange = exchange
        self.data: Dict[str, ItemWallet] = dict()
        self.update_at = btime.time()
        config = AppConfig.get()
        self.margin_add_rate = config.get('margin_add_rate') or 0.667
        '出现亏损时，在亏损百分比后追加保证金'

    def set_wallets(self, **kwargs):
        for key, val in kwargs.items():
            if isinstance(val, ItemWallet):
                self.data[key] = val
            elif isinstance(val, (int, float)):
                self.data[key] = ItemWallet(key, available=val)
            else:
                raise ValueError(f'unsupport val type: {key} {type(val)}')

    def cost_ava(self, od_key: str, symbol: str, amount: float, negative: bool = False, after_ts: float = 0,
                 min_rate: float = 0.1) -> float:
        '''
        从某个币的可用余额中扣除，添加到pending中，仅用于回测
        :param od_key: 锁定的键
        :param symbol: 币代码
        :param amount: 金额
        :param negative: 是否允许负数余额（空单用到）
        :param after_ts: 是否要求更新时间戳
        :param min_rate: 最低扣除比率
        :return 实际扣除数量
        '''
        if self.update_at + 1 < after_ts:
            logger.warning(f'wallet expired: expect > {after_ts}, delay: {after_ts - self.update_at} ms')
        if symbol not in self.data:
            self.data[symbol] = ItemWallet(symbol)
        wallet = self.data[symbol]
        src_amount = wallet.available
        if src_amount >= amount or negative:
            # 余额充足，或允许负数，直接扣除
            real_cost = amount
        elif src_amount / amount > min_rate:
            # 差额在近似允许范围内，扣除实际值
            real_cost = src_amount
        else:
            raise LackOfCash(amount, f'wallet {symbol} balance {src_amount:.5f} < {amount:.5f}')
        logger.debug('cost_ava wallet %s.%s %f - %f', od_key, symbol, wallet.available, real_cost)
        wallet.available -= real_cost
        wallet.pendings[od_key] = real_cost
        self.update_at = btime.time()
        return real_cost

    def cost_frozen(self, od_key: str, symbol: str, amount: float, after_ts: float = 0):
        '''
        此方法不用于合约
        从frozen中扣除，如果不够，从available扣除剩余部分
        扣除后，添加到pending中
        '''
        if self.update_at + 1 < after_ts:
            logger.warning(f'wallet expired: expect > {after_ts}, delay: {after_ts - self.update_at} ms')
        if symbol not in self.data:
            return 0
        wallet = self.data[symbol]
        frozen_amt = wallet.frozens.get(od_key, 0)
        if frozen_amt:
            del wallet.frozens[od_key]
        # 将冻结的剩余部分归还到available，正负都有可能
        logger.debug('cost_frozen wallet %s.%s %f + %f', od_key, symbol, wallet.available, frozen_amt - amount)
        wallet.available += frozen_amt - amount
        real_cost = amount
        if wallet.available < 0:
            real_cost += wallet.available
            wallet.available = 0
        wallet.pendings[od_key] = real_cost
        self.update_at = btime.time()
        return real_cost

    def confirm_pending(self, od_key: str, src_key: str, src_amount: float, tgt_key: str, tgt_amount: float,
                        to_frozen: bool = False):
        '''
        从src中确认扣除，添加到tgt的余额中
        '''
        self.update_at = btime.time()
        src, tgt = self.data.get(src_key), self.data.get(tgt_key)
        if not src:
            return False
        if not tgt:
            tgt = ItemWallet(tgt_key)
            self.data[tgt_key] = tgt
        pending_amt = src.pendings.get(od_key, 0)
        if not pending_amt:
            return False
        left_pending = pending_amt - src_amount
        del src.pendings[od_key]
        logger.debug('confirm_pending wallet %s.%s %f + %f', od_key, src_key, src.available, left_pending)
        src.available += left_pending  # 剩余pending归还到available，（正负都可能）
        if to_frozen:
            tgt.frozens[od_key] = tgt_amount
        else:
            tgt.available += tgt_amount
            # logger.info(f'confirm_pending tgt wallet {od_key}.{tgt_key} {tgt.available} + {tgt_amount}')
        return True

    def cancel(self, od_key: str, symbol: str, add_amount: float = 0, from_pending: bool = True):
        '''
        取消对币种的数量锁定(frozens/pendings)，重新加到available上
        '''
        self.update_at = btime.time()
        wallet = self.data.get(symbol)
        if not wallet:
            return
        src_dic = wallet.pendings if from_pending else wallet.frozens
        src_amount = src_dic.get(od_key) or 0
        if src_amount:
            # logger.info(f'cancel wallet {od_key}.{symbol} {wallet.available} + {src_amount} + {add_amount}')
            del src_dic[od_key]
        src_amount += add_amount
        tag = 'pending' if from_pending else 'frozen'
        wallet.available += src_amount
        logger.debug('cancel %s %f to ava, %s, %s, final: %.4f', tag, src_amount, od_key, symbol, wallet.available)

    def enter_od(self, od: InOutOrder, after_ts=0):
        '''
        实盘和模拟都执行，实盘时可防止过度消费
        如果余额不足，会发出异常
        需要调用confirm_od_enter确认。也可调用cancel取消
        '''
        od_key = od.key
        # 如果余额不足会发出异常
        exs = ExSymbol.get_by_id(od.sid)
        if od.enter.amount:
            legal_cost = od.enter.amount * MarketPrice.get(od.symbol)
        else:
            legal_cost = od.get_info('legal_cost')
        is_future = exs.market == 'future'
        if is_future or not od.short:
            # 期货合约，现货多单锁定quote
            if legal_cost < MIN_STAKE_AMOUNT:
                raise ValueError(f'margin cost must >= {MIN_STAKE_AMOUNT}, cur: {legal_cost:.2f}')
            if is_future:
                # 期货合约，名义价值=保证金*杠杆
                legal_cost /= od.leverage
            quote_cost = self.get_amount_by_legal(exs.quote_code, legal_cost)
            quote_cost = self.cost_ava(od_key, exs.quote_code, quote_cost, after_ts=after_ts)
            quote_margin = quote_cost  # 计算名义数量
            if is_future:
                quote_margin *= od.leverage
            od.quote_cost = self.exchange.pres_cost(od.symbol, quote_margin)
            if od.get_info('wallet_left'):
                item = self.data.get(exs.quote_code)
                if item:
                    od.set_info(wallet_left=item.available)
        else:
            # 现货空单，锁定base，允许金额为负
            base_cost = self.get_amount_by_legal(exs.base_code, legal_cost)
            base_cost = self.cost_ava(od_key, exs.base_code, base_cost, negative=True, after_ts=after_ts)
            od.enter.amount = base_cost
        return legal_cost

    def confirm_od_enter(self, od: InOutOrder, enter_price: float):
        if btime.prod_mode():
            return
        exs = ExSymbol.get_by_id(od.sid)
        sub_od = od.enter
        quote_amount = enter_price * sub_od.amount
        if exs.market == 'future':
            # 期货合约，只锁定定价币，不涉及base币的增加
            quote_amount /= od.leverage
            self.confirm_pending(od.key, exs.quote_code, quote_amount, exs.quote_code, quote_amount, True)
        elif od.short:
            quote_amount *= (1 - sub_od.fee)
            self.confirm_pending(od.key, exs.base_code, sub_od.amount, exs.quote_code, quote_amount, True)
        else:
            base_amt = sub_od.amount * (1 - sub_od.fee)
            self.confirm_pending(od.key, exs.quote_code, quote_amount, exs.base_code, base_amt)

    def exit_od(self, od: InOutOrder, base_amount: float, after_ts=0):
        if btime.prod_mode():
            return
        exs = ExSymbol.get_by_id(od.sid)
        if exs.market == 'future':
            # 期货合约，不涉及base币的变化。退出订单时，对锁定的定价币平仓释放
            pass
        elif od.short:
            # 现货空单，从quote的frozen卖，计算到quote的available，从base的pending未成交部分取消
            self.cancel(od.key, exs.base_code)
            # 这里不用预先扣除，价格可能为None
        else:
            # 现货多单，从base的available卖，计算到quote的available，从quote的pending未成交部分取消
            wallet = self.get(exs.base_code, od.enter.create_at)
            if 0 < wallet.available < base_amount or abs(wallet.available / base_amount - 1) <= 0.01:
                base_amount = wallet.available
                # 取消quote的pending未成交部分
                self.cancel(od.key, exs.quote_code)
            if base_amount:
                self.cost_ava(od.key, exs.base_code, base_amount, after_ts=after_ts, min_rate=0.01)

    def confirm_od_exit(self, od: InOutOrder, exit_price: float):
        if btime.prod_mode():
            return
        exs = ExSymbol.get_by_id(od.sid)
        sub_od = od.exit
        if exs.market == 'future':
            # 期货合约不涉及base币的变化。退出订单时，对锁定的定价币平仓释放
            self.cancel(od.key, exs.quote_code, add_amount=od.profit, from_pending=False)
        elif od.short:
            # 空单，优先从quote的frozen买，不兑换为base，再换算为quote的avaiable
            org_amount = od.enter.filled  # 这里应该取卖单的数量，才能完全平掉
            if org_amount:
                # 执行quote买入，中和base的欠债
                self.cost_frozen(od.key, exs.quote_code, org_amount * exit_price)
            if exit_price < od.enter.price:
                # 空单，出场价低于入场价，有利润，将冻结的利润置为available
                self.cancel(od.key, exs.quote_code, from_pending=False)
            quote_amount = exit_price * org_amount
            self.confirm_pending(od.key, exs.quote_code, quote_amount, exs.base_code, org_amount)
        else:
            # 多单，从base的avaiable卖，兑换为quote的available
            quote_amount = exit_price * sub_od.amount * (1 - sub_od.fee)
            self.confirm_pending(od.key, exs.base_code, sub_od.amount, exs.quote_code, quote_amount)

    def cut_part(self, src_key: str, tgt_key: str, symbol: str, rate: float):
        item = self.data.get(symbol)
        if not item:
            return
        if src_key in item.pendings:
            cut_amt = item.pendings[src_key] * rate
            item.pendings[tgt_key] = cut_amt
            item.pendings[src_key] -= cut_amt
        if src_key in item.frozens:
            cut_amt = item.frozens[src_key] * rate
            item.frozens[tgt_key] = cut_amt
            item.frozens[src_key] -= cut_amt

    def update_ods(self, od_list: List[InOutOrder]):
        """
        更新订单。目前只针对期货合约订单，需要更新合约订单的保证金比率。
        传入的订单必然都是同一个定价币的订单
        保证金比率： (仓位名义价值 * 维持保证金率 - 维持保证金速算数) / (钱包余额 + 未实现盈亏)
        钱包余额 = 初始净划入余额（含初始保证金） + 已实现盈亏 + 净资金费用 - 手续费
        """
        if not od_list:
            return
        # 所有订单都是同一个定价币，提前获取此币的钱包
        exs = ExSymbol.get_by_id(od_list[0].sid)
        wallet = self.get(exs.quote_code)
        # 计算是否爆仓
        tot_profit = sum(od.profit for od in od_list)
        wallet.unrealized_pol = tot_profit
        wallet.used_upol = 0
        if tot_profit < 0:
            margin_ratio = abs(tot_profit) / wallet.total()
            if margin_ratio > 0.99:
                # 总亏损超过总资产，爆仓
                self.on_acc_bomb(exs.quote_code, od_list)
                return
        from banbot.exchange import get_exchange
        exchange = get_exchange(exs.exchange, exs.market)
        for od in od_list:
            if not od.enter.filled:
                continue
            cur_price = MarketPrice.get(exs.symbol)
            # 计算名义价值
            quote_value = od.enter.filled * cur_price
            # 计算当前所需保证金
            cur_margin = quote_value / od.leverage
            # 判断价格走势和开单方向是否相同
            is_good = (cur_price - od.enter.average) * (-1 if od.short else 1)
            if is_good < 0:
                # 价格走势不同，产生亏损，判断是否自动补充保证金
                if od.profit > 0:
                    raise ValueError(f'od profit should < 0: {od}, profit: {od.profit}')
                # 计算维持保证金=名义价值*维持保证金率-维持保证金额
                min_margin = exchange.min_margin(quote_value)  # 要求的最低保证金
                if abs(od.profit) >= (cur_margin - min_margin) * self.margin_add_rate:
                    # 当亏损达到初始保证金比例时，为此订单增加保证金避免强平
                    loss_pct = round(self.margin_add_rate * 100)
                    logger.debug('loss %d%% %s %.5f %.5f', loss_pct, od, od.profit, cur_margin)
                    cur_margin -= od.profit
            # 价格走势和预期相同。所需保证金增长
            try:
                wallet.set_margin(od.key, cur_margin)
            except LackOfCash as e:
                logger.debug('cash lack, add margin fail: %s %.5f', od.key, e.amount)

    def on_acc_bomb(self, coin: str, od_list: List[InOutOrder]):
        """
        账户爆仓，相关订单退出，钱包重置。
        """
        wallet = self.get(coin)
        for od in od_list:
            od.local_exit(ExitTags.bomb)
        wallet.reset()
        raise AccountBomb(coin)

    def get(self, symbol: str, after_ts: float = 0):
        if self.update_at + 1 < after_ts:
            logger.warning(f'wallet expired: expect > {after_ts}, delay: {after_ts - self.update_at} ms')
        if symbol not in self.data:
            self.data[symbol] = ItemWallet(symbol)
        return self.data[symbol]

    def get_amount_by_legal(self, symbol: str, legal_cost: float):
        '''
        根据花费的USDT计算需要的数量，并返回可用数量
        :param symbol: 产品，不是交易对。如：USDT
        :param legal_cost: 花费法币金额（一般是USDT）
        '''
        return legal_cost / MarketPrice.get(symbol)

    def _calc_legal(self, item_amt: Callable, symbols: Iterable[str] = None):
        if symbols:
            data = {k: self.data[k] for k in symbols if k in self.data}
        else:
            data = self.data
        amounts, coins, prices = [], [], []
        for key, item in data.items():
            if key.find('USD') >= 0:
                price = 1
            else:
                price = MarketPrice.get(key)
            amounts.append(item_amt(item) * price)
            coins.append(key)
            prices.append(price)
        return amounts, coins, prices

    def ava_legal(self, symbols: Iterable[str] = None):
        return sum(self._calc_legal(lambda x: x.available, symbols)[0])

    def total_legal(self, symbols: Iterable[str] = None, with_upol=False):
        return sum(self._calc_legal(lambda x: x.total(with_upol), symbols)[0])

    def profit_legal(self, symbols: Iterable[str] = None):
        return sum(self._calc_legal(lambda x: x.unrealized_pol, symbols)[0])

    def get_withdraw_legal(self, symbols: Iterable[str] = None):
        return sum(self._calc_legal(lambda x: x.withdraw, symbols)[0])

    def withdraw_legal(self, amount: float, symbols: Iterable[str] = None):
        """
        从余额提现，从而禁止一部分钱开单。
        """
        amounts, coins, prices = self._calc_legal(lambda x: x.available, symbols)
        total = sum(amounts)
        draw_amts = [(a / total) * amount / prices[i] for i, a in enumerate(amounts)]
        for i, amt in enumerate(draw_amts):
            item = self.get(coins[i])
            draw_amt = min(amt, item.available)
            item.withdraw += draw_amt
            item.available -= draw_amt

    def fiat_value(self, *symbols, with_upol=False):
        '''
        返回给定币种的对法币价值。为空时返回所有币种
        '''
        if not symbols:
            symbols = list(self.data.keys())
        total_val = 0
        for symbol in symbols:
            item = self.data.get(symbol)
            if not item:
                continue
            total_val += item.fiat_value(with_upol)
        return total_val

    def __str__(self):
        from io import StringIO
        builder = StringIO()
        for key, item in self.data.items():
            pend_sum = sum([v for k, v in item.pendings.items()])
            frozen_sum = sum([v for k, v in item.frozens.items()])
            builder.write(f"{key}: {item.available:.4f}|{pend_sum:.4f}|{frozen_sum:.4f} ")
        return builder.getvalue()


class CryptoWallet(WalletsLocal):
    def __init__(self, config: dict, exchange: CryptoExchange):
        super(CryptoWallet, self).__init__(exchange)
        self.config = config
        self._symbols = set()

    def _update_local(self, balances: dict):
        message = []
        for symbol in self._symbols:
            state: dict = balances.get(symbol)
            if not state or not state.get('total'):
                continue
            free, used = state.get('free') or 0, state.get('used') or 0
            if not free and not used:
                # 未开单时，free和used都是None，这时total就是free
                free = state.get('total')
            key = 'pendings' if self.exchange.market_type == 'future' else 'frozens'
            args = {'coin': symbol, 'available': free, key: {'*': used}}
            self.data[symbol] = ItemWallet(**args)
            if free + used < 0.00001:
                continue
            message.append(f'{symbol}: {free}/{used}')
        return '  '.join(message)

    async def init(self, pairs: Iterable[str]):
        for p in pairs:
            self._symbols.update(split_symbol(p))
        await self.update_balance()

    async def update_balance(self):
        balances = await self.exchange.fetch_balance()
        self.update_at = btime.time()
        logger.info('update balances: %s', self._update_local(balances))

    @loop_forever
    async def watch_balance_forever(self):
        try:
            balances = await self.exchange.watch_balance()
        except ccxt.NetworkError as e:
            logger.error(f'watch balance net error: {e}')
            return
        self.update_at = btime.time()
        result = self._update_local(balances)
        if result:
            logger.info('update balances: %s', result)
