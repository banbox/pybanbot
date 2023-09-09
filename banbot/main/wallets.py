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
from banbot.storage import InOutOrder, ExSymbol, split_symbol
from banbot.main.addons import *
from banbot.util import btime
from banbot.util.common import logger


@dataclass
class ItemWallet:
    available: float = 0
    '可用余额'
    pendings: Dict[str, float] = field(default_factory=dict)
    '买入卖出时锁定金额，键可以是订单id'
    frozens: Dict[str, float] = field(default_factory=dict)
    '空单等长期冻结金额，键可以是订单id'

    @property
    def total(self):
        sum_val = self.available
        for k, v in self.pendings.items():
            sum_val += v
        for k, v in self.frozens.items():
            sum_val += v
        return sum_val


class WalletsLocal:
    obj: 'WalletsLocal' = None

    def __init__(self):
        WalletsLocal.obj = self
        self.data: Dict[str, ItemWallet] = dict()
        self.update_at = btime.time()
        self.refill_margin = AppConfig.get().get('refill_margin', True)

    def set_wallets(self, **kwargs):
        for key, val in kwargs.items():
            if isinstance(val, ItemWallet):
                self.data[key] = val
            elif isinstance(val, (int, float)):
                self.data[key] = ItemWallet(available=val)
            else:
                raise ValueError(f'unsupport val type: {key} {type(val)}')

    def cost_ava(self, od_key: str, symbol: str, amount: float, negative: bool = False, after_ts: float = 0,
                 min_rate: float = 0.1) -> float:
        '''
        从某个币的可用余额中扣除，仅用于回测
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
            self.data[symbol] = ItemWallet()
        wallet = self.data[symbol]
        src_amount = wallet.available
        if src_amount >= amount or negative:
            # 余额充足，或允许负数，直接扣除
            real_cost = amount
        elif src_amount / amount > min_rate:
            # 差额在近似允许范围内，扣除实际值
            real_cost = src_amount
        else:
            return 0
        # logger.info(f'cost_ava wallet {key}.{symbol} {wallet.available} - {real_cost}')
        wallet.available -= real_cost
        wallet.pendings[od_key] = real_cost
        self.update_at = btime.time()
        return real_cost

    def cost_frozen(self, od_key: str, symbol: str, amount: float, after_ts: float = 0):
        '''
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
        # logger.info(f'cost_frozen wallet {od_key}.{symbol} {wallet.available} + {frozen_amt - amount}')
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
        if not src or not tgt:
            return False
        pending_amt = src.pendings.get(od_key, 0)
        if not pending_amt:
            return False
        left_pending = pending_amt - src_amount
        del src.pendings[od_key]
        # logger.info(f'confirm_pending wallet {od_key}.{src_key} {src.available} + {left_pending}')
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
        src_amount = src_dic.get(od_key)
        if not src_amount:
            return
        # logger.info(f'cancel wallet {od_key}.{symbol} {wallet.available} + {src_amount} + {add_amount}')
        src_amount += add_amount
        del src_dic[od_key]
        wallet.available += src_amount

    def enter_od(self, exs: ExSymbol, sigin: dict, od_key: str, after_ts=0):
        '''
        实盘和模拟都执行，实盘时可防止过度消费
        '''
        is_short = sigin.get('short')
        legal_cost = sigin.pop('legal_cost')
        is_future = exs.market == 'future'
        if is_future or not is_short:
            # 期货合约，现货多单锁定quote
            if legal_cost < MIN_STAKE_AMOUNT:
                raise ValueError(f'margin cost must >= {MIN_STAKE_AMOUNT}, cur: {legal_cost:.2f}')
            if is_future:
                # 期货合约，名义价值=保证金*杠杆
                legal_cost /= sigin['leverage']
            quote_cost = self.get_amount_by_legal(exs.quote_code, legal_cost)
            quote_cost = self.cost_ava(od_key, exs.quote_code, quote_cost, after_ts=after_ts)
            if not quote_cost:
                logger.debug('wallet %s empty: %f', exs.symbol, quote_cost)
                return 0
            quote_margin = quote_cost  # 计算名义数量
            if is_future:
                quote_margin *= sigin['leverage']
            wallet = self.data[exs.quote_code]
            sigin['wallet_left'] = wallet.available
            sigin['quote_cost'] = quote_margin
        else:
            # 现货空单，锁定base，允许金额为负
            base_cost = self.get_amount_by_legal(exs.base_code, legal_cost)
            base_cost = self.cost_ava(od_key, exs.base_code, base_cost, negative=True, after_ts=after_ts)
            sigin['enter_amount'] = base_cost
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
            if od.exit_tag == 'bomb':
                # 爆仓，将利润置为0-全部占用保证金
                wallet = self.data.get(exs.quote_code)
                od.profit = 0 - wallet.frozens[od.key]
                od.profit_rate = od.profit / (od.enter.price * od.enter.amount)
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

    def update_ods(self, od_list: List[InOutOrder]):
        '''
        更新订单。目前只针对期货合约订单，需要更新合约订单的保证金比率。
        保证金比率： (仓位名义价值 * 维持保证金率 - 维持保证金速算数) / (钱包余额 + 未实现盈亏)
        钱包余额 = 初始净划入余额（含初始保证金） + 已实现盈亏 + 净资金费用 - 手续费
        '''
        if btime.prod_mode():
            return
        from banbot.exchange.crypto_exchange import get_exchange
        bomb_ods = []
        for od in od_list:
            if not od.enter.filled or od.exit_tag == 'bomb':
                continue
            exs = ExSymbol.get_by_id(od.sid)
            exchange = get_exchange(exs.exchange, exs.market)
            quote_amount = od.enter.filled * MarketPrice.get(exs.symbol)
            min_margin = exchange.min_margin(quote_amount)  # 要求的最低保证金
            wallet = self.get(exs.quote_code)
            frozen_val = wallet.frozens.get(od.key)
            if not frozen_val:
                logger.error(f'no frozen_val for {od.key} {od.status} {wallet}')
                continue
            if self.refill_margin:
                # 自动从可用余额中填补保证金
                init_margin = od.enter.filled * od.enter.price / od.leverage  # 初始保证金
                if frozen_val + od.profit < init_margin:
                    # 合约亏损，从available中填充
                    add_val = min(wallet.available, init_margin - (frozen_val + od.profit))
                    frozen_val += add_val
                    wallet.frozens[od.key] = frozen_val
                    # logger.info(f'update_ods wallet {exs.quote_code} {wallet.available} - {add_val}, profit: {od.profit}')
                    wallet.available -= add_val
            # 这里分母不用再加钱包余额，上面自动从钱包扣除了。这里再加会导致多个订单重用余额
            od.margin_ratio = min_margin / (frozen_val + od.profit)
            if od.margin_ratio >= 0.999999:
                # 保证金比率100%，爆仓
                wallet.frozens[od.key] = 0
                bomb_ods.append(od)
                logger.info(f'bamb: {od.key}')
        return bomb_ods

    def get(self, symbol: str, after_ts: float = 0):
        if self.update_at + 1 < after_ts:
            logger.warning(f'wallet expired: expect > {after_ts}, delay: {after_ts - self.update_at} ms')
        if symbol not in self.data:
            self.data[symbol] = ItemWallet()
        return self.data[symbol]

    def get_amount_by_legal(self, symbol: str, legal_cost: float):
        '''
        根据花费的USDT计算需要的数量，并返回可用数量
        :param symbol: 产品，不是交易对。如：USDT
        :param legal_cost: 花费法币金额（一般是USDT）
        '''
        return legal_cost / MarketPrice.get(symbol)

    def total_legal(self, symbols: Iterable[str] = None):
        legal_sum = 0
        if symbols:
            data = {k: self.data[k] for k in symbols if k in self.data}
        else:
            data = self.data
        for key, item in data.items():
            if key.find('USD') >= 0:
                price = 1
            else:
                price = MarketPrice.get(key)
            legal_sum += item.total * price
        return legal_sum

    def _position(self, wallet: Optional[ItemWallet], prefix: str, side: str = None, enter_tag: str = None):
        if not wallet:
            return 0
        amt_list = list(wallet.pendings.items()) + list(wallet.frozens.items())
        total_amount = 0
        for key, amount in amt_list:
            if key.startswith(prefix):
                continue
            ent_side, tag, ent_at = key.split(':')[2:]
            if side and side != ent_side:
                continue
            if enter_tag and enter_tag != tag:
                continue
            total_amount += amount
        return total_amount

    def position(self, symbol: str, strategy: str, side: str = None, enter_tag: str = None):
        '''
        获取指定条件的仓位；仅支持回测
        '''
        prefix = f'{symbol}|{strategy}|'
        base_s, quote_s = split_symbol(symbol)
        base_amount = self._position(self.data.get(base_s), prefix, side, enter_tag)
        quote_amount = self._position(self.data.get(quote_s), prefix, side, enter_tag)
        legal_cost = 0
        if base_amount:
            legal_cost += base_amount * MarketPrice.get(symbol)
        if quote_amount:
            legal_cost += quote_amount * MarketPrice.get(quote_s)
        return legal_cost

    def fiat_value(self, *symbols):
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
            total_val += item.total * MarketPrice.get(symbol)
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
        super(CryptoWallet, self).__init__()
        self.exchange = exchange
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
            args = {'available': free, key: {'*': used}}
            self.data[symbol] = ItemWallet(**args)
            if free + used < 0.00001:
                continue
            message.append(f'{symbol}: {free}/{used}')
        return '  '.join(message)

    async def init(self, pairs: List[str]):
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
