#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : binance.py
# Author: anyongjin
# Date  : 2023/11/4
from banbot.main.od_manager.lives.base import *


class BinanceOrderMgr(LiveOrderMgr):
    def __init__(self, config: dict, exchange: CryptoExchange, wallets: CryptoWallet, data_hd: LiveDataProvider,
                 callback: Callable):
        super().__init__(config, exchange, wallets, data_hd, callback)

    async def _apply_exg_order(self, od: Order, data: dict):
        info: dict = data['info']
        state = info['X']
        if state == 'NEW':
            return
        cur_ts = info.get('E') or info.get('T')  # 合约订单没有E
        if cur_ts < od.update_at:
            # 收到的订单更新不一定按服务器端顺序。故早于已处理的时间戳的跳过
            return
        od.update_at = cur_ts  # 记录上次更新的时间戳，避免旧数据覆盖新数据
        od.amount = float(info['q'])
        if state in {'CANCELED', 'REJECTED', 'EXPIRED', 'EXPIRED_IN_MATCH'}:
            od.update_props(status=OrderStatus.Close)
        inout_status = None
        if state in {'FILLED', 'PARTIALLY_FILLED'}:
            od_status = OrderStatus.Close if state == 'FILLED' else OrderStatus.PartOk
            filled, fee_val = float(info['z']), float(info['n'])
            if self.market_type == 'future':
                average = float(info['ap'])
            else:
                average = float(info['Z']) / filled
            kwargs = dict(status=od_status, order_type=info['o'], filled=filled, average=average)
            if od_status == OrderStatus.Close:
                kwargs['price'] = kwargs['average']
            if fee_val:
                fee_name = info['N'] or ''
                kwargs['fee_type'] = fee_name
                if od.symbol.endswith(fee_name):
                    # 期货市场手续费以USD计算
                    fee_val /= average
                kwargs['fee'] = fee_val / filled
            od.update_props(**kwargs)
            mtaker = 'maker' if info['m'] else 'taker'
            fee_key = f'{od.symbol}_{mtaker}'
            self.exchange.pair_fees[fee_key] = od.fee_type, od.fee
            if od_status == OrderStatus.Close:
                if od.enter:
                    inout_status = InOutStatus.FullEnter
                else:
                    inout_status = InOutStatus.FullExit
            self.last_ts = btime.time()
        else:
            logger.error('unknown bnb order status: %s, %s', state, data)
            return
        inout_od: InOutOrder = await InOutOrder.get(od.inout_id)
        if inout_status:
            inout_od.status = inout_status
        if inout_status == InOutStatus.FullExit:
            self._finish_order(inout_od)
            logger.debug('fire exg od: %s', inout_od)
            await self._cancel_trigger_ods(od)
            await self._fire(inout_od, od.enter)

    async def _exit_by_exg_order(self, trade: dict) -> bool:
        info: dict = trade['info']
        pair, filled = trade['symbol'], float(info['z'])
        if not filled:
            # 忽略未成交的交易
            return False
        is_short = info.get('ps') == 'SHORT'
        od_side = trade['side']
        open_ods = await InOutOrder.open_orders(pairs=pair)
        open_ods = [od for od in open_ods if od.short == is_short and od.enter.side != od_side]
        if not open_ods:
            # 没有同方向，相反操作的订单
            return False
        is_reduce_only = info.get('R', False)
        od_price, od_time = float(info['ap']), trade['timestamp']
        order_id, od_type = trade['order'], trade['type']
        fee_name = info['N'] or ''
        fee_val = float(info['n'])
        if fee_val:
            fee_name = info['N'] or ''
            if pair.endswith(fee_name):
                # 期货市场手续费以USD计算
                fee_val /= od_price
            fee_val /= filled
        for iod in open_ods:
            # 尝试平仓
            filled, part = await self._try_fill_exit(iod, filled, od_price, od_time, order_id,
                                                     od_type, fee_name, fee_val)
            if part.status == InOutStatus.FullExit:
                self._finish_order(part)
                logger.info('exit : %s by third %s', part, trade)
                await self._fire(part, False)
            if filled <= min_dust:
                break
        if not is_reduce_only and filled > min_dust and self.allow_take_over:
            # 有剩余数量，创建相反订单
            exs = ExSymbol.get(self.name, self.market_type, pair)
            iod = self._create_inout_od(exs, is_short, od_price, filled, od_type, fee_val, fee_name, od_time,
                                        OrderStatus.Close, order_id)
            if iod:
                await iod.save()
                logger.debug('enter for left: %s', iod)
                await self._fire(iod, True)
        return True

    async def _trace_exg_order(self, trade: dict):
        info: dict = trade['info']
        if info.get('R', False):
            # 忽略只减仓订单
            return
        state: str = info['X']
        if state != 'FILLED':
            # 只对完全入场的尝试跟踪
            return
        side: str = info['S']
        position: str = info.get('ps')
        if self.market_type == 'future':
            if position == 'LONG' and side == 'SELL' or position == 'SHORT' and side == 'BUY':
                # 忽略平仓的订单
                return
        else:
            if side == 'SELL':
                # 现货市场卖出即平仓，忽略平仓
                return
        exs = ExSymbol.get(self.name, self.market_type, trade['symbol'])
        od_status = OrderStatus.Close if state == 'FILLED' else OrderStatus.PartOk
        filled, fee_val = float(info['z']), float(info['n'])
        if self.market_type == 'future':
            average = float(info['ap'])
        else:
            average = float(info['Z']) / filled
        fee_name = info['N'] or ''
        if fee_name and exs.symbol.endswith(fee_name):
            # 期货市场手续费以USD计算
            fee_val /= average
        fee_rate = fee_val / filled
        is_short = position == 'SHORT'
        od = self._create_inout_od(exs, is_short, average, filled, info['o'], fee_rate, fee_name,
                                   btime.time_ms(), od_status, trade['id'])
        if od:
            await od.save()
            logger.debug('enter od: %s', od)
            await self._fire(od, True)
            stg_list = BotGlobal.pairtf_stgs.get(f'{exs.symbol}_{od.timeframe}')
            stg = next((stg for stg in stg_list if stg.name == self.take_over_stgy), None)
            if stg:
                stg.init_third_od(od)
        return od
