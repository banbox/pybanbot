#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : main.py
# Author: anyongjin
# Date  : 2023/3/30
import asyncio

import aiofiles as aiof

from banbot.storage.orders import *
from banbot.util.common import logger, SingletonArg
from banbot.main.wallets import CryptoWallet, WalletsLocal
from banbot.strategy.base import BaseStrategy
from banbot.data.data_provider import *
from banbot.util.misc import *


class OrderBook():
    def __init__(self, **kwargs):
        self.bids = kwargs.get('bids')
        self.asks = kwargs.get('asks')
        self.symbol = kwargs.get('symbol')
        self.timestamp = kwargs.get('timestamp') or btime.time()

    def limit_price(self, side: str, depth: float):
        data_arr = self.bids if side == 'buy' else self.asks
        vol_sum, last_price = 0, 0
        for price, amount in data_arr:
            vol_sum += amount
            last_price = price
            if vol_sum >= depth:
                break
        if vol_sum < depth:
            logger.warning(f'depth not enough, require: {depth:.5f} cur: {vol_sum:.5f}, len: {len(data_arr)}')
        return last_price


class OrderManager(metaclass=SingletonArg):
    def __init__(self, config: dict, exg_name: str, wallets: WalletsLocal, data_hd: DataProvider):
        self.config = config
        self.name = exg_name
        self.wallets = wallets
        self.data_hold = data_hd
        self.prices = dict()  # 所有产品对法币的价格
        self.open_orders: Dict[str, InOutOrder] = dict()  # 尚未退出的订单
        self.his_list: List[InOutOrder] = []  # 历史已完成或取消的订单
        self.network_cost = 0.6  # 模拟网络延迟
        self.callbacks: List[Callable] = []
        self.dump_path = os.path.join(config['data_dir'], 'live/orders.json')
        self.fatal_stop = dict()
        self._load_fatal_stop()
        self.disabled = False

    def _load_fatal_stop(self):
        fatal_cfg = self.config.get('fatal_stop')
        if not fatal_cfg:
            return
        for k, v in fatal_cfg.items():
            self.fatal_stop[int(k)] = v

    async def _fire(self, od: InOutOrder, enter: bool):
        for func in self.callbacks:
            await run_async(func, od, enter)

    async def try_dump(self):
        if not self.dump_path:
            return
        result = dict(
            open_ods=[od.to_dict() for key, od in self.open_orders.items()],
            his_ods=[od.to_dict() for od in self.his_list],
        )
        async with aiof.open(self.dump_path, 'wb') as fout:
            await fout.write(orjson.dumps(result))

    async def enter_exit_pair_orders(self, pair: str, enters: List[Tuple[str, str, float]],
                                     exits: List[Tuple[str, str]], exit_keys: Dict[str, str])\
            -> Tuple[List[InOutOrder], List[InOutOrder]]:
        '''
        批量创建指定交易对的订单
        :param pair: 交易对
        :param enters: [(strategy, enter_tag, cost), ...]
        :param exits: [(strategy, exit_tag), ...]
        :param exit_keys: Dict(order_key, exit_tag)
        :return:
        '''
        price = self.data_hold.get_latest_ohlcv(pair)[ccol]
        enter_ods, exit_ods = [], []
        if enters and btime.run_mode not in NORDER_MODES:
            for stg_name, enter_tag, cost in enters:
                enter_ods.append(await self.enter_order(stg_name, pair, enter_tag, cost, price))
        if exits:
            for stg_name, exit_tag in exits:
                exit_ods.extend(await self.exit_open_orders(exit_tag, price, stg_name, pair))
        if exit_keys:
            for key, ext_tag in exit_keys.items():
                od = self.open_orders.get(key)
                if not od:
                    logger.warning(f'order not found to exit: {key}')
                    continue
                exit_ods.append(await self.exit_order(od, ext_tag, price))
        enter_ods = [od for od in enter_ods if od]
        exit_ods = [od for od in exit_ods if od]
        return enter_ods, exit_ods

    async def enter_order(self, strategy: str, pair: str, tag: str, cost: float, price: Union[float, Callable]
                          ) -> Optional[InOutOrder]:
        lock_key = f'{pair}_{tag}_{strategy}'
        if lock_key in self.open_orders or btime.run_mode in NORDER_MODES:
            # 同一交易对，同一策略，同一信号，只允许一个订单
            return
        quote_cost = self.wallets.get_avaiable_by_cost(pair.split('/')[1], cost)
        if not quote_cost:
            return
        if self.disabled:
            # 触发系统交易熔断时，禁止入场，允许出场
            logger.warning(f'order enter forbid, fatal stop, {strategy} {pair} {tag} {cost:.3f}')
            return
        if TradeLock.get_pair_locks(lock_key):
            logger.warning(f'fail to create order: {lock_key} is locked')
            return
        if callable(price):
            price = await run_async(price)
        amount = quote_cost / price
        od = InOutOrder(
            symbol=pair,
            enter_price=price,
            enter_amount=amount,
            enter_tag=tag,
            enter_at=bar_num.get(),
            strategy=strategy
        )
        self.open_orders[lock_key] = od
        TradeLock.lock(lock_key, btime.now() + btime.timedelta(minutes=1))
        logger.trade_info(f'order {od.symbol} {od.enter_tag} {od.enter.price} cost: {cost:.2f}')
        await self._enter_order(od)
        return od

    async def _enter_order(self, od: InOutOrder):
        pass

    async def _fill_pending_enter(self, arr: np.ndarray, od: InOutOrder):
        enter_price = self._sim_market_price(od.symbol, arr)
        quote_amount = enter_price * od.enter.amount
        _, base_s, quote_s, timeframe = get_cur_symbol()
        self.wallets.update_wallets(**{quote_s: -quote_amount, base_s: od.enter.amount})
        od.status = InOutStatus.FullEnter
        od.enter_at = bar_num.get()
        od.enter.filled = od.enter.amount
        od.enter.average = enter_price
        od.enter.status = OrderStatus.Close
        if not od.enter.price:
            od.enter.price = enter_price
        # TODO: 根据交易对情况设置手续费
        TradeLock.unlock(od.key)
        await self._fire(od, True)
        await self.try_dump()

    async def _fill_pending_exit(self, arr: np.ndarray, od: InOutOrder):
        exit_price = self._sim_market_price(od.symbol, arr)
        quote_amount = exit_price * od.enter.amount
        pair, base_s, quote_s, timeframe = get_cur_symbol()
        self.wallets.update_wallets(**{quote_s: quote_amount, base_s: -od.enter.amount})
        od.status = InOutStatus.FullExit
        od.exit_at = bar_num.get()
        od.update_exit(
            status=OrderStatus.Close,
            price=exit_price,
            filled=od.enter.amount,
            average=exit_price,
        )
        # TODO: 根据交易对情况设置手续费
        self._finish_order(od)
        await self._fire(od, False)
        await self.try_dump()

    def _sim_market_price(self, pair: str, arr: np.ndarray) -> float:
        '''
        计算从收到bar数据，到订单提交到交易所的时间延迟：对应的价格。
        :return:
        '''
        rate = min(1, self.network_cost / timeframe_secs.get())
        if arr is not None:
            candle = arr[-1]
            return candle[ocol] * (1 - rate) + candle[ccol] * rate
        else:
            candle = self.data_hold.get_latest_ohlcv(pair)
            return candle[ocol]

    async def fill_pending_orders(self, arr: Optional[np.ndarray]):
        '''
        填充等待交易所响应的订单。不可用于实盘；可用于回测、模拟实盘等。
        :param arr:
        :return:
        '''
        if btime.run_mode == btime.RunMode.LIVE:
            raise RuntimeError('fill_pending_orders unavaiable in LIVE mode')
        for od in list(self.open_orders.values()):
            if od.exit_tag and od.exit and od.exit.status != OrderStatus.Close:
                await self._fill_pending_exit(arr, od)
            elif od.enter.status != OrderStatus.Close:
                await self._fill_pending_enter(arr, od)

    def get_open_orders(self, strategy: str = None, pair: str = None):
        if not self.open_orders:
            return []
        result = []
        for key, od in self.open_orders.items():
            if pair and not key.startswith(pair):
                continue
            if strategy and od.strategy != strategy:
                continue
            result.append(od)
        return result

    async def exit_open_orders(self, exit_tag: str, price: float, strategy: str = None, pair: str = None)\
            -> List[InOutOrder]:
        order_list = self.get_open_orders(strategy, pair)
        result = []
        for od in order_list:
            if await self.exit_order(od, exit_tag, price):
                result.append(od)
        return result

    async def exit_order(self, od: InOutOrder, exit_tag: str, price: Union[float, Callable]) -> Optional[InOutOrder]:
        if od.exit_tag:
            return
        od.exit_tag = exit_tag
        od.exit_at = bar_num.get()
        if not price:
            # 为提供价格时，以最低价卖出（即吃单方）
            candle = self.data_hold.get_latest_ohlcv(od.symbol)
            hprice, lprice = candle[hcol: ccol]
            price = lprice * 2 - hprice
        elif callable(price):
            price = await run_async(price)
        od.update_exit(price=price)
        cost = od.exit.price * od.exit.amount
        logger.trade_info(f'exit {od.symbol} {od.exit_tag} {od.exit.price} got: {cost:.2f}')
        await self._exit_order(od)
        return od

    async def _exit_order(self, od: InOutOrder):
        '''
        调用此方法前，已将exit_tag写入到订单中
        :param od:
        :return:
        '''
        pass

    def _finish_order(self, od: InOutOrder):
        if od.key in self.open_orders:
            self.open_orders.pop(od.key)
        od.profit_rate = float(od.exit.price / od.enter.price) - 1
        od.profit = float((od.exit.price - od.enter.price) * od.enter.amount)
        self.his_list.append(od)

    def update_by_bar(self, pair_arr: np.ndarray):
        if self.open_orders:
            # 更新订单利润
            for tag in self.open_orders:
                self.open_orders[tag].update_by_bar(pair_arr)
        # 更新价格
        pair, base_s, quote_s, timeframe = get_cur_symbol()
        if quote_s.find('USD') >= 0:
            self.prices[base_s] = float(pair_arr[-1, ccol])

    def calc_custom_exits(self, pair_arr: np.ndarray, strategy: BaseStrategy) -> Dict[str, str]:
        result = dict()
        if not self.open_orders:
            return result
        pair, _, _, _ = get_cur_symbol()
        cur_strategy = strategy.name
        # 调用策略的自定义退出判断
        for od in list(self.open_orders.values()):
            if not od.can_close() or od.strategy != cur_strategy:
                continue
            if ext_tag := strategy.custom_exit(pair_arr, od):
                result[od.key] = ext_tag
        return result

    def check_fatal_stop(self):
        for check_mins, bad_ratio in self.fatal_stop.items():
            fatal_loss = self.calc_fatal_loss(check_mins)
            if fatal_loss >= bad_ratio:
                logger.error(f'{bar_end_time.get()} fatal loss {fatal_loss * 100:.2f}% in {check_mins} mins, Disable!')
                self.disabled = True
                break

    def calc_fatal_loss(self, back_mins: int) -> float:
        '''
        计算系统级别最近n分钟内，账户余额损失百分比
        :param back_mins:
        :return:
        '''
        fin_loss = 0
        min_timestamp = btime.to_utcstamp(btime.now() - btime.timedelta(minutes=back_mins))
        for i in range(len(self.his_list) - 1, -1, -1):
            od = self.his_list[i]
            if od.enter.timestamp < min_timestamp:
                break
            fin_loss += od.profit
        if fin_loss >= 0:
            return 0
        fin_loss = abs(fin_loss)
        return fin_loss / (fin_loss + self.get_legal_value())

    def _symbol_price(self, symbol: str):
        if symbol not in self.prices:
            raise RuntimeError(f'{symbol} price to USD unknown')
        return self.prices[symbol]

    def _get_legal_value(self, symbol: str):
        if symbol not in self.wallets.data:
            return 0
        amount = sum(self.wallets.data[symbol])
        if symbol.find('USD') >= 0:
            return amount
        elif not amount:
            return 0
        return amount * self._symbol_price(symbol)

    def get_legal_value(self, symbol: str = None):
        '''
        获取某个产品的法定价值。USDT直接返回。BTC等需要计算。
        :param symbol:
        :return:
        '''
        if symbol:
            return self._get_legal_value(symbol)
        else:
            result = 0
            for key in self.wallets.data:
                result += self._get_legal_value(key)
            return result


class LiveOrderManager(OrderManager):
    def __init__(self, config: dict, exchange: CryptoExchange, wallets: CryptoWallet, data_hd: LiveDataProvider):
        super(LiveOrderManager, self).__init__(config, exchange.name, wallets, data_hd)
        self.exchange = exchange
        self.exg_orders: Dict[str, Order] = dict()
        self.max_market_rate = config.get('max_market_rate', 0.0001)
        self.odbook_ttl: int = config.get('odbook_ttl', 500)
        self.odbooks: Dict[str, OrderBook] = dict()
        # 限价单的深度对应的秒级成交量
        self.limit_vol_secs = config.get('limit_vol_secs', 10)

    async def _get_odbook_price(self, pair: str, side: str, depth: float):
        '''
        获取订单簿指定深度价格。用于生成限价单价格
        :param pair:
        :param side:
        :param depth:
        :return:
        '''
        odbook = self.odbooks.get(pair)
        if not odbook or odbook.timestamp + self.odbook_ttl < time.time() * 1000:
            od_res = await self.exchange.fetch_order_book(pair, 1000)
            self.odbooks[pair] = OrderBook(**od_res)
        return self.odbooks[pair].limit_price(side, depth)

    async def calc_price(self, pair: str):
        # 如果taker的费用为0，直接使用市价单，否则获取订单簿，使用限价单
        candle = self.data_hold.get_latest_ohlcv(pair)
        high_price, low_price, close_price, vol_amount = candle[hcol: vcol + 1]
        fees = self.exchange.calc_funding_fee(pair, 'limit', 'buy', vol_amount, close_price)
        if fees['rate'] > self.max_market_rate and btime.run_mode in TRADING_MODES:
            # 手续费率超过指定市价单费率，使用限价单
            # 取过去300s数据计算；限价单深度=min(60*每秒平均成交量, 最后30s总成交量)
            his_ohlcvs = await self.exchange.fetch_ohlcv(pair, '1s', limit=300)
            vol_arr = np.array(his_ohlcvs)[:, vcol]
            depth = min(np.average(vol_arr) * self.limit_vol_secs * 2, np.sum(vol_arr[-self.limit_vol_secs:]))
            buy_price = await self._get_odbook_price(pair, 'buy', depth)
            sell_price = await self._get_odbook_price(pair, 'sell', depth)
        else:
            buy_price = high_price * 2 - low_price
            sell_price = low_price * 2 - high_price
        return buy_price, sell_price

    def _make_async_price(self, pair: str):
        _buy_price, _sell_price = None, None

        async def buy_price():
            nonlocal _buy_price, _sell_price
            if _buy_price is None:
                _buy_price, _sell_price = await self.calc_price(pair)
            return _buy_price

        async def sell_price():
            nonlocal _buy_price, _sell_price
            if _sell_price is None:
                _buy_price, _sell_price = await self.calc_price(pair)
            return _sell_price
        return buy_price, sell_price

    async def enter_exit_pair_orders(self, pair: str, enters: List[Tuple[str, str, float]],
                                     exits: List[Tuple[str, str]], exit_keys: Dict[str, str]):
        '''
        批量创建指定交易对的订单
        :param pair: 交易对
        :param enters: [(strategy, enter_tag, cost), ...]
        :param exits: [(strategy, exit_tag), ...]
        :param exit_keys: Dict(order_key, exit_tag)
        :return:
        '''
        if self.disabled and not (exits or exit_keys):
            logger.warning('system enter disabled')
            return
        run_tasks = []
        buy_price, sell_price = self._make_async_price(pair)
        if enters and btime.run_mode not in NORDER_MODES:
            for stg_name, enter_tag, cost in enters:
                run_tasks.append(self.enter_order(stg_name, pair, enter_tag, cost, buy_price))
        if exits:
            for stg_name, exit_tag in exits:
                for od in self.get_open_orders(stg_name, pair):
                    run_tasks.append(self.exit_order(od, exit_tag, sell_price))
        if exit_keys:
            for key, ext_tag in exit_keys.items():
                od = self.open_orders.get(key)
                if od.can_close():
                    continue
                if not od:
                    logger.warning(f'order not found to exit: {key}')
                    continue
                run_tasks.append(self.exit_order(od, ext_tag, sell_price))
        enter_ods, exit_ods = [], []
        for task in asyncio.as_completed(run_tasks):
            od = await task
            if not od:
                continue
            if od.enter_at == bar_num.get():
                enter_ods.append(od)
            else:
                exit_ods.append(od)
        return enter_ods, exit_ods

    def _update_subod_by_ccxtres(self, od: InOutOrder, is_enter: bool, order: dict):
        sub_od = od.enter if is_enter else od.exit
        logger.info(f'create order res: {order}')
        sub_od.order_id = order["id"]
        self.exg_orders[f'{od.symbol}_{sub_od.order_id}'] = sub_od
        order_status, fee, filled = order.get('status'), order.get('fee'), float(order.get('filled', 0))
        if filled > 0:
            filled_price = safe_value_fallback(order, 'average', 'price', sub_od.price)
            sub_od.update(average=filled_price, filled=filled, status=OrderStatus.PartOk)
            od.status = InOutStatus.PartEnter if is_enter else InOutStatus.PartExit
            if fee and fee.get('cost'):
                sub_od.fee = fee.get('cost')
                sub_od.fee_type = fee.get('currency')
            # 下单后立刻有成交的，认为是taker方（ccxt返回的信息中未明确）
            fee_key = f'{od.symbol}_taker'
            self.exchange.pair_fees[fee_key] = sub_od.fee_type, sub_od.fee
        if order_status in {'expired', 'rejected', 'closed'}:
            sub_od.status = OrderStatus.Close
            od.status = InOutStatus.FullEnter if is_enter else InOutStatus.FullExit
            if filled == 0:
                logger.warning(f'{od} is {order_status} by {self.name}, no filled')
        if od.status == InOutStatus.FullExit:
            self._finish_order(od)

    async def _create_exg_order(self, od: InOutOrder, is_enter: bool):
        sub_od = od.enter if is_enter else od.exit
        side, amount, price = sub_od.side, sub_od.amount, sub_od.price
        order = await self.exchange.create_limit_order(od.symbol, side, amount, price)
        self._update_subod_by_ccxtres(od, is_enter, order)
        await self._fire(od, is_enter)
        await self.try_dump()

    async def _enter_order(self, od: InOutOrder):
        TradeLock.unlock(od.key)
        if btime.run_mode == btime.RunMode.LIVE:
            await self._create_exg_order(od, True)

    async def _exit_order(self, od: InOutOrder):
        if btime.run_mode == btime.RunMode.LIVE:
            await self._create_exg_order(od, False)

    async def _update_bnb_order(self, od: Order, data: dict):
        info = data['info']
        state = info['X']
        if state == 'NEW':
            return
        if state in {'CANCELED', 'REJECTED', 'EXPIRED', 'EXPIRED_IN_MATCH'}:
            od.update(status=OrderStatus.Close)
        inout_od = self.open_orders[od.inout_key]
        if state in {'FILLED', 'PARTIALLY_FILLED'}:
            od_status = OrderStatus.Close if state == 'FILLED' else OrderStatus.PartOk
            filled, total_cost, fee_val = float(info['z']), float(info['Z']), float(info['n'])
            kwargs = dict(status=od_status, order_type=info['o'], filled=filled, average=total_cost/filled)
            if not od.price and od_status == OrderStatus.Close:
                kwargs['price'] = kwargs['average']
            if info['N']:
                kwargs['fee_type'] = info['N']
            if fee_val:
                kwargs['fee'] = od.fee + fee_val
            od.update(**kwargs)
            mtaker = 'maker' if info['m'] else 'taker'
            fee_key = f'{od.symbol}_{mtaker}'
            self.exchange.pair_fees[fee_key] = od.fee_type, od.fee
            if od_status == OrderStatus.Close:
                if od.enter:
                    inout_od.status = InOutStatus.FullEnter
                else:
                    inout_od.status = InOutStatus.FullExit
        else:
            logger.error(f'unknown bnb order status: {state}, {data}')
            return
        if inout_od.status == InOutStatus.FullExit:
            self._finish_order(inout_od)
        await self._fire(inout_od, od.enter)
        await self.try_dump()

    async def _update_order(self, od: Order, data: dict):
        if self.name.find('binance') >= 0:
            await self._update_bnb_order(od, data)
        else:
            raise ValueError(f'unsupport exchange to update order: {self.name}')

    @loop_forever
    async def listen_orders_forever(self):
        orders = await self.exchange.watch_my_trades()
        for data in orders:
            key = f"{data['symbol']}_{data['order']}"
            if key not in self.exg_orders:
                logger.warning(f'update order {key} not found in {self.name} {self.exg_orders.keys()}')
                continue
            await self._update_order(self.exg_orders[key], data)

    @loop_forever
    async def trail_open_orders_forever(self):
        timeouts = self.config.get('limit_vol_secs', 5) * 2
        await self._trail_open_orders(timeouts)
        if btime.run_mode != RunMode.LIVE:
            # 非实盘模式下，这里应该用sleep，否则在预热阶段会出现死循环
            await asyncio.sleep(timeouts)
        else:
            await btime.sleep(timeouts)

    async def _trail_open_orders(self, timeouts: int):
        '''
        跟踪未关闭的订单，根据市场价格及时调整，避免长时间无法成交
        :return:
        '''
        if not self.open_orders or btime.run_mode != RunMode.LIVE:
            return
        exp_orders: Dict[str, List[InOutOrder]] = dict()
        for od in list(self.open_orders.values()):
            if od.exit and od.exit_tag and od.exit.filled < od.exit.amount:
                # 未完全退出
                if btime.time() - od.exit.timestamp < timeouts:
                    # 尚未超时，本次不处理
                    continue
                if od.symbol not in exp_orders:
                    exp_orders[od.symbol] = []
                exp_orders[od.symbol].append(od)
            elif od.enter.filled < od.enter.amount:
                if btime.time() - od.enter.timestamp < timeouts:
                    # 尚未超时，本次不处理
                    continue
                if od.symbol not in exp_orders:
                    exp_orders[od.symbol] = []
                exp_orders[od.symbol].append(od)
        if not exp_orders:
            return
        for pair, od_list in exp_orders.items():
            buy_price, sell_price = self._make_async_price(pair)
            buy_price = await buy_price()
            sell_price = await sell_price()
            for od in od_list:
                if od.exit and od.exit_tag:
                    od.exit.timestamp = btime.time()
                    logger.info(f'change exit order {od.key} price: {od.exit.price} -> {sell_price}')
                    od.exit.price = sell_price
                    if not od.exit.order_id:
                        await self._exit_order(od)
                    else:
                        left_amount = od.exit.amount - od.exit.filled
                        res = await self.exchange.edit_limit_order(od.exit.order_id, od.symbol, od.exit.side,
                                                                   left_amount, sell_price)
                        self._update_subod_by_ccxtres(od, False, res)
                else:
                    od.enter.timestamp = btime.time()
                    logger.info(f'change enter order {od.key} price: {od.enter.price} -> {buy_price}')
                    od.enter.price = buy_price
                    if not od.enter.order_id:
                        await self._enter_order(od)
                    else:
                        left_amount = od.enter.amount - od.enter.filled
                        res = await self.exchange.edit_limit_order(od.enter.order_id, od.symbol, od.enter.side,
                                                                   left_amount, buy_price)
                        self._update_subod_by_ccxtres(od, True, res)
