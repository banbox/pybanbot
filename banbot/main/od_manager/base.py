#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : base.py
# Author: anyongjin
# Date  : 2023/11/4

from banbot.data.provider import *
from banbot.main.wallets import WalletsLocal
from banbot.storage.orders import *
from banbot.util.misc import *
from banbot.util.common import SingletonArg

min_dust = 0.00000001


class OrderBook:
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
            logger.warning('depth not enough, require: {0:.5f} cur: {1:.5f}, len: {2}', depth, vol_sum, len(data_arr))
        return last_price


class OrderManager(metaclass=SingletonArg):
    def __init__(self, config: dict, exchange: CryptoExchange, wallets: WalletsLocal,
                 data_hd: DataProvider, callback: Callable):
        self.config = config
        self.exchange = exchange
        self.market_type = config.get('market_type')
        self.leverage = config.get('leverage', 3)  # 杠杆倍数
        self.od_type = config.get('order_type', OrderType.Market.value)
        self.name = exchange.name
        self.wallets = wallets
        self.data_mgr = data_hd
        self.callback = callback
        self.max_open_orders = config.get('max_open_orders', 100)
        self.fatal_stop = dict()
        self.last_ts = 0  # 记录上次订单时间戳，方便对比钱包时间戳是否正确
        self._load_fatal_stop()
        self.disable_until = 0
        '禁用交易，到指定时间再允许，13位毫秒时间戳'
        self.fatal_stop_hours: float = config.get('fatal_stop_hours', 8)
        '全局止损时禁止时间，默认8小时'
        self.pair_fee_limits = AppConfig.obj.exchange_cfg.get('pair_fee_limits')
        if self.market_type == 'future' and OrderType(self.od_type) is OrderType.Limit:
            raise ValueError('only market order type is supported for future (as watch trades is not avaiable on bnb)')

    def _load_fatal_stop(self):
        fatal_cfg = self.config.get('fatal_stop')
        if not fatal_cfg:
            return
        for k, v in fatal_cfg.items():
            self.fatal_stop[int(k)] = v

    def _fire(self, od: InOutOrder, enter: bool):
        pair_tf = f'{self.name}_{self.market_type}_{od.symbol}_{od.timeframe}'
        with TempContext(pair_tf):
            try:
                self.callback(od, enter)
            except Exception:
                logger.exception(f'fire od callback fail {od.id} {od}, enter: {enter} {traceback.format_stack()}')

    def get_context(self, od: InOutOrder):
        pair_tf = f'{self.name}_{self.market_type}_{od.symbol}_{od.timeframe}'
        return get_context(pair_tf)

    def allow_pair(self, pair: str) -> bool:
        if self.disable_until > btime.utcstamp():
            # 触发系统交易熔断时，禁止入场，允许出场
            logger.warning('order enter forbid, fatal stop, %s', pair)
            return False
        return pair not in BotGlobal.forbid_pairs

    async def try_dump(self):
        pass

    async def process_orders(self, pair_tf: str, enters: List[Tuple[str, dict]],
                             exits: List[Tuple[str, dict]], edit_triggers: List[Tuple[InOutOrder, str]] = None) \
            -> Tuple[List[InOutOrder], List[InOutOrder]]:
        '''
        批量创建指定交易对的订单
        :param pair_tf: 交易对
        :param enters: [(strategy, enter_tag, cost), ...]
        :param exits: [(strategy, exit_tag), ...]
        :param edit_triggers: 编辑止损止盈订单
        :return:
        '''
        if enters or exits:
            logger.debug('bar signals: %s %s', enters, exits)
            ctx = get_context(pair_tf)
            exs, _ = get_cur_symbol(ctx)
            enter_ods, exit_ods = [], []
            if enters:
                if btime.allow_order_enter(ctx) and self.allow_pair(exs.symbol):
                    if len(BotCache.open_ods) < self.max_open_orders:
                        for stg_name, sigin in enters:
                            try:
                                ent_od = await self.enter_order(ctx, stg_name, sigin, do_check=False)
                                enter_ods.append(ent_od)
                            except LackOfCash as e:
                                logger.warning(f'enter fail, lack of cash: {e} {stg_name} {sigin}')
                                await self.on_lack_of_cash()
                            except Exception as e:
                                logger.exception(f'enter fail: {e} {stg_name} {sigin}')
                else:
                    logger.debug('pair %s enter not allow: %s', exs.symbol, enters)
            if exits:
                for stg_name, sigout in exits:
                    exit_ods.extend(await self.exit_open_orders(sigout, None, stg_name, exs.symbol,
                                                                with_unopen=True))
        else:
            enter_ods, exit_ods = [], []
        if btime.run_mode in btime.LIVE_MODES:
            enter_ods = [od.clone() for od in enter_ods if od]
            exit_ods = [od.clone() for od in exit_ods if od]
        for od in enter_ods:
            BotCache.open_ods[od.id] = od
        if edit_triggers:
            edit_triggers = [(od, prefix) for od, prefix in edit_triggers if od not in exit_ods]
            self.submit_triggers(edit_triggers)
        return enter_ods, exit_ods

    def submit_triggers(self, triggers: List[Tuple[InOutOrder, str]]):
        """提交止损止盈单到交易所（这里加入队列）"""
        for od, prefix in triggers:
            if od.enter.status < OrderStatus.PartOk:
                # 订单未成交时，不应该挂止损
                continue
            self._put_order(od, OrderJob.ACT_EDITTG, prefix)

    async def enter_order(self, ctx: Context, strategy: str, sigin: dict, do_check=True) -> Optional[InOutOrder]:
        '''
        策略产生入场信号，执行入场订单。（目前仅支持做多）
        :param ctx:
        :param strategy:
        :param sigin: tag,short,legal_cost,cost_rate, stoploss_price, takeprofit_price
        :param do_check: 是否执行入场检查
        :return:
        '''
        if 'short' not in sigin:
            if self.market_type == 'spot':
                sigin['short'] = False
            else:
                raise ValueError(f'`short` is required in market: {self.market_type}')
        elif self.market_type == 'spot' and sigin.get('short'):
            # 现货市场，忽略做空单
            raise ValueError('short order unavaiable in spot market')
        exs, timeframe = get_cur_symbol(ctx)
        if do_check and (not btime.allow_order_enter(ctx) or not self.allow_pair(exs.symbol)):
            raise RuntimeError('pair %s enter not allowed' % exs.symbol)
        tag = sigin.pop('tag')
        if 'leverage' not in sigin and self.market_type == 'future':
            sigin['leverage'] = self.leverage
        od = InOutOrder(
            **sigin,
            sid=exs.id,
            symbol=exs.symbol,
            timeframe=timeframe,
            enter_tag=tag,
            enter_at=btime.time_ms(),
            init_price=MarketPrice.get(exs.symbol),
            strategy=strategy
        )
        await od.save()
        if btime.run_mode in LIVE_MODES:
            logger.info('enter order {0} {1} cost: {2:.2f}', od.symbol, od.enter_tag, od.enter_cost)
        self._put_order(od, OrderJob.ACT_ENTER)
        return od

    def _put_order(self, od: InOutOrder, action: str, data: str = None):
        pass

    async def _find_open_orders(self, sigout: dict, strategy: str = None, pairs: Union[str, List[str]] = None,
                                is_force=False, od_dir: str = None, ):
        is_exact = False
        if 'order_id' in sigout:
            is_exact = True
            order_list = [await InOutOrder.get(sigout.pop('order_id'))]
        else:
            order_list = await InOutOrder.open_orders(strategy, pairs)
        if not is_exact:
            # 精确指定退出订单ID时，忽略方向过滤
            if od_dir == 'both':
                pass
            elif od_dir == 'long' or sigout.get('short') is False:
                order_list = [od for od in order_list if not od.short]
            elif od_dir == 'short' or sigout.get('short'):
                order_list = [od for od in order_list if od.short]
            elif self.market_type != 'spot':
                raise ValueError(f'`od_dir` is required in market: {self.market_type}')
        if 'enter_tag' in sigout:
            enter_tag = sigout.pop('enter_tag')
            order_list = [od for od in order_list if od.enter_tag == enter_tag]
        if not is_force:
            # 非强制退出，筛选可退出订单
            order_list = [od for od in order_list if od.can_close()]
        return order_list

    async def exit_open_orders(self, sigout: dict, price: Optional[float] = None, strategy: str = None,
                               pairs: Union[str, List[str]] = None, is_force=False, od_dir: str = None,
                               with_unopen=False) -> List[InOutOrder]:
        order_list = await self._find_open_orders(sigout, strategy, pairs, is_force, od_dir)
        result = []
        if not order_list:
            return result
        # 计算要退出的数量
        all_amount = sum(od.enter.filled for od in order_list)
        exit_amount = all_amount
        if sigout.get('unopen_only'):
            # 只退出尚未入场的订单（挂单）
            exit_amount = 0
            with_unopen = True
        elif 'amount' in sigout:
            exit_amount = sigout.pop('amount')
        elif 'exit_rate' in sigout:
            exit_amount = all_amount * sigout.pop('exit_rate')
        for od in order_list:
            if not od.enter.filled:
                if with_unopen:
                    await self.exit_order(od, copy.copy(sigout), price)
                continue
            cur_ext_rate = exit_amount / od.enter.filled
            if cur_ext_rate < 0.01:
                break
            try:
                exit_amount -= od.enter.filled
                cur_out = copy.copy(sigout)
                if cur_ext_rate < 0.99:
                    cur_out['amount'] = od.enter.filled * cur_ext_rate
                res_od = await self.exit_order(od, cur_out, price)
                if res_od:
                    result.append(res_od)
            except Exception as e:
                logger.error(f'exit order fail: {e} {od}')
        return result

    async def exit_order(self, od: InOutOrder, sigout: dict, price: Optional[float] = None) -> Optional[InOutOrder]:
        if od.exit_tag:
            logger.debug('order already exit, skip: %s', od.key)
            return
        exit_amt = sigout.get('amount')
        if exit_amt:
            exit_rate = exit_amt / od.get_exit_amount()
            if exit_rate < 0.99:
                # 要退出的部分不足99%，分割出一个小订单，用于退出。
                part = od.cut_part(exit_amt)
                # 这里part的key和原始的一样，所以part作为src_key
                tgt_key, src_key = od.key, part.key
                exs = ExSymbol.get_by_id(od.sid)
                self.wallets.cut_part(src_key, tgt_key, exs.base_code, (1 - exit_rate))
                self.wallets.cut_part(src_key, tgt_key, exs.quote_code, (1 - exit_rate))
                return await self.exit_order(part, sigout, price)
        od.exit_tag = sigout.pop('tag')
        od.exit_at = btime.time_ms()
        if price:
            sigout['price'] = price
        od.update_exit(**sigout)
        await od.save()
        if btime.run_mode in LIVE_MODES:
            logger.info('exit order {0} {1}', od, od.exit_tag)
        self._put_order(od, OrderJob.ACT_EXIT)
        return od

    def _finish_order(self, od: InOutOrder):
        # fee_rate = (od.enter.fee or 0) + (od.exit.fee or 0)
        od.update_profits()
        if od.id in BotCache.open_ods:
            del BotCache.open_ods[od.id]
            logger.debug(f'remove open key {od.key} _finish_order')
        # if self.pair_fee_limits and fee_rate and od.symbol not in self.forbid_pairs:
        #     limit_fee = self.pair_fee_limits.get(od.symbol)
        #     if limit_fee is not None and fee_rate > limit_fee * 2:
        #         self.forbid_pairs.add(od.symbol)
        #         logger.error('%s fee Over limit: %f', od.symbol, self.pair_fee_limits.get(od.symbol, 0))
        od.save_mem()

    def update_by_bar(self, all_opens: List[InOutOrder], pair: str, timeframe: str, row):
        """使用价格更新订单的利润等。可能会触发爆仓：AccountBomb"""
        price = float(row[ccol])
        if btime.run_mode not in LIVE_MODES:
            self.wallets.update_at = btime.time()
        cur_orders = [od for od in all_opens if od.symbol == pair]
        # 更新订单利润
        for od in cur_orders:
            od.update_profits(price)

    async def on_lack_of_cash(self):
        pass

    async def check_fatal_stop(self):
        if self.disable_until >= btime.utcstamp():
            return
        async with dba():
            for check_mins, bad_ratio in self.fatal_stop.items():
                fatal_loss = await self.calc_fatal_loss(check_mins)
                if fatal_loss >= bad_ratio:
                    logger.error(
                        f'{check_mins}分钟内损失{(fatal_loss * 100):.2f}%，禁止下单{self.fatal_stop_hours}小时!')
                    self.disable_until = btime.utcstamp() + self.fatal_stop_hours * 60 * 60000
                    break

    async def calc_fatal_loss(self, back_mins: int) -> float:
        '''
        计算系统级别最近n分钟内，账户余额损失百分比
        :param back_mins:
        :return:
        '''
        fin_loss = 0
        min_time_ms = btime.to_utcstamp(btime.now() - btime.timedelta(minutes=back_mins), ms=True)
        min_time_ms = max(min_time_ms, BotGlobal.start_at)
        his_orders = await InOutOrder.his_orders()
        for i in range(len(his_orders) - 1, -1, -1):
            od = his_orders[i]
            if od.enter.create_at < min_time_ms:
                break
            fin_loss += od.profit
        if fin_loss >= 0:
            return 0
        fin_loss = abs(fin_loss)
        total_legal = self.wallets.total_legal()
        return fin_loss / (fin_loss + total_legal)

    def cleanup(self):
        pass
