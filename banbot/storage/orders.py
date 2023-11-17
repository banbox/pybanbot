#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : trades.py
# Author: anyongjin
# Date  : 2023/3/21
import math
from dataclasses import dataclass
from banbot.compute.sta_inds import *
from banbot.util.tf_utils import *
from banbot.storage.base import *
from banbot.storage.bot_task import BotTask
from banbot.util.misc import del_dict_prefix
from banbot.util import btime
from banbot.storage.common import BotGlobal
from banbot.storage.extension import InfoPart


class OrderStatus:
    Init = 0
    PartOk = 1
    Close = 2  # cancel, expired, close; 部分成交也可能是这个状态


class InOutStatus:
    Init = 0
    PartEnter = 1
    FullEnter = 2
    PartExit = 3
    FullExit = 4


class EnterTags:
    third = 'third'
    '第三方开单，不确定'
    user_open = 'user_open'
    '用户主动开单'


class ExitTags:
    bot_stop = 'bot_stop'
    '机器人停止强制平仓（回测）'
    force_exit = 'force_exit'
    '发生错误时强制平仓'
    user_exit = 'user_exit'
    '用户主动平仓'
    third = 'third'
    '第三方平仓：从收到的推送流中更新订单状态'
    fatal_err = 'fatal_err'
    '退出时交易所返回错误，无法追踪订单状态'
    pair_del = 'pair_del'
    '交易对删除时平仓'
    unknown = 'unknown'
    '未知原因'
    bomb = 'bomb'
    '账户爆仓'
    stoploss = 'stoploss'
    '止损'
    takeprofit = 'takeprofit'
    '止盈'
    data_stuck = 'data_stuck'
    '数据反馈超时'


class Order(BaseDbModel):
    '''
    交易所订单；一次买入（卖出）就会产生一个订单
    同一交易所下，symbol+order_id可唯一确定一个订单。
    '''

    __tablename__ = 'exorder'

    __table_args__ = (
        sa.Index('idx_od_task_id', 'task_id'),
        sa.Index('idx_od_inout_id', 'inout_id'),
        sa.Index('idx_od_status', 'status'),
    )
    # 插入后更新obj的default值到对应列
    __mapper_args__ = {'eager_defaults': True}

    id = Column(sa.Integer, primary_key=True)
    task_id = Column(sa.Integer)
    inout_id = Column(sa.Integer)
    symbol = Column(sa.String(50))
    enter = Column(sa.Boolean, default=False)
    order_type = Column(sa.String(50))
    order_id = Column(sa.String(164))  # 交易所订单ID，如修改订单会变化，记录最新的值
    side = Column(sa.String(10))
    'buy/sell'
    create_at = Column(sa.BIGINT)  # 创建时间，13位整型时间戳
    price = Column(sa.Float)
    '入场价格，市价单此项为空'
    average = Column(sa.Float)
    '平均成交价格'
    amount = Column(sa.Float)
    'base币的数量；这里无需扣除手续费，这里和实际钱包到账金额不同'
    filled = Column(sa.Float)
    '已成交数量，这里不用扣除手续费，完全成交时和amount相等'
    status = Column(sa.SMALLINT, default=OrderStatus.Init)
    fee = Column(sa.Float, default=0)
    fee_type = Column(sa.String(10))
    update_at = Column(sa.BIGINT)
    '13位，上次更新的交易所时间戳，如果trade小于此值，则是旧的数据不更新'

    def cut_part(self, cut_rate: float, fill=True) -> 'Order':
        part = Order(task_id=self.task_id, symbol=self.symbol, enter=self.enter,
                     order_type=self.order_type, order_id=self.order_id, side=self.side,
                     create_at=self.create_at, price=self.price, average=self.average,
                     amount=self.amount * cut_rate, fee=self.fee, fee_type=self.fee_type,
                     update_at=self.update_at)
        self.amount -= part.amount
        if fill and self.filled:
            if self.filled <= part.amount:
                part.filled = self.filled
                self.filled = 0
            else:
                part.filled = part.amount
                self.filled -= part.amount
        elif self.filled > self.amount:
            part.filled = self.filled - self.amount
            self.filled = self.amount
        if part.filled >= part.amount:
            part.status = OrderStatus.Close
        elif part.filled:
            part.status = OrderStatus.PartOk
        else:
            part.status = OrderStatus.Init
        return part

    @orm.reconstructor
    def __init__(self, **kwargs):
        from banbot.storage import BotTask
        if self.order_type or self.id:
            # 从数据库读取映射对象。这里不用设置，否则会覆盖数据库的值
            data = dict()
        else:
            cur_stamp = math.floor(btime.time() * 1000)
            data = dict(enter=False, status=OrderStatus.Init, fee=0, task_id=BotTask.cur_id,
                        side='buy', filled=0, create_at=cur_stamp, update_at=0)
        kwargs = {**data, **kwargs}
        super(Order, self).__init__(**kwargs)

    def __str__(self):
        if not self.amount:
            fill_pct = 0
        else:
            fill_pct = int((self.filled or 0) * 100 // self.amount)
        return f'{self.side} {(self.amount or 0):.5f}[{fill_pct}%] at {self.price}'


class InOutOrder(BaseDbModel, InfoPart):
    '''
    策略逻辑订单（包含入场、出场两个Order）
    为避免过度复杂，不支持市价单按定价金额买入（需按基准产品数量买入）
    一个交易所的所有订单维护在一个OrderManager中
    '''
    __tablename__ = 'iorder'

    __table_args__ = (
        sa.Index('idx_io_task_id', 'task_id'),
        sa.Index('idx_io_status', 'status'),
    )
    # 插入后更新obj的default值到对应列
    __mapper_args__ = {'eager_defaults': True}

    _open_ods: ClassVar[Dict[int, 'InOutOrder']] = dict()
    _his_ods: ClassVar[List['InOutOrder']] = []
    _next_id: ClassVar[int] = 1

    id = Column(sa.Integer, primary_key=True)
    task_id = Column(sa.Integer)
    symbol = Column(sa.String(50))
    sid = Column(sa.Integer)
    timeframe = Column(sa.String(5))
    short = Column(sa.Boolean)
    '是否是做空单'
    status = Column(sa.SMALLINT, default=InOutStatus.Init)
    '交易加锁的键，阻止相同键同时下单'
    enter_tag = Column(sa.String(30))
    init_price = Column(sa.Float)
    '发出信号时入场价格，仅用于策略后续计算'
    quote_cost = Column(sa.Float)
    '花费定价币金额，当价格不确定时，可先不设置amount，后续通过此字段计算amount'
    exit_tag = Column(sa.String(30))
    leverage = Column(sa.Integer, default=1)
    '杠杆倍数；现货杠杆和期货合约都可使用'
    enter_at = Column(sa.BIGINT)
    '13位时间戳，策略决定入场时间戳'
    exit_at = Column(sa.BIGINT)
    '13位时间戳，策略决定出场时间戳'
    strategy = Column(sa.String(20))
    stg_ver = Column(sa.Integer, default=0)
    profit_rate = Column(sa.Float, default=0)
    profit = Column(sa.Float, default=0)

    @orm.reconstructor
    def __init__(self, **kwargs):
        self.enter: Optional[Order] = None
        self.exit: Optional[Order] = None
        self.margin_ratio = 0  # 合约的保证金比率
        enter_kwargs = del_dict_prefix(kwargs, 'enter_', 'at', 'tag')
        exit_kwargs = del_dict_prefix(kwargs, 'exit_', 'at', 'tag')
        from_db = super().init_infos(self, kwargs)
        if from_db:
            # 从数据库创建映射的值，无需设置，否则会覆盖数据库值
            super(InOutOrder, self).__init__(**kwargs)
            return
        # 仅针对新创建的订单执行下面初始化
        from banbot.storage import BotTask
        from banbot.strategy.resolver import get_strategy
        data = dict(status=InOutStatus.Init, profit_rate=0, profit=0, task_id=BotTask.cur_id, leverage=1)
        stg = get_strategy(kwargs.get('strategy'))
        if stg:
            data['stg_ver'] = stg.version
        kwargs = {**data, **kwargs}
        super(InOutOrder, self).__init__(**kwargs)
        live_mode = btime.run_mode in btime.LIVE_MODES
        if not live_mode:
            self.id = InOutOrder._next_id
            InOutOrder._next_id += 1
        sub_com = dict(task_id=kwargs.get('task_id'), symbol=kwargs.get('symbol'))
        sub_com['inout_id'] = self.id
        enter_kwargs['side'] = 'sell' if self.short else 'buy'
        self.enter: Order = Order(**enter_kwargs, **sub_com, enter=True)
        if 'exit_amount' in kwargs:
            self.exit = Order(**exit_kwargs, **sub_com)

    @property
    def key(self):
        '''
        获取一个唯一标识某个订单的字符串
        币种:策略:方向:入场tag:入场时间戳
        '''
        side = 'short' if self.short else 'long'
        return f'{self.symbol}|{self.strategy}|{side}|{self.enter_tag}|{self.enter_at}'

    def _elp_num_offset(self, time_ms: int):
        from banbot.storage import ExSymbol
        exs = ExSymbol.get_by_id(self.sid)
        ctx = get_context(f'{exs.exchange}_{exs.market}_{exs.symbol}_{self.timeframe}')
        tf_secs = tf_to_secs(self.timeframe)
        return round((ctx[bar_time][1] - time_ms) / tf_secs / 1000)

    @property
    def elp_num_enter(self):
        return self._elp_num_offset(self.enter_at)

    @property
    def elp_num_exit(self):
        if not self.exit_at:
            return -1
        return self._elp_num_offset(self.exit_at)

    @property
    def enter_cost(self):
        '''
        获取订单花费的金额（名义价值）
        如果未入场则返回0
        '''
        if not self.enter or not self.enter.filled:
            return 0
        return self.enter.filled * (self.enter.average or self.enter.price or self.init_price)

    @property
    def enter_cost_real(self):
        '''
        获取订单真实花费（现货模式等同enter_cost，期货模式下除以leverage）
        '''
        cost = self.enter_cost
        if self.leverage <= 1:
            return cost
        return cost / self.leverage

    def can_close(self):
        '''
        订单正在退出、或刚入场需等到下个bar退出
        '''
        if self.exit_tag:
            return False
        if self.timeframe == 'ws':
            return True
        return self.elp_num_enter > 0

    def pending_type(self, timeouts: int):
        if self.exit and self.exit_tag:
            if btime.time() - self.exit.create_at < timeouts or self.exit.status == OrderStatus.Close:
                # 尚未超时，或者订单已关闭，本次不处理
                return
            return 'exit'
        elif self.enter.status != OrderStatus.Close:
            if btime.time() - self.enter.create_at < timeouts:
                # 尚未超时，本次不处理
                return
            return 'enter'

    def update_exit(self, **kwargs):
        if not self.exit:
            kwargs.update(dict(
                symbol=self.enter.symbol,
                inout_id=self.enter.inout_id,
                side='sell' if self.enter.side == 'buy' else 'buy',
            ))
            if 'amount' not in kwargs and self.enter.filled:
                # 未提供时，默认全部卖出。（这里模拟手续费扣除）
                # 这里amount传入可能为0，不能通过get('amount')方式判断
                kwargs['amount'] = self.get_exit_amount()
            self.exit = Order(**kwargs)
        else:
            self.exit.update_props(**kwargs)

    def get_exit_amount(self):
        """
        获取可平仓的最大金额
        """
        if BotGlobal.market_type == 'future':
            return self.enter.filled
        return self.enter.filled * (1 - self.enter.fee)

    def calc_profit(self, price: float = None):
        """返回利润（未扣除手续费）"""
        if not self.status or not self.enter or not self.enter.average or not self.enter.filled:
            return 0
        if price is None:
            if self.exit:
                price = self.exit.average or self.exit.price or self.enter.average
            else:
                price = self.enter.average
        ent_quote_value = self.enter.average * self.enter.filled
        profit_val = self.enter.filled * price - ent_quote_value
        if self.short:
            profit_val = 0 - profit_val
        return profit_val

    def update_profits(self, price: float = None):
        """
        此方法由接口调用，策略中不应该调用此方法。
        :param price:
        :return:
        """
        profit_val = self.calc_profit(price)
        if not profit_val:
            return
        enter_fee = (self.enter.fee or 0) if self.enter else 0
        exit_fee = (self.exit.fee or 0) if self.exit else 0
        self.profit = profit_val - enter_fee - exit_fee
        ent_price = self.enter.average or self.enter.price or self.init_price
        ent_quote_value = ent_price * self.enter.filled
        if self.leverage:
            ent_quote_value /= self.leverage
        self.profit_rate = self.profit / ent_quote_value

    def cut_part(self, enter_amt: float, exit_amt: float = 0) -> 'InOutOrder':
        '''
        从当前订单分割出一个小的InOutOrder，解决买入一次，需要分多次卖出的问题。
        '''
        enter_rate = enter_amt / self.enter.amount
        exit_rate = exit_amt / self.exit.amount if self.exit else 0
        if not self.quote_cost:
            self.quote_cost = 0
        part = InOutOrder(task_id=self.task_id, symbol=self.symbol, sid=self.sid, timeframe=self.timeframe,
                          short=self.short, status=self.status, enter_tag=self.enter_tag, init_price=self.init_price,
                          quote_cost=self.quote_cost * enter_rate, leverage=self.leverage, enter_at=self.enter_at,
                          strategy=self.strategy, stg_ver=self.stg_ver, info=self.info)
        self.enter_at += 1  # 原来订单的enter_at需要+1，防止和拆分的子订单冲突。
        self.quote_cost -= part.quote_cost
        part_enter = self.enter.cut_part(enter_rate)
        part_enter.inout_id = part.id
        part.enter = part_enter
        if not exit_rate and self.exit and self.exit.amount > self.enter.amount:
            exit_rate = (self.exit.amount - self.enter.amount) / self.exit.amount
        if exit_rate:
            part.exit_at = self.exit_at
            part.exit_tag = self.exit_tag
            part_exit = self.exit.cut_part(exit_rate)
            part_exit.inout_id = part.id
            part.exit = part_exit
        return part

    async def _save_to_db(self):
        from banbot.storage.biz import BotCache
        if self.status < InOutStatus.FullExit:
            sess = dba.session
            if not self.id:
                sess.add(self)
                await sess.flush()
            need_flush = False
            if self.enter and not self.enter.id:
                if not self.enter.inout_id:
                    self.enter.inout_id = self.id
                sess.add(self.enter)
                need_flush = True
            if self.exit and not self.exit.id:
                if not self.exit.inout_id:
                    self.exit.inout_id = self.id
                sess.add(self.exit)
                need_flush = True
            if need_flush:
                await sess.flush()
            BotCache.open_ods[self.id] = self.detach(sess)
        elif self.id in BotCache.open_ods:
            del BotCache.open_ods[self.id]

    def _save_to_mem(self):
        from banbot.storage.biz import BotCache
        if self.status < InOutStatus.FullExit:
            self._open_ods[self.id] = self
            BotCache.open_ods[self.id] = self
        else:
            if self.id in self._open_ods:
                self._open_ods.pop(self.id)
                if self.enter.filled:
                    self._his_ods.append(self)
            if self.id in BotCache.open_ods:
                del BotCache.open_ods[self.id]

    def save_mem(self):
        if btime.run_mode not in btime.LIVE_MODES:
            self._save_to_mem()
        return self

    async def save(self):
        if btime.run_mode not in btime.LIVE_MODES:
            self._save_to_mem()
        else:
            await self._save_to_db()
        return self

    def local_exit(self, tag: str, price: float = None, status_msg: str = None):
        '''
        在本地强制退出订单，立刻生效，无需等到下一个bar。这里不涉及钱包更新，钱包需要自行更新。
        '''
        amount = self.enter.filled
        if not price:
            from banbot.main.addons import MarketPrice
            price = MarketPrice.get(self.symbol) or self.enter.average or self.enter.price or self.init_price
        if not self.exit_at:
            self.exit_at = btime.time_ms()
        self.exit_tag = tag
        self.update_exit(
            tag=tag,
            update_at=btime.time_ms(),
            status=OrderStatus.Close,
            amount=amount,
            filled=amount,
            price=price,
            average=price,
        )
        self.status = InOutStatus.FullExit
        self.update_profits(price)
        if status_msg:
            self.set_info(status_msg=status_msg)

    async def force_exit(self, tag: str = None, status_msg: str = None):
        '''
        强制退出订单，如已买入，则以市价单退出。如买入未成交，则取消挂单，如尚未提交，则直接删除订单
        生成模式：提交请求到交易所。
        模拟模式：在下一个bar出现后完成退出
        '''
        from banbot.main.od_manager import LiveOrderMgr, LocalOrderManager
        if not tag:
            tag = ExitTags.force_exit
        if status_msg:
            self.set_info(status_msg=status_msg)
        exit_dic = dict(tag=tag, order_type=OrderType.Market.value)
        if self.exit:
            if btime.prod_mode():
                # 实盘模式，提交到交易所平仓
                self.exit_tag = None
                await LiveOrderMgr.obj.exit_order(self, exit_dic)
            else:
                # 模拟模式，从订单管理器平仓
                await LocalOrderManager.obj.force_exit(self)
        else:
            if btime.prod_mode():
                await LiveOrderMgr.obj.exit_order(self, exit_dic)
            else:
                await LocalOrderManager.obj.exit_order(self, exit_dic)

    def detach(self, sess: SqlSession, keep_map=False):
        result = detach_obj(sess, self, keep_map=keep_map)
        if self.enter:
            result.enter = detach_obj(sess, self.enter, keep_map=keep_map)
        if self.exit:
            result.exit = detach_obj(sess, self.exit, keep_map=keep_map)
        return result

    async def attach(self, sess: SqlSession) -> 'InOutOrder':
        if self in sess:
            return self
        db_od = await sess.merge(self)
        if self.enter:
            db_od.enter = await sess.merge(self.enter)
        if self.exit:
            db_od.exit = await sess.merge(self.exit)
        return db_od

    def update_by(self, other: 'InOutOrder'):
        self.update_props(**other.dict(origin=True))
        if other.enter:
            self.enter.update_props(**other.enter.dict())
        if other.exit:
            self.exit.update_props(**other.exit.dict())

    def dict(self, only: List[Union[str, sa.Column]] = None, skips: List[Union[str, sa.Column]] = None,
             flat_sub: bool = False, origin: bool = False):
        from banbot.util.misc import add_dict_prefix
        result = super().dict(only, skips)
        if origin:
            return result
        in_price = self.enter.average or self.enter.price or self.init_price
        in_amount = self.enter.filled or self.enter.amount
        if in_amount:
            result['enter_cost'] = in_amount * in_price
        else:
            result['enter_cost'] = self.enter_cost
        if self.infos:
            result.update(**self.infos)
        del result['info']
        if not self.exit_at:
            result['duration'] = 0
        else:
            result['duration'] = round(self.exit_at - self.enter_at) / 1000
        ent_data = self.enter.dict()
        if not flat_sub:
            result['enter'] = ent_data
        else:
            ent_data = add_dict_prefix(ent_data, 'enter_')
            result.update(**ent_data)
        if self.exit:
            ext_data = self.exit.dict()
            if not flat_sub:
                result['exit'] = ext_data
            else:
                ext_data = add_dict_prefix(ext_data, 'exit_')
                result.update(**ext_data)
        return result

    @classmethod
    async def get_orders(cls, strategy: str = None, pairs: Union[str, List[str]] = None, status: str = None,
                         close_after: int = None, close_before: int = None, sess: SqlSession = None)\
            -> List['InOutOrder']:
        if btime.run_mode in btime.LIVE_MODES:
            from banbot.storage.bot_task import BotTask
            return await get_db_orders(strategy, pairs, status, close_after, close_before, sess=sess)
        else:
            if status == 'his':
                candicates = cls._his_ods
            elif status:
                candicates: List[InOutOrder] = list(cls._open_ods.values())
            else:
                candicates = cls._his_ods + list(cls._open_ods.values())
            if not strategy and not pairs:
                return candicates
            if strategy:
                candicates = [od for od in candicates if od.strategy == strategy]
            if close_after:
                candicates = [od for od in candicates if od.exit_at > close_after]
            if close_before:
                candicates = [od for od in candicates if od.exit_at < close_before]
            if pairs:
                if isinstance(pairs, six.string_types):
                    candicates = [od for od in candicates if od.symbol == pairs]
                else:
                    pairs = set(pairs)
                    candicates = [od for od in candicates if od.symbol in pairs]
            return candicates

    @classmethod
    async def open_orders(cls, strategy: str = None, pairs: Union[str, List[str]] = None) -> List['InOutOrder']:
        """
        仅用于机器人交易和回测。兼容内存存储和数据库存储的订单。
        api接口等应直接使用get_bot_orders从数据库查询订单.
        """
        return await cls.get_orders(strategy, pairs, 'open')

    @classmethod
    async def his_orders(cls) -> List['InOutOrder']:
        """
        仅用于机器人交易和回测。兼容内存存储和数据库存储的订单。
        api接口等应直接使用get_bot_orders从数据库查询订单.
        """
        return await cls.get_orders(status='his')

    @classmethod
    async def get_pair_performance(cls, start_ms: int, stop_ms: int) -> List[Dict[str, Any]]:
        from itertools import groupby
        his_ods = await cls.get_orders(status='his', close_after=start_ms, close_before=stop_ms)
        his_ods = sorted(his_ods, key=lambda x: x.symbol)
        gps = groupby(his_ods, key=lambda x: x.symbol)
        result = []
        all_pairs = copy.copy(BotGlobal.pairs)
        for key, gp in gps:
            gp_items = list(gp)
            profit_sum = sum(od.profit for od in gp_items)
            amount_sum = sum(od.enter_cost_real for od in gp_items)
            if key in all_pairs:
                all_pairs.remove(key)
            profit_pct = profit_sum / amount_sum if amount_sum else 0
            result.append(dict(
                pair=key,
                profit_sum=profit_sum,
                profit_pct=profit_pct,
                close_num=len(gp_items)
            ))
        for pair in all_pairs:
            result.append(dict(
                pair=pair,
                profit_sum=0,
                profit_pct=0,
                close_num=0
            ))
        return result

    @classmethod
    async def get(cls, od_id: int):
        if btime.run_mode in btime.LIVE_MODES:
            sess = dba.session
            # 这里不使用sess.get(InOutOrder, od_id)方式，避免读取缓存
            op_od = (await sess.execute(select(InOutOrder).where(InOutOrder.id == od_id).limit(1))).scalar()
            if not op_od:
                return op_od
            ex_ods = list(await sess.scalars(select(Order).where(Order.inout_id == od_id)))
            op_od.enter = next((o for o in ex_ods if o.enter), None)
            op_od.exit = next((o for o in ex_ods if not o.enter), None)
            logger.debug(f'InoutOrder.get: {od_id}: {op_od}, {sess}')
            return op_od
        op_od = cls._open_ods.get(od_id)
        if op_od is not None:
            return op_od
        return next((od for od in cls._his_ods if od.id == od_id), None)

    @classmethod
    async def dump_to_db(cls):
        save_ods = cls._his_ods + list(cls._open_ods.values())
        logger.info(f'dump {len(save_ods)} orders to db...')
        await insert_orders_to_db(save_ods)

    def __str__(self):
        return f'[{self.key}] {self.enter} || {self.exit}'

    def __repr__(self):
        return self.__str__()


@dataclass
class OrderJob:
    ACT_ENTER: ClassVar[str] = 'enter'
    ACT_EXIT: ClassVar[str] = 'exit'
    ACT_EDITTG: ClassVar[str] = 'edit_trigger'
    od_id: int
    action: str
    data: str = None


def get_od_sign(od: InOutOrder):
    """获取订单的状态签名，只跟踪是否有新对象创建"""
    od_sign, ent_sign, exit_sign = None, None, None
    od_sign = od.id or 0
    if od.enter:
        ent_sign = od.enter.id or 0
    if od.exit:
        exit_sign = od.exit.id or 0
    return od_sign, ent_sign, exit_sign


class InOutTracer:
    """
    跟踪订单的前后状态，对比是否有新建的ORM对象需要保存到数据库的。
    """
    def __init__(self, ods: List[InOutOrder] = None):
        self.state = dict()
        self.orders = ods or []
        self._set_state()

    def _set_state(self):
        for od in self.orders:
            self.state[id(od)] = get_od_sign(od)

    def trace(self, ods: List[InOutOrder]):
        for od in ods:
            self.state[id(od)] = get_od_sign(od)
        od_set = set(self.orders)
        od_set.update(ods)
        self.orders = list(od_set)

    def get_changes(self):
        result = []
        for od in self.orders:
            old_sign = self.state.get(id(od))
            if old_sign:
                cur_sign = get_od_sign(od)
                if old_sign == cur_sign:
                    continue
            result.append(od)
        return result

    async def save(self):
        chg_ods = self.get_changes()
        for od in chg_ods:
            await od.save()
        self._set_state()


def get_order_filters(task_id: int = 0, strategy: str = None, pairs: Union[str, List[str]] = None, status: str = None,
                      close_after: int = None, close_before: int = None, filters=None):
    where_list = []
    if task_id:
        where_list.append(InOutOrder.task_id == task_id)
    if status:
        if status == 'his':
            where_list.append(InOutOrder.status == InOutStatus.FullExit)
        else:
            where_list.append(InOutOrder.status < InOutStatus.FullExit)
    if strategy:
        where_list.append(InOutOrder.strategy == strategy)
    if close_after:
        where_list.append(InOutOrder.exit_at > close_after)
    if close_before:
        where_list.append(InOutOrder.exit_at < close_before)
    if pairs:
        if isinstance(pairs, six.string_types):
            where_list.append(InOutOrder.symbol == pairs)
        else:
            where_list.append(InOutOrder.symbol.in_(set(pairs)))
    if filters:
        where_list.extend(filters)
    return where_list


async def get_db_orders(strategy: str = None, pairs: Union[str, List[str]] = None, status: str = None,
                        close_after: int = None, close_before: int = None, task_id: int = -1,
                        filters=None, limit=0, offset=0, order_by=None, sess: SqlSession = None) -> List[InOutOrder]:
    '''
    此方法仅用于订单管理器获取数据库订单，会自动关联Order到InOutOrder。
    :param task_id: 任务ID，不提供时默认取BotTask.cur_id
    :param strategy: 策略
    :param pairs: 交易对，字符串或列表
    :param status: open/his
    :param close_after: 毫秒时间戳
    :param close_before: 毫秒时间戳
    :param filters: 额外筛选条件
    :param limit:
    :param offset:
    :param order_by:
    :param sess: 指定的Db会话
    '''
    if task_id < 0:
        task_id = BotTask.cur_id
    if not sess:
        sess = dba.session
    where_list = get_order_filters(task_id, strategy, pairs, status, close_after, close_before, filters)
    query = select(InOutOrder).where(*where_list)
    if order_by is not None:
        query = query.order_by(order_by)
    if offset:
        query = query.offset(offset)
    if limit:
        query = query.limit(limit)
    io_rows: List[InOutOrder] = list((await sess.scalars(query)).all())
    io_ids = {row.id for row in io_rows}
    if not io_ids:
        return []
    ex_filters = [Order.task_id == task_id, Order.inout_id.in_(io_ids)]
    stmt = select(Order).where(*ex_filters)
    ex_ods = (await sess.scalars(stmt)).all()
    ex_enters = {od.inout_id: od for od in ex_ods if od.enter}
    ex_exits = {od.inout_id: od for od in ex_ods if not od.enter}
    for row in io_rows:
        row.enter = ex_enters.get(row.id)
        row.exit = ex_exits.get(row.id)
    return io_rows


async def insert_orders_to_db(orders: List[InOutOrder]):
    sess = dba.session
    for od in orders:
        od.id = None
        sess.add(od)
    await sess.flush()
    for od in orders:
        if od.enter:
            od.enter.inout_id = od.id
            sess.add(od.enter)
        if od.exit:
            od.exit.inout_id = od.id
            sess.add(od.exit)
    await sess.flush()

