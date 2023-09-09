#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : trades.py
# Author: anyongjin
# Date  : 2023/3/21
import json
from dataclasses import dataclass
from banbot.compute.sta_inds import *
from banbot.exchange.exchange_utils import tf_to_secs
from banbot.storage.base import *
from banbot.util.misc import del_dict_prefix
from banbot.util.redis_helper import AsyncRedis
from banbot.util import btime
from banbot.storage.common import BotGlobal


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


class ExitTags:
    bot_stop = 'bot_stop'
    '机器人停止强制平仓（回测）'
    force_exit = 'force_exit'
    '发生错误时强制平仓'
    user_exit = 'user_exit'
    '用户主动平仓'
    fatal_err = 'fatal_err'
    '退出时交易所返回错误，无法追踪订单状态'
    pair_del = 'pair_del'
    '交易对删除时平仓'


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
    order_type = Column(sa.String(50), default='limit')
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
    fee = Column(sa.Float)
    fee_type = Column(sa.String(10))
    update_at = Column(sa.BIGINT)
    '13位，上次更新的交易所时间戳，如果trade小于此值，则是旧的数据不更新'

    @orm.reconstructor
    def __init__(self, **kwargs):
        from banbot.storage import BotTask
        if self.order_type:
            # 从数据库读取映射对象。这里不用设置，否则会覆盖数据库的值
            data = dict()
        else:
            data = dict(enter=False, order_type='limit', status=OrderStatus.Init, fee=0, task_id=BotTask.cur_id,
                        side='buy', filled=0, create_at=btime.time(), update_at=btime.time())
        kwargs = {**data, **kwargs}
        super(Order, self).__init__(**kwargs)

    def lock(self):
        redis = AsyncRedis()
        return redis.lock(f'order_{self.id}', with_conn=True)

    def __str__(self):
        if not self.amount:
            fill_pct = 0
        else:
            fill_pct = int((self.filled or 0) * 100 // self.amount)
        return f'{self.side} {(self.amount or 0):.5f}[{fill_pct}%] at {self.price}'


class InOutOrder(BaseDbModel):
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
    info = Column(sa.String(1024))

    @orm.reconstructor
    def __init__(self, **kwargs):
        self.enter: Optional[Order] = None
        self.exit: Optional[Order] = None
        self.margin_ratio = 0  # 合约的保证金比率
        db_keys = set(self.__table__.columns.keys())
        tmp_keys = {k for k in kwargs if k not in db_keys and not k.startswith('enter_') and not k.startswith('exit_')}
        self.infos: Dict = {k: kwargs.pop(k) for k in tmp_keys}
        if self.id:
            # 从数据库创建映射的值，无需设置，否则会覆盖数据库值
            super(InOutOrder, self).__init__(**kwargs)
            if self.info:
                # 数据库初始化的必然只包含列名，这里可以直接覆盖
                self.infos: Dict = json.loads(self.info)
            return
        # 仅针对新创建的订单执行下面初始化
        if self.infos:
            # 自行实例化的对象，忽略info参数
            kwargs['info'] = json.dumps(self.infos)
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
        enter_kwargs = del_dict_prefix(kwargs, 'enter_')
        enter_kwargs['inout_id'] = self.id
        enter_kwargs['side'] = 'sell' if self.short else 'buy'
        self.enter: Order = Order(**enter_kwargs, enter=True)
        if 'exit_amount' in kwargs:
            exit_kwargs = del_dict_prefix(kwargs, 'exit_')
            exit_kwargs['inout_id'] = self.id
            self.exit = Order(**exit_kwargs)

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
        return round((ctx[bar_time][0] - time_ms) / tf_secs / 1000)

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
        '''
        if self.quote_cost:
            return self.quote_cost
        return self.enter.amount * (self.enter.price or self.init_price)

    @property
    def enter_cost_real(self):
        '''
        获取订单真实花费（现货模式等同enter_cost，期货模式下除以leverage）
        '''
        cost = self.enter_cost
        if self.leverage <= 1:
            return cost
        return cost / self.leverage

    @property
    def enter_amount(self):
        '''
        获取入场标的的数量
        '''
        if self.enter.amount:
            return self.enter.amount
        return self.quote_cost / (self.enter.price or self.init_price)

    def can_close(self):
        if self.exit_tag:
            return False
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
                kwargs['amount'] = self.enter.filled * (1 - self.enter.fee)
            self.exit = Order(**kwargs)
        else:
            if self.exit.filled and kwargs.get('filled'):
                # 已有部分退出，传入新的退出成交时，重新计算average
                cur_filled = kwargs.get('filled')
                cur_price = kwargs.get('average') or kwargs.get('price')
                if not cur_price:
                    raise ValueError('price is require to update exit')
                total_fill = cur_filled + self.exit.filled
                if total_fill > self.exit.amount:
                    raise ValueError(f'exit filled fail: {self.exit.filled:.5f}/{self.exit.amount:.5f} cur: {cur_filled:.5f}')
                kwargs['average'] = (cur_price * cur_filled + self.exit.filled * self.exit.average) / total_fill
                kwargs['filled'] = total_fill
            self.exit.update_props(**kwargs)

    def update_by_price(self, price: float):
        '''
        此方法由接口调用，策略中不应该调用此方法。
        :param price:
        :return:
        '''
        if not self.status or not self.enter.price or not self.enter.filled:
            return
        ent_fee_rate = self.enter.fee
        ent_quote_amount = self.enter.price * self.enter.amount
        if BotGlobal.market_type == 'future':
            # 期货市场，手续费以定价币计算
            get_amount = self.enter.filled
            fee_cost = ent_quote_amount * ent_fee_rate
            if self.status == InOutStatus.FullExit:
                fee_cost += get_amount * price * self.exit.fee
        else:
            get_amount = self.enter.filled * (1 - ent_fee_rate)  # 入场后的数量
            if self.status == InOutStatus.FullExit:
                # 已完全退出
                get_amount *= (1 - self.exit.fee)  # 出场后的数量
            fee_cost = 0
        # TODO: 当定价货币不是USD时，这里需要计算对应USD的利润
        self.profit = get_amount * price - ent_quote_amount - fee_cost
        if self.short:
            self.profit = 0 - self.profit
        if self.leverage:
            ent_quote_amount /= self.leverage
        self.profit_rate = self.profit / ent_quote_amount

    def _save_to_db(self):
        sess = db.session
        if self.status < InOutStatus.FullExit:
            if not self.id:
                sess.add(self)
                sess.flush()
            if self.enter and not self.enter.id:
                if not self.enter.inout_id:
                    self.enter.inout_id = self.id
                sess.add(self.enter)
            if self.exit and not self.exit.id:
                if not self.exit.inout_id:
                    self.exit.inout_id = self.id
                sess.add(self.exit)
        sess.commit()

    def _save_to_mem(self):
        if self.status < InOutStatus.FullExit:
            self._open_ods[self.id] = self
        else:
            if self.id in self._open_ods:
                self._open_ods.pop(self.id)
            self._his_ods.append(self)

    def save(self):
        if btime.run_mode not in btime.LIVE_MODES:
            self._save_to_mem()
        else:
            self._save_to_db()
        return self

    def get_info(self, key: str, def_val=None):
        if not self.infos:
            if self.info:
                self.infos: Dict = json.loads(self.info)
            else:
                return def_val
        return self.infos.get(key, def_val)

    def set_info(self, **kwargs):
        self.infos.update(kwargs)
        self.info = json.dumps(self.infos)

    def local_exit(self, tag: str, price: float = None, status_msg: str = None):
        '''
        在本地强制退出订单。这里不涉及钱包更新，钱包需要自行更新。
        '''
        amount = self.enter.filled
        if not price:
            price = self.enter.average or self.enter.price or self.init_price
        if not self.exit_at:
            self.exit_at = btime.time_ms()
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
        self.update_by_price(price)
        if status_msg:
            self.set_info(status_msg=status_msg)
        self.save()

    def force_exit(self, status_msg: str = None):
        '''
        强制退出订单，如已买入，则以市价单退出。如买入未成交，则取消挂单，如尚未提交，则直接删除订单
        '''
        from banbot.main.od_manager import LiveOrderManager, LocalOrderManager
        if status_msg:
            self.set_info(status_msg=status_msg)
        if self.exit:
            if btime.prod_mode():
                # 实盘模式，提交到交易所平仓
                price_rate = 100 if self.short else 0.01
                new_price = (self.exit.price or self.enter.price) * price_rate
                self.exit_tag = None
                LiveOrderManager.obj.exit_order(self, dict(tag=ExitTags.force_exit), new_price)
            else:
                # 模拟模式，从订单管理器平仓
                LocalOrderManager.obj.force_exit(self)
        else:
            if btime.prod_mode():
                LiveOrderManager.obj.exit_order(self, dict(tag=ExitTags.force_exit))
            else:
                LocalOrderManager.obj.exit_order(self, dict(tag=ExitTags.force_exit))
        db.session.commit()

    def detach(self, sess: SqlSession):
        detach_obj(sess, self)
        if self.enter:
            detach_obj(sess, self.enter)
        if self.exit:
            detach_obj(sess, self.exit)
        return self

    @classmethod
    def get_orders(cls, strategy: str = None, pairs: Union[str, List[str]] = None, status: str = None,
                   close_after: int = None)\
            -> List['InOutOrder']:
        if btime.run_mode in btime.LIVE_MODES:
            from banbot.storage.bot_task import BotTask
            return get_db_orders(BotTask.cur_id, strategy, pairs, status, close_after)
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
            if pairs:
                if isinstance(pairs, six.string_types):
                    candicates = [od for od in candicates if od.symbol == pairs]
                else:
                    pairs = set(pairs)
                    candicates = [od for od in candicates if od.symbol in pairs]
            return candicates

    @classmethod
    def open_orders(cls, strategy: str = None, pairs: Union[str, List[str]] = None) -> List['InOutOrder']:
        return cls.get_orders(strategy, pairs, 'open')

    @classmethod
    def his_orders(cls) -> List['InOutOrder']:
        return cls.get_orders(status='his')

    @classmethod
    def get_overall_performance(cls, minutes=None) -> List[Dict[str, Any]]:
        from itertools import groupby
        close_after = None
        if minutes:
            close_after = btime.utcstamp() - minutes * 60000
        his_ods = cls.get_orders(status='his', close_after=close_after)
        his_ods = sorted(his_ods, key=lambda x: x.symbol)
        gps = groupby(his_ods, key=lambda x: x.symbol)
        result = []
        for key, gp in gps:
            gp_items = list(gp)
            profit_sum = sum(od.profit for od in gp_items)
            amount_sum = sum(od.enter_cost_real for od in gp_items)
            result.append(dict(
                pair=key,
                profit_ratio=profit_sum,
                profit_pct=profit_sum / amount_sum,
                count=len(gp_items)
            ))
        return result

    @classmethod
    def get(cls, sess: SqlSession, od_id: int):
        if btime.run_mode in btime.LIVE_MODES:
            op_od = sess.query(InOutOrder).get(od_id)
            if not op_od:
                return op_od
            ex_ods = sess.query(Order).filter(Order.inout_id == od_id).all()
            op_od.enter = next((o for o in ex_ods if o.enter), None)
            op_od.exit = next((o for o in ex_ods if not o.enter), None)
            return op_od
        op_od = cls._open_ods.get(od_id)
        if op_od is not None:
            return op_od
        return next((od for od in cls._his_ods if od.id == od_id), None)

    @classmethod
    def dump_to_db(cls):
        save_ods = cls._his_ods + list(cls._open_ods.values())
        logger.info(f'dump {len(save_ods)} orders to db...')
        insert_orders_to_db(save_ods)

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


def get_order_filters(task_id: int = 0, strategy: str = None, pairs: Union[str, List[str]] = None, status: str = None,
                      close_after: int = None, filters=None):
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
    if pairs:
        if isinstance(pairs, six.string_types):
            where_list.append(InOutOrder.symbol == pairs)
        else:
            where_list.append(InOutOrder.symbol.in_(set(pairs)))
    if filters:
        where_list.extend(filters)
    return where_list


def get_db_orders(task_id: int, strategy: str = None, pairs: Union[str, List[str]] = None, status: str = None,
                  close_after: int = None, filters=None, limit=0, offset=0, order_by=None) -> List[InOutOrder]:
    '''
    此方法仅用于订单管理器获取数据库订单，会自动关联Order到InOutOrder。
    '''
    sess = db.session
    where_list = get_order_filters(task_id, strategy, pairs, status, close_after, filters)
    query = sess.query(InOutOrder).filter(*where_list)
    if order_by:
        query = query.order_by(order_by)
    if offset:
        query = query.offset(offset)
    if limit:
        query = query.limit(limit)
    io_rows: List[InOutOrder] = query.all()
    io_ids = {row.id for row in io_rows}
    ex_filters = [Order.task_id == task_id, Order.inout_id.in_(io_ids)]
    ex_ods = sess.query(Order).filter(*ex_filters).order_by(Order.inout_id).all()
    ex_enters = {od.inout_id: od for od in ex_ods if od.enter}
    ex_exits = {od.inout_id: od for od in ex_ods if not od.enter}
    for row in io_rows:
        row.enter = ex_enters.get(row.id)
        row.exit = ex_exits.get(row.id)
    return io_rows


def insert_orders_to_db(orders: List[InOutOrder]):
    sess = db.session
    for od in orders:
        od.id = None
        sess.add(od)
    sess.flush()
    for od in orders:
        if od.enter:
            od.enter.inout_id = od.id
            sess.add(od.enter)
        if od.exit:
            od.exit.inout_id = od.id
            sess.add(od.exit)
    sess.flush()
    sess.commit()

