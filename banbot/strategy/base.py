#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : BaseTipper.py
# Author: anyongjin
# Date  : 2023/3/1
from banbot.strategy.common import *
from banbot.rpc import Notify, NotifyType  # noqa
from banbot.storage import ExSymbol
from banbot.main.addons import MarketPrice  # noqa
from banbot.config import UserConfig


_stg_args_info = []

_job_args_info = [
    dict(field='open_long', val_type='bool', title='开多'),
    dict(field='open_short', val_type='bool', title='开空'),
    dict(field='close_long', val_type='bool', title='平多'),
    dict(field='close_short', val_type='bool', title='平空'),
    dict(field='exg_stoploss', val_type='bool', title='止损单'),
    dict(field='long_sl_price', val_type='float', title='做多止损'),
    dict(field='short_sl_price', val_type='float', title='做空止损'),
    dict(field='exg_takeprofit', val_type='bool', title='止盈单'),
    dict(field='long_tp_price', val_type='float', title='做多止盈'),
    dict(field='short_tp_price', val_type='float', title='做空止盈'),
]


def _restore_args(target, args_info: List[dict], cache: dict):
    """从配置中恢复参数到指定对象"""
    import builtins
    for item in args_info:
        name = item.get('field')
        if not name or name not in cache:
            continue
        cache_val = cache[name]
        if cache_val is None:
            continue
        type_name = item.get('val_type')
        if type_name:
            val_type = getattr(builtins, type_name)
            cache_val = val_type(cache_val)
        setattr(target, name, cache_val)


class BaseStrategy:
    '''
    策略基类。每个交易对，每种时间帧，每个策略，对应一个策略实例。
    可直接在策略的__init__中存储本策略在此交易对和交易维度上的缓存信息。不会和其他交易对冲突。
    '''
    run_timeframes = []
    '指定运行周期，从里面计算最小符合分数的周期，不提供尝试使用config.json中run_timeframes或Kline.agg_list'
    params = []
    '传入参数'
    warmup_num = 60
    min_tfscore = 0.8
    nofee_tfscore = 0.6
    max_fee = 0.002
    stake_amount = 0
    '每笔下单金额基数'

    args_info: List[dict] = []
    '此策略通用可参数的描述信息，用于机器人面板中修改通用参数'

    job_args_info: List[dict] = []
    '此策略在当前币种下的可配置参数信息；用于机器人面板修改'

    _restored = False
    '指示此策略的通用参数，是否已从本地缓存的配置中恢复'

    version = 1
    __source__: str = ''

    def __init__(self, config: dict):
        self.config = config
        self.state = dict()
        '仅在当前bar生效的临时缓存'
        self._state_fn = dict()
        self.bar_signals: Dict[str, float] = dict()
        '当前bar产生的信号及其价格'
        self.entrys: List[dict] = []
        '策略创建的入场信号'
        self.exits: List[dict] = []
        '策略创建的出场信号'
        self.calc_num = 0
        self.orders: List[InOutOrder] = []
        '当前打开的订单'
        symbol, timeframe = get_cur_symbol(catch_err=True)
        self.symbol: ExSymbol = symbol
        '当前处理的币种'
        self.timeframe: str = timeframe
        '当前处理的时间周期'
        self.enter_tags: Set[str] = set()
        '已入场订单的标签'
        self.enter_num = 0
        '记录已提交入场订单数量，避免访问数据库过于频繁'
        self.open_long = True
        '是否允许开多'
        self.open_short = True
        '是否允许开空'
        self.close_long = True
        '是否允许平多'
        self.close_short = True
        '是否允许平空'
        self.exg_stoploss = True
        '是否允许交易所止损'
        self.long_sl_price = None
        '做多止损价格'
        self.short_sl_price = None
        '做空止损价格'
        self.exg_takeprofit = True
        '是否允许交易所止盈'
        self.long_tp_price = None
        '做多止盈价格'
        self.short_tp_price = None
        '做空止盈价格'
        self._restore_config()

    def _restore_config(self):
        """从用户面板配置中恢复策略在当前job的状态"""
        config = UserConfig.get()
        pair_jobs: dict = config.get('pair_jobs') or dict()
        cur_key = f'{self.symbol.symbol}_{self.name}'
        stg_config: dict = pair_jobs.get(cur_key)
        if not stg_config:
            return
        _restore_args(self, self.get_job_args_info(), stg_config)
        self._restore_args()

    def apply_args(self, job_args: dict):
        """更新job参数"""
        _restore_args(self, self.get_job_args_info(), job_args)

    @classmethod
    def get_job_args_info(cls):
        return _job_args_info + cls.job_args_info

    @classmethod
    def get_args_info(cls):
        return _stg_args_info + cls.args_info

    @classmethod
    def _restore_args(cls):
        """从用户面板缓存的配置中恢复策略的公共参数"""
        if cls._restored:
            return
        config = UserConfig.get()
        stg_config: dict = config.get('strategy')
        if not stg_config:
            return
        _restore_args(cls, cls.get_args_info(), stg_config)

    def _calc_state(self, key: str, *args, **kwargs):
        if key not in self.state:
            self.state[key] = self._state_fn[key](*args, **kwargs)
        return self.state[key]

    def on_bar(self, arr: np.ndarray):
        '''
        计算指标。用于后续入场出场信号判断使用。
        此方法必须被派生类重写后调用。
        此方法应尽可能减少访问数据库。很多策略同时运行此方法会导致db连接过多。
        :param arr:
        :return:
        '''
        self.entrys = []
        self.exits = []
        self.state = dict()
        self.bar_signals = dict()
        self.calc_num += 1

    def open_order(self, tag: str,
                   short: bool = False,
                   limit: float = None,
                   cost_rate: float = 1.,
                   legal_cost: float = None,
                   leverage: int = None,
                   amount: float = None,
                   stoploss: float = None,
                   takeprofit: float = None,
                   **kwargs):
        '''
        打开一个订单。默认开多。如需开空short=False
        :param tag: 入场信号
        :param short: 是否做空，默认False
        :param limit: 限价单入场价格，指定时订单将作为限价单提交
        :param cost_rate: 开仓倍率、默认按配置1倍。用于计算legal_cost
        :param legal_cost: 花费法币金额。指定时忽略cost_rate
        :param leverage: 杠杆倍数。为空时使用默认配置的杠杆
        :param amount: 入场标的数量，由legal_cost和price计算
        :param stoploss: 止损价格，不为空时在交易所提交一个止损单
        :param takeprofit: 止盈价格，不为空时在交易所提交一个止盈单。
        '''
        if short and not self.open_short or not short and not self.open_long:
            tag = 'short' if short else 'long'
            logger.warning(f'[{self.name}] open {tag} is disabled for {self.symbol}')
            return
        od_args = dict(tag=tag, short=short, **kwargs)
        if leverage:
            od_args['leverage'] = leverage
        if limit:
            if not np.isfinite(limit):
                raise ValueError(f'`limit` should be a valid number, current: {limit}')
            od_args['enter_order_type'] = OrderType.Limit.value
            od_args['enter_price'] = limit
        if amount:
            od_args['enter_amount'] = amount
        else:
            if legal_cost:
                od_args['legal_cost'] = legal_cost
            else:
                od_args['cost_rate'] = cost_rate
                od_args['legal_cost'] = self.custom_cost(od_args)
        fix_sl_price = self.short_sl_price if short else self.long_sl_price
        if self.exg_stoploss and fix_sl_price:
            od_args['stoploss_price'] = fix_sl_price
        elif stoploss:
            if self.exg_stoploss:
                od_args['stoploss_price'] = stoploss
            else:
                logger.warning(f'[{self.name}] stoploss on exchange is disabled for {self.symbol}')
        fix_tp_price = self.short_tp_price if short else self.long_tp_price
        if self.exg_takeprofit and fix_tp_price:
            od_args['takeprofit_price'] = fix_tp_price
        elif takeprofit:
            if self.exg_takeprofit:
                od_args['takeprofit_price'] = takeprofit
            else:
                logger.warning(f'[{self.name}] takeprofit on exchange is disabled for {self.symbol}')
        self.entrys.append(od_args)
        self.enter_num += 1

    def close_orders(self, tag: str,
                     short: bool = False,
                     limit: float = None,
                     exit_rate: float = 1.,
                     amount: float = None,
                     enter_tag: str = None,
                     order_id: int = None,
                     unopen_only: bool = None
                     ):
        '''
        退出若干订单。默认平多。如需平空short=True
        :param tag: 退出原因
        :param short: 是否平空，默认否
        :param limit: 限价单退出价格，指定时订单将作为限价单提交
        :param exit_rate: 退出比率，默认100%即所有订单全部退出
        :param amount: 要退出的标的数量。指定时exit_rate无效
        :param enter_tag: 只退出入场信号为enter_tag的订单
        :param order_id: 只退出指定订单
        :param unopen_only: True时只退出尚未入场的订单
        '''
        if short and not self.close_short or not short and not self.close_long:
            tag = 'short' if short else 'long'
            logger.warning(f'[{self.name}] close {tag} is disabled for {self.symbol}')
            return
        exit_args = dict(tag=tag, short=short)
        if amount:
            exit_args['amount'] = amount
        elif exit_rate < 1:
            exit_args['exit_rate'] = exit_rate
        if unopen_only is not None:
            exit_args['unopen_only'] = unopen_only
        if limit:
            if not np.isfinite(limit):
                raise ValueError(f'`limit` should be a valid number, current: {limit}')
            exit_args['order_type'] = OrderType.Limit.value
            exit_args['price'] = limit
        if enter_tag:
            exit_args['enter_tag'] = enter_tag
        if order_id:
            exit_args['order_id'] = order_id
        self.exits.append(exit_args)

    def custom_exit(self, arr: np.ndarray, od: InOutOrder) -> Optional[bool]:
        return None

    def check_custom_exits(self, pair_arr: np.ndarray) -> List[Tuple[InOutOrder, str]]:
        # 调用策略的自定义退出判断
        edit_ods = []
        skip_takeprofit = 0
        skip_stoploss = 0
        for od in self.orders:
            if not od.can_close():
                continue
            sl_price = od.get_info('stoploss_price')
            tp_price = od.get_info('takeprofit_price')
            sigout = self.custom_exit(pair_arr, od)
            if not sigout:
                # 检查是否需要修改条件单
                fix_sl_price = self.short_sl_price if od.short else self.long_sl_price
                new_sl_price = fix_sl_price or od.get_info('stoploss_price')
                fix_tp_price = self.short_tp_price if od.short else self.long_tp_price
                new_tp_price = fix_tp_price or od.get_info('takeprofit_price')
                if new_sl_price != sl_price:
                    if self.exg_stoploss:
                        edit_ods.append((od, 'stoploss_'))
                        od.set_info(stoploss_price=new_sl_price)
                    else:
                        skip_stoploss += 1
                        od.set_info(stoploss_price=None)
                if new_tp_price != tp_price:
                    if self.exg_takeprofit:
                        edit_ods.append((od, 'takeprofit_'))
                        od.set_info(takeprofit_price=new_tp_price)
                    else:
                        skip_takeprofit += 1
                        od.set_info(takeprofit_price=None)
        if skip_stoploss:
            logger.warning(f'[{self.name}] {self.symbol} stoploss on exchange is disabled, {skip_stoploss} orders')
        if skip_takeprofit:
            logger.warning(f'[{self.name}] {self.symbol} takeprofit on exchange is disabled, {skip_takeprofit} orders')
        return edit_ods

    def get_trig_ods(self):
        """根据当前job的止损价止盈价，检查是否有订单需要下止损止盈单"""
        edit_ods = []
        for od in self.orders:
            new_sl_price = (self.short_sl_price if od.short else self.long_sl_price) if self.exg_stoploss else None
            new_tp_price = (self.short_tp_price if od.short else self.long_tp_price) if self.exg_takeprofit else None
            sl_price = od.get_info('stoploss_price')
            tp_price = od.get_info('takeprofit_price')
            if new_sl_price and new_sl_price != sl_price:
                edit_ods.append((od, 'stoploss_'))
                od.set_info(stoploss_price=new_sl_price)
            else:
                logger.info(f'new stoploss same, skip: {new_sl_price}')
            if new_tp_price and new_tp_price != tp_price:
                edit_ods.append((od, 'takeprofit_'))
                od.set_info(takeprofit_price=new_tp_price)
        logger.info(f'get_trig_ods test {len(self.orders)} orders')
        return edit_ods

    def on_bot_stop(self):
        pass

    def position(self, side: str = None, enter_tag: str = None):
        '''
        获取仓位大小，返回基于基准金额的倍数。
        '''
        # 从订单成本计算仓位
        open_ods = self.orders
        if enter_tag:
            open_ods = [od for od in open_ods if od.enter_tag == enter_tag]
        if side and side != 'both':
            is_short = side == 'short'
            open_ods = [od for od in open_ods if od.short == is_short]
        if open_ods:
            total_cost = sum(od.enter_cost for od in open_ods)
        else:
            total_cost = 0
        return total_cost / self.get_stake_amount()

    def init_third_od(self, od: InOutOrder):
        pass

    @classmethod
    def send_notify(cls, msg: str, with_pair=True):
        '''
        发送策略消息通知
        '''
        if BotGlobal.is_warmup:
            return
        if with_pair:
            exs, tf = get_cur_symbol()
            msg = f'{exs.symbol} {tf} {msg}'
        logger.info(f'send strategy_msg: {msg}')
        Notify.send(type=NotifyType.STRATEGY_MSG, msg=msg)

    @property
    def name(self):
        return self.__class__.__name__

    @classmethod
    def get_stake_amount(cls):
        if not cls.stake_amount:
            cls.stake_amount = AppConfig.get().get('stake_amount', 1000)
        return cls.stake_amount

    @classmethod
    def custom_cost(cls, sigin: dict) -> float:
        '''
        返回自定义的此次订单花费金额（基于法定币，如USDT、RMB）
        :param sigin:
        :return:
        '''
        rate = sigin.get('cost_rate', 1)
        return cls.get_stake_amount() * rate

    @classmethod
    def pick_timeframe(cls, exg_name: str, symbol: str, tfscores: List[Tuple[str, float]]) -> Optional[str]:
        if not tfscores:
            return None
        from banbot.exchange.crypto_exchange import get_exchange
        fee_rate = get_exchange(exg_name).calc_fee(symbol, OrderType.Market.value)['rate']
        if fee_rate > cls.max_fee:
            return
        min_score = cls.min_tfscore if fee_rate else cls.nofee_tfscore
        for tf, score in tfscores:
            if score >= min_score:
                return tf
