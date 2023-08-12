#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : BaseTipper.py
# Author: anyongjin
# Date  : 2023/3/1
import random
import sys
from banbot.storage import Overlay
from banbot.strategy.common import *


class BaseStrategy:
    '''
    策略基类。每个交易对，每种时间帧，每个策略，对应一个策略实例。
    可直接在策略的__init__中存储本策略在此交易对和交易维度上的缓存信息。不会和其他交易对冲突。
    '''
    run_timeframes = []
    '指定运行周期，从里面计算最小符合分数的周期，不提供尝试使用config.json中run_timeframes或Kline.agg_list'
    paintFields: Dict[str, str] = dict()
    '要绘制显示到K线图的字段列表'
    warmup_num = 600
    min_tfscore = 0.8
    nofee_tfscore = 0.6
    max_fee = 0.002
    stop_loss = 0.05
    skip_exit_on_enter = True
    skip_enter_on_exit = True
    version = 1

    def __init__(self, config: dict):
        self.config = config
        self._cross_up: Dict[str, List[int]] = dict()  # 记录信号向上交叉，由负变正
        self._cross_down: Dict[str, List[int]] = dict()  # 记录信号向下交叉，由正变负
        self._cross_last: Dict[str, float] = dict()  # 记录信号交叉点
        self.state = dict()  # 仅在当前bar生效的临时缓存
        self._state_fn = dict()
        self.bar_signals: Dict[str, float] = dict()  # 当前bar产生的信号及其价格
        self.orders: List[InOutOrder] = []  # 打开的订单，下单未成交的在内，离场未成交的不在内
        self.calc_num = 0
        self.base_cost = self.config.get('stake_amount', 1000)  # 每笔下单金额基数

    def _calc_state(self, key: str, *args, **kwargs):
        if key not in self.state:
            self.state[key] = self._state_fn[key](*args, **kwargs)
        return self.state[key]

    def _log_cross(self, tag: str, value: float):
        '''
        记录指定标签的每一次交叉。
        '''
        if not np.isfinite(value) or value == 0:
            return
        prev_val = self._cross_last.get(tag)
        self._cross_last[tag] = value
        if prev_val is None:
            return
        if prev_val * value < 0:
            cross_dic = self._cross_up if value > 0 else self._cross_down
            if tag not in cross_dic:
                cross_dic[tag] = []
            cross_dic[tag].append(bar_num.get())
            if len(cross_dic[tag]) > 200:
                cross_dic[tag] = cross_dic[tag][-100:]

    def _cross_dist(self, tag: str, dirt=0):
        '''
        获取指定标签，与上次交叉的距离。如果尚未发生交叉，返回一个极大值
        dirt：0表示不限制；1上穿；-1下穿
        '''
        up_dist, down_dist = sys.maxsize, sys.maxsize
        if dirt >= 0:
            pos = self._cross_up.get(tag)
            if pos:
                up_dist = bar_num.get() - pos[-1]
        if dirt <= 0:
            pos = self._cross_down.get(tag)
            if pos:
                down_dist = bar_num.get() - pos[-1]
        return min(up_dist, down_dist)

    def on_bar(self, arr: np.ndarray):
        '''
        计算指标。用于后续入场出场信号判断使用。
        :param arr:
        :return:
        '''
        self.state = dict()
        self.bar_signals = dict()
        self.calc_num += 1

    def _update_inds(self, arr: np.ndarray, *args):
        cur_close, cur_row = arr[-1, ccol], arr[-1]
        for ind in args:
            dim_sub = arr.ndim - ind.input_dim
            if dim_sub == 1:
                in_val = cur_row
            elif dim_sub == 0:
                in_val = arr
            elif dim_sub == 2:
                in_val = cur_close
            else:
                raise ValueError(f'unsupport dim sub: {dim_sub} from {type(ind)}')
            ind(in_val)

    def on_entry(self, arr: np.ndarray) -> Optional[dict]:
        '''
        时间升序，最近的是最后一个
        :param arr:
        :return: InOutOrder的属性。额外：tag,short,legal_cost,cost_rate
        '''
        pass

    def custom_cost(self, sigin: dict) -> float:
        '''
        返回自定义的此次订单花费金额（基于法定币，如USDT、RMB）
        :param sigin:
        :return:
        '''
        rate = sigin.get('cost_rate', 1)
        return self.base_cost * rate

    def on_exit(self, arr: np.ndarray) -> Optional[dict]:
        '''
        检查是否有退出信号。
        :return: Order的属性，额外：tag, for_short
        '''
        pass

    def custom_exit(self, arr: np.ndarray, od: InOutOrder) -> Optional[dict]:
        return None

    def update_orders(self, enters: List[InOutOrder], exits: List[InOutOrder]):
        cur_name = self.name
        cur_exts = [od for od in exits if od.strategy == cur_name]
        for od in cur_exts:
            if od.strategy != cur_name:
                continue
            try:
                idx = self.orders.index(od)
                self.orders.pop(idx)
            except ValueError:
                continue
        self.orders.extend([od for od in enters if od.strategy == cur_name])

    def on_bot_stop(self):
        pass

    @property
    def name(self):
        return self.__class__.__name__

    @classmethod
    def pick_timeframe(cls, exg_name: str, symbol: str, tfscores: List[Tuple[str, float]]) -> Optional[str]:
        if not tfscores:
            return None
        from banbot.exchange.crypto_exchange import get_exchange
        fee_rate = get_exchange(exg_name).calc_fee(symbol, 'market')['rate']
        if fee_rate > cls.max_fee:
            return
        min_score = cls.min_tfscore if fee_rate else cls.nofee_tfscore
        for tf, score in tfscores:
            if score >= min_score:
                return tf

    @classmethod
    def paint(cls, name: str, points: List[OlayPoint], paneId: str = 'candle_pane', mode: str = 'normal',
              lock: bool = True, groupId='ban_stg', styles: dict = None, visible: bool = True,
              zLevel: int = 1000, extendData=None, ):
        '''
        绘制内容到K线图上
        :param name: 覆盖物名称: arc, circle, line, polygon, rect, text, rectText, segment, simpleTag, ...
        :param points: 定位覆盖物的点
        :param paneId: 所属子图，默认candle_pane是主K线图
        :param mode: `normal`，`weak_magnet`，`strong_magnet`
        :param lock: 是否锁定，不触发事件，默认锁定
        :param groupId: 分组ID
        :param styles: 样式配置字典，参考：https://klinecharts.com/guide/styles.html 中 "overlay:"，会自动包装name属性
        :param visible: 是否可见，默认true
        :param zLevel: 显示层级，越大显示越上面
        :param extendData: 扩展数据
        '''
        exs, timeframe = get_cur_symbol()
        if styles:
            styles = dict(name=styles)
        olay_data = dict(
            id=f'{cls.__name__}_{cls.version}_{name}_{bar_num.get()}_{random.randrange(100, 999)}',
            name=name,
            points=[p.dict() for p in points],
            paneId=paneId,
            mode=mode,
            lock=lock,
            groupId=groupId,
            styles=styles,
            visible=visible,
            zLevel=zLevel,
            extendData=extendData
        )
        Overlay.create(0, exs.id, timeframe, olay_data)

    @classmethod
    def draw_circle(cls, value: float, radius: int, paneId: str = 'candle_pane', mode: str = 'normal',
              lock: bool = True, groupId='ban_stg', styles: dict = None, visible: bool = True,
              zLevel: int = 1000, extendData=None, ):
        bar_start, bar_stop = bar_time.get()
        end_ms = bar_start + radius * (bar_stop - bar_start)
        points = [OlayPoint(bar_start, value), OlayPoint(end_ms, value)]
        cls.paint('circle', points, paneId, mode, lock, groupId, styles, visible, zLevel, extendData)
