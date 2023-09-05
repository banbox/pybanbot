#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : BaseTipper.py
# Author: anyongjin
# Date  : 2023/3/1
import random
from banbot.strategy.common import *
from banbot.main.wallets import WalletsLocal
from banbot.rpc import Notify, NotifyType  # noqa


class BaseStrategy:
    '''
    策略基类。每个交易对，每种时间帧，每个策略，对应一个策略实例。
    可直接在策略的__init__中存储本策略在此交易对和交易维度上的缓存信息。不会和其他交易对冲突。
    '''
    run_timeframes = []
    '指定运行周期，从里面计算最小符合分数的周期，不提供尝试使用config.json中run_timeframes或Kline.agg_list'
    paintFields: Dict[str, str] = dict()
    '要绘制显示到K线图的字段列表[key, plotType]'
    params = []
    '传入参数'
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
        self.state = dict()  # 仅在当前bar生效的临时缓存
        self._state_fn = dict()
        self.bar_signals: Dict[str, float] = dict()  # 当前bar产生的信号及其价格
        self.calc_num = 0
        self.base_cost = self.config.get('stake_amount', 1000)  # 每笔下单金额基数

    def _calc_state(self, key: str, *args, **kwargs):
        if key not in self.state:
            self.state[key] = self._state_fn[key](*args, **kwargs)
        return self.state[key]

    def on_bar(self, arr: np.ndarray):
        '''
        计算指标。用于后续入场出场信号判断使用。
        :param arr:
        :return:
        '''
        self.state = dict()
        self.bar_signals = dict()
        self.calc_num += 1

    def on_entry(self, arr: np.ndarray) -> Optional[dict]:
        '''
        时间升序，最近的是最后一个
        :param arr:
        :return: InOutOrder的属性。额外：tag,short,legal_cost,cost_rate, stoploss_price, takeprofit_price
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
        :return: Order的属性，额外：tag, short
        '''
        pass

    def custom_exit(self, arr: np.ndarray, od: InOutOrder) -> Optional[dict]:
        return None

    def on_bot_stop(self):
        pass

    def position(self, side: str = None, enter_tag: str = None):
        '''
        获取仓位大小，返回基于基准金额的倍数。
        '''
        symbol = symbol_tf.get().split('_')[2]
        legal_cost = WalletsLocal.obj.position(symbol, self.name, side, enter_tag)
        return legal_cost / self.base_cost

    def init_third_od(self, od: InOutOrder):
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
        from banbot.storage import Overlay
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
