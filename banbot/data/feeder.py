#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : base.py
# Author: anyongjin
# Date  : 2023/3/28

from banbot.exchange.crypto_exchange import *
from banbot.data.tools import *
from banbot.data.wacther import *


class DataFeeder(Watcher):
    '''
    每个Feeder对应一个交易对。可包含多个时间维度。
    支持动态添加时间维度。
    回测模式：根据Feeder的下次更新时间，按顺序调用执行回调。
    实盘模式：订阅此交易对时间周期的新数据，被唤起时执行回调。
    支持预热数据。每个策略+交易对全程单独预热，不可交叉预热，避免btime被污染。
    LiveFeeder新交易对和新周期都需要预热；HistFeeder仅新周期需要预热
    '''
    def __init__(self, pair: str, warm_secs: int, timeframes: List[str], callback: Callable, auto_prefire=False):
        super(DataFeeder, self).__init__(pair, callback)
        self.warm_secs = warm_secs
        self.states: List[PairTFCache] = []
        self.auto_prefire = auto_prefire
        self.sub_tflist(*timeframes)

    def sub_tflist(self, *timeframes):
        '''
        订阅新的时间周期。此方法不做预热。
        '''
        added_tfs = {st.timeframe for st in self.states}
        timeframes = [tf for tf in timeframes if tf not in added_tfs]
        if not timeframes:
            return
        # 检查是否有效并添加到states
        tf_pairs = [(tf, tf_to_secs(tf)) for tf in timeframes]
        tf_pairs = sorted(tf_pairs, key=lambda x: x[1])
        new_states = [PairTFCache(tf, tf_sec) for tf, tf_sec in tf_pairs]
        min_tf = new_states[0]
        check_list = self.states
        if self.states and self.states[0].tf_secs < min_tf.tf_secs:
            min_tf = self.states[0]
            check_list = new_states
        for sta in check_list:
            err_msg = f'{sta.timeframe} of {self.pair} must be integer multiples of the first ({min_tf.timeframe})'
            assert sta.tf_secs % min_tf.tf_secs == 0, err_msg
        self.states.extend(new_states)
        self.states = sorted(self.states, key=lambda x: x.tf_secs)
        return timeframes

    async def warm_tfs(self, warm_secs: int, *timeframes) -> Optional[int]:
        '''
        预热周期数据。当动态添加周期到已有的HistDataFeeder时，应调用此方法预热数据。
        LiveFeeder在初始化时也应当调用此函数
        :retrun 返回结束的时间戳（即下一个bar开始时间戳）
        '''
        raise NotImplementedError

    def on_new_data(self, details: List, fetch_intv: int) -> bool:
        # 获取从上次间隔至今期间，更新的子序列
        state = self.states[0]
        prefire = 0.1 if self.auto_prefire else 0
        if fetch_intv < state.tf_secs:
            ohlcvs = [state.wait_bar] if state.wait_bar else []
            ohlcvs, last_finish = build_ohlcvc(details, state.tf_secs, prefire, ohlcvs=ohlcvs)
        elif fetch_intv == state.tf_secs:
            ohlcvs, last_finish = details, True
        else:
            raise RuntimeError(f'fetch interval {fetch_intv} should <= min_tf: {state.tf_secs}')
        # 子序列周期维度<=当前维度。当收到spider发送的数据时，这里可能是3个或更多ohlcvs
        min_finished = self._on_state_ohlcv(state, ohlcvs, last_finish)
        if len(self.states) > 1:
            # 对于第2个及后续的粗粒度。从第一个得到的OHLC更新
            # 即使第一个没有完成，也要更新更粗周期维度，否则会造成数据丢失
            if fetch_intv < state.tf_secs:
                # 这里应该保留最后未完成的数据
                ohlcvs, _ = build_ohlcvc(details, state.tf_secs, prefire)
            else:
                ohlcvs = details
            for state in self.states[1:]:
                cur_ohlcvs = [state.wait_bar] if state.wait_bar else []
                prefire = 0.05 if self.auto_prefire else 0
                cur_ohlcvs, last_finish = build_ohlcvc(ohlcvs, state.tf_secs, prefire, ohlcvs=cur_ohlcvs)
                self._on_state_ohlcv(state, cur_ohlcvs, last_finish)
        return bool(min_finished)


class HistDataFeeder(DataFeeder):
    '''
    历史数据反馈器。是文件反馈器和数据库反馈器的基类。
    可通过next_at属性获取下个bar的达到时间。按到达时间对所有Feeder排序。
    然后针对第一个Feeder，直接调用对象即可触发数据回调
    '''
    def __init__(self, pair: str, warm_secs: int, timeframes: List[str], callback: Callable, auto_prefire=False,
                 timerange: Optional[TimeRange] = None):
        super(HistDataFeeder, self).__init__(pair, warm_secs, timeframes, callback, auto_prefire)
        if not timerange:
            timerange = TimeRange(btime.time(), btime.time())
        elif not timerange.stopts:
            timerange.stopts = btime.time()
        if self.warm_secs:
            timerange = TimeRange(timerange.startts - self.warm_secs, timerange.stopts)
        self.timerange = timerange
        self.total_len = 0
        self.row_id = 0
        self._next_arr = []  # 下个bar的数据

    @property
    def next_at(self):
        if not self._next_arr:
            return sys.maxsize
        return self._next_arr[-1][0]

    def _set_next(self):
        pass

    def __call__(self, *args, **kwargs):
        ret_arr = self._next_arr
        self.on_new_data(ret_arr, self.states[0].tf_secs)
        self._set_next()
        return ret_arr


class FileDataFeeder(HistDataFeeder):

    def __init__(self, pair: str, warm_secs: int, timeframes: List[str], callback: Callable, data_dir: str,
                 auto_prefire=False, timerange: Optional[TimeRange] = None):
        super(FileDataFeeder, self).__init__(pair, warm_secs, timeframes, callback, auto_prefire, timerange)
        self.data_dir = data_dir
        self.min_tfsecs = self.states[0].tf_secs
        self.data_path, self.fetch_tfsecs = parse_data_path(self.data_dir, self.pair, self.min_tfsecs)
        self.dataframe = load_file_range(self.data_path, self.timerange)
        self.total_len = len(self.dataframe)
        self._set_next()

    def _set_next(self):
        if self.row_id >= self.total_len:
            self._next_arr = []
            return
        if self.min_tfsecs == self.fetch_tfsecs:
            ret_arr = [self.dataframe.iloc[self.row_id].values.tolist()]
            self.row_id += 1
        else:
            back_len = round(self.min_tfsecs / self.fetch_tfsecs)
            ret_arr = self.dataframe.iloc[self.row_id: self.row_id + back_len].values.tolist()
            self.row_id += back_len
        self._next_arr = ret_arr

    async def warm_tfs(self, warm_secs: int, *timeframes) -> Optional[int]:
        tr = TimeRange(btime.time() - warm_secs, btime.time() + 0.1)
        ohlcv_arr: List[Tuple] = load_file_range(self.data_path, tr).values.tolist()
        max_end_ms = 0
        if not ohlcv_arr:
            return
        for tf in timeframes:
            tf_secs = tf_to_secs(tf)
            if tf_secs > self.fetch_tfsecs:
                ohlcv_arr, _ = build_ohlcvc(ohlcv_arr, tf_secs)
            self._fire_callback(ohlcv_arr, tf, tf_secs)
            max_end_ms = max(max_end_ms, ohlcv_arr[-1][0] + tf_secs * 1000)
        return max_end_ms


class DBDataFeeder(HistDataFeeder):

    def __init__(self, pair: str, warm_secs: int, timeframes: List[str], callback: Callable, auto_prefire=False,
                 timerange: Optional[TimeRange] = None):
        super(DBDataFeeder, self).__init__(pair, warm_secs, timeframes, callback, auto_prefire, timerange)
        self._offset_ts = self.timerange.startts * 1000
        self.exg_name = AppConfig.get()['exchange']['name']
        self._batch_size = 300
        self._row_id = 0
        self._cache_arr = []
        self._calc_total()
        self._set_next()

    def _calc_total(self):
        from banbot.storage import KLine
        start_ts, stop_ts = KLine.query_range(self.exg_name, self.pair)
        if start_ts and stop_ts:
            vstart = max(start_ts, self.timerange.startts * 1000)
            vend = min(stop_ts, self.timerange.stopts * 1000)
            if vstart < vend:
                div_m = self.states[0].tf_secs * 1000
                self.total_len = (vend - vstart) // div_m + 1

    def _set_next(self):
        if self._row_id >= len(self._cache_arr):
            # 缓存结束，重新读取数据库
            if self.row_id >= self.total_len:
                self._next_arr = []
                return
            from banbot.storage import KLine
            end = self.timerange.stopts * 1000
            min_tf = self.states[0].timeframe
            self._cache_arr = KLine.query(self.exg_name, self.pair, min_tf, self._offset_ts, end, self._batch_size)
            self._row_id = 0
            if not self._cache_arr:
                self.total_len = self.row_id
                self._next_arr = []
                return
            first_ts = self._cache_arr[0][0]
            delay_num = int(first_ts - self._offset_ts) // self.states[0].tf_secs // 1000
            if delay_num > 1:
                logger.error(f'candles start from {btime.to_datestr(first_ts)}, lack: {delay_num}')
            self._offset_ts = self._cache_arr[-1][0] + 1
        self._next_arr = [self._cache_arr[self._row_id]]
        self._row_id += 1
        self.row_id += 1

    async def warm_tfs(self, warm_secs: int, *timeframes) -> Optional[int]:
        end_ms = int(btime.time() * 1000) + 1
        start_ms = end_ms - warm_secs * 1000
        max_end_ms = 0
        exchange = get_exchange(self.exg_name)
        for tf in timeframes:
            tf_secs = tf_to_secs(tf)
            ohlcv_arr = await auto_fetch_ohlcv(exchange, self.pair, tf, start_ms, end_ms)
            if not ohlcv_arr:
                logger.warning(f"warm {self.pair} {tf} fail, no data...")
                continue
            self._fire_callback(ohlcv_arr, tf, tf_secs)
            max_end_ms = max(max_end_ms, ohlcv_arr[-1][0] + tf_secs * 1000)
        return max_end_ms


class LiveDataFeader(DBDataFeeder):
    '''
    每个Feeder对应一个交易对。可包含多个时间维度。
    支持动态添加时间维度。
    实盘模式：订阅此交易对时间周期的新数据，被唤起时执行回调。
    支持返回预热数据。每个策略+交易对全程单独预热，不可交叉预热，避免btime被污染。

    通过redis检查此交易对是否已在监听刷新，如没有则发消息给爬虫监听。
    '''

    def __init__(self, pair: str, warm_secs: int, timeframes: List[str], callback: Callable):
        # 实盘数据的auto_prefire在爬虫端进行。
        super(LiveDataFeader, self).__init__(pair, warm_secs, timeframes, callback)



