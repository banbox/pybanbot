#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : base.py
# Author: anyongjin
# Date  : 2023/3/28
import numpy as np
import pandas as pd

from banbot.exchange.crypto_exchange import *
from banbot.exchange.exchange_utils import *
from banbot.util import btime
from banbot.config.timerange import TimeRange


class PairTFCache:
    def __init__(self, timeframe: str, tf_decs: int):
        self.timeframe = timeframe
        self.tf_secs = tf_decs
        self.last_check = None  # 上次检查更新的时间戳
        self.check_interval = get_check_interval(self.tf_secs)
        self.bar_row = None

    def need_check(self) -> bool:
        return not self.last_check or self.last_check + self.check_interval <= btime.time()


class PairDataFeeder:
    '''
    用于记录交易对的数据更新状态
    '''
    def __init__(self, pair: str, timeframes: List[str], auto_prefire=False):
        tf_pairs = [(tf, timeframe_to_seconds(tf)) for tf in timeframes]
        tf_pairs = sorted(tf_pairs, key=lambda x: x[1])
        for _, tf_secs in tf_pairs[1:]:
            err_msg = f'{_} of {pair} must be integer multiples of the first ({tf_pairs[0][0]})'
            assert tf_secs % tf_pairs[0][1] == 0, err_msg
        self.pair = pair
        self.states = [PairTFCache(tf, tf_sec) for tf, tf_sec in tf_pairs]
        self.warmup_num = 600  # 默认前600个作为预热，可根据策略设置
        self.prefire_secs = 0
        self.min_interval = min(s.check_interval for s in self.states)
        self.auto_prefire = auto_prefire
        self.callback = None
        self._per_details = None
        self._fetch_intv = None

    async def _get_feeds(self) -> Tuple[np.ndarray, float]:
        raise NotImplementedError(f'_get_feeds in {self.__class__.__name__}')

    async def try_fetch(self):
        '''
        对当前交易对检查是否需要从交易所拉取数据
        :return:
        '''
        assert self.callback, '`callback` is not set!'
        self._per_details = None
        self._fetch_intv = None
        # 第一个最小粒度，从api或ws更新行情数据
        if not self.states[0].need_check():
            return
        self._per_details, self._fetch_intv = await self._get_feeds()

    async def try_update(self) -> bool:
        if self._per_details is None:
            return False
        details, fetch_intv = self._per_details, self._fetch_intv
        state = self.states[0]
        prefire = 0.1 if self.auto_prefire else 0
        if fetch_intv < state.tf_secs:
            ohlcvs = [state.bar_row] if state.bar_row else []
            ohlcvs = build_ohlcvc(details, state.tf_secs, prefire, ohlcvs=ohlcvs)
            self.prefire_secs = prefire * state.tf_secs
        else:
            ohlcvs = details
            self.prefire_secs = 0
        state.last_check = btime.time()
        if not state.bar_row or ohlcvs[-1][0] == state.bar_row[0]:
            state.bar_row = ohlcvs[-1]
        elif ohlcvs[0][0] == state.bar_row[0]:
            state.bar_row = ohlcvs[0]
        if ohlcvs[-1][0] > state.bar_row[0] or fetch_intv == state.tf_secs:
            await self._fire_callback(state)
            state.bar_row = ohlcvs[-1]
        else:
            # 当前蜡烛未完成，后续更粗粒度也不会完成，直接退出
            return False
        # 对于第2个及后续的粗粒度。从第一个得到的OHLC更新
        if fetch_intv < state.tf_secs:
            ohlcvs = build_ohlcvc(details, state.tf_secs, prefire)
        else:
            ohlcvs = details
        for state in self.states[1:]:
            cur_ohlcvs = [state.bar_row] if state.bar_row else []
            prefire = 0.05 if self.auto_prefire else 0
            cur_ohlcvs = build_ohlcvc(ohlcvs, state.tf_secs, prefire, ohlcvs=cur_ohlcvs)
            state.last_check = btime.time()
            if state.bar_row and cur_ohlcvs[-1][0] > state.bar_row[0]:
                await self._fire_callback(state)
            state.bar_row = cur_ohlcvs[-1]
        return True

    async def _fire_callback(self, state: PairTFCache):
        bar_end_secs = state.bar_row[0] / 1000 + state.tf_secs
        if btime.run_mode in TRADING_MODES:
            if bar_end_secs + state.tf_secs < btime.time():
                # 当蜡烛的触发时间过于滞后时，输出错误信息
                delay = btime.time() - bar_end_secs
                logger.error('{0}/{1} bar is too late, delay:{2}', self.pair, state.timeframe, delay)
        else:
            btime.cur_timestamp = bar_end_secs
        await self.callback(self.pair, state.timeframe, state.bar_row)


class LocalPairDataFeeder(PairDataFeeder):

    def __init__(self, pair: str, timeframes: List[str], data_dir: str, auto_prefire=False,
                 timerange: Optional[TimeRange] = None):
        import pandas as pd
        super(LocalPairDataFeeder, self).__init__(pair, timeframes, auto_prefire)
        self.data_dir = data_dir
        self.fetch_tfsecs = 0
        self.dataframe: Optional[pd.DataFrame] = None
        self.data_path: Optional[str] = None
        self.row_id = 0
        self.timerange = timerange
        self._check_data()
        self.min_interval = max(self.min_interval, self.fetch_tfsecs)
        self.total_len = 0

    def _check_data(self):
        req_tfsecs = self.states[0].tf_secs
        base_s, quote_s = self.pair.split('/')
        try_list = []
        for tf in NATIVE_TFS:
            cur_secs = timeframe_to_seconds(tf)
            if cur_secs > req_tfsecs:
                break
            if req_tfsecs % cur_secs != 0:
                continue
            try_list.append(tf)
            data_path = os.path.join(self.data_dir, f'{base_s}_{quote_s}-{tf}.feather')
            if not os.path.isfile(data_path):
                continue
            self.data_path = data_path
            self.fetch_tfsecs = cur_secs
            break
        if not self.data_path:
            raise ValueError(f'no data found, try: {try_list} in {self.data_dir}')

    def _load_sml_data(self):
        import pandas as pd
        df = pd.read_feather(self.data_path)
        if df.date.dtype != 'int64':
            df['date'] = df['date'].apply(lambda x: int(x.timestamp() * 1000))
        start_ts, end_ts = df.iloc[0]['date'] // 1000, df.iloc[-1]['date'] // 1000
        logger.info('loading data %s, range: %d-%d', os.path.basename(self.data_path), start_ts, end_ts)
        if self.timerange:
            if self.timerange.startts:
                df = df[df['date'] >= self.timerange.startts * 1000]
            if self.timerange.stopts:
                df = df[df['date'] <= self.timerange.stopts * 1000]
            if len(df):
                start_ts, end_ts = df.iloc[0]['date'] // 1000, df.iloc[-1]['date'] // 1000
                tfrom, tto = btime.to_datestr(start_ts), btime.to_datestr(end_ts)
                logger.info('truncate data from %s(%d) to %s(%d)', tfrom, start_ts, tto, end_ts)
            else:
                tfrom, tto = btime.to_datestr(self.timerange.startts), btime.to_datestr(self.timerange.stopts)
                raise ValueError('no data found after truncate from %s to %s', tfrom, tto)
        self.dataframe = df
        self.total_len = len(df)
        return df

    async def _get_feeds(self):
        if self.dataframe is None:
            self._load_sml_data()
        if self.row_id >= len(self.dataframe):
            raise EOFError()
        req_tfsecs = self.states[0].tf_secs
        if req_tfsecs == self.fetch_tfsecs:
            ret_arr = [self.dataframe.iloc[self.row_id].values.tolist()]
            self.row_id += 1
        else:
            back_len = round(req_tfsecs / self.fetch_tfsecs)
            ret_arr = self.dataframe.iloc[self.row_id: self.row_id + back_len].values.tolist()
            self.row_id += back_len
        return ret_arr, self.fetch_tfsecs

    def load_data(self, timeframe: Optional[str] = None):
        '''
        加载指定timeframe的数据，返回DataFrame。仅用于jupyter-lab中测试。
        :param timeframe:
        :return:
        '''
        if not timeframe:
            timeframe = self.states[0].timeframe
        tf_secs = timeframe_to_seconds(timeframe)
        df = self._load_sml_data()
        if self.fetch_tfsecs == tf_secs:
            return df.reset_index(drop=True)
        if tf_secs % self.fetch_tfsecs > 0:
            raise ValueError(f'unsupport timeframe: {timeframe}, min tf secs: {self.fetch_tfsecs}')
        details = df.values.tolist()
        rows = build_ohlcvc(details, tf_secs)
        return pd.DataFrame(rows[:-1], columns=df.columns.tolist())


class LivePairDataFeader(PairDataFeeder):

    def __init__(self, pair: str, timeframes: List[str], exchange: CryptoExchange, auto_prefire=False):
        super(LivePairDataFeader, self).__init__(pair, timeframes, auto_prefire)
        # 超过3s检查一次的（1分钟及以上维度），通过API获取；否则通过Websocket获取
        self.is_ws = self.min_interval <= 2
        self.exchange = exchange
        self.next_since = None
        self.is_warmed = False
        self.warm_data = None
        self.warm_id = 0
        self.back_rmode = btime.run_mode

    async def _get_feeds(self):
        state = self.states[0]
        if not self.is_warmed:
            # 先请求前900周期数据作为预热
            if self.warm_data is None:
                fetch_num = round(self.states[-1].tf_secs / state.tf_secs) * self.warmup_num
                assert 0 < fetch_num <= 1000, 'fetch warmup num > 1000'
                # 这里请求网络，内部有超时逻辑，故临时切换到实时模式
                with btime.TempRunMode(RunMode.DRY_RUN):
                    self.warm_data = await self.exchange.fetch_ohlcv_plus(self.pair, state.timeframe, limit=fetch_num)
            if self.warm_id >= len(self.warm_data):
                logger.info('warm up complete with %d', self.warmup_num)
                self.is_warmed = True
                self.next_since = self.warm_data[-1][0] + state.tf_secs + 1
                del self.warm_data
                btime.run_mode = self.back_rmode
                btime.cur_timestamp = btime.time()
            else:
                details = [self.warm_data[self.warm_id]]
                self.warm_id += 1
                return details, state.tf_secs
        try:
            if not self.is_ws:
                # 这里不设置limit，如果外部修改了更新间隔，这里能及时输出期间所有的数据，避免出现delay
                details = await self.exchange.fetch_ohlcv(self.pair, '1s', since=self.next_since)
                self.next_since = details[-1][0] + 1
                detail_interval = 1
            else:
                details = await self.exchange.watch_trades(self.pair)
                detail_interval = 0.1
            return details, detail_interval
        except ccxt.NetworkError:
            logger.exception(f'get live data exception: {self.pair}')
            return None, None

