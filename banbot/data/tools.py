#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : tools.py
# Author: anyongjin
# Date  : 2023/2/28
import asyncio
import datetime
import os

import ccxt
import pandas as pd
from banbot.util.common import logger
from banbot.config.timerange import TimeRange
from banbot.config.consts import *
from banbot.exchange.exchange_utils import *
from typing import Tuple, List


def trades_to_ohlcv(trades: List[dict]) -> List[Tuple[int, float, float, float, float, float, int]]:
    result = []
    for trade in trades:
        price = trade['price']
        result.append((trade['timestamp'], price, price, price, price, trade['amount'], 1))
    return result


def build_ohlcvc(details: List[Tuple], tf_secs: int, prefire: float = 0., since=None, ohlcvs=None):
    '''
    从交易或子OHLC数组中，构建或更新更粗粒度OHLC数组。
    :param details: 子OHLC列表。[[t,o,h,l,c,v,cnt], ...]
    :param tf_secs: 指定要构建的时间粒度，单位：秒
    :param prefire: 是否提前触发构建完成；用于在特定信号时早于其他交易者提早发出信号
    :param since:
    :param ohlcvs: 已有的待更新数组
    :return:
    '''
    ms = tf_secs * 1000
    off_ms = round(ms * prefire)
    ohlcvs = ohlcvs or []
    (timestamp, copen, high, low, close, volume, count) = (0, 1, 2, 3, 4, 5, 6)
    raw_ts = []
    for detail in details:
        row = list(detail)
        # 按给定粒度重新格式化时间戳
        raw_ts.append(row[timestamp])
        row[timestamp] = (row[timestamp] + off_ms) // ms * ms
        if since and row[timestamp] < since:
            continue
        if not ohlcvs or (row[timestamp] >= ohlcvs[-1][timestamp] + ms):
            # moved to a new timeframe -> create a new candle from opening trade
            ohlcvs.append(row)
        else:
            prow = ohlcvs[-1]
            # still processing the same timeframe -> update opening trade
            prow[high] = max(prow[high], row[high])
            prow[low] = min(prow[low], row[low])
            prow[close] = row[close]
            prow[volume] += row[volume]
            if len(row) > count:
                prow[count] += row[count]
    last_finish = False
    if len(raw_ts) >= 2:
        # 至少有2个，判断最后一个bar是否结束：假定details中每个bar间隔相等，最后一个bar+间隔属于下一个规划区间，则认为最后一个bar结束
        ts_interval = raw_ts[-1] - raw_ts[-2]
        finish_ts = (raw_ts[-1] + ts_interval + off_ms) // ms * ms
        last_finish = finish_ts > ohlcvs[-1][0]
    return ohlcvs, last_finish


def convert_bnb_klines_datas(data_dir: str, timeframe: str, group_num: int = 7):
    '''
    将币安下载的K线数据(csv，每个文件是一天的数据)转为feather格式，只保留关键列。
    时间戳，开盘价，最高价，最低价，收盘价，成交量，笔数，主动买入量
    :param data_dir:
    :param timeframe:
    :param group_num:
    :return:
    '''
    import pandas as pd
    names = os.listdir(data_dir)
    fea_tf = f'-{timeframe}-'
    csv_names = [name for name in names if name.endswith('.csv') and name.find(fea_tf) > 0]
    csv_names = sorted(csv_names)
    # tgt_names = [name for name in names if name.endswith('.feather') and name.find(fea_tf) > 0]
    data_list = []
    pair, date_val, start_date = None, None, None

    def merge_and_save():
        nonlocal data_list, start_date
        merge_df = pd.concat(data_list, ignore_index=True, sort=False)
        start_text, end_text = start_date.strftime("%Y-%m-%d"), date_val.strftime("%Y-%m-%d")
        out_name = f'{pair}{fea_tf}{start_text}-{end_text}.feather'
        merge_df.to_feather(os.path.join(data_dir, out_name), compression='lz4')
        logger.info('saved: %s', out_name)
        data_list = []
        start_date = None
    for i, name in enumerate(csv_names):
        cpair, cdate = (os.path.splitext(name)[0]).split(fea_tf)
        cdata_val = datetime.datetime.strptime(cdate, '%Y-%m-%d').date()
        if date_val and ((cdata_val - date_val).days != 1 or pair != cpair):
            # 和上一个时间不连续
            logger.warning('{0} date not continus : {1} -- {2}', cpair, date_val, cdata_val)
            merge_and_save()
        elif len(data_list) >= group_num:
            merge_and_save()
        usecols = ['date', 'open', 'high', 'low', 'close', 'volume', 'count', 'long_vol']
        col_ids = [0, 1, 2, 3, 4, 5, 8, 9]
        max_col_num = max(col_ids) + 1
        df = pd.read_csv(os.path.join(data_dir, name), header=None, usecols=list(range(max_col_num)))
        df.columns = [f'col{i}' if i not in col_ids else usecols[col_ids.index(i)] for i in range(max_col_num)]
        df = df[usecols]
        df['date'] = pd.to_datetime(df['date'], utc=True, unit='ms')
        data_list.append(df)
        pair, date_val = cpair, cdata_val
        if not start_date:
            start_date = cdata_val
    merge_and_save()


def parse_data_path(data_dir: str, pair: str, tf_secs: int) -> Tuple[str, int]:
    '''
    从给定的目录中，尝试查找给定交易对，符合时间维度的文件
    '''
    base_s, quote_s = pair.split('/')
    try_list = []
    for tf in NATIVE_TFS:
        cur_secs = tf_to_secs(tf)
        if cur_secs > tf_secs:
            break
        if tf_secs % cur_secs != 0:
            continue
        try_list.append(tf)
        data_path = os.path.join(data_dir, f'{base_s}_{quote_s}-{tf}.feather')
        if not os.path.isfile(data_path):
            continue
        return data_path, cur_secs
    raise FileNotFoundError(f'no data found, try: {try_list} in {data_dir}')


def load_file_range(data_path: str, tr: Optional[TimeRange]) -> pd.DataFrame:
    '''
    加载数据文件，必须是feather，按给定的范围截取并返回DataFrame
    '''
    df = pd.read_feather(data_path)
    if df.date.dtype != 'int64':
        df['date'] = df['date'].apply(lambda x: int(x.timestamp() * 1000))
    start_ts, end_ts = df.iloc[0]['date'] // 1000, df.iloc[-1]['date'] // 1000
    logger.info('loading data %s, range: %d-%d', os.path.basename(data_path), start_ts, end_ts)
    if tr:
        if tr.startts:
            df = df[df['date'] >= tr.startts * 1000]
        if tr.stopts:
            df = df[df['date'] <= tr.stopts * 1000]
        if len(df):
            start_ts, end_ts = df.iloc[0]['date'] // 1000, df.iloc[-1]['date'] // 1000
            tfrom, tto = btime.to_datestr(start_ts), btime.to_datestr(end_ts)
            logger.info('truncate data from %s(%d) to %s(%d)', tfrom, start_ts, tto, end_ts)
        else:
            tfrom, tto = btime.to_datestr(tr.startts), btime.to_datestr(tr.stopts)
            raise FileNotFoundError('no data found after truncate from %s to %s', tfrom, tto)
    return df


def load_pair_file_range(data_dir: str, pair: str, timeframe: str, ts_from: float, ts_to: float) -> pd.DataFrame:
    '''
    从给定目录下，查找交易对的指定时间维度数据，按时间段截取返回.
    '''
    tf_secs = tf_to_secs(timeframe)
    trange = TimeRange.parse_timerange(f'{ts_from}-{ts_to}')
    try:
        data_path, fetch_tfsecs = parse_data_path(data_dir, pair, tf_secs)
        df = load_file_range(data_path, trange)
    except FileNotFoundError:
        return pd.DataFrame()
    if fetch_tfsecs == tf_secs:
        return df.reset_index(drop=True)
    if tf_secs % fetch_tfsecs > 0:
        raise ValueError(f'unsupport timeframe: {timeframe}, min tf secs: {fetch_tfsecs}')
    details = df.values.tolist()
    rows, last_finish = build_ohlcvc(details, tf_secs)
    if not last_finish:
        rows = rows[:-1]
    return pd.DataFrame(rows, columns=df.columns.tolist())


def load_data_range_from_bt(btres: dict):
    '''
    加载指定timeframe的数据，返回DataFrame。仅用于jupyter-lab中测试。
    :param btres: 回测的结果
    :return:
    '''
    return load_pair_file_range(
        btres['data_dir'],
        btres['pair'],
        btres['timeframe'],
        btres['ts_from'],
        btres['ts_to'],
    )


async def fetch_api_ohlcv(exchange, pair: str, timeframe: str, start_ms: int, end_ms: int, show_info: bool = True):
    '''
    按给定时间段下载交易对的K线数据。
    :param exchange:
    :param pair:
    :param timeframe:
    :param start_ms: 毫秒时间戳（含）
    :param end_ms: 毫秒时间戳（不含）
    :param show_info: 是否显示进度信息
    :return:
    '''
    from tqdm import tqdm
    tf_msecs = tf_to_secs(timeframe) * 1000
    assert start_ms > 1000000000000, '`start_ts` must be milli seconds'
    fetch_num = (end_ms - start_ms) // tf_msecs
    batch_size = min(1000, fetch_num + 5)
    req_times = fetch_num / batch_size
    if show_info:
        start_text, end_text = btime.to_datestr(start_ms), btime.to_datestr(end_ms)
        logger.info(f'fetch ohlcv {pair}/{timeframe} {start_text} - {end_text}')
        if req_times < 3:
            show_info = False
    since, total_tr = round(start_ms), end_ms - start_ms
    result = []
    pbar = tqdm(total=100, unit='%') if show_info else None
    while since + tf_msecs <= end_ms:
        data = await exchange.fetch_ohlcv(pair, timeframe, since=since, limit=batch_size)
        if not len(data):
            break
        result.extend(data)
        cur_end = round(data[-1][0])
        if show_info:
            pbar.update(round((cur_end - since) * 100 / total_tr))
        since = cur_end + 1
    if show_info:
        pbar.close()
    end_pos = len(result) - 1
    while end_pos >= 0 and result[end_pos][0] >= end_ms:
        end_pos -= 1
    return result[:end_pos + 1]


async def download_to_db(exchange, pair: str, timeframe: str, start_ms: int, end_ms: int, check_exist=True,
                         allow_lack: float = 0.):
    '''
    从交易所下载K线数据到数据库。
    跳过已有部分，同时保持数据连续
    '''
    if timeframe not in {'1m', '1h'}:
        raise RuntimeError(f'can only download kline: 1m or 1h, current: {timeframe}')
    exg_name = exchange.name
    from banbot.storage import KLine, ExSymbol, KHole
    start_ms = await ExSymbol.get_valid_start(exg_name, pair, start_ms)
    start_ms, end_ms = KHole.get_down_range(exg_name, pair, timeframe, start_ms, end_ms)
    if not start_ms:
        return
    if check_exist:
        old_start, old_end = KLine.query_range(exg_name, pair, timeframe)
        if old_start and old_end > old_start:
            if start_ms < old_start:
                # 直接抓取start_ms - old_start的数据，避免出现空洞；可能end_ms>old_end，还需要下载后续数据
                cur_end = round(old_start)
                predata = await fetch_api_ohlcv(exchange, pair, timeframe, start_ms, cur_end)
                KLine.insert(exg_name, pair, timeframe, predata)
                KLine.log_candles_conts(exg_name, pair, timeframe, start_ms, cur_end, predata)
                start_ms = old_end
            elif end_ms > old_end:
                if (end_ms - old_end) / (end_ms - start_ms) <= allow_lack:
                    # 最新部分缺失的较少，不再请求交易所，节省时间
                    return
                # 直接抓取old_end - end_ms的数据，避免出现空洞；前面没有需要再下次的数据了。可直接退出
                predata = await fetch_api_ohlcv(exchange, pair, timeframe, old_end, end_ms)
                KLine.insert(exg_name, pair, timeframe, predata)
                KLine.log_candles_conts(exg_name, pair, timeframe, old_end, end_ms, predata)
                return
            else:
                # 要下载的数据全部存在，直接退出
                return
    newdata = await fetch_api_ohlcv(exchange, pair, timeframe, start_ms, end_ms)
    KLine.insert(exg_name, pair, timeframe, newdata, check_exist)
    KLine.log_candles_conts(exg_name, pair, timeframe, start_ms, end_ms, newdata)


async def download_to_file(exchange, pair: str, timeframe: str, start_ms: int, end_ms: int, out_dir: str):
    fname = pair.replace('/', '_') + '-' + timeframe + '.feather'
    data_path = os.path.join(out_dir, fname)
    columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df_list = []
    if os.path.isfile(data_path):
        df = pd.read_feather(data_path)
        if df.date.dtype != 'int64':
            df['date'] = df['date'].apply(lambda x: int(x.timestamp() * 1000))
        old_start = df.iloc[0]['date']
        df_list.append(df)
        if start_ms < old_start:
            predata = await fetch_api_ohlcv(exchange, pair, timeframe, start_ms, old_start)
            if predata:
                df_list.insert(0, pd.DataFrame(predata, columns=columns))
        start_ms = round(df.iloc[-1]['date']) + 1
        if start_ms >= end_ms:
            return
    newdata = await fetch_api_ohlcv(exchange, pair, timeframe, start_ms, end_ms)
    if newdata:
        df_list.append(pd.DataFrame(newdata, columns=columns))
    df = pd.concat(df_list).reset_index(drop=True)
    df.to_feather(data_path, compression='lz4')


async def auto_fetch_ohlcv(exchange, pair: str, timeframe: str, start_ms: Optional[int] = None,
                           end_ms: Optional[int] = None, limit: Optional[int] = None,
                           allow_lack: float = 0.):
    '''
    获取给定交易对，给定时间维度，给定范围的K线数据。
    先尝试从本地读取，不存在时从交易所下载，然后返回。
    :param exchange:
    :param pair:
    :param timeframe:
    :param start_ms: 毫秒
    :param end_ms: 毫秒
    :param limit:
    :param allow_lack: 最大允许缺失的最新数据比例。设置此项避免对很小数据缺失时的不必要网络请求
    :return:
    '''
    from banbot.storage import KLine
    tf_msecs = tf_to_secs(timeframe) * 1000
    if not end_ms:
        end_ms = int(btime.time() * 1000)
    end_ms = end_ms // tf_msecs * tf_msecs
    if not start_ms:
        start_ms = end_ms - tf_msecs * limit
    start_ms = start_ms // tf_msecs * tf_msecs
    down_tf = KLine.get_down_tf(timeframe)
    await download_to_db(exchange, pair, down_tf, start_ms, end_ms, allow_lack=allow_lack)
    return KLine.query(exchange.name, pair, timeframe, start_ms, end_ms)


async def bulk_ohlcv_do(exg, symbols: List[str], timeframe: str, kwargs: Union[dict, List[dict]],
                        callback: Callable):
    '''
    批量下载并处理K线数据。
    :param exg: 交易所对象
    :param symbols: 所有待处理的交易对
    :param timeframe: 下载处理的K线维度
    :param kwargs: 下载K线的额外参数，支持：start_ms,end_ms,limit,allow_lack
    :param callback: 获得K线数据后回调处理函数，接受参数：ohlcv_arr, symbol, timeframe, **kwargs
    '''
    from banbot.util.misc import parallel_jobs
    if isinstance(kwargs, dict):
        kwargs = [kwargs] * len(symbols)
    for rid in range(0, len(symbols), MAX_CONC_OHLCV):
        # 批量下载，提升效率
        batch = symbols[rid: rid + MAX_CONC_OHLCV]
        args_list = [((exg, pair, timeframe), kwargs[rid + i]) for i, pair in enumerate(batch)]
        task_iter = parallel_jobs(auto_fetch_ohlcv, args_list)
        for f in task_iter:
            res = await f
            callback(res['data'], res['args'][1], res['args'][2], **res['kwargs'])


if __name__ == '__main__':
    cdata_dir = r'E:\trade\freqtd_data\user_data\spec_data\bnb1s'
    convert_bnb_klines_datas(cdata_dir, '1s')
