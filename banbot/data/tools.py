#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : tools.py
# Author: anyongjin
# Date  : 2023/2/28
import asyncio
import datetime
import math
import os

import six

from banbot.config.timerange import TimeRange
from banbot.exchange.crypto_exchange import CryptoExchange
from banbot.exchange.exchange_utils import *
from banbot.storage.symbols import ExSymbol
from banbot.storage.base import SqlSession
from banbot.util.common import logger
from banbot.util.misc import LazyTqdm


def trades_to_ohlcv(trades: List[dict]) -> List[Tuple[int, float, float, float, float, float, int]]:
    result = []
    for trade in trades:
        price = trade['price']
        result.append((trade['timestamp'], price, price, price, price, trade['amount'], 1))
    return result


def build_ohlcvc(details: List[Tuple], tf_secs: int, prefire: float = 0., since=None, ohlcvs=None, with_count=True,
                 in_tf_msecs: int = 0):
    '''
    从交易或子OHLC数组中，构建或更新更粗粒度OHLC数组。
    :param details: 子OHLC列表。[[t,o,h,l,c,v,cnt], ...]
    :param tf_secs: 指定要构建的时间粒度，单位：秒
    :param prefire: 是否提前触发构建完成；用于在特定信号时早于其他交易者提早发出信号
    :param since:
    :param ohlcvs: 已有的待更新数组
    :param with_count: 是否添加交易数量
    :param in_tf_msecs: 传入的蜡烛的间隔。未提供时计算
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
            ohlcvs.append(row if with_count else row[:count])
        else:
            prow = ohlcvs[-1]
            # still processing the same timeframe -> update opening trade
            prow[high] = max(prow[high], row[high])
            prow[low] = min(prow[low], row[low])
            prow[close] = row[close]
            prow[volume] += row[volume]
            if with_count and len(row) > count:
                prow[count] += row[count]
    last_finish = False
    if len(raw_ts) >= 2:
        # 至少有2个，判断最后一个bar是否结束：假定details中每个bar间隔相等，最后一个bar+间隔属于下一个规划区间，则认为最后一个bar结束
        in_tf_msecs = raw_ts[-1] - raw_ts[-2]
    if in_tf_msecs:
        finish_ts = (raw_ts[-1] + in_tf_msecs + off_ms) // ms * ms
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


def load_file_range(data_path: str, tr: Optional[TimeRange]):
    '''
    加载数据文件，必须是feather，按给定的范围截取并返回DataFrame
    '''
    import pandas as pd
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


def load_pair_file_range(data_dir: str, pair: str, timeframe: str, ts_from: float, ts_to: float):
    '''
    从给定目录下，查找交易对的指定时间维度数据，按时间段截取返回.
    '''
    import pandas as pd
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


async def fetch_api_ohlcv(exchange: CryptoExchange, pair: str, timeframe: str, start_ms: int, end_ms: int,
                          pbar: LazyTqdm = None):
    '''
    按给定时间段下载交易对的K线数据。
    :param exchange:
    :param pair:
    :param timeframe:
    :param start_ms: 毫秒时间戳（含）
    :param end_ms: 毫秒时间戳（不含）
    :param pbar: tqdm进度条
    :return:
    '''
    tf_msecs = tf_to_secs(timeframe) * 1000
    assert start_ms > 1000000000000, '`start_ts` must be milli seconds'
    max_end_ts = (end_ms // tf_msecs - 1) * tf_msecs  # 最后一个bar的时间戳，达到此bar时停止，避免额外请求
    if start_ms > max_end_ts:
        return []
    fetch_num = (end_ms - start_ms) // tf_msecs
    batch_size = min(1000, fetch_num + 5)
    if pbar is not None:
        start_text, end_text = btime.to_datestr(start_ms), btime.to_datestr(end_ms)
        logger.info(f'fetch ohlcv {pair}/{timeframe} {start_text} - {end_text}')
    since = round(start_ms)
    result = []
    retry_cnt = 0
    while since + tf_msecs <= end_ms:
        try:
            data = await exchange.fetch_ohlcv(pair, timeframe, since=since, limit=batch_size)
        except ccxt.NetworkError as e:
            retry_cnt += 1
            if retry_cnt > 1:
                raise e
            logger.error(f'down ohlcv error: {e}, retry {retry_cnt}')
            await asyncio.sleep(3)
            continue
        retry_cnt = 0
        if not len(data):
            break
        result.extend(data)
        cur_end = round(data[-1][0])
        if pbar is not None:
            pbar.update()
        if cur_end >= max_end_ts:
            break
        since = cur_end + 1
    end_pos = len(result) - 1
    while end_pos >= 0 and result[end_pos][0] >= end_ms:
        end_pos -= 1
    return result[:end_pos + 1]


async def download_to_db(exchange, exs: ExSymbol, timeframe: str, start_ms: int, end_ms: int, check_exist=True,
                         allow_lack: float = 0., pbar: Union[LazyTqdm, str] = 'auto', is_parallel=False,
                         sess: SqlSession = None) -> int:
    '''
    从交易所下载K线数据到数据库。
    跳过已有部分，同时保持数据连续
    '''
    from banbot.storage.klines import KLine, KHole, dba
    if timeframe not in KLine.down_tfs:
        raise RuntimeError(f'can only download kline: {KLine.down_tfs}, current: {timeframe}')
    from banbot.util.common import MeasureTime
    measure = MeasureTime()
    measure.start_for('valid_start')
    start_ms = await exs.get_valid_start(start_ms)
    measure.start_for('down_range')
    is_temp = False
    if not sess:
        sess = dba.new_session() if is_parallel else None
        is_temp = True
    start_ms, end_ms = await KHole.get_down_range(exs, timeframe, start_ms, end_ms, sess=sess)
    if not start_ms:
        return 0
    if isinstance(pbar, six.string_types) and pbar == 'auto':
        pbar = LazyTqdm()
    try:
        down_count = 0
        if check_exist:
            measure.start_for('query_range')
            old_start, old_end = await KLine.query_range(exs.id, timeframe, sess=sess)
            if old_start and old_end > old_start:
                if start_ms < old_start:
                    # 直接抓取start_ms - old_start的数据，避免出现空洞；可能end_ms>old_end，还需要下载后续数据
                    cur_end = round(old_start)
                    measure.start_for('fetch_start')
                    predata = await fetch_api_ohlcv(exchange, exs.symbol, timeframe, start_ms, cur_end, pbar)
                    measure.start_for('insert_start')
                    down_count += await KLine.insert(exs.id, timeframe, predata, sess=sess)
                    measure.start_for('candle_conts_start')
                    await KLine.log_candles_conts(exs, timeframe, start_ms, cur_end, predata, sess=sess)
                    start_ms = old_end
                elif end_ms > old_end:
                    if (end_ms - old_end) / (end_ms - start_ms) <= allow_lack:
                        # 最新部分缺失的较少，不再请求交易所，节省时间
                        return down_count
                    # 直接抓取old_end - end_ms的数据，避免出现空洞；前面没有需要再下次的数据了。可直接退出
                    measure.start_for('fetch_end')
                    predata = await fetch_api_ohlcv(exchange, exs.symbol, timeframe, old_end, end_ms, pbar)
                    measure.start_for('insert_end')
                    down_count += await KLine.insert(exs.id, timeframe, predata, sess=sess)
                    measure.start_for('candle_conts_end')
                    await KLine.log_candles_conts(exs, timeframe, old_end, end_ms, predata, sess=sess)
                    return down_count
                else:
                    # 要下载的数据全部存在，直接退出
                    return down_count
        measure.start_for('fetch_all')
        newdata = await fetch_api_ohlcv(exchange, exs.symbol, timeframe, start_ms, end_ms, pbar)
        measure.start_for('insert_all')
        down_count += await KLine.insert(exs.id, timeframe, newdata, check_exist, sess=sess)
        measure.start_for('candle_conts_all')
        await KLine.log_candles_conts(exs, timeframe, start_ms, end_ms, newdata, sess=sess)
        measure.print_all(min_cost=0.01)
    finally:
        if sess and is_temp:
            await asyncio.shield(sess.close())
    return down_count


async def download_to_file(exchange, pair: str, timeframe: str, start_ms: int, end_ms: int, out_dir: str):
    import pandas as pd
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


def parse_down_args(timeframe: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None,
                    limit: Optional[int] = None, with_unfinish: bool = False):
    tf_msecs = tf_to_secs(timeframe) * 1000
    if start_ms:
        fix_start_ms = start_ms // tf_msecs * tf_msecs
        if start_ms > fix_start_ms:
            start_ms = fix_start_ms + tf_msecs
        if limit and not end_ms:
            end_ms = start_ms + tf_msecs * limit
    if not end_ms:
        end_ms = int(btime.time() * 1000)
    factor = end_ms / tf_msecs
    factor_int = math.ceil(factor) if with_unfinish else math.floor(factor)
    end_ms = factor_int * tf_msecs
    if not start_ms:
        start_ms = end_ms - tf_msecs * limit
    return start_ms, end_ms


async def auto_fetch_ohlcv(exchange, exs: ExSymbol, timeframe: str, start_ms: Optional[int] = None,
                           end_ms: Optional[int] = None, limit: Optional[int] = None,
                           allow_lack: float = 0., with_unfinish: bool = False, pbar: Union[LazyTqdm, str] = 'auto',
                           is_parallel=False, sess: SqlSession = None):
    '''
    获取给定交易对，给定时间维度，给定范围的K线数据。
    先尝试从本地读取，不存在时从交易所下载，然后返回。
    :param exchange:
    :param exs:
    :param timeframe:
    :param start_ms: 毫秒
    :param end_ms: 毫秒
    :param limit:
    :param allow_lack: 最大允许缺失的最新数据比例。设置此项避免对很小数据缺失时的不必要网络请求
    :param with_unfinish: 是否附加未完成的数据
    :param pbar: 进度条
    :param is_parallel: 如果是并发执行，写入时需要单独的session
    :return:
    '''
    from banbot.storage import KLine
    start_ms, end_ms = parse_down_args(timeframe, start_ms, end_ms, limit, with_unfinish)
    down_tf = KLine.get_down_tf(timeframe)
    await download_to_db(exchange, exs, down_tf, start_ms, end_ms, allow_lack=allow_lack, pbar=pbar,
                         is_parallel=is_parallel, sess=sess)
    return await KLine.query(exs, timeframe, start_ms, end_ms, with_unfinish=with_unfinish, sess=sess)


async def fast_bulk_ohlcv(exg: CryptoExchange, symbols: List[str], timeframe: str,
                          start_ms: int = None, end_ms: int = None, limit: int = None,
                          callback: Callable = None, **kwargs):
    '''
    快速批量获取K线。先下载所有需要的币种，然后批量查询再分组返回。
    适用于币种较多，且需要的开始结束时间一致，且大部分已下载的情况。
    '''
    from banbot.storage import KLine
    exs_list = await ExSymbol.ensures(exg.name, exg.market_type, symbols)
    exs_map = {item.id: item for item in exs_list}
    item_ranges = await KLine.load_kline_ranges()
    start_ms, end_ms = parse_down_args(timeframe, start_ms, end_ms, limit)
    # 筛选需要下载的币种
    down_pairs = []
    for sid, tf in item_ranges:
        if tf != timeframe or sid not in exs_map:
            continue
        cur_start, cur_stop = item_ranges[(sid, tf)]
        if cur_start <= start_ms <= end_ms <= cur_stop:
            continue
        down_pairs.append(exs_map.get(sid))
    if down_pairs:
        # 分批次执行下载。
        from banbot.util.misc import parallel_jobs
        pbar = LazyTqdm()
        kwargs['pbar'] = pbar
        kwargs['is_parallel'] = True
        down_tf = KLine.get_down_tf(timeframe)
        for rid in range(0, len(down_pairs), MAX_CONC_OHLCV):
            # 批量下载，提升效率
            batch = down_pairs[rid: rid + MAX_CONC_OHLCV]
            args_list = [((exg, exs, down_tf, start_ms, end_ms), kwargs) for i, exs in enumerate(batch)]
            task_iter = parallel_jobs(download_to_db, args_list)
            for f in task_iter:
                await f
        pbar.close()
    if not callback:
        return
    # 查询全部数据然后分组返回
    result = await KLine.query_batch(list(exs_map.keys()), timeframe, start_ms, end_ms)
    result = sorted(result, key=lambda x: x[-1])
    prev_sid, ohlcvs = 0, []
    ret_args = dict(start_ms=start_ms, end_ms=end_ms, limit=limit)
    for bar in result:
        if bar[-1] != prev_sid:
            if ohlcvs and prev_sid in exs_map:
                callback(ohlcvs, exs_map.get(prev_sid), timeframe, **ret_args)
            prev_sid = bar[-1]
            ohlcvs = []
        ohlcvs.append(bar[:-1])
    if ohlcvs and prev_sid in exs_map:
        callback(ohlcvs, exs_map.get(prev_sid), timeframe, **ret_args)


async def bulk_ohlcv_do(exg: CryptoExchange, symbols: List[str], timeframe: str, kwargs: Union[dict, List[dict]],
                        callback: Callable = None):
    '''
    批量下载并处理K线数据。
    :param exg: 交易所对象
    :param symbols: 所有待处理的交易对
    :param timeframe: 下载处理的K线维度
    :param kwargs: 下载K线的额外参数，支持：start_ms,end_ms,limit,allow_lack
    :param callback: 获得K线数据后回调处理函数，接受参数：ohlcv_arr, symbol, timeframe, **kwargs
    '''
    from banbot.util.misc import parallel_jobs, run_async
    from banbot.storage import dba, select
    pbar = LazyTqdm()
    if isinstance(kwargs, dict):
        kwargs['pbar'] = pbar
        kwargs['is_parallel'] = True
        kwargs = [kwargs] * len(symbols)
    else:
        for kw in kwargs:
            kw['pbar'] = pbar
            kw['is_parallel'] = True
    sess = dba.session
    fts = [ExSymbol.exchange == exg.name, ExSymbol.symbol.in_(set(symbols)), ExSymbol.market == exg.market_type]
    exs_list = list(await sess.scalars(select(ExSymbol).filter(*fts)))
    if len(symbols) < len(exs_list):
        keep_pairs = {s.symbol for s in exs_list}
        del_pairs = set(symbols).difference(keep_pairs)
        logger.warning(f'{len(del_pairs)} pairs removed in bulk_ohlcv_do, as not in exsymbol: {del_pairs}')
    for rid in range(0, len(exs_list), MAX_CONC_OHLCV):
        # 批量下载，提升效率
        batch = exs_list[rid: rid + MAX_CONC_OHLCV]
        args_list = [((exg, exs, timeframe), kwargs[rid + i]) for i, exs in enumerate(batch)]
        task_iter = parallel_jobs(auto_fetch_ohlcv, args_list)
        for f in task_iter:
            res = await f
            if callback:
                await run_async(callback, res['data'], res['args'][1], res['args'][2], **res['kwargs'])
    pbar.close()

