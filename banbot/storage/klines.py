#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : klines.py
# Author: anyongjin
# Date  : 2023/4/24
import six

from banbot.storage.base import *
from typing import Callable, Tuple, ClassVar, Dict
from banbot.exchange.exchange_utils import tf_to_secs
from banbot.util import btime


class BarAgg:
    def __init__(self, tf: str, tbl: str, agg_from: Optional[str], agg_start: Optional[str],
                 agg_end: Optional[str], agg_every: Optional[str], comp_before: str = None,
                 retention: str = 'all'):
        self.tf = tf
        self.secs = tf_to_secs(tf)
        self.tbl = tbl
        self.agg_from = agg_from
        self.agg_start = agg_start
        self.agg_end = agg_end
        self.agg_every = agg_every
        self.comp_before = comp_before
        self.retention = retention
        self.is_view = comp_before is None  # 超表可压缩，故传入压缩参数的认为是超表


class KLine(BaseDbModel):
    '''
    K线数据超表，存储1m维度数据
    还有一个超表kline_1h存储1h维度数据
    '''
    __tablename__ = 'kline_1m'
    _sid_range_map: ClassVar[Dict[Tuple[int, str], Tuple[int, int]]] = dict()
    _tname: ClassVar[str] = 'kline_1m'

    sid = Column(sa.Integer, primary_key=True)
    time = Column(sa.DateTime, primary_key=True)
    open = Column(sa.FLOAT)
    high = Column(sa.FLOAT)
    low = Column(sa.FLOAT)
    close = Column(sa.FLOAT)
    volume = Column(sa.FLOAT)

    agg_list = [
        # 全部使用超表，自行在插入时更新依赖表。因连续聚合无法按sid刷新，在按sid批量插入历史数据后刷新时性能较差
        BarAgg('1m', 'kline_1m', None, None, None, None, '3 days', '2 months'),
        BarAgg('5m', 'kline_5m', '1m', '20m', '1m', '1m', '3 days', '2 months'),
        BarAgg('15m', 'kline_15m', '5m', '1h', '5m', '5m', '6 days', '4 months'),
        BarAgg('1h', 'kline_1h', '15m', '3h', '15m', '15m', '2 months', '2 years'),
        BarAgg('1d', 'kline_1d', '1h', '3d', '1h', '1h', '3 years', '20 years'),
    ]

    agg_map: Dict[str, BarAgg] = {v.tf: v for v in agg_list}

    _insert_conflict = '''
    ON CONFLICT (sid, time)
DO UPDATE SET 
open = EXCLUDED.open,
high = EXCLUDED.high,
low = EXCLUDED.low,
close = EXCLUDED.close,
volume = EXCLUDED.volume'''

    _candle_agg = '''
  first(open, time) AS open,  
  max(high) AS high,
  min(low) AS low, 
  last(close, time) AS close,
  sum(volume) AS volume'''

    @classmethod
    def _agg_sql(cls, intv: str, base_tbl: str, where_str: str = ''):
        return f'''
SELECT sid, time_bucket(INTERVAL '{intv}', time) AS time, {cls._candle_agg}
FROM {base_tbl} {where_str}
GROUP BY sid, 2 
ORDER BY sid, 2'''

    @classmethod
    def _init_hypertbl(cls, conn: sa.Connection, tbl: BarAgg):
        statements = [
            f"SELECT create_hypertable('{tbl.tbl}','time');",
            '''ALTER TABLE {tbl.tbl} SET (
              timescaledb.compress,
              timescaledb.compress_orderby = 'time DESC',
              timescaledb.compress_segmentby = 'sid'
            );''',
            f"SELECT add_compression_policy('{tbl.tbl}', INTERVAL '{tbl.comp_before}');",
        ]
        for stat in statements:
            conn.execute(sa.text(stat))
            conn.commit()
        # 设置数据丢弃
        db_retention = AppConfig.get()['database'].get('retention')
        if db_retention and db_retention != 'all':
            conn.execute(sa.text(f"SELECT add_retention_policy('{tbl.tbl}', INTERVAL '{tbl.retention}');"))
            conn.commit()

    @classmethod
    def init_tbl(cls, conn: sa.Connection):
        cls._init_hypertbl(conn, cls.agg_list[0])
        # 创建连续聚合及更新策略
        for item in cls.agg_list[1:]:
            if not item.is_view:
                create_sql = f'CREATE TABLE {item.tbl} (LIKE {cls._tname} INCLUDING ALL);'
                conn.execute(sa.text(create_sql))
                conn.commit()
                # 初始化超表
                cls._init_hypertbl(conn, item)
                continue
            stat_create = f'''
CREATE MATERIALIZED VIEW kline_{item.tf} 
WITH (timescaledb.continuous) AS 
SELECT sid, time_bucket(INTERVAL '{item.tf}', time) AS time, {cls._candle_agg}
FROM {item.agg_from}
GROUP BY sid, 2 
ORDER BY sid, 2'''
            stat_policy = f'''
SELECT add_continuous_aggregate_policy('kline_{item.tf}',
  start_offset => INTERVAL '{item.agg_start}',
  end_offset => INTERVAL '{item.agg_end}',
  schedule_interval => INTERVAL '{item.agg_every}');'''
            conn.execute(sa.text(stat_create))
            conn.commit()
            conn.execute(sa.text(stat_policy))
            conn.commit()

    @classmethod
    def drop_tbl(cls, conn: sa.Connection):
        '''
        删除所有的K线数据表；超表+连续聚合
        '''
        for tbl in cls.agg_list[::-1]:
            if tbl.is_view:
                conn.execute(sa.text(f"drop MATERIALIZED view if exists {tbl.tbl} CASCADE"))
            else:
                conn.execute(sa.text(f"drop table if exists {tbl.tbl}"))

    @classmethod
    def _get_sid(cls, exg_name: str, symbol: Union[str, int]):
        sid = symbol
        if isinstance(symbol, six.string_types):
            from banbot.storage.symbols import SymbolTF
            sid = SymbolTF.get_id(exg_name, symbol)
        return sid

    @classmethod
    def _query_hyper(cls, sid: int, timeframe: str, dct_sql: str, gp_sql: Union[str, Callable]):
        conn = db.session.connection()
        if timeframe in cls.agg_map:
            stmt = dct_sql.format(
                tbl=cls.agg_map[timeframe].tbl,
                sid=sid
            )
            return conn.execute(sa.text(stmt))
        else:
            # 时间帧没有直接符合的，从最接近的子timeframe聚合
            tf_secs = tf_to_secs(timeframe)
            sub_tf, sub_tbl = None, None
            for item in cls.agg_list[::-1]:
                if tf_secs % item.secs == 0:
                    sub_tf, sub_tbl = item.tf, item.tbl
                    break
            if not sub_tbl:
                raise RuntimeError(f'unsupport timeframe {timeframe}')
            if callable(gp_sql):
                gp_sql = gp_sql()
            stmt = gp_sql.format(
                tbl=sub_tbl,
                sid=sid,
            )
            return conn.execute(sa.text(stmt))

    @classmethod
    def query(cls, exg_name: str, pair: str, timeframe: str, start_ms: int, end_ms: Optional[int] = None,
              limit: Optional[int] = None):
        tf_secs = tf_to_secs(timeframe)
        max_end_ms = end_ms
        if limit:
            limit_end_ms = start_ms + tf_secs * limit * 1000
            max_end_ms = end_ms or btime.time() * 1000
            end_ms = min(limit_end_ms, max_end_ms)

        start_ts, end_ts = start_ms / 1000, end_ms / 1000

        dct_sql = f'''
select (extract(epoch from time) * 1000)::float as time,open,high,low,close,volume from {{tbl}}
where sid={{sid}} and time >= to_timestamp({start_ts}) and time < to_timestamp({end_ts})
order by time'''

        def gen_gp_sql():
            return f'''
                select (extract(epoch from time_bucket('{timeframe}', time)) * 1000)::float AS time,
                  {cls._candle_agg} from {{tbl}}
                where sid={{sid}} and time >= to_timestamp({start_ts}) and time < to_timestamp({end_ts})
                group by time
                order by time'''

        sid = cls._get_sid(exg_name, pair)
        rows = cls._query_hyper(sid, timeframe, dct_sql, gen_gp_sql).fetchall()
        if not len(rows) and max_end_ms - end_ms > tf_secs * 1000:
            rows = cls.query(exg_name, pair, timeframe, end_ms, max_end_ms, limit)
        return rows

    @classmethod
    def _load_symbol_ranges(cls):
        dct_sql = '''
select sid,
(extract(epoch from min(time)) * 1000)::float, 
(extract(epoch from max(time)) * 1000)::float 
from {tbl}
group by 1'''
        sess = db.session
        for item in cls.agg_list:
            sql_text = dct_sql.format(tbl=item.tbl)
            rows = sess.execute(sa.text(sql_text)).fetchall()
            for sid, min_time, max_time in rows:
                cache_key = sid, item.tf
                if max_time:
                    # 这里记录蜡烛对应的结束时间
                    max_time += tf_to_secs(item.tf) * 1000
                    min_time, max_time = round(min_time), round(max_time)
                cls._sid_range_map[cache_key] = min_time, max_time
        return cls._sid_range_map

    @classmethod
    def _fetch_range(cls, sid: int, timeframe: str):
        dct_sql = 'select (extract(epoch from min(time)) * 1000)::float, ' \
                  '(extract(epoch from max(time)) * 1000)::float from {tbl} where sid={sid}'
        min_time, max_time = cls._query_hyper(sid, timeframe, dct_sql, '').fetchone()
        if max_time:
            # 这里记录蜡烛对应的结束时间
            max_time += tf_to_secs(timeframe) * 1000
            min_time, max_time = round(min_time), round(max_time)
        cache_key = sid, timeframe
        cls._sid_range_map[cache_key] = min_time, max_time
        return min_time, max_time

    @classmethod
    def query_range(cls, exg_name: str, symbol: Union[str, int], timeframe: str = '1m')\
            -> Tuple[Optional[int], Optional[int]]:
        sid = cls._get_sid(exg_name, symbol)
        cache_key = sid, timeframe
        cache_val = cls._sid_range_map.get(cache_key)
        if not cache_val:
            min_time, max_time = cls._fetch_range(sid, timeframe)
        else:
            min_time, max_time = cache_val
        return min_time, max_time

    @classmethod
    def _update_range(cls, sid: int, timeframe: str, start_ms: int, end_ms: int):
        '''
        更新sid+timeframe对应的数据区间。end_ms应为最后一个bar对应的结束时间，而非开始时间
        '''
        cache_key = sid, timeframe
        if cache_key not in cls._sid_range_map:
            cls._sid_range_map[cache_key] = start_ms, end_ms
        else:
            old_start, old_stop = cls._sid_range_map[cache_key]
            if old_start and old_stop:
                if old_stop < start_ms or end_ms < old_start:
                    raise ValueError(f'incontinus range: {cache_key}, old range: [{old_start}, {old_stop}], '
                                     f'new: [{start_ms}, {end_ms}]')
                cls._sid_range_map[cache_key] = min(start_ms, old_start), max(end_ms, old_stop)
            else:
                cls._sid_range_map[cache_key] = start_ms, end_ms

    @classmethod
    def get_down_tf(cls, tf: str):
        '''
        获取指定周期对应的下载的时间周期。
        只有1m和1h允许下载并写入超表。其他维度都是由这两个维度聚合得到。
        '''
        tf_secs = tf_to_secs(tf)
        m1_secs, h1_secs = 60, 3600
        if tf_secs >= h1_secs:
            if tf_secs % h1_secs > 0:
                raise RuntimeError(f'unsupport timeframe: {tf}')
            return '1h'
        if tf_secs < m1_secs or tf_secs % m1_secs > 0:
            raise RuntimeError(f'unsupport timeframe: {tf}')
        return '1m'

    @classmethod
    def insert(cls, exg_name: str, pair: str, timeframe: str, rows: List[Tuple], skip_in_range=True):
        if not rows:
            return
        if timeframe not in {'1m', '1h'}:
            raise RuntimeError(f'can only insert kline: 1m or 1h, current: {timeframe}')
        intv_secs = tf_to_secs(timeframe)
        rows = sorted(rows, key=lambda x: x[0])
        if len(rows) > 1:
            # 检查间隔是否正确
            row_intv = (rows[1][0] - rows[0][0]) // 1000
            if row_intv != intv_secs:
                raise ValueError(f'insert kline must be {timeframe} interval, current: {row_intv} s')
        sid = cls._get_sid(exg_name, pair)
        start_ms, end_ms = rows[0][0], rows[-1][0] + intv_secs * 1000
        old_start, old_stop = cls.query_range(exg_name, sid, timeframe)
        if old_start and old_stop:
            # 插入的数据应该和已有数据连续，避免出现空洞。
            if old_stop < start_ms or end_ms < old_start:
                raise ValueError(f'insert incontinus data, sid: {sid}, old range: [{old_start}, {old_stop}], '
                                 f'insert: [{start_ms}, {end_ms}]')
            if skip_in_range:
                rows = [r for r in rows if not (old_start <= r[0] < old_stop)]
        ins_rows = []
        for r in rows:
            row_ts = btime.to_datetime(r[0])
            ins_rows.append(dict(
                sid=sid, time=row_ts, open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5],
            ))
        sess = db.session
        ins_tbl = cls.agg_map[timeframe].tbl
        ins_cols = "sid, time, open, high, low, close, volume"
        places = ":sid, :time, :open, :high, :low, :close, :volume"
        insert_sql = f"insert into {ins_tbl} ({ins_cols}) values ({places}) {cls._insert_conflict}"
        sess.execute(sa.text(insert_sql), ins_rows)
        sess.commit()
        # 更新区间
        cls._update_range(sid, timeframe, start_ms, end_ms)
        # 刷新相关的连续聚合
        cls._refresh_conti_agg(sid, timeframe, start_ms, end_ms)

    @classmethod
    def _refresh_conti_agg(cls, sid: int, from_level: str, start_ms: int, end_ms: int):
        agg_keys = {'1m'}
        from_secs = tf_to_secs(from_level)
        for item in cls.agg_list:
            if item.secs <= from_secs or item.agg_from not in agg_keys:
                # 跳过过小维度；跳过无关的连续聚合
                continue
            start_align = start_ms // 1000 // item.secs * item.secs
            end_align = end_ms // 1000 // item.secs * item.secs
            if start_align == end_align < start_ms:
                # 没有出现新的bar数据，无需更新
                # 前2个相等，说明：插入的数据所属bar尚未完成。
                # start_align < start_ms说明：插入的数据不是所属bar的第一个数据
                continue
            agg_keys.add(item.tf)
        agg_keys.remove('1m')
        if not agg_keys:
            return
        from banbot.storage.base import init_db
        with init_db().connect() as conn:
            # 这里必须从连接池重新获取连接，不能使用sess的连接，否则会导致sess无效
            bak_iso_level = conn.get_isolation_level()
            conn.execution_options(isolation_level='AUTOCOMMIT')
            for tf in agg_keys:
                cls._refresh_agg(conn, cls.agg_map[tf], sid, start_ms, end_ms)
            conn.commit()
            conn.execution_options(isolation_level=bak_iso_level)

    @classmethod
    def _refresh_agg(cls, conn: Union[SqlSession, sa.Connection], tbl: BarAgg, sid: int,
                     start_ms: int, end_ms: int, agg_from: str = None):
        if not agg_from:
            agg_from = 'kline_' + tbl.agg_from
        tf_msecs = tbl.secs * 1000
        # 有可能start_ms刚好是下一个bar的开始，前一个需要-1
        win_start = f"'{btime.to_datetime((start_ms // tf_msecs - 1) * tf_msecs)}'"
        win_end = f"'{btime.to_datetime(end_ms // tf_msecs * tf_msecs)}'"
        if tbl.is_view:
            # 刷新连续聚合（连续聚合不支持按sid筛选刷新，性能批量插入历史数据时性能较差）
            stmt = f"CALL refresh_continuous_aggregate('{tbl.tbl}', {win_start}, {win_end});"
        else:
            # 聚合数据到超表中
            select_sql = f'''
SELECT sid, time_bucket(INTERVAL '{tbl.tf}', time) AS time, {cls._candle_agg}
FROM {agg_from}
where sid={sid} and time >= {win_start} and time < {win_end}
GROUP BY sid, 2 
ORDER BY sid, 2'''
            ins_cols = "sid, time, open, high, low, close, volume"
            stmt = f'''INSERT INTO {tbl.tbl} ({ins_cols}) {select_sql} {cls._insert_conflict};'''
        try:
            conn.execute(sa.text(stmt))
            return cls._fetch_range(sid, tbl.tf)
        except Exception as e:
            logger.error(f'refresh conti agg error: {e}, {tbl.tf}')

    @classmethod
    def _find_sid_hole(cls, sess: SqlSession, timeframe: str, sid: int, since: float, until: float = 0., as_ts=True):
        tbl = f'kline_{timeframe}'
        batch_size, true_intv = 1000, tf_to_secs(timeframe) * 1.
        prev_date = None
        res_holes = []
        while True:
            hole_sql = f"SELECT time FROM {tbl} where sid={sid} and time >= to_timestamp({since}) "
            if until:
                hole_sql += f'and time < to_timestamp({until}) '
            hole_sql += f"ORDER BY time limit {batch_size};"
            rows = sess.execute(sa.text(hole_sql)).fetchall()
            if not len(rows):
                break
            off_idx = 0
            if prev_date is None:
                prev_date = rows[0][0]
                off_idx = 1
            for row in rows[off_idx:]:
                cur_intv = (row[0] - prev_date).total_seconds()
                if cur_intv > true_intv:
                    if as_ts:
                        hole_start, hold_end = btime.to_utcstamp(prev_date), btime.to_utcstamp(row[0])
                    else:
                        hole_start, hold_end = prev_date, row[0]
                    res_holes.append((hole_start, hold_end))
                elif cur_intv < true_intv:
                    logger.warning(f'invalid kline interval: {cur_intv:.3f}, sid: {sid}')
                prev_date = row[0]
            since = btime.to_utcstamp(prev_date) + 0.1
        return res_holes

    @classmethod
    async def _fill_tf_hole(cls, timeframe: str):
        from banbot.data.tools import download_to_db
        from banbot.storage.symbols import SymbolTF
        from banbot.exchange.crypto_exchange import get_exchange
        sess = db.session
        tbl = f'kline_{timeframe}'
        gp_sql = f'SELECT sid, extract(epoch from min(time))::float as mtime FROM {tbl} GROUP BY sid;'
        sid_rows = sess.execute(sa.text(gp_sql)).fetchall()
        if not len(sid_rows):
            return
        for sid, start_ts in sid_rows:
            hole_list = cls._find_sid_hole(sess, timeframe, sid, start_ts)
            if not hole_list:
                continue
            old_holes = sess.query(KHole).filter(KHole.sid == sid, KHole.timeframe == timeframe).all()
            old_holes = [(btime.to_utcstamp(row.start), btime.to_utcstamp(row.stop)) for row in old_holes]
            hole_list = [h for h in hole_list if h not in old_holes]
            if not hole_list:
                continue
            stf: SymbolTF = sess.query(SymbolTF).get(sid)
            exchange = get_exchange(stf.exchange)
            for hole in hole_list:
                start_ms, end_ms = int(hole[0] * 1000) + 1, int(hole[1] * 1000)
                logger.warning(f'filling hole: {stf.symbol}, {start_ms} - {end_ms}')
                await download_to_db(exchange, stf.symbol, '1m', start_ms, end_ms, check_exist=False)
                real_holes = cls._find_sid_hole(sess, timeframe, sid, hole[0], hole[1] + 0.1, as_ts=False)
                if not real_holes:
                    continue
                logger.warning(f'REAL Holes: {stf.symbol} {real_holes}')
                for rhole in real_holes:
                    sess.add(KHole(sid=sid, timeframe=timeframe, start=rhole[0], stop=rhole[1]))
                sess.commit()

    @classmethod
    async def fill_holes(cls):
        logger.info('find and fill holes for kline...')
        for item in cls.agg_list:
            await cls._fill_tf_hole(item.tf)

    @classmethod
    def sync_timeframes(cls):
        '''
        检查各kline表的数据一致性，如果低维度数据比高维度多，则聚合更新到高维度
        应在机器人启动时调用
        '''
        start = time.monotonic()
        sid_ranges = cls._load_symbol_ranges()
        if not sid_ranges:
            return
        logger.info('try sync timeframes for klines...')
        sid_list = [(*k, *v) for k, v in sid_ranges.items()]
        sid_list = sorted(sid_list, key=lambda x: x[0])
        all_tf = set(map(lambda x: x.tf, cls.agg_list))
        first = sid_list[0]
        tf_ranges = {first[1]: (tf_to_secs(first[1]), *first[2:])}  # {tf: (tf_secs, min_time, max_time)}
        prev_sid = sid_list[0][0]

        def call_sync():
            lack_tf = all_tf.difference(tf_ranges.keys())
            for tf in lack_tf:
                tf_ranges[tf] = (tf_to_secs(tf), 0, 0)
            tf_list = [(k, *v) for k, v in tf_ranges.items()]
            tf_list = sorted(tf_list, key=lambda x: x[1])
            cls._sync_kline_sid(prev_sid, tf_list)

        for it in sid_list[1:]:
            if it[0] != prev_sid:
                call_sync()
                prev_sid = it[0]
                tf_ranges.clear()
            tf_ranges[it[1]] = (tf_to_secs(it[1]), *it[2:])
        call_sync()
        cost = time.monotonic() - start
        logger.info(f'sync timeframes cost: {cost:.2f} s')

    @classmethod
    def _sync_kline_sid(cls, sid: int, tf_list: List[Tuple[str, int, int, int]]):
        '''
        检查给定的sid，各个周期数据是否有不一致，尝试用低维度数据更新高维度数据
        '''
        sess = db.session
        for i in range(len(tf_list) - 1):
            if not tf_list[i][2]:
                continue
            cls._sync_kline_sid_tf(sess, sid, tf_list, i)
        sess.commit()

    @classmethod
    def _sync_kline_sid_tf(cls, sess: SqlSession, sid: int, tf_list: List[Tuple[str, int, int, int]], base_id: int):
        '''
        检查给定sid的指定周期数据，是否可更新更大周期数据。必要时进行更新。
        '''
        stf, secs, start, end = tf_list[base_id]
        base_tbl = f'kline_{stf}'
        for i, par in enumerate(tf_list):
            if i <= base_id:
                continue
            ptf, psec, pstart, pend = par
            agg_tbl = cls.agg_map[ptf]
            tf_msecs = psec * 1000
            base_start = start // tf_msecs * tf_msecs
            base_end = end // tf_msecs * tf_msecs
            if not pstart or not pend:
                min_time, max_time = cls._refresh_agg(sess, agg_tbl, sid, base_start, base_end, base_tbl)
                tf_list[i] = (*par[:2], min_time, max_time)
                continue
            min_time, max_time = None, None
            if base_start < pstart:
                min_time, max_time = cls._refresh_agg(sess, agg_tbl, sid, base_start, pstart, base_tbl)
            if base_end > pend:
                min_time, max_time = cls._refresh_agg(sess, agg_tbl, sid, pend, base_end, base_tbl)
            if min_time:
                tf_list[i] = (*par[:2], min_time, max_time)


class KHole(BaseDbModel):
    '''
    交易所的K线数据可能有时因公司业务或系统故障等原因，出现空洞；记录到这里避免重复下载
    '''
    __tablename__ = 'khole'

    __table_args__ = (
        sa.Index('idx_khole_sid_tf', 'sid', 'timeframe'),
        sa.UniqueConstraint('sid', 'timeframe', 'start', name='idx_khole_sid_tf_start'),
    )

    id = Column(sa.Integer, primary_key=True)
    sid = Column(sa.Integer)
    timeframe = Column(sa.String(5))
    start = Column(sa.DateTime)
    stop = Column(sa.DateTime)
