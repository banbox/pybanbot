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


class KLine(BaseDbModel):
    __tablename__ = 'kline'
    _sid_range_map: ClassVar[Dict[int, Tuple[int, int]]] = dict()
    interval: ClassVar[int] = 60000  # 每行是1m维度

    sid = Column(sa.Integer, primary_key=True)
    time = Column(sa.DateTime, primary_key=True)
    open = Column(sa.FLOAT)
    high = Column(sa.FLOAT)
    low = Column(sa.FLOAT)
    close = Column(sa.FLOAT)
    volume = Column(sa.FLOAT)

    agg_intvs = [
        ('5m', '20m', '1m', '1m', 'kline'),
        ('15m', '1h', '5m', '5m', 'kline_5m'),
        ('1d', '3d', '15m', '15m', 'kline_15m')]

    tf_tbls = {'1m': 'kline', '5m': 'kline_5m', '15m': 'kline_15m', '1d': 'kline_1d'}

    tf_list = [
        ('1m', 'kline', 60),
        ('5m', 'kline_5m', 300),
        ('15m', 'kline_15m', 900),
        ('1d', 'kline_1d', 1440),
    ]

    @classmethod
    def init_tbl(cls, conn: sa.Connection):
        statements = [
            "SELECT create_hypertable('kline','time');",
            '''ALTER TABLE kline SET (
              timescaledb.compress,
              timescaledb.compress_orderby = 'time DESC',
              timescaledb.compress_segmentby = 'sid'
            );''',
            "SELECT add_compression_policy('kline', INTERVAL '3 days');",
        ]
        for stat in statements:
            conn.execute(sa.text(stat))
            conn.commit()
        # 设置数据丢弃
        db_retention = AppConfig.get()['database'].get('retention')
        if db_retention and db_retention != 'all':
            conn.execute(sa.text(f"SELECT add_retention_policy('kline', INTERVAL '{db_retention}');"))
            conn.commit()
        # 创建连续聚合及更新策略
        for item in cls.agg_intvs:
            intv, start, end, every, base_tbl = item
            stat_create = f'''
CREATE MATERIALIZED VIEW kline_{intv} 
WITH (timescaledb.continuous) AS 
SELECT 
  sid,
  time_bucket(INTERVAL '{intv}', time) AS time,
  first(open, time) AS open,  
  max(high) AS high,
  min(low) AS low, 
  last(close, time) AS close,
  sum(volume) AS volume
FROM {base_tbl} 
GROUP BY sid, 2 
ORDER BY sid, 2'''
            stat_policy = f'''
SELECT add_continuous_aggregate_policy('kline_{intv}',
  start_offset => INTERVAL '{start}',
  end_offset => INTERVAL '{end}',
  schedule_interval => INTERVAL '{every}');'''
            conn.execute(sa.text(stat_create))
            conn.commit()
            conn.execute(sa.text(stat_policy))
            conn.commit()

    @classmethod
    def drop_tbl(cls, conn: sa.Connection):
        intv, start, end, every, base_tbl = cls.agg_intvs[0]
        vname = f'kline_{intv}'
        conn.execute(sa.text(f"drop MATERIALIZED view if exists {vname} CASCADE"))
        conn.execute(sa.text(f"drop table if exists kline"))

    @classmethod
    def _get_sid(cls, exg_name: str, symbol: Union[str, int]):
        sid = symbol
        if isinstance(symbol, six.string_types):
            from banbot.storage.symbols import SymbolTF
            sid = SymbolTF.get_id(exg_name, symbol)
        return sid

    @classmethod
    def _query_hyper(cls, exg_name: str, symbol: Union[str, int], timeframe: str,
                     dct_sql: str, gp_sql: Union[str, Callable]):
        sid = cls._get_sid(exg_name, symbol)
        conn = db.session.connection()
        if timeframe in cls.tf_tbls:
            stmt = dct_sql.format(
                tbl=cls.tf_tbls[timeframe],
                sid=sid
            )
            return conn.execute(sa.text(stmt))
        else:
            # 时间帧没有直接符合的，从最接近的子timeframe聚合
            tf_secs = tf_to_secs(timeframe)
            sub_tf, sub_tbl = None, None
            for stf, stbl, stf_secs in cls.tf_list[::-1]:
                if tf_secs % stf_secs == 0:
                    sub_tf, sub_tbl = stf, stbl
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
              limit: int = 3000):
        tf_secs = tf_to_secs(timeframe)
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
                  first(open, time) AS open,  
                  max(high) AS high,
                  min(low) AS low, 
                  last(close, time) AS close,
                  sum(volume) AS volume
                  from {{tbl}}
                where sid={{sid}} and time >= to_timestamp({start_ts}) and time < to_timestamp({end_ts})
                group by time
                order by time'''
        rows = cls._query_hyper(exg_name, pair, timeframe, dct_sql, gen_gp_sql).fetchall()
        if not len(rows) and max_end_ms - end_ms > tf_secs * 1000:
            rows = cls.query(exg_name, pair, timeframe, end_ms, max_end_ms, limit)
        return rows

    @classmethod
    def query_range(cls, exg_name: str, symbol: Union[str, int]) -> Tuple[Optional[int], Optional[int]]:
        sid = cls._get_sid(exg_name, symbol)
        cache_val = cls._sid_range_map.get(sid)
        if not cache_val:
            dct_sql = 'select (extract(epoch from min(time)) * 1000)::float, ' \
                      '(extract(epoch from max(time)) * 1000)::float from {tbl} where sid={sid}'
            min_time, max_time = cls._query_hyper(exg_name, sid, '1m', dct_sql, '').fetchone()
            cls._sid_range_map[sid] = min_time, max_time
        else:
            min_time, max_time = cache_val
        return min_time, max_time

    @classmethod
    def _update_range(cls, sid: int, start_ms: int, end_ms: int):
        if sid not in cls._sid_range_map:
            cls._sid_range_map[sid] = start_ms, end_ms
        else:
            old_start, old_stop = cls._sid_range_map[sid]
            if old_start and old_stop:
                old_start -= cls.interval
                old_stop += cls.interval
                if old_stop < start_ms or end_ms < old_start:
                    raise ValueError(f'incontinus range, sid: {sid}, old range: [{old_start}, {old_stop}], '
                                     f'new: [{start_ms}, {end_ms}]')
                cls._sid_range_map[sid] = min(start_ms, old_start), max(end_ms, old_stop)
            else:
                cls._sid_range_map[sid] = start_ms, end_ms

    @classmethod
    def insert(cls, exg_name: str, pair: str, rows: List[Tuple]):
        if not rows:
            return
        rows = sorted(rows, key=lambda x: x[0])
        if len(rows) > 1:
            # 检查是否是1分钟间隔
            row_interval = rows[1][0] - rows[0][0]
            if row_interval != cls.interval:
                raise ValueError(f'insert kline must be 1m interval, current: {row_interval/1000:.1f}s')
        sid = cls._get_sid(exg_name, pair)
        start_ms, end_ms = rows[0][0], rows[-1][0]
        old_start, old_stop = cls.query_range(exg_name, sid)
        if old_start and old_stop:
            # 插入的数据应该和已有数据连续，避免出现空洞。
            old_start -= cls.interval
            old_stop += cls.interval
            if old_stop < start_ms or end_ms < old_start:
                raise ValueError(f'insert incontinus data, sid: {sid}, old range: [{old_start}, {old_stop}], '
                                 f'insert: [{start_ms}, {end_ms}]')
        sess = db.session
        ins_rows = []
        for r in rows:
            row_ts = btime.to_datetime(r[0])
            ins_rows.append(dict(
                sid=sid, time=row_ts, open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5],
            ))
        sess.bulk_insert_mappings(KLine, ins_rows)
        sess.commit()
        # 更新区间
        cls._update_range(sid, start_ms, end_ms)
        # 刷新连续聚合
        range_ms = (end_ms - start_ms) / 1000
        start_dt = btime.to_datetime(start_ms)
        conn = sess.connection()
        conn.commit()
        bak_iso_level = conn.get_isolation_level()
        conn.execution_options(isolation_level='AUTOCOMMIT')
        for intv in cls.agg_intvs:
            tf = intv[0]
            tf_secs = tf_to_secs(tf)
            if range_ms < tf_secs * 0.3:
                continue
            cur_end_ms = int(end_ms // tf_secs * tf_secs) + tf_secs * 1000 + 1
            cur_end_dt = btime.to_datetime(cur_end_ms)
            stmt = f"CALL refresh_continuous_aggregate('kline_{tf}', '{start_dt}', '{cur_end_dt}');"
            conn.execute(sa.text(stmt))
        conn.commit()
        conn.execution_options(isolation_level=bak_iso_level)

    @classmethod
    def _find_sid_hole(cls, conn: sa.Connection, sid: int, since: float, until: float = 0., as_ts=True):
        batch_size, true_intv = 1000, cls.interval / 1000
        last_date = None
        res_holes = []
        while True:
            hole_sql = f"SELECT time FROM kline where sid={sid} and time >= to_timestamp({since}) "
            if until:
                hole_sql += f'and time < to_timestamp({until}) '
            hole_sql += f"ORDER BY time limit {batch_size};"
            rows = conn.execute(sa.text(hole_sql)).fetchall()
            if not len(rows):
                break
            off_idx = 0
            if last_date is None:
                last_date = rows[0][0]
                off_idx = 1
            for row in rows[off_idx:]:
                cur_intv = (row[0] - last_date).total_seconds()
                if cur_intv > true_intv:
                    if as_ts:
                        hole_start, hold_end = btime.to_utcstamp(last_date), btime.to_utcstamp(row[0])
                    else:
                        hole_start, hold_end = last_date, row[0]
                    res_holes.append((hole_start, hold_end))
                elif cur_intv < true_intv:
                    logger.warning(f'invalid kline interval: {cur_intv:.3f}, sid: {sid}')
                last_date = row[0]
            since = btime.to_utcstamp(last_date) + 0.1
        return res_holes

    @classmethod
    async def fill_holes(cls):
        from banbot.data.tools import download_to_db
        from banbot.storage.symbols import SymbolTF
        from banbot.exchange.crypto_exchange import get_exchange
        logger.info('find and fill holes for kline...')
        sess = db.session
        conn = sess.connection()
        gp_sql = 'SELECT sid, extract(epoch from min(time))::float as mtime FROM kline GROUP BY sid;'
        sid_rows = conn.execute(sa.text(gp_sql)).fetchall()
        if not len(sid_rows):
            return
        for sid, start_ts in sid_rows:
            hole_list = cls._find_sid_hole(conn, sid, start_ts)
            if not hole_list:
                continue
            old_holes = sess.query(KHole).filter(KHole.sid == sid).all()
            old_holes = [(btime.to_utcstamp(row.start), btime.to_utcstamp(row.stop)) for row in old_holes]
            hole_list = [h for h in hole_list if h not in old_holes]
            if not hole_list:
                continue
            stf: SymbolTF = sess.query(SymbolTF).get(sid)
            exchange = get_exchange(stf.exchange)
            for hole in hole_list:
                start_ms, end_ms = int(hole[0] * 1000) + 1, int(hole[1] * 1000)
                logger.warning(f'filling hole: {stf.symbol}, {start_ms} - {end_ms}')
                await download_to_db(exchange, stf.symbol, start_ms, end_ms, check_exist=False)
                real_holes = cls._find_sid_hole(conn, sid, hole[0], hole[1] + 0.1, as_ts=False)
                if not real_holes:
                    continue
                logger.warning(f'REAL Holes: {stf.symbol} {real_holes}')
                for rhole in real_holes:
                    sess.add(KHole(sid=sid, start=rhole[0], stop=rhole[1]))
                sess.commit()


class KHole(BaseDbModel):
    '''
    交易所的K线数据可能有时因公司业务或系统故障等原因，出现空洞；记录到这里避免重复下载
    '''
    __tablename__ = 'khole'

    sid = Column(sa.Integer, primary_key=True)
    start = Column(sa.DateTime)
    stop = Column(sa.DateTime)
