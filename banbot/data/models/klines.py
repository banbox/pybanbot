#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : klines.py
# Author: anyongjin
# Date  : 2023/4/24
import six

from banbot.data.models.base import *
from typing import Callable, Tuple, ClassVar, Dict
from banbot.exchange.exchange_utils import timeframe_to_seconds
from banbot.util import btime


class KLine(BaseDbModel):
    __tablename__ = 'kline'
    _sid_range_map: ClassVar[Dict[int, Tuple[int, int]]] = dict()
    _interval: ClassVar[int] = 60000  # 每行是1m维度

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
    def _get_sid(cls, symbol: Union[str, int]):
        sid = symbol
        if isinstance(symbol, six.string_types):
            from banbot.data.models.symbols import SymbolTF
            sid = SymbolTF.get_id(symbol)
        return sid

    @classmethod
    def _query_hyper(cls, symbol: Union[str, int], timeframe: str, dct_sql: str, gp_sql: Union[str, Callable]):
        sid = cls._get_sid(symbol)
        conn = db_conn()
        if timeframe in cls.tf_tbls:
            stmt = dct_sql.format(
                tbl=cls.tf_tbls[timeframe],
                sid=sid
            )
            return conn.execute(sa.text(stmt))
        else:
            # 时间帧没有直接符合的，从最接近的子timeframe聚合
            tf_secs = timeframe_to_seconds(timeframe)
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
    def query(cls, pair: str, timeframe: str, start_ms: int, end_ms: int):
        dct_sql = f'''
select time,open,high,low,close,volume from {{tbl}}
where sid={{sid}} and time >= {start_ms} and time < {end_ms}
order by time'''

        def gen_gp_sql():
            tf_secs = timeframe_to_seconds(timeframe)
            big_start_ms = int(start_ms // tf_secs * tf_secs)
            return f'''
                select time_bucket('{timeframe}', time) AS time,
                  first(open, time) AS open,  
                  max(high) AS high,
                  min(low) AS low, 
                  last(close, time) AS close,
                  sum(volume) AS volume
                  from {{tbl}}
                where sid={{sid}} and time >= {big_start_ms} and time < {end_ms}
                group by time
                order by time'''
        rows = cls._query_hyper(pair, timeframe, dct_sql, gen_gp_sql).fetchall()
        return [r for r in rows]

    @classmethod
    def query_range(cls, symbol: Union[str, int]) -> Tuple[Optional[int], Optional[int]]:
        sid = cls._get_sid(symbol)
        cache_val = cls._sid_range_map.get(sid)
        if not cache_val:
            dct_sql = 'select min(time), max(time) from {tbl} where sid={sid}'
            min_time, max_time = cls._query_hyper(sid, '1m', dct_sql, '').fetchone()
            if min_time is not None and max_time is not None:
                min_time = btime.to_utcstamp(min_time, ms=True, round_int=True)
                max_time = btime.to_utcstamp(max_time, ms=True, round_int=True)
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
                old_start -= cls._interval
                old_stop += cls._interval
                if old_stop < start_ms or end_ms < old_start:
                    raise ValueError(f'incontinus range, sid: {sid}, old range: [{old_start}, {old_stop}], '
                                     f'new: [{start_ms}, {end_ms}]')
                cls._sid_range_map[sid] = min(start_ms, old_start), max(end_ms, old_stop)
            else:
                cls._sid_range_map[sid] = start_ms, end_ms

    @classmethod
    def insert(cls, pair: str, rows: List[Tuple]):
        if not rows:
            return
        rows = sorted(rows, key=lambda x: x[0])
        if len(rows) > 1:
            # 检查是否是1分钟间隔
            row_interval = rows[1][0] - rows[0][0]
            if row_interval != cls._interval:
                raise ValueError(f'insert kline must be 1m interval, current: {row_interval/1000:.1f}s')
        sid = cls._get_sid(pair)
        start_ms, end_ms = rows[0][0], rows[-1][0]
        old_start, old_stop = cls.query_range(sid)
        if old_start and old_stop:
            # 插入的数据应该和已有数据连续，避免出现空洞。
            old_start -= cls._interval
            old_stop += cls._interval
            if old_stop < start_ms or end_ms < old_start:
                raise ValueError(f'insert incontinus data, sid: {sid}, old range: [{old_start}, {old_stop}], '
                                 f'insert: [{start_ms}, {end_ms}]')
        sess = db_sess()
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
        conn = sess.connection(execution_options=dict(isolation_level="AUTOCOMMIT"))
        for intv in cls.agg_intvs:
            tf = intv[0]
            tf_secs = timeframe_to_seconds(tf)
            if range_ms < tf_secs * 0.3:
                continue
            cur_end_ms = int(end_ms // tf_secs * tf_secs) + tf_secs * 1000 + 1
            cur_end_dt = btime.to_datetime(cur_end_ms)
            stmt = f"CALL refresh_continuous_aggregate('kline_{tf}', '{start_dt}', '{cur_end_dt}');"
            conn.execute(sa.text(stmt))
        conn.commit()
        conn.close()
