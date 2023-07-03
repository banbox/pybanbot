#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : klines.py
# Author: anyongjin
# Date  : 2023/4/24
import math
from datetime import datetime
from typing import Callable, Tuple, ClassVar

from banbot.exchange.exchange_utils import tf_to_secs
from banbot.storage.base import *
from banbot.util import btime
from banbot.storage.symbols import ExSymbol


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
            f'CREATE INDEX "{tbl.tbl}_sid" ON "{tbl.tbl}" USING btree ("sid");',
            f"SELECT create_hypertable('{tbl.tbl}','time');",
            f'''ALTER TABLE {tbl.tbl} SET (
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
        # 创建未完成kline表，存储所有币的所有周期未完成bar
        un_tname = 'kline_un'
        exc_sqls = [
            f'CREATE TABLE {un_tname} (LIKE {cls._tname} INCLUDING ALL);',
            f'alter table {un_tname} add column timeframe varchar(30);',
            f'CREATE INDEX "{un_tname}_tf" ON "{un_tname}" USING btree ("timeframe");',
            f"SELECT create_hypertable('{un_tname}','time');",
            # sqlalchemy创建的表，附带主键默认是类名+"_pkey"后缀
            f"ALTER TABLE {un_tname} DROP CONSTRAINT kline_un_pkey;",
            f"ALTER TABLE {un_tname} ADD CONSTRAINT kline_un_pkey PRIMARY KEY (sid, time, timeframe);"
        ]
        for stat in exc_sqls:
            conn.execute(sa.text(stat))
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
    def _query_hyper(cls, timeframe: str, dct_sql: str, gp_sql: Union[str, Callable], **kwargs):
        conn = db.session.connection()
        if timeframe in cls.agg_map:
            stmt = dct_sql.format(tbl=cls.agg_map[timeframe].tbl, **kwargs)
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
            stmt = gp_sql.format(tbl=sub_tbl, **kwargs)
            return conn.execute(sa.text(stmt))

    @classmethod
    def query(cls, exs: ExSymbol, timeframe: str, start_ms: int, end_ms: int,
              limit: Optional[int] = None, with_unfinish: bool = False):
        tf_msecs = tf_to_secs(timeframe) * 1000
        max_end_ms = end_ms
        if limit:
            end_ms = min(start_ms + tf_msecs * limit, end_ms)

        start_ts, end_ts = start_ms / 1000, end_ms / 1000

        dct_sql = f'''
select (extract(epoch from time) * 1000)::float as time,open,high,low,close,volume from {{tbl}}
where sid={{sid}} and time >= to_timestamp({start_ts}) and time < to_timestamp({end_ts})
order by time'''

        def gen_gp_sql():
            return f'''
                select (extract(epoch from time_bucket('{timeframe}', time)) * 1000)::float AS gtime,
                  {cls._candle_agg} from {{tbl}}
                where sid={{sid}} and time >= to_timestamp({start_ts}) and time < to_timestamp({end_ts})
                group by gtime
                order by gtime'''

        rows = cls._query_hyper(timeframe, dct_sql, gen_gp_sql, sid=exs.id).fetchall()
        rows = [list(r) for r in rows]
        max_finish_ts = (btime.utcstamp() // tf_msecs - 1) * tf_msecs
        if not len(rows) and max_end_ms - end_ms > tf_msecs:
            rows = cls.query(exs, timeframe, end_ms, max_end_ms, limit)
        elif with_unfinish and rows and rows[-1][0] == max_finish_ts:
            unfinish_ts = (max_finish_ts + tf_msecs) // 1000
            dct_sql = f'''
            select (extract(epoch from time) * 1000)::float as time,open,high,low,close,volume from kline_un
            where sid={exs.id} and timeframe='{timeframe}' and time >= to_timestamp({unfinish_ts})
            order by time limit 1'''
            conn = db.session.connection()
            row = conn.execute(sa.text(dct_sql)).fetchone()
            if row:
                rows.append(list(row))
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
        min_time, max_time = cls._query_hyper(timeframe, dct_sql, '', sid=sid).fetchone()
        if max_time:
            # 这里记录蜡烛对应的结束时间
            max_time += tf_to_secs(timeframe) * 1000
            min_time, max_time = round(min_time), round(max_time)
        cache_key = sid, timeframe
        cls._sid_range_map[cache_key] = min_time, max_time
        return min_time, max_time

    @classmethod
    def query_range(cls, sid: int, timeframe: str)\
            -> Tuple[Optional[int], Optional[int]]:
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
    def get_latest_stamps(cls, timeframe: str, max_off: int = 100) -> Dict[int, float]:
        '''
        获取某个时间周期，所以交易对的最新一个时间戳。只查询前max_off * timeframe段时间，避免数据太多性能太差
        '''
        conn = db.session.connection()
        stop_ts = btime.utcstamp() / 1000
        start_ts = stop_ts - tf_to_secs(timeframe) * max_off
        tbl = cls.agg_map[timeframe].tbl
        dct_sql = f'''
select distinct on(sid) sid, (extract(epoch from time) * 1000)::float from {tbl}
where "time" > to_timestamp({start_ts}) and "time" < to_timestamp({stop_ts}) 
ORDER BY sid, "time" desc'''
        rows = conn.execute(sa.text(dct_sql)).fetchall()
        return {r[0]: r[1] for r in rows}

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
    def insert(cls, sid: int, timeframe: str, rows: List[Tuple], skip_in_range=True):
        if not rows:
            return
        if timeframe not in {'1m', '1h'}:
            raise RuntimeError(f'can only insert kline: 1m or 1h, current: {timeframe}')
        intv_msecs = tf_to_secs(timeframe) * 1000
        rows = sorted(rows, key=lambda x: x[0])
        if len(rows) > 1:
            # 检查间隔是否正确
            row_intv = rows[1][0] - rows[0][0]
            if row_intv != intv_msecs:
                raise ValueError(f'insert kline must be {timeframe} interval, current: {row_intv} s')
        start_ms, end_ms = rows[0][0], rows[-1][0] + intv_msecs
        old_start, old_stop = cls.query_range(sid, timeframe)
        if old_start and old_stop:
            # 插入的数据应该和已有数据连续，避免出现空洞。
            if old_stop < start_ms or end_ms < old_start:
                raise ValueError(f'insert incontinus data, sid: {sid}, old range: [{old_start}, {old_stop}], '
                                 f'insert: [{start_ms}, {end_ms}]')
            if skip_in_range:
                rows = [r for r in rows if not (old_start <= r[0] < old_stop)]
        if not rows:
            return
        ins_rows = []
        for r in rows:
            row_ts = btime.to_datetime(r[0])
            ins_rows.append(dict(
                sid=sid, time=row_ts, open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5],
            ))
        if not ins_rows:
            return
        sess = db.session
        ins_tbl = cls.agg_map[timeframe].tbl
        ins_cols = "sid, time, open, high, low, close, volume"
        places = ":sid, :time, :open, :high, :low, :close, :volume"
        insert_sql = f"insert into {ins_tbl} ({ins_cols}) values ({places}) {cls._insert_conflict}"
        try:
            sess.execute(sa.text(insert_sql), ins_rows)
        except Exception as e:
            if str(e).find('not supported on compressed chunks') >= 0:
                logger.error(f"insert compressed, call `pause_compress` first, {ins_tbl} sid:{sid} {start_ms}-{end_ms}")
            else:
                raise
        sess.commit()
        # 更新区间
        cls._update_range(sid, timeframe, start_ms, end_ms)
        # 刷新相关的连续聚合
        cls._refresh_conti_agg(sid, timeframe, start_ms, end_ms)

    @classmethod
    def pause_compress(cls, tbl_list: List[str]) -> List[int]:
        sess = db.session
        result = []
        for tbl in tbl_list:
            get_job_id = f"""
    SELECT j.job_id FROM timescaledb_information.jobs j
    WHERE j.proc_name = 'policy_compression' AND j.hypertable_name = '{tbl}'"""
            job_id = sess.execute(sa.text(get_job_id)).scalar()
            if job_id:
                # 暂停压缩任务
                sess.execute(sa.text(f'SELECT alter_job({job_id}, scheduled => false);'))
                # 解压缩涉及的块
                decps_sql = f'''SELECT decompress_chunk(i, true) FROM show_chunks('{tbl}') i ;'''
                sess.execute(sa.text(decps_sql))
                result.append(job_id)
            else:
                logger.warning(f"no compress job id found {tbl}")
        sess.commit()
        return result

    @classmethod
    def restore_compress(cls, jobs: List[int]):
        if not jobs:
            return
        sess = db.session
        for job_id in jobs:
            # 启动压缩任务（不会立刻执行压缩）
            sess.execute(sa.text(f'SELECT alter_job({job_id}, scheduled => true);'))
        sess.commit()

    @classmethod
    def _refresh_conti_agg(cls, sid: int, from_level: str, start_ms: int, end_ms: int):
        agg_keys = {from_level}
        from_secs = tf_to_secs(from_level)
        for item in cls.agg_list:
            if item.secs <= from_secs or item.agg_from not in agg_keys:
                # 跳过过小维度；跳过无关的连续聚合
                continue
            start_align = start_ms // 1000 // item.secs * item.secs
            end_align = end_ms // 1000 // item.secs * item.secs
            if start_align == end_align < start_ms // 1000:
                # 没有出现新的bar数据，无需更新
                # 前2个相等，说明：插入的数据所属bar尚未完成。
                # start_align < start_ms说明：插入的数据不是所属bar的第一个数据
                cls._update_unfinish(item, sid, start_align, end_ms)
            else:
                agg_keys.add(item.tf)
        agg_keys.remove(from_level)
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
    def _update_unfinish(cls, item: BarAgg, sid: int, start_align: int, end_ms: int):
        sess = db.session
        sess.execute(sa.text(f"DELETE from kline_un where sid={sid} and timeframe='{item.tf}';"))
        # 从已完成的子周期中统计筛选数据
        agg_from = 'kline_' + item.agg_from
        select_sql = f'''
                        SELECT sid, time_bucket(INTERVAL '{item.tf}', time) AS time, {cls._candle_agg}
                        FROM {agg_from}
                        where sid={sid} and time >= to_timestamp({start_align}) and time < to_timestamp({end_ms / 1000})
                        GROUP BY sid, 2 
                        ORDER BY sid, 2 limit 1'''
        finish_row = sess.execute(sa.text(select_sql)).fetchone()
        merge_rows = [tuple(finish_row)] if finish_row else []
        # 从未完成的自周期中查询bar
        unfinish = sess.execute(sa.text(f'''
                        SELECT sid,time,open,high,low,close,volume FROM kline_un
                        where sid={sid} and timeframe='{item.tf}' and time >= to_timestamp({start_align})
                        limit 1''')).fetchone()
        if unfinish:
            merge_rows.append(tuple(unfinish))
        if not merge_rows:
            sess.commit()
            return
        if len(merge_rows) > 1:
            mg_cols = list(zip(*merge_rows))
            cur_bar = (mg_cols[0][0], mg_cols[1][0], mg_cols[2][0], max(mg_cols[3]), min(mg_cols[4]),
                           mg_cols[5][-1], sum(mg_cols[6]))
        else:
            cur_bar = merge_rows[0]
        ins_cols = "sid, time, open, high, low, close, volume, timeframe"
        bar_time = btime.to_utcstamp(cur_bar[1])
        places = f"{cur_bar[0]}, to_timestamp({bar_time}), {cur_bar[2]}, {cur_bar[3]}, {cur_bar[4]}, {cur_bar[5]}, {cur_bar[6]}, '{item.tf}'"
        insert_sql = f"insert into kline_un ({ins_cols}) values ({places})"
        sess.execute(sa.text(insert_sql))
        sess.commit()

    @classmethod
    def _refresh_agg(cls, conn: Union[SqlSession, sa.Connection], tbl: BarAgg, sid: int,
                     start_ms: int, end_ms: int, agg_from: str = None):
        tf_msecs = tbl.secs * 1000
        # 有可能start_ms刚好是下一个bar的开始，前一个需要-1
        start_ms = (start_ms // tf_msecs - 1) * tf_msecs
        end_ms = end_ms // tf_msecs * tf_msecs
        old_start, old_end = cls.query_range(sid, tbl.tf)
        if old_start and old_end > old_start:
            # 避免出现空洞或数据错误
            start_ms = min(start_ms, old_end)
            end_ms = max(end_ms, old_start)
        if not agg_from:
            agg_from = 'kline_' + tbl.agg_from
        win_start = f"'{btime.to_datetime(start_ms)}'"
        win_end = f"'{btime.to_datetime(end_ms)}'"
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
    def log_candles_conts(cls, exs: ExSymbol, timeframe: str, start_ms: int, end_ms: int, candles: list):
        '''
        检查下载的蜡烛数据是否连续，如不连续记录到khole中
        '''
        tf_msecs = tf_to_secs(timeframe) * 1000
        # 取 >= start_ms的第一个bar开始时间
        start_ms = math.ceil(start_ms / tf_msecs) * tf_msecs
        # 取 < end_ms的最后一个bar开始时间
        end_ms = (math.ceil(end_ms / tf_msecs) - 1) * tf_msecs
        if start_ms > end_ms:
            return
        until_ms = end_ms + tf_msecs
        holes = []
        if not candles:
            holes = [(start_ms, until_ms)]
        else:
            if candles[0][0] > start_ms:
                holes.append((start_ms, candles[0][0]))
            prev_date = candles[0][0]
            for row in candles[1:]:
                cur_intv = row[0] - prev_date
                if cur_intv > tf_msecs:
                    holes.append((prev_date + tf_msecs, row[0]))
                elif cur_intv < tf_msecs:
                    logger.warning(f'invalid kline interval: {cur_intv:.3f}, {exs}/{timeframe}')
                prev_date = row[0]
            if prev_date != end_ms:
                holes.append((prev_date + tf_msecs, until_ms))
        if not holes:
            return
        sid = exs.id
        holes = [(btime.to_datetime(h[0]), btime.to_datetime(h[1])) for h in holes]
        holes = [KHole(sid=sid, timeframe=timeframe, start=h[0], stop=h[1]) for h in holes]
        sess = db.session
        old_holes: List[KHole] = sess.query(KHole).filter(KHole.sid == sid, KHole.timeframe == timeframe).all()
        holes.extend(old_holes)
        holes.sort(key=lambda x: x.start)
        merged: List[KHole] = []
        for h in holes:
            if not merged or merged[-1].stop < h.start:
                merged.append(h)
            else:
                prev = merged[-1]
                old_h, hole = prev, h
                if h.id:
                    old_h, hole = h, prev
                if hole.id:
                    sess.delete(hole)
                if hole.stop > old_h.stop:
                    old_h.stop = hole.stop
                merged[-1] = old_h
        for m in merged:
            if not m.id:
                sess.add(m)
        sess.commit()

    @classmethod
    def _find_sid_hole(cls, sess: SqlSession, timeframe: str, sid: int, since: datetime, until: datetime = None):
        tbl = f'kline_{timeframe}'
        batch_size, true_intv = 1000, tf_to_secs(timeframe) * 1.
        intv_delta = btime.timedelta(seconds=true_intv)
        prev_date = None
        res_holes = []
        hole_sql = f"SELECT time FROM {tbl} where sid={sid} and time >= :since "
        if until:
            hole_sql += f'and time < :stop '
        hole_sql += f"ORDER BY time limit {batch_size};"
        while True:
            args = dict(since=since, stop=until) if until else dict(since=since)
            rows = sess.execute(sa.text(hole_sql), args).fetchall()
            if not len(rows):
                break
            off_idx = 0
            if prev_date is None:
                prev_date = rows[0][0]
                off_idx = 1
            for row in rows[off_idx:]:
                cur_intv = (row[0] - prev_date).total_seconds()
                if cur_intv > true_intv:
                    res_holes.append((prev_date + intv_delta, row[0]))
                elif cur_intv < true_intv:
                    logger.warning(f'invalid kline interval: {cur_intv:.3f}, sid: {sid}')
                prev_date = row[0]
            since = prev_date + btime.timedelta(seconds=0.1)
        return res_holes

    @classmethod
    async def _fill_tf_hole(cls, timeframe: str):
        from banbot.data.tools import download_to_db
        from banbot.storage.symbols import ExSymbol
        from banbot.exchange.crypto_exchange import get_exchange
        sess = db.session
        tbl = f'kline_{timeframe}'
        gp_sql = f'SELECT sid, min(time) FROM {tbl} GROUP BY sid;'
        sid_rows = sess.execute(sa.text(gp_sql)).fetchall()
        if not len(sid_rows):
            return
        down_tf = KLine.get_down_tf(timeframe)
        down_tfsecs = tf_to_secs(down_tf)
        for sid, start_dt in sid_rows:
            hole_list = cls._find_sid_hole(sess, timeframe, sid, start_dt)
            if not hole_list:
                continue
            fts = [KHole.sid == sid, KHole.timeframe == timeframe]
            old_holes = sess.query(KHole).filter(*fts).order_by(KHole.start).all()
            res_holes = []
            for h in hole_list:
                start, stop = get_unknown_range(h[0], h[1], old_holes)
                if start < stop:
                    res_holes.append((start, stop))
            if not res_holes:
                continue
            stf: ExSymbol = sess.query(ExSymbol).get(sid)
            exchange = get_exchange(stf.exchange, stf.market)
            for hole in res_holes:
                start_dt, end_dt = btime.to_datestr(hole[0]), btime.to_datestr(hole[1])
                logger.warning(f'filling hole: {stf.symbol}, {start_dt} - {end_dt}')
                start_ms = btime.to_utcstamp(hole[0], True, True)
                end_ms = btime.to_utcstamp(hole[1], True, True)
                sub_arr = cls.query(stf, down_tf, start_ms, end_ms)
                true_len = (end_ms - start_ms) // down_tfsecs // 1000
                if true_len == len(sub_arr):
                    # 子维度周期数据存在，直接归集更新
                    cls._refresh_agg(sess, cls.agg_map[timeframe], sid, start_ms, end_ms, f'kline_{down_tf}')
                else:
                    logger.info(f'db sub tf {down_tf} not enough {len(sub_arr)} < {true_len}, downloading..')
                    await download_to_db(exchange, stf, down_tf, start_ms, end_ms, check_exist=False)

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
        if cost > 2:
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
    start = Column(sa.DateTime)  # 从第一个缺失的bar时间戳记录
    stop = Column(sa.DateTime)  # 记录到最后一个确实的bar的结束时间戳（即下一个有效bar的时间戳）

    @classmethod
    def get_down_range(cls, exs: ExSymbol, timeframe: str, start_ms: int, stop_ms: int) -> Tuple[int, int]:
        sess = db.session
        start = btime.to_datetime(start_ms)
        stop = btime.to_datetime(stop_ms)
        fts = [KHole.sid == exs.id, KHole.timeframe == timeframe, KHole.stop > start, KHole.start < stop]
        holes: List[KHole] = sess.query(KHole).filter(*fts).order_by(KHole.start).all()
        start, stop = get_unknown_range(start, stop, holes)
        if start >= stop:
            return 0, 0
        return btime.to_utcstamp(start, True, True), btime.to_utcstamp(stop, True, True)


def get_unknown_range(start, stop, holes: List[KHole]):
    for h in holes:
        if h.start <= start:
            start = max(start, h.stop)
        elif h.stop >= stop:
            stop = min(stop, h.start)
        if start >= stop:
            return start, stop
    return start, stop
