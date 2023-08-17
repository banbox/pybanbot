#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : klines.py
# Author: anyongjin
# Date  : 2023/4/24
import math
import collections
import asyncio
from typing import Callable, Tuple, ClassVar, Iterable, Deque

from banbot.exchange.exchange_utils import tf_to_secs
from banbot.storage.base import *
from banbot.util import btime
from banbot.storage.symbols import ExSymbol
from asyncio import Future


class DisContiError(Exception):
    '''
    插入K线时不连续错误
    '''
    def __init__(self, key, old_rg: Tuple[int, int], new_rg: Tuple[int, int]):
        super(DisContiError, self).__init__()
        self.key = key
        self.old_rg = old_rg
        self.new_rg = new_rg

    def __str__(self):
        return f'incontinus insert: {self.key}, old: {self.old_rg}, new: {self.new_rg}'

    def __repr__(self):
        return f'incontinus insert: {self.key}, old: {self.old_rg}, new: {self.new_rg}'


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

    def has_finish(self, start_ms: int, end_ms: int):
        start_align = start_ms // 1000 // self.secs * self.secs
        end_align = end_ms // 1000 // self.secs * self.secs
        # 没有出现新的完成的bar数据，无需更新
        # 前2个相等，说明：插入的数据所属bar尚未完成。
        # start_align < start_ms说明：插入的数据不是所属bar的第一个数据
        return not (start_align == end_align < start_ms // 1000)

    def __str__(self):
        return f'{self.tbl} agg_from: {self.agg_from}'


class KLine(BaseDbModel):
    '''
    K线数据超表，存储1m维度数据
    还有一个超表kline_1h存储1h维度数据
    '''
    __tablename__ = 'kline_1m'
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

    down_tfs = {'1m', '1h', '1d'}

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

    _listeners: Dict[str, Deque[Future]] = dict()
    '监听蜡烛插入数据库的消费者队列'

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
        # 创建未完成kline表，存储所有币的所有周期未完成bar；不需要是超表
        exc_sqls = [
            'DROP TABLE IF EXISTS "kline_un";',
            '''
CREATE TABLE "kline_un" (
  "sid" int4 NOT NULL,
  "start_ms" bigint NOT NULL,
  "stop_ms" bigint NOT NULL,
  "timeframe" varchar(5) NOT NULL,
  "open" float8,
  "high" float8,
  "low" float8,
  "close" float8,
  "volume" float8
);''',
            'CREATE INDEX "kline_un_sid_tf_idx" ON "kline_un"("sid", "timeframe");',
            "ALTER TABLE kline_un ADD CONSTRAINT pk_kline_un PRIMARY KEY (sid, start_ms, timeframe);"
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
    def _get_sub_tf(cls, timeframe: str) -> Tuple[str, str]:
        tf_secs = tf_to_secs(timeframe)
        for item in cls.agg_list[::-1]:
            if item.secs >= tf_secs:
                continue
            if tf_secs % item.secs == 0:
                return item.tf, item.tbl
        raise RuntimeError(f'unsupport timeframe {timeframe}')

    @classmethod
    def _query_hyper(cls, timeframe: str, dct_sql: str, gp_sql: Union[str, Callable], **kwargs):
        conn = db.session.connection()
        if timeframe in cls.agg_map:
            stmt = dct_sql.format(tbl=cls.agg_map[timeframe].tbl, **kwargs)
            return conn.execute(sa.text(stmt))
        else:
            # 时间帧没有直接符合的，从最接近的子timeframe聚合
            sub_tf, sub_tbl = cls._get_sub_tf(timeframe)
            if callable(gp_sql):
                gp_sql = gp_sql()
            stmt = gp_sql.format(tbl=sub_tbl, **kwargs)
            return conn.execute(sa.text(stmt))

    @classmethod
    def query(cls, exs: ExSymbol, timeframe: str, start_ms: int, end_ms: int,
              limit: Optional[int] = None, with_unfinish: bool = False):
        tf_secs = tf_to_secs(timeframe)
        tf_msecs = tf_secs * 1000
        max_end_ms = end_ms
        if limit:
            end_ms = min(start_ms + tf_msecs * limit, end_ms)

        start_ts, end_ts = start_ms / 1000, end_ms / 1000
        # 计算最新未完成bar的时间戳
        finish_end_ts = end_ts // tf_secs * tf_secs
        unfinish_ts = int(btime.utctime() // tf_secs * tf_secs)
        if finish_end_ts > unfinish_ts:
            finish_end_ts = unfinish_ts

        dct_sql = f'''
select (extract(epoch from time) * 1000)::bigint as time,open,high,low,close,volume from {{tbl}}
where sid={{sid}} and time >= to_timestamp({start_ts}) and time < to_timestamp({finish_end_ts})
order by time'''

        def gen_gp_sql():
            return f'''
                select (extract(epoch from time_bucket('{timeframe}', time, origin => '1970-01-01')) * 1000)::bigint AS gtime,
                  {cls._candle_agg} from {{tbl}}
                where sid={{sid}} and time >= to_timestamp({start_ts}) and time < to_timestamp({finish_end_ts})
                group by gtime
                order by gtime'''

        rows = cls._query_hyper(timeframe, dct_sql, gen_gp_sql, sid=exs.id).fetchall()
        rows = [list(r) for r in rows]
        if not len(rows) and max_end_ms - end_ms > tf_msecs:
            rows = cls.query(exs, timeframe, end_ms, max_end_ms, limit)
        elif with_unfinish and rows and rows[-1][0] // 1000 + tf_secs == unfinish_ts:
            un_bar, _ = cls._get_unfinish(exs.id, timeframe, unfinish_ts, unfinish_ts + tf_secs)
            if un_bar:
                rows.append(list(un_bar))
        return rows

    @classmethod
    def _recalc_ranges(cls, *tf_list: str) -> Dict[Tuple[int, str], Tuple[int, int]]:
        dct_sql = '''
select sid,
(extract(epoch from min(time)) * 1000)::bigint, 
(extract(epoch from max(time)) * 1000)::bigint 
from {tbl}
group by 1'''
        sess = db.session
        if not tf_list:
            tf_list = [item.tf for item in cls.agg_list]
        # 删除旧的kinfo
        result = dict()
        tf_texts = ', '.join([f"'{tf}'" for tf in tf_list])
        del_sql = f"delete from kinfo where timeframe in ({tf_texts})"
        sess.execute(sa.text(del_sql))
        for tf in tf_list:
            sql_text = dct_sql.format(tbl=f'kline_{tf}')
            rows = sess.execute(sa.text(sql_text)).fetchall()
            for sid, min_time, max_time in rows:
                cache_key = sid, tf
                # 这里记录蜡烛对应的结束时间
                max_time += tf_to_secs(tf) * 1000
                min_time, max_time = int(min_time), int(max_time)
                result[cache_key] = min_time, max_time
                sess.add(KInfo(sid=sid, timeframe=tf, start=min_time, stop=max_time))
        sess.commit()
        return result

    @classmethod
    def load_kline_ranges(cls) -> Dict[Tuple[int, str], Tuple[int, int]]:
        sess = db.session
        rows: Iterable[KInfo] = sess.query(KInfo).all()
        if not rows:
            return cls._recalc_ranges()
        result = dict()
        for row in rows:
            cache_key = row.sid, row.timeframe
            result[cache_key] = row.start, row.stop
        return result

    @classmethod
    def _update_range(cls, sid: int, timeframe: str, start_ms: int, end_ms: int, force_new: bool = False)\
            -> Tuple[int, int]:
        '''
        更新sid+timeframe对应的数据区间。end_ms应为最后一个bar对应的结束时间，而非开始时间
        :param force_new: 是否强制刷新范围后，再尝试更新
        '''
        cache_key = sid, timeframe
        if force_new:
            cls._recalc_ranges(timeframe)
        sess = db.session
        fts = [KInfo.sid == sid, KInfo.timeframe == timeframe]
        kinfo: KInfo = sess.query(KInfo).filter(*fts).first()
        if kinfo:
            old_start, old_end = kinfo.start, kinfo.stop
            if start_ms >= old_start and end_ms <= old_end:
                # 未超出已有范围，不更新直接返回
                return old_start, old_end
            if old_end < start_ms or end_ms < old_start:
                if not force_new:
                    logger.info('incontinus insert detect, try refresh range...')
                    return cls._update_range(sid, timeframe, start_ms, end_ms, True)
                raise DisContiError(cache_key, (old_start, old_end), (start_ms, end_ms))
            else:
                kinfo.start = min(kinfo.start, start_ms)
                kinfo.stop = max(kinfo.stop, end_ms)
        else:
            kinfo = KInfo(sid=sid, timeframe=timeframe, start=start_ms, stop=end_ms)
            sess.add(kinfo)
        new_start, new_stop = kinfo.start, kinfo.stop
        sess.commit()
        return new_start, new_stop

    @classmethod
    def query_range(cls, sid: int, timeframe: str) -> Tuple[Optional[int], Optional[int]]:
        if timeframe not in cls.agg_map:
            # 当查询聚合周期时，最小相邻周期计算
            timeframe = cls._get_sub_tf(timeframe)[0]
        sess = db.session
        fts = [KInfo.sid == sid, KInfo.timeframe == timeframe]
        kinfo: KInfo = sess.query(KInfo).filter(*fts).first()
        if kinfo:
            return kinfo.start, kinfo.stop
        else:
            return None, None

    @classmethod
    def get_down_tf(cls, tf: str):
        '''
        获取指定周期对应的下载的时间周期。
        只有1m和1h允许下载并写入超表。其他维度都是由这两个维度聚合得到。
        '''
        from banbot.exchange.exchange_utils import secs_min, secs_hour, secs_day
        tf_secs = tf_to_secs(tf)
        if tf_secs >= secs_day:
            if tf_secs % secs_day:
                raise RuntimeError(f'unsupport timeframe: {tf}')
            return '1d'
        if tf_secs >= secs_hour:
            if tf_secs % secs_hour > 0:
                raise RuntimeError(f'unsupport timeframe: {tf}')
            return '1h'
        if tf_secs < secs_min or tf_secs % secs_min > 0:
            raise RuntimeError(f'unsupport timeframe: {tf}')
        return '1m'

    @classmethod
    def force_insert(cls, sid: int, timeframe: str, rows: List[Tuple]):
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
        try:
            sess.execute(sa.text(insert_sql), ins_rows)
        except Exception as e:
            if str(e).find('not supported on compressed chunks') >= 0:
                intv_msecs = tf_to_secs(timeframe) * 1000
                start_ms, end_ms = rows[0][0], rows[-1][0] + intv_msecs
                logger.error(f"insert compressed, call `pause_compress` first, {ins_tbl} sid:{sid} {start_ms}-{end_ms}")
            else:
                raise
        sess.commit()

    @classmethod
    def insert(cls, sid: int, timeframe: str, rows: List[Tuple], skip_in_range=True) -> int:
        '''
        单个bar插入，耗时约38ms
        '''
        if not rows:
            return 0
        if timeframe not in cls.down_tfs:
            raise RuntimeError(f'can only insert kline: {cls.down_tfs}, current: {timeframe}')
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
                raise DisContiError((sid, timeframe), (old_start, old_stop), (start_ms, end_ms))
            if skip_in_range:
                rows = [r for r in rows if not (old_start <= r[0] < old_stop)]
        if not rows:
            return 0
        cls.force_insert(sid, timeframe, rows)
        # 更新区间
        n_start, n_end = cls._update_range(sid, timeframe, start_ms, end_ms)
        # 刷新相关的连续聚合
        tf_new_ranges = cls._refresh_conti_agg(sid, timeframe, start_ms, end_ms, rows)
        if not old_stop or n_end > old_stop:
            tf_new_ranges.insert(0, (timeframe, old_stop or n_start, n_end))
        if tf_new_ranges and cls._listeners:
            # 有可能的监听者，发出查询数据发出事件
            sess = db.session
            exs: ExSymbol = sess.get(ExSymbol, sid)
            exg_name, market, symbol = exs.exchange, exs.market, exs.symbol
            for tf, n_start, n_end in tf_new_ranges:
                cls._on_new_bars(exg_name, market, symbol, tf)
        return len(rows)

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
    def _refresh_conti_agg(cls, sid: int, from_level: str, start_ms: int, end_ms: int, sub_bars: List[tuple]) -> List[Tuple[str, int, int]]:
        '''
        刷新连续聚合。返回大周期有新数据的区间
        '''
        from banbot.util.common import MeasureTime
        measure = MeasureTime()
        agg_keys = [from_level]
        from_secs = tf_to_secs(from_level)
        now_stamp = int(btime.utctime())  # 当前13位整型时间戳
        rev_sub_bars = sub_bars[::-1]  # 逆序，提高后续遍历速度
        for item in cls.agg_list:
            if item.secs <= from_secs:
                # 跳过过小维度；跳过无关的连续聚合
                continue
            measure.start_for(f'upd_un_{item.tf}')
            end_align = end_ms // 1000 // item.secs * item.secs
            if item.has_finish(start_ms, end_ms) and item.agg_from in agg_keys:
                agg_keys.append(item.tf)
            unbar_start_ts = now_stamp // item.secs * item.secs
            if end_align >= unbar_start_ts:
                # 仅当数据涉及当前周期未完成bar时，才尝试更新；仅传入相关的bar，提高效率
                unbar_start_ms = unbar_start_ts * 1000
                if sub_bars and sub_bars[-1][0] >= unbar_start_ms:
                    care_sub_bars = sub_bars
                else:
                    stop_id = next((i for i, r in enumerate(rev_sub_bars) if r[0] < unbar_start_ms), len(sub_bars))
                    care_sub_bars = rev_sub_bars[:stop_id][::-1]
                cls._update_unfinish(item, sid, start_ms, end_ms, care_sub_bars)
        agg_keys.remove(from_level)
        if not agg_keys:
            return []
        measure.start_for(f'get_db')
        from banbot.storage.base import init_db
        with init_db().connect() as conn:
            # 这里必须从连接池重新获取连接，不能使用sess的连接，否则会导致sess无效
            bak_iso_level = conn.get_isolation_level()
            conn.execution_options(isolation_level='AUTOCOMMIT')
            tf_ins_list = []
            for tf in agg_keys:
                measure.start_for(f'refresh_agg_{tf}')
                n_start, n_end, o_start, o_end = cls.refresh_agg(conn, cls.agg_map[tf], sid, start_ms, end_ms)
                if not o_end or n_end > o_end:
                    # 记录有新数据的周期
                    tf_ins_list.append((tf, o_end or n_start, n_end))
            measure.start_for(f'commit')
            conn.commit()
            conn.execution_options(isolation_level=bak_iso_level)
            return tf_ins_list
            # measure.print_all()

    @classmethod
    def _get_unfinish(cls, sid: int, timeframe: str, start_ts: int, end_ts: int, mode: str = 'query'):
        '''
        查询给定周期的未完成bar。给定周期可以是保存的周期1m,5m,15m,1h,1d；也可以是聚合周期如4h,3d
        此方法两种用途：query用户查询最新数据（可能是聚合周期）；calc从子周期更新大周期的未完成bar（不可能是聚合周期）
        :param sid:
        :param timeframe:
        :param start_ts: 10位秒级时间戳
        :param end_ts: 10位秒级时间戳
        :param mode: query|calc
        '''
        if mode == 'calc' and timeframe not in cls.agg_map:
            raise RuntimeError(f'{timeframe} not allowed in `calc` mode')
        sess = db.session
        merge_rows = []
        tf_secs = tf_to_secs(timeframe)
        un_tf = timeframe
        bar_end_ms = 0
        if mode == 'calc' or timeframe not in cls.agg_map:
            # 从已完成的子周期中统计筛选数据。
            from_tf = cls._get_sub_tf(timeframe)[0]
            agg_from = 'kline_' + from_tf
            sel_sql = f'''SELECT (extract(epoch from time) * 1000)::bigint AS start_ms,
                            open,high,low,close,volume FROM {agg_from}
                            where sid={sid} and time >= to_timestamp({start_ts}) and time < to_timestamp({end_ts})'''
            sub_rows = sess.execute(sa.text(sel_sql)).fetchall()
            sub_rows = [tuple(r) for r in sub_rows]
            from banbot.data.tools import build_ohlcvc
            merge_rows, last_finish = build_ohlcvc(sub_rows, tf_secs)
            if sub_rows:
                bar_end_ms = sub_rows[-1][0] + tf_to_secs(from_tf) * 1000
            un_tf = from_tf  # 未完成bar从子周期查询
        # 从未完成的周期/子周期中查询bar
        unfinish = sess.execute(sa.text(f'''
                                SELECT start_ms,open,high,low,close,volume,stop_ms FROM kline_un
                                where sid={sid} and timeframe='{un_tf}' and start_ms >= {int(start_ts * 1000)}
                                limit 1''')).fetchone()
        if unfinish:
            merge_rows.append(tuple(unfinish)[:-1])
            bar_end_ms = max(bar_end_ms, unfinish[-1])
        if not merge_rows:
            return None, bar_end_ms
        if len(merge_rows) > 1:
            ts_col, opens, highs, lows, closes, volumes = list(zip(*merge_rows))
            cur_bar = ts_col[0], opens[0], max(highs), min(lows), closes[-1], sum(volumes)
        else:
            cur_bar = merge_rows[0]
        return cur_bar, bar_end_ms

    @classmethod
    def _update_unfinish(cls, item: BarAgg, sid: int, start_ms: int, end_ms: int, sml_bars: List[Tuple]):
        '''
        :param start_ms: 毫秒时间戳，子周期插入数据的开始时间
        :param end_ms: 毫秒时间戳，子周期bar的截止时间（非bar的开始时间）
        :param sml_bars: 子周期插入的bars，可能包含超出start范围的旧数据
        '''
        sess = db.session
        tf_secs = tf_to_secs(item.tf)
        tf_msecs = tf_secs * 1000
        bar_finish = end_ms % tf_msecs == 0
        where_sql = f"where sid={sid} and timeframe='{item.tf}';"
        from_where = f"from kline_un {where_sql}"
        if bar_finish:
            # 当前周期已完成，kline_un中删除即可
            sess.execute(sa.text(f"DELETE {from_where}"))
            sess.commit()
            return
        bar_start_ts = start_ms // tf_msecs * tf_secs
        bar_end_ts = end_ms // tf_msecs * tf_secs
        if bar_start_ts == bar_end_ts:
            # 当子周期插入开始结束时间戳，对应到当前周期，属于同一个bar时，才执行快速更新
            rows = sess.execute(sa.text(f"select start_ms,open,high,low,close,volume,stop_ms {from_where}")).fetchall()
            rows = [tuple(r) for r in rows]
            if len(rows) == 1 and rows[0][-1] == start_ms:
                # 当本次插入开始时间戳，和未完成bar结束时间戳完全匹配时，认为有效
                from banbot.data.tools import build_ohlcvc
                old_un = rows[0]
                cur_bars, last_finish = build_ohlcvc(sml_bars, tf_secs)
                if len(cur_bars) == 1:
                    new_un = cur_bars[0]
                    phigh = max(old_un[2], new_un[2])
                    plow = min(old_un[3], new_un[3])
                    pclose = new_un[4]
                    vol_sum = old_un[5] + new_un[5]
                    sess.execute(sa.text(f'''
update "kline_un" set high={phigh},low={plow},
  close={pclose},volume={vol_sum},stop_ms={end_ms}
  {where_sql}'''))
                    sess.commit()
                    return
            elif start_ms % tf_msecs == 0:
                # 当插入的bar是第一个时，也认为有效。直接插入
                if len(rows):
                    sess.execute(sa.text(f"DELETE {from_where}"))
                from banbot.data.tools import build_ohlcvc
                cur_bars, last_finish = build_ohlcvc(sml_bars, tf_secs)
                if len(cur_bars) == 1:
                    new_un = cur_bars[0]
                    ins_cols = "sid, start_ms, stop_ms, open, high, low, close, volume, timeframe"
                    places = f"{sid}, {new_un[0]}, {end_ms}, {new_un[1]}, {new_un[2]}, {new_un[3]}, " \
                             f"{new_un[4]}, {new_un[5]}, '{item.tf}'"
                    insert_sql = f"insert into kline_un ({ins_cols}) values ({places})"
                    sess.execute(sa.text(insert_sql))
                    sess.commit()
                    return
        # logger.info(f'slow kline_un: {sid} {item.tf} {start_ms} {end_ms}')
        # 当快速更新不可用时，从子周期归集
        sess.execute(sa.text(f"DELETE {from_where}"))
        cur_bar, bar_end_ms = cls._get_unfinish(sid, item.tf, bar_end_ts, bar_end_ts + tf_secs, 'calc')
        if not cur_bar:
            sess.commit()
            return
        ins_cols = "sid, start_ms, stop_ms, open, high, low, close, volume, timeframe"
        places = f"{sid}, {cur_bar[0]}, {bar_end_ms}, {cur_bar[1]}, {cur_bar[2]}, {cur_bar[3]}, " \
                 f"{cur_bar[4]}, {cur_bar[5]}, '{item.tf}'"
        insert_sql = f"insert into kline_un ({ins_cols}) values ({places})"
        sess.execute(sa.text(insert_sql))
        sess.commit()

    @classmethod
    def refresh_agg(cls, conn: Union[SqlSession, sa.Connection], tbl: BarAgg, sid: int,
                    org_start_ms: int, org_end_ms: int, agg_from: str = None):
        tf_msecs = tbl.secs * 1000
        start_ms = org_start_ms // tf_msecs * tf_msecs
        # 有可能start_ms刚好是下一个bar的开始，前一个需要-1
        agg_start = start_ms - tf_msecs
        end_ms = org_end_ms // tf_msecs * tf_msecs
        if start_ms == end_ms < org_start_ms:
            # 没有出现新的完成的bar数据，无需更新
            # 前2个相等，说明：插入的数据所属bar尚未完成。
            # start_ms < org_start_ms说明：插入的数据不是所属bar的第一个数据
            return None, None, None, None
        old_start, old_end = cls.query_range(sid, tbl.tf)
        if old_start and old_end > old_start:
            # 避免出现空洞或数据错误
            agg_start = min(agg_start, old_end)
            end_ms = max(end_ms, old_start)
        if not agg_from:
            agg_from = 'kline_' + tbl.agg_from
        win_start = f"to_timestamp({agg_start / 1000})"
        win_end = f"to_timestamp({end_ms / 1000})"
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
        except Exception as e:
            logger.exception(f'refresh conti agg error: {e}, {sid} {tbl.tf}')
            return old_start, old_end, old_start, old_end
        new_start, new_end = cls._update_range(sid, tbl.tf, start_ms, end_ms)
        return new_start, new_end, old_start, old_end

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
    async def wait_bars(cls, exg_name: str, market: str, pair: str, timeframe: str):
        '''
        监听指定币，指定周期的新的bar。四个参数都可为*，表示任意。
        '''
        key = f'{exg_name}_{market}_{pair}_{timeframe}'
        if key not in cls._listeners:
            cls._listeners[key] = collections.deque()
        queue = cls._listeners[key]
        fut = asyncio.get_running_loop().create_future()
        queue.append(fut)
        try:
            return await fut
        finally:
            queue.remove(fut)

    @classmethod
    def _on_new_bars(cls, exg_name: str, market: str, pair: str, timeframe: str):
        '''
        有新的bar被写入到数据库，触发相应的监听者。
        '''
        if not cls._listeners:
            return []
        waiters = []
        for exg in ['*', exg_name]:
            for m in ['*', market]:
                for p in ['*', pair]:
                    for t in ['*', timeframe]:
                        waits = cls._listeners.get(f'{exg}_{m}_{p}_{t}')
                        if waits:
                            waiters.extend(waits)
        if not waiters:
            return
        data = [exg_name, market, pair, timeframe]
        for fut in waiters:
            if not fut.done():
                fut.set_result(data)


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


class KInfo(BaseDbModel):
    '''
    记录K线所有周期的相关数据，如：开始结束时间。（每次查询开始结束时间非常慢）
    '''
    __tablename__ = 'kinfo'

    sid = Column(sa.Integer, primary_key=True)
    timeframe = Column(sa.String(5), primary_key=True)
    start = Column(sa.BigInteger)  # 第一个bar的13位时间戳
    stop = Column(sa.BigInteger)  # 最后一个bar的13位时间戳 + tf_secs * 1000
