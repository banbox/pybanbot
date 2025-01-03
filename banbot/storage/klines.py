#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : klines.py
# Author: anyongjin
# Date  : 2023/4/24
import collections

import pytz

from banbot.storage.base import *
from banbot.util import btime
from banbot.storage.symbols import ExSymbol
from banbot.util.tf_utils import *
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

    sid = mapped_column(sa.Integer, primary_key=True)
    time = mapped_column(type_=sa.TIMESTAMP(timezone=True), primary_key=True)
    open = mapped_column(sa.FLOAT)
    high = mapped_column(sa.FLOAT)
    low = mapped_column(sa.FLOAT)
    close = mapped_column(sa.FLOAT)
    volume = mapped_column(sa.FLOAT)

    agg_list = [
        # 全部使用超表，自行在插入时更新依赖表。因连续聚合无法按sid刷新，在按sid批量插入历史数据后刷新时性能较差
        BarAgg('1m', 'kline_1m', None, None, None, None, '2 months', '12 months'),
        BarAgg('5m', 'kline_5m', '1m', '20m', '1m', '1m', '2 months', '12 months'),
        BarAgg('15m', 'kline_15m', '5m', '1h', '5m', '5m', '3 months', '16 months'),
        BarAgg('1h', 'kline_1h', None, None, None, None, '6 months', '3 years'),
        BarAgg('1d', 'kline_1d', '1h', '3d', '1h', '1h', '3 years', '20 years'),
    ]

    down_tfs = {'1m', '15m', '1h', '1d'}

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
    async def _init_hypertbl(cls, sess: SqlSession, tbl: BarAgg):
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
            await sess.execute(sa.text(stat))
            await sess.flush()
        # 设置数据丢弃
        db_retention = AppConfig.get()['database'].get('retention')
        if db_retention and db_retention != 'all':
            await sess.execute(sa.text(f"SELECT add_retention_policy('{tbl.tbl}', INTERVAL '{tbl.retention}');"))
            await sess.flush()

    @classmethod
    async def init_tbl(cls, sess: SqlSession):
        await cls._init_hypertbl(sess, cls.agg_list[0])
        # 创建连续聚合及更新策略
        for item in cls.agg_list[1:]:
            create_sql = f'CREATE TABLE {item.tbl} (LIKE {cls._tname} INCLUDING ALL);'
            await sess.execute(sa.text(create_sql))
            await sess.flush()
            # 初始化超表
            await cls._init_hypertbl(sess, item)
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
            await sess.execute(sa.text(stat))
        await sess.flush()

    @classmethod
    async def drop_tbl(cls, sess: SqlSession):
        '''
        删除所有的K线数据表；超表+连续聚合
        '''
        for tbl in cls.agg_list[::-1]:
            await sess.execute(sa.text(f"drop table if exists {tbl.tbl}"))

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
    async def _query_hyper(cls, timeframe: str, dct_sql: str, gp_sql: Union[str, Callable],
                           sess: Optional[SqlSession] = None, **kwargs):
        if not sess:
            sess = dba.session
        if timeframe in cls.agg_map:
            stmt = dct_sql.format(tbl=cls.agg_map[timeframe].tbl, **kwargs)
            return await sess.execute(sa.text(stmt))
        else:
            # 时间帧没有直接符合的，从最接近的子timeframe聚合
            sub_tf, sub_tbl = cls._get_sub_tf(timeframe)
            if callable(gp_sql):
                gp_sql = gp_sql()
            stmt = gp_sql.format(tbl=sub_tbl, **kwargs)
            return await sess.execute(sa.text(stmt))

    @classmethod
    async def query(cls, exs: ExSymbol, timeframe: str, start_ms: int, end_ms: int, limit: Optional[int] = None,
                    with_unfinish: bool = False, sess: SqlSession = None):
        tf_secs = tf_to_secs(timeframe)
        tf_msecs = tf_secs * 1000
        max_end_ms = end_ms
        if limit:
            end_ms = min(start_ms + tf_msecs * limit, end_ms)

        start_ts, end_ts = start_ms / 1000, end_ms / 1000
        # 计算最新未完成bar的时间戳
        finish_end_ts = align_tfsecs(end_ts, tf_secs)
        unfinish_ts = align_tfsecs(btime.utctime(), tf_secs)
        if finish_end_ts > unfinish_ts:
            finish_end_ts = unfinish_ts

        dct_sql = f'''
select (extract(epoch from time) * 1000)::bigint as time,open,high,low,close,volume from {{tbl}}
where sid={{sid}} and time >= to_timestamp({start_ts}) and time < to_timestamp({finish_end_ts})
order by time'''

        def gen_gp_sql():
            origin = get_tfalign_origin(timeframe)[0]
            return f'''
                select (extract(epoch from time_bucket('{timeframe}', time, origin => '{origin}')) * 1000)::bigint AS gtime,
                  {cls._candle_agg} from {{tbl}}
                where sid={{sid}} and time >= to_timestamp({start_ts}) and time < to_timestamp({finish_end_ts})
                group by gtime
                order by gtime'''
        if not sess:
            sess = dba.session
        rows = (await cls._query_hyper(timeframe, dct_sql, gen_gp_sql, sid=exs.id, sess=sess)).all()
        rows = [list(r) for r in rows]
        if not len(rows) and max_end_ms - end_ms > tf_msecs:
            rows = await cls.query(exs, timeframe, end_ms, max_end_ms, limit, sess=sess)
        elif with_unfinish and rows and rows[-1][0] // 1000 + tf_secs == unfinish_ts:
            un_bar, _ = await cls._get_unfinish(sess, exs.id, timeframe, unfinish_ts, unfinish_ts + tf_secs)
            if un_bar:
                rows.append(list(un_bar))
        return rows

    @classmethod
    async def query_batch(cls, exs_ids: Iterable[int], timeframe: str, start_ms: int, end_ms: int,
                          sess: SqlSession = None):
        '''
        批量查询sid的ohlcv。
        返回的按time, sid升序。
        sid作为最后一列返回。
        '''
        if not exs_ids:
            return []
        tf_secs = tf_to_secs(timeframe)

        start_ts, end_ts = start_ms / 1000, end_ms / 1000
        # 计算最新未完成bar的时间戳
        finish_end_ts = align_tfsecs(end_ts, tf_secs)
        unfinish_ts = align_tfsecs(btime.utctime(), tf_secs)
        if finish_end_ts > unfinish_ts:
            finish_end_ts = unfinish_ts

        dct_sql = f'''
select (extract(epoch from time) * 1000)::bigint as time,open,high,low,close,volume,sid from {{tbl}}
where time >= to_timestamp({start_ts}) and time < to_timestamp({finish_end_ts}) and sid in ({{sid}}) 
order by time, sid'''

        def gen_gp_sql():
            return f'''
                select (extract(epoch from time_bucket('{timeframe}', time, origin => '1970-01-01')) * 1000)::bigint AS gtime,
                  {cls._candle_agg},sid from {{tbl}}
                where time >= to_timestamp({start_ts}) and time < to_timestamp({finish_end_ts}) and sid in ({{sid}})
                group by gtime, sid
                order by gtime, sid'''

        sid_text = ', '.join(map(lambda x: str(x), exs_ids))
        rows = (await cls._query_hyper(timeframe, dct_sql, gen_gp_sql, sid=sid_text, sess=sess)).all()
        return [list(r) for r in rows]

    @classmethod
    async def _recalc_ranges(cls, *tf_list: str, sess: SqlSession = None) -> Dict[Tuple[int, str], Tuple[int, int]]:
        dct_sql = '''
select sid,
(extract(epoch from min(time)) * 1000)::bigint, 
(extract(epoch from max(time)) * 1000)::bigint 
from {tbl}
group by 1'''
        if not sess:
            sess = dba.session
        if not tf_list:
            tf_list = [item.tf for item in cls.agg_list]
        # 删除旧的kinfo
        result = dict()
        tf_texts = ', '.join([f"'{tf}'" for tf in tf_list])
        del_sql = f"delete from kinfo where timeframe in ({tf_texts})"
        await sess.execute(sa.text(del_sql))
        await sess.flush()
        for tf in tf_list:
            sql_text = dct_sql.format(tbl=f'kline_{tf}')
            rows = (await sess.execute(sa.text(sql_text))).fetchall()
            for sid, min_time, max_time in rows:
                cache_key = sid, tf
                # 这里记录蜡烛对应的结束时间
                max_time += tf_to_secs(tf) * 1000
                min_time, max_time = int(min_time), int(max_time)
                result[cache_key] = min_time, max_time
                sess.add(KInfo(sid=sid, timeframe=tf, start=min_time, stop=max_time))
        await sess.flush()
        return result

    @classmethod
    async def load_kline_ranges(cls) -> Dict[Tuple[int, str], Tuple[int, int]]:
        sess = dba.session
        rows: Iterable[KInfo] = (await sess.scalars(select(KInfo))).all()
        if not rows:
            return await cls._recalc_ranges(sess=sess)
        result = dict()
        for row in rows:
            cache_key = row.sid, row.timeframe
            result[cache_key] = row.start, row.stop
        return result

    @classmethod
    async def _update_range(cls, sid: int, timeframe: str, start_ms: int, end_ms: int, force_new: bool = False,
                            sess: SqlSession = None)\
            -> Tuple[int, int]:
        '''
        更新sid+timeframe对应的数据区间。end_ms应为最后一个bar对应的结束时间，而非开始时间
        :param force_new: 是否强制刷新范围后，再尝试更新
        '''
        cache_key = sid, timeframe
        if not sess:
            sess = dba.session
        if force_new:
            await cls._recalc_ranges(timeframe, sess=sess)
        old_start, old_end = await cls.query_range(sid, timeframe, sess)
        if old_end or old_start:
            if start_ms >= old_start and end_ms <= old_end:
                # 未超出已有范围，不更新直接返回
                return old_start, old_end
            if old_end < start_ms or end_ms < old_start:
                if not force_new:
                    logger.info('incontinus insert detect, try refresh range...')
                    return await cls._update_range(sid, timeframe, start_ms, end_ms, True, sess=sess)
                raise DisContiError(cache_key, (old_start, old_end), (start_ms, end_ms))
            else:
                fts = [KInfo.sid == sid, KInfo.timeframe == timeframe]
                new_start = min(old_start, start_ms)
                new_stop = max(old_end, end_ms)
                upds = dict(start=new_start, stop=new_stop)
                stmt = update(KInfo).where(*fts).values(**upds)
                await sess.execute(stmt)
        else:
            stmt = insert(KInfo).values(sid=sid, timeframe=timeframe, start=start_ms, stop=end_ms)
            await sess.execute(stmt)
            new_start, new_stop = start_ms, end_ms
        return new_start, new_stop

    @classmethod
    async def query_range(cls, sid: int, timeframe: str, sess: SqlSession = None) -> Tuple[Optional[int], Optional[int]]:
        if timeframe not in cls.agg_map:
            # 当查询聚合周期时，最小相邻周期计算
            timeframe = cls._get_sub_tf(timeframe)[0]
        if not sess:
            sess = dba.session
        fts = [KInfo.sid == sid, KInfo.timeframe == timeframe]
        stmt = select(KInfo.start, KInfo.stop).where(*fts).limit(1)
        rows = await sess.execute(stmt)
        row = rows.first()
        if row:
            return row[0], row[1]
        else:
            return None, None

    @classmethod
    def get_down_tf(cls, tf: str):
        '''
        获取指定周期对应的下载的时间周期。
        只有1m和1h允许下载并写入超表。其他维度都是由这两个维度聚合得到。
        '''
        from banbot.util.tf_utils import secs_min, secs_hour, secs_day
        tf_secs = tf_to_secs(tf)
        if tf_secs >= secs_day:
            if tf_secs % secs_day:
                raise RuntimeError(f'unsupport timeframe: {tf}')
            return '1d'
        if tf_secs >= secs_hour:
            if tf_secs % secs_hour > 0:
                raise RuntimeError(f'unsupport timeframe: {tf}')
            return '1h'
        if tf_secs >= secs_min * 15:
            return '15m'
        if tf_secs < secs_min or tf_secs % secs_min > 0:
            raise RuntimeError(f'unsupport timeframe: {tf}')
        return '1m'

    @classmethod
    async def force_insert(cls, sess: SqlSession, sid: int, timeframe: str, rows: List[Tuple]):
        ins_rows = []
        for r in rows:
            row_ts = btime.to_datetime(r[0])
            ins_rows.append(dict(
                sid=sid, time=row_ts, open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5],
            ))
        ins_tbl = cls.agg_map[timeframe].tbl
        ins_cols = "sid, time, open, high, low, close, volume"
        places = ":sid, :time, :open, :high, :low, :close, :volume"
        insert_sql = f"insert into {ins_tbl} ({ins_cols}) values ({places}) {cls._insert_conflict}"
        try:
            await sess.execute(sa.text(insert_sql), ins_rows)
        except Exception as e:
            if str(e).find('not supported on compressed chunks') >= 0:
                intv_msecs = tf_to_secs(timeframe) * 1000
                start_ms, end_ms = rows[0][0], rows[-1][0] + intv_msecs
                err_msg = f"insert compressed, add `--no-compress` and retry, {ins_tbl} sid:{sid} {start_ms}-{end_ms}"
                raise ValueError(err_msg)
            else:
                raise
        await sess.flush()

    @classmethod
    async def insert(cls, sid: int, timeframe: str, rows: List[Tuple], skip_in_range=True,
                     sess: Optional[SqlSession] = None) -> int:
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
        old_start, old_stop = await cls.query_range(sid, timeframe, sess=sess)
        if old_start and old_stop:
            # 插入的数据应该和已有数据连续，避免出现空洞。
            if old_stop < start_ms or end_ms < old_start:
                raise DisContiError((sid, timeframe), (old_start, old_stop), (start_ms, end_ms))
            if skip_in_range:
                rows = [r for r in rows if not (old_start <= r[0] < old_stop)]
        if not rows:
            return 0
        if not sess:
            sess = dba.session
        await cls.force_insert(sess, sid, timeframe, rows)
        # 更新区间
        n_start, n_end = await cls._update_range(sid, timeframe, start_ms, end_ms, sess=sess)
        # 刷新相关的连续聚合
        tf_new_ranges = []
        if not old_stop or n_end > old_stop:
            tf_new_ranges.append((timeframe, old_stop or n_start, n_end))
        await cls._refresh_conti_agg(sess, sid, timeframe, start_ms, end_ms, rows, tf_new_ranges)
        return len(rows)

    @classmethod
    @contextlib.asynccontextmanager
    async def decompress(cls, tbl_list: List[str] = None):
        start = time.monotonic()
        async with dba.autocommit() as sess:
            logger.info('pause compress ing...')
            jobs = await cls._pause_compress(sess, tbl_list)
        cost = time.monotonic() - start
        logger.info(f'pause {len(jobs)} compress jobs ok, cost: {cost:.3f} secs')
        try:
            yield
        finally:
            logger.info('restore compress ...')
            async with dba.autocommit() as sess:
                await cls._restore_compress(sess, jobs)

    @classmethod
    async def _pause_compress(cls, sess: SqlSession, tbl_list: List[str] = None) -> List[int]:
        if not tbl_list:
            tbl_list = [t.tbl for t in KLine.agg_list]
        tbl_list = set(tbl_list)
        get_job_id = f"""
            SELECT j.hypertable_name, j.job_id FROM timescaledb_information.jobs j
            WHERE j.proc_name = 'policy_compression'"""
        rows = (await sess.execute(sa.text(get_job_id))).all()
        result = []
        for row in rows:
            if row[0] not in tbl_list:
                continue
            job_id = row[1]
            # 暂停压缩任务
            await sess.execute(sa.text(f'SELECT alter_job({job_id}, scheduled => false);'))
            # 解压缩涉及的块
            decps_sql = f'''SELECT decompress_chunk(i, true) FROM show_chunks('{row[0]}') i ;'''
            await sess.execute(sa.text(decps_sql))
            result.append(job_id)
        await sess.flush()
        return result

    @classmethod
    async def _restore_compress(cls, sess: SqlSession, jobs: List[int]):
        if not jobs:
            return
        for job_id in jobs:
            # 启动压缩任务（不会立刻执行压缩）
            await sess.execute(sa.text(f'SELECT alter_job({job_id}, scheduled => true);'))
        await sess.flush()

    @classmethod
    async def _refresh_conti_agg(cls, sess: SqlSession, sid: int, from_level: str, start_ms: int, end_ms: int,
                                 sub_bars: List[tuple], tf_new_ranges):
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
                await cls._update_unfinish(sess, item, sid, start_ms, end_ms, care_sub_bars)
        agg_keys.remove(from_level)
        if not agg_keys:
            return
        measure.start_for(f'get_db')

        for tf in agg_keys:
            measure.start_for(f'refresh_agg_{tf}')
            args = [sess, cls.agg_map[tf], sid, start_ms, end_ms]
            n_start, n_end, o_start, o_end = await cls.refresh_agg(*args)
            if not o_end or n_end > o_end:
                # 记录有新数据的周期
                tf_new_ranges.append((tf, o_end or n_start, n_end))
        measure.start_for(f'commit')
        if tf_new_ranges and cls._listeners:
            # 有可能的监听者，发出查询数据发出事件
            exs: ExSymbol = await ExSymbol.safe_get_by_id(sid, sess)
            exg_name, market, symbol = exs.exchange, exs.market, exs.symbol
            for tf, n_start, n_end in tf_new_ranges:
                cls._on_new_bars(exg_name, market, symbol, tf)
        # measure.print_all()

    @classmethod
    async def _get_unfinish(cls, sess: SqlSession, sid: int, timeframe: str, start_ts: int, end_ts: int,
                            mode: str = 'query'):
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
            sub_rows = (await sess.execute(sa.text(sel_sql))).fetchall()
            sub_rows = [tuple(r) for r in sub_rows]
            from banbot.data.tools import build_ohlcvc
            merge_rows, last_finish = build_ohlcvc(sub_rows, tf_secs)
            if sub_rows:
                bar_end_ms = sub_rows[-1][0] + tf_to_secs(from_tf) * 1000
            un_tf = from_tf  # 未完成bar从子周期查询
        # 从未完成的周期/子周期中查询bar
        unfinish = (await sess.execute(sa.text(f'''
                                SELECT start_ms,open,high,low,close,volume,stop_ms FROM kline_un
                                where sid={sid} and timeframe='{un_tf}' and start_ms >= {int(start_ts * 1000)}
                                limit 1'''))).fetchone()
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
    async def _update_unfinish(cls, sess: SqlSession, item: BarAgg, sid: int, start_ms: int, end_ms: int, sml_bars: List[Tuple]):
        '''
        :param start_ms: 毫秒时间戳，子周期插入数据的开始时间
        :param end_ms: 毫秒时间戳，子周期bar的截止时间（非bar的开始时间）
        :param sml_bars: 子周期插入的bars，可能包含超出start范围的旧数据
        '''
        tf_secs = tf_to_secs(item.tf)
        tf_msecs = tf_secs * 1000
        bar_finish = end_ms % tf_msecs == 0
        where_sql = f"where sid={sid} and timeframe='{item.tf}';"
        from_where = f"from kline_un {where_sql}"
        if bar_finish:
            # 当前周期已完成，kline_un中删除即可
            await sess.execute(sa.text(f"DELETE {from_where}"))
            await sess.flush()
            return
        bar_start_ts = align_tfsecs(start_ms // 1000, tf_secs)
        bar_end_ts = align_tfsecs(end_ms // 1000, tf_secs)
        if bar_start_ts == bar_end_ts:
            # 当子周期插入开始结束时间戳，对应到当前周期，属于同一个bar时，才执行快速更新
            stmt = sa.text(f"select start_ms,open,high,low,close,volume,stop_ms {from_where}")
            rows = (await sess.execute(stmt)).fetchall()
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
                    await sess.execute(sa.text(f'''
update "kline_un" set high={phigh},low={plow},
  close={pclose},volume={vol_sum},stop_ms={end_ms}
  {where_sql}'''))
                    await sess.flush()
                    return
            elif start_ms % tf_msecs == 0:
                # 当插入的bar是第一个时，也认为有效。直接插入
                if len(rows):
                    await sess.execute(sa.text(f"DELETE {from_where}"))
                    await sess.flush()
                from banbot.data.tools import build_ohlcvc
                cur_bars, last_finish = build_ohlcvc(sml_bars, tf_secs)
                if len(cur_bars) == 1:
                    new_un = cur_bars[0]
                    ins_cols = "sid, start_ms, stop_ms, open, high, low, close, volume, timeframe"
                    places = f"{sid}, {new_un[0]}, {end_ms}, {new_un[1]}, {new_un[2]}, {new_un[3]}, " \
                             f"{new_un[4]}, {new_un[5]}, '{item.tf}'"
                    insert_sql = f"insert into kline_un ({ins_cols}) values ({places})"
                    await sess.execute(sa.text(insert_sql))
                    await sess.flush()
                    return
        # logger.info(f'slow kline_un: {sid} {item.tf} {start_ms} {end_ms}')
        # 当快速更新不可用时，从子周期归集
        await sess.execute(sa.text(f"DELETE {from_where}"))
        await sess.flush()
        cur_bar, bar_end_ms = await cls._get_unfinish(sess, sid, item.tf, bar_end_ts, bar_end_ts + tf_secs, 'calc')
        if not cur_bar:
            await sess.flush()
            return
        ins_cols = "sid, start_ms, stop_ms, open, high, low, close, volume, timeframe"
        places = f"{sid}, {cur_bar[0]}, {bar_end_ms}, {cur_bar[1]}, {cur_bar[2]}, {cur_bar[3]}, " \
                 f"{cur_bar[4]}, {cur_bar[5]}, '{item.tf}'"
        insert_sql = f"insert into kline_un ({ins_cols}) values ({places})"
        # 先删除旧的无效的记录
        await sess.execute(sa.text(f"DELETE {from_where}"))
        await sess.flush()
        await sess.execute(sa.text(insert_sql))
        await sess.flush()

    @classmethod
    async def refresh_agg(cls, sess: SqlSession, tbl: BarAgg, sid: int,
                          org_start_ms: int, org_end_ms: int, agg_from: str = None):
        tf_msecs = tbl.secs * 1000
        start_ms = align_tfmsecs(org_start_ms, tf_msecs)
        # 有可能start_ms刚好是下一个bar的开始，前一个需要-1
        agg_start = start_ms - tf_msecs
        end_ms = align_tfmsecs(org_end_ms, tf_msecs)
        if start_ms == end_ms < org_start_ms:
            # 没有出现新的完成的bar数据，无需更新
            # 前2个相等，说明：插入的数据所属bar尚未完成。
            # start_ms < org_start_ms说明：插入的数据不是所属bar的第一个数据
            return None, None, None, None
        old_start, old_end = await cls.query_range(sid, tbl.tf, sess)
        if old_start and old_end > old_start:
            # 避免出现空洞或数据错误
            agg_start = min(agg_start, old_end)
            end_ms = max(end_ms, old_start)
        if not agg_from:
            agg_from = 'kline_' + tbl.agg_from
        win_start = f"to_timestamp({agg_start / 1000})"
        win_end = f"to_timestamp({end_ms / 1000})"
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
            await sess.execute(sa.text(stmt))
        except Exception as e:
            # 如果遇到insert into with on conflict on compressed block 错误，可在启动时，调用Kline.pause_compress暂时解压缩
            logger.exception(f'refresh conti agg error: {e}, {sid} {tbl.tf}')
            return old_start, old_end, old_start, old_end
        new_start, new_end = await cls._update_range(sid, tbl.tf, start_ms, end_ms, sess=sess)
        return new_start, new_end, old_start, old_end

    @classmethod
    async def log_candles_conts(cls, exs: ExSymbol, timeframe: str, start_ms: int, end_ms: int, candles: list,
                                sess: SqlSession = None):
        '''
        检查下载的蜡烛数据是否连续，如不连续记录到khole中
        '''
        tf_msecs = tf_to_secs(timeframe) * 1000
        # 取 >= start_ms的第一个bar开始时间
        align_start = align_tfmsecs(start_ms, tf_msecs)
        if start_ms % tf_msecs:
            align_start += tf_msecs
        start_ms = align_start
        # 取 < end_ms的最后一个bar开始时间
        end_ms = min(end_ms, btime.time_ms())
        end_ms = align_tfmsecs(end_ms, tf_msecs)
        if start_ms > end_ms:
            return
        holes = []
        if not candles:
            holes = [(start_ms, end_ms)]
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
            if end_ms - prev_date > tf_msecs:
                holes.append((prev_date + tf_msecs, end_ms))
        if not holes:
            return
        if not sess:
            sess = dba.session
        sid = exs.id
        holes = [KHole(sid=sid, timeframe=timeframe, start=h[0], stop=h[1]) for h in holes]
        fts = [KHole.sid == sid, KHole.timeframe == timeframe]
        old_holes = await get_holes(sess, fts)
        holes.extend(old_holes)
        holes.sort(key=lambda x: x.start)
        merged: List[KHole] = []
        for h in holes:
            if not merged or merged[-1].stop < h.start:
                # 与前一个洞不连续，出现缺口
                merged.append(h)
            else:
                # 与前一个洞连续，需要删除一个合并成更大的洞
                prev = merged[-1]
                old_h, hole = prev, h
                if h.id:
                    old_h, hole = h, prev
                if hole.id:
                    await sess.delete(hole)
                    await sess.flush()
                if hole.stop > old_h.stop:
                    old_h.stop = hole.stop
                merged[-1] = old_h
        for m in merged:
            if m.start == m.stop:
                if m.id:
                    await sess.delete(m)
                continue
            if not m.id:
                if m.stop > btime.time_ms():
                    logger.error(f'hole.stop exceed cur time, bad: {m}, all: {merged}')
                    await sess.rollback()
                    return
                sess.add(m)
        await sess.flush()

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

    id = mapped_column(sa.Integer, primary_key=True)
    sid = mapped_column(sa.Integer)
    timeframe = mapped_column(sa.String(5))
    start = mapped_column(sa.BIGINT)  # 从第一个缺失的bar时间戳记录
    stop = mapped_column(sa.BIGINT)  # 记录到最后一个确实的bar的结束时间戳（即下一个有效bar的时间戳）

    @classmethod
    async def get_down_range(cls, exs: ExSymbol, timeframe: str, start_ms: int, stop_ms: int,
                             sess: SqlSession = None) -> Tuple[int, int]:
        if not sess:
            sess = dba.session
        fts = [KHole.sid == exs.id, KHole.timeframe == timeframe, KHole.stop > start_ms, KHole.start < stop_ms]
        holes: List[KHole] = await get_holes(sess, fts, KHole.start)
        start_ms, stop_ms = get_unknown_range(start_ms, stop_ms, holes)
        if start_ms >= stop_ms:
            return 0, 0
        return start_ms, stop_ms

    def __str__(self):
        return f'{self.sid}|{self.timeframe}|{self.start}|{self.stop}'

    def __repr__(self):
        return f'{self.sid}|{self.timeframe}|{self.start}|{self.stop}'


def get_unknown_range(start_ms: int, stop_ms: int, holes: List[KHole]) -> Tuple[int, int]:
    for h in holes:
        if h.start <= start_ms:
            start_ms = max(start_ms, h.stop)
        elif h.stop >= stop_ms:
            stop_ms = min(stop_ms, h.start)
        if start_ms >= stop_ms:
            return start_ms, stop_ms
    return start_ms, stop_ms


async def get_holes(sess: SqlSession, fts: List[Any], od_by=None) -> List[KHole]:
    stmt = select(KHole).where(*fts)
    if od_by:
        stmt = stmt.order_by(od_by)
    return list((await sess.scalars(stmt)).all())


class KInfo(BaseDbModel):
    '''
    记录K线所有周期的相关数据，如：开始结束时间。（每次查询开始结束时间非常慢）
    '''
    __tablename__ = 'kinfo'

    sid = mapped_column(sa.Integer, primary_key=True)
    timeframe = mapped_column(sa.String(5), primary_key=True)
    start = mapped_column(sa.BigInteger)  # 第一个bar的13位时间戳
    stop = mapped_column(sa.BigInteger)  # 最后一个bar的13位时间戳 + tf_secs * 1000
