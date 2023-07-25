#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : toolbox.py
# Author: anyongjin
# Date  : 2023/7/25
from banbot.storage.klines import *
from banbot.data.tools import *


def _find_sid_hole(sess: SqlSession, timeframe: str, sid: int, since: datetime, until: datetime = None):
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


async def _fill_tf_hole(timeframe: str):
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
        hole_list = _find_sid_hole(sess, timeframe, sid, start_dt)
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
            sub_arr = KLine.query(stf, down_tf, start_ms, end_ms)
            true_len = (end_ms - start_ms) // down_tfsecs // 1000
            if true_len == len(sub_arr):
                # 子维度周期数据存在，直接归集更新
                KLine.refresh_agg(sess, KLine.agg_map[timeframe], sid, start_ms, end_ms, f'kline_{down_tf}')
            else:
                logger.info(f'db sub tf {down_tf} not enough {len(sub_arr)} < {true_len}, downloading..')
                await download_to_db(exchange, stf, down_tf, start_ms, end_ms, check_exist=False)


async def fill_holes():
    logger.info('find and fill holes for kline...')
    for item in KLine.agg_list:
        await _fill_tf_hole(item.tf)


def sync_timeframes():
    '''
    检查各kline表的数据一致性，如果低维度数据比高维度多，则聚合更新到高维度
    应在机器人启动时调用
    '''
    start = time.monotonic()
    sid_ranges = KLine.load_kline_ranges()
    if not sid_ranges:
        return
    logger.info('try sync timeframes for klines...')
    sid_list = [(*k, *v) for k, v in sid_ranges.items()]
    sid_list = sorted(sid_list, key=lambda x: x[0])
    all_tf = set(map(lambda x: x.tf, KLine.agg_list))
    first = sid_list[0]
    tf_ranges = {first[1]: (tf_to_secs(first[1]), *first[2:])}  # {tf: (tf_secs, min_time, max_time)}
    prev_sid = sid_list[0][0]

    def call_sync():
        lack_tf = all_tf.difference(tf_ranges.keys())
        for tf in lack_tf:
            tf_ranges[tf] = (tf_to_secs(tf), 0, 0)
        tf_list = [(k, *v) for k, v in tf_ranges.items()]
        tf_list = sorted(tf_list, key=lambda x: x[1])
        _sync_kline_sid(prev_sid, tf_list)

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


def _sync_kline_sid(sid: int, tf_list: List[Tuple[str, int, int, int]]):
    '''
    检查给定的sid，各个周期数据是否有不一致，尝试用低维度数据更新高维度数据
    '''
    sess = db.session
    for i in range(len(tf_list) - 1):
        if not tf_list[i][2]:
            continue
        _sync_kline_sid_tf(sess, sid, tf_list, i)
    sess.commit()


def _sync_kline_sid_tf(sess: SqlSession, sid: int, tf_list: List[Tuple[str, int, int, int]], base_id: int):
    '''
    检查给定sid的指定周期数据，是否可更新更大周期数据。必要时进行更新。
    '''
    stf, secs, start, end = tf_list[base_id]
    base_tbl = f'kline_{stf}'
    for i, par in enumerate(tf_list):
        if i <= base_id:
            continue
        ptf, psec, pstart, pend = par
        agg_tbl = KLine.agg_map[ptf]
        if not pstart or not pend:
            min_time, max_time = KLine.refresh_agg(sess, agg_tbl, sid, start, end, base_tbl)
            tf_list[i] = (*par[:2], min_time, max_time)
            continue
        min_time, max_time = None, None
        if start < pstart:
            min_time, max_time = KLine.refresh_agg(sess, agg_tbl, sid, start, pstart, base_tbl)
        if end > pend:
            min_time, max_time = KLine.refresh_agg(sess, agg_tbl, sid, pend, end, base_tbl)
        if min_time:
            tf_list[i] = (*par[:2], min_time, max_time)


def correct_ohlcvs():
    '''
    遍历数据库的所有K线，从小周期到大周期。
    检查是否有大周期bar数据不正确，有则从小周期重新归集。
    '''
    with db():
        sess = db.session
        all_ranges = KLine.load_kline_ranges()
        all_sids = {k[0] for k in all_ranges.keys()}
        for sid in all_sids:
            exs: ExSymbol = sess.query(ExSymbol).get(sid)
            if not exs:
                logger.warning(f'no ExSymbol for: {sid}')
                continue
            exs = detach_obj(sess, exs)
            for agg in KLine.agg_list[1:]:
                sub_range = all_ranges.get((sid, agg.agg_from))
                if not sub_range:
                    continue
                _correct_sid_tf_ohlcv(exs, agg.agg_from, agg.tf, sub_range[0], sub_range[1])


def _correct_sid_tf_ohlcv(exs: ExSymbol, sub_tf: str, timeframe: str, start_ms: int, end_ms: int):
    # 计算子周期一次处理的数量
    batch_num = 3000
    sub_tfmsecs = tf_to_secs(sub_tf) * 1000
    tf_secs = tf_to_secs(timeframe)
    tf_msecs = tf_secs * 1000
    assert tf_msecs % sub_tfmsecs == 0
    factor = int(round(tf_msecs / sub_tfmsecs))
    batch_num = batch_num // factor * factor

    # 计算此次父周期涉及的时间范围
    big_start = start_ms // tf_msecs * tf_msecs
    if big_start < start_ms:
        big_start += tf_msecs
    big_end = end_ms // tf_msecs * tf_msecs

    cur_start = big_start
    while cur_start < big_end:
        # 筛选当前批次子周期和大周期ohlcv，对比分析更新
        batch_end = cur_start + batch_num * sub_tfmsecs
        sub_ohlcvs = KLine.query(exs, sub_tf, cur_start, batch_end)
        ohlcvs = KLine.query(exs, timeframe, cur_start, batch_end)
        agg_ohlcvs, last_finish = build_ohlcvc(sub_ohlcvs, tf_secs)
        if not last_finish:
            agg_ohlcvs = agg_ohlcvs[:-1]
        _correct_ohlcv_range(exs.id, timeframe, ohlcvs, agg_ohlcvs)
        cur_start = batch_end


def _correct_ohlcv_range(sid: int, timeframe: str, ohlcvs: List[Tuple], agg_ohlcvs: List[Tuple]):
    '''
    对比分析给定大周期的ohlcv和从子周期聚合的ohlcv，纠正错误的蜡烛数据
    '''
    cur_id, agg_id = 0, 0
    ins_rows = []
    sess = db.session
    while cur_id < len(ohlcvs) and agg_id < len(agg_ohlcvs):
        cur_bar, agg_bar = ohlcvs[cur_id], agg_ohlcvs[agg_id]
        if cur_bar[0] < agg_bar[0]:
            cur_id += 1
            continue
        if agg_bar[0] < cur_bar[0]:
            ins_rows.append(agg_bar)
            agg_id += 1
            continue
        price_chg = max(
            abs(agg_bar[1] - cur_bar[1]),
            abs(agg_bar[2] - cur_bar[2]),
            abs(agg_bar[3] - cur_bar[3]),
            abs(agg_bar[4] - cur_bar[4]),
        )
        chg_pct = price_chg / max(agg_bar[2], cur_bar[2])
        price_valid = chg_pct < 0.01 and price_chg < 1
        vol_chg = agg_bar[5] - cur_bar[5]
        if not vol_chg:
            vol_pct = 0
        else:
            vol_pct = vol_chg / max(agg_bar[5], cur_bar[5])
        if price_valid and vol_pct < 0.03:
            agg_id += 1
            cur_id += 1
            continue
        cols = f'open=:open,high=:high,low=:low,close=:close,volume=:volume'
        bar_ts = cur_bar[0] // 1000
        upd_sql = f'update kline_{timeframe} set {cols} where sid={sid} and "time"=to_timestamp({bar_ts})'
        params = dict(open=agg_bar[1], high=agg_bar[2], low=agg_bar[3], close=agg_bar[4], volume=agg_bar[5])
        exc_res = sess.execute(sa.text(upd_sql), params)
        logger.info(f'update {exc_res.rowcount} bar: {sid} {timeframe} {agg_bar} for {cur_bar}')
        agg_id += 1
        cur_id += 1
    sess.commit()
    if agg_id < len(agg_ohlcvs):
        ins_rows.extend(agg_ohlcvs[agg_id:])
    if ins_rows:
        KLine.force_insert(sid, timeframe, ins_rows)
        ins_ts = [row[0] // 1000 for row in ins_rows]
        logger.info(f'insert {sid} {timeframe}: {ins_ts}')
