#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : od_analyze.py
# Author: anyongjin
# Date  : 2023/5/7

import numpy as np

from banbot.config import AppConfig
from banbot.config.consts import *
from banbot.storage import *
from banbot.util.common import logger


def calc_overlap_rate(od1, od2):
    x1, x2 = od1.enter_at, od1.exit_at
    y1, y2 = od2.enter_at, od2.exit_at
    overlap = max(0, min(x2, y2) - max(x1, y1))
    if overlap <= 0:
        return 0
    return overlap / (x2 - x1) / 2 + overlap / (y2 - y1) / 2


async def compare_task_orders(bt_task_id: int, live_task_ids: List[int]):
    sess = dba.session
    live_stats = select(InOutOrder).where(InOutOrder.task_id.in_(set(live_task_ids)))
    live_ods: List[InOutOrder] = list(await sess.scalars(live_stats))
    bt_od_stat = select(InOutOrder).where(InOutOrder.task_id == bt_task_id)
    bt_ods: List[InOutOrder] = list(await sess.scalars(bt_od_stat))
    logger.info(f'load orders ok, backtest: {len(bt_ods)}, live: {len(live_ods)}')
    # 查找回测订单和实盘订单的最匹配对。（按入场时间和离场时间）
    pair_mats, unmat_lives = [], []
    for lod in live_ods:
        if not lod.exit_at:
            lod.exit_at = lod.enter_at + 1
        mid_time = (lod.enter_at + lod.exit_at) / 2
        pos_bts = [od for od in bt_ods if od.enter_at < mid_time < od.exit_at]
        if not pos_bts:
            unmat_lives.append(lod)
            continue
        if len(pos_bts) == 1:
            pair_mats.append((lod.id, pos_bts[0].id, calc_overlap_rate(lod, pos_bts[0])))
            continue
        candis = []
        for btod in pos_bts:
            candis.append((btod.id, calc_overlap_rate(lod, btod)))
        candis = sorted(candis, key=lambda x: x[1], reverse=True)
        pair_mats.append((lod.id, *candis[0]))
    from itertools import groupby
    pair_mats = sorted(pair_mats, key=lambda x: x[1])
    gps = groupby(pair_mats, key=lambda x: x[1])
    od_mats = []
    for key, gp in gps:
        gp_list = list(gp)
        if len(gp_list) == 1:
            od_mats.append(gp_list[0])
            continue
        gp_list = sorted(gp_list, key=lambda x: x[2], reverse=True)
        od_mats.append(gp_list[0])
    logger.info(f"{len(pair_mats)} candicates to {len(od_mats)} pairs, unmatch lives: {len(unmat_lives)}")
    # 分析实盘订单和回测订单的匹配程度。
    # Step 1：已匹配的订单，重合程度
    from banbot.util.num_utils import cluster_kmeans
    from tabulate import tabulate
    mat_scores = [mat[2] for mat in od_mats]
    row_gps, centers = cluster_kmeans(np.array(mat_scores), 7)
    score_gps = groupby(sorted(row_gps))
    sgps = [(key, len(list(sgp))) for key, sgp in score_gps]
    sgps = sorted(sgps, key=lambda x: x[1], reverse=True)
    gp_rows = [(num, centers[gp] * 100) for gp, num in sgps]
    gp_head = ['Group Num', 'Match Score']
    hd_fmts = ['d', '.1f']
    gp_text = tabulate(gp_rows, gp_head, 'orgtbl', hd_fmts)
    print(' Order Match Scores '.center(len(gp_text.splitlines()[0]), '='))
    print(gp_text)
    avg_score = sum(mat_scores) * 100 / len(mat_scores)
    logger.info(f"Avg Match Score: {avg_score:.1f}")


async def compare_orders(task_ids: List[int], task_hash: str):
    from banbot.strategy.resolver import StrategyResolver
    config = AppConfig.get()
    sess = dba.session
    if not task_ids:
        if not task_hash:
            StrategyResolver.load_run_jobs(config, [])
            task_hash = BotGlobal.stg_hash
            logger.info(f'use current task hash: {task_hash}')
        where_list = [BotTask.stg_hash == task_hash]
        filter_text = 'task_hash:' + task_hash
    else:
        where_list = [BotTask.id.in_(set(task_ids))]
        filter_text = 'task_ids:' + str(task_ids)
    bt_wheres = where_list + [BotTask.mode == RunMode.BACKTEST.value]
    qtask_st = select(BotTask).where(*bt_wheres).order_by(BotTask.create_at.desc()).limit(1)
    bt_task: BotTask = (await sess.scalars(qtask_st)).first()
    if not bt_task:
        logger.error('no Backtest Task found for ' + filter_text)
        return
    logger.info(f'found Backtest Task: {bt_task.id}')
    live_wheres = where_list + [BotTask.mode == RunMode.PROD.value]
    lv_stat = select(BotTask).where(*live_wheres).order_by(BotTask.id)
    live_tasks: List[BotTask] = list(await sess.scalars(lv_stat))
    if not live_tasks:
        logger.error('no live tasks found for ' + filter_text)
        return
    logger.info(f'found live tasks: {[t.id for t in live_tasks]}')
    lv_tids = [ltask.id for ltask in live_tasks]
    await compare_task_orders(bt_task.id, lv_tids)


async def run_od_compare(args: Dict[str, Any]):
    from banbot.storage.base import init_db
    init_db()
    task_hash, task_id = args.get('task_hash'), args.get('task_id')
    async with dba():
        await compare_orders(task_id, task_hash)

