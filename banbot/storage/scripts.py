#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : scripts.py
# Author: anyongjin
# Date  : 2023/4/24
from typing import Any, Dict, List

from banbot.storage import *
from banbot.storage.base import BaseDbModel, sa, init_db
from banbot.util.common import logger

all_tables = [ExSymbol, KLine, KHole, BotTask, Order, InOutOrder]
tbl_map: Dict[str, BaseDbModel] = dict()
for tbl in all_tables:
    tbl_map[tbl.__tablename__.lower()] = tbl
    tbl_map[tbl.__class__.__name__.lower()] = tbl


def rebuild_db(tables: list = None, skip_exist=True):
    logger.info('start rebuild tables...')
    if not tables:
        tables = all_tables
    bandb = init_db()
    bandb.echo = True
    conn = db.session.connection()
    conn.commit()
    conn.execution_options(isolation_level='AUTOCOMMIT')
    exist_tbls = [tbl for tbl in tables if sa.inspect(bandb).has_table(tbl.__tablename__)]
    if skip_exist and exist_tbls:
        tables = list(set(tables) - set(exist_tbls))
    if not tables:
        print('No Tables need to create, all exists!')
        return
    print('=======  Tables to Create:', [t.__name__ for t in tables])
    print('=======  Database:', bandb.url)
    if not skip_exist and exist_tbls:
        del_names = ','.join(tbl.__tablename__ for tbl in exist_tbls)
        print(f'*******  Tables Would Be Deleted: {del_names} ********')
    flag = input('input `yes` to continue:\n')
    if not flag or flag.strip() != 'yes':
        raise Exception('user cancled')
    if not skip_exist and exist_tbls:
        left_tbls = []
        for t in exist_tbls:
            if hasattr(t, 'drop_tbl'):
                t.drop_tbl(conn)
            else:
                left_tbls.append(t)
        if left_tbls:
            BaseDbModel.metadata.drop_all(bandb, [t.__table__ for t in left_tbls])
    BaseDbModel.metadata.create_all(bandb, [t.__table__ for t in tables])
    # 执行表的初始化
    for t in tables:
        if hasattr(t, 'init_tbl'):
            t.init_tbl(conn)
    logger.info('rebuild db complete')


def _parse_tbls(args: Dict[str, Any]) -> List[BaseDbModel]:
    tables = []
    tbl_names = args.get('tables')
    if not tbl_names:
        return tables
    for tname in tbl_names:
        tbl = tbl_map.get(tname)
        if not tbl:
            raise RuntimeError(f'unknown table: {tname}')
        tables.append(tbl)
    return tables


def exec_dbcmd(args: Dict[str, Any]):
    from banbot.config import AppConfig
    AppConfig.init_by_args(args)
    action = args['action']
    with db():
        if action == 'rebuild':
            rebuild_db(_parse_tbls(args), not args['force'])
        else:
            raise RuntimeError(f'unsupport db action: {action}')
