#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : scripts.py
# Author: anyongjin
# Date  : 2023/4/24

from banbot.storage import *
from banbot import storage
from banbot.storage.base import BaseDbModel, sa, init_db
from banbot.util.common import logger
from banbot.util.misc import get_module_classes


all_tables = get_module_classes(storage, BaseDbModel)
tbl_map: Dict[str, BaseDbModel] = dict()
for tbl in all_tables:
    tbl_map[tbl.__tablename__.lower()] = tbl
    tbl_map[tbl.__class__.__name__.lower()] = tbl


def get_db_tbls(conn):
    inspector = sa.inspect(conn)
    return set(inspector.get_table_names())


async def get_fail_tables():
    async with init_db().connect() as conn:
        db_tbls = await conn.run_sync(get_db_tbls)
        exist_tbls = [tbl for tbl in all_tables if tbl.__tablename__ in db_tbls]
    return list(set(all_tables) - set(exist_tbls))


async def rebuild_db(tables: list = None, skip_exist=True, require_confirm=True):
    logger.info('start rebuild tables...')
    if not tables:
        tables = all_tables
    engine = init_db()
    bandb = engine.sync_engine
    async with engine.connect() as conn:
        bak_iso_level = await conn.get_isolation_level()
        await conn.execution_options(isolation_level='AUTOCOMMIT')
        exist_tbls = [tbl for tbl in tables if sa.inspect(bandb).has_table(tbl.__tablename__)]
        if skip_exist and exist_tbls:
            tables = list(set(tables) - set(exist_tbls))
        if not tables:
            print('No Tables need to create, all exists!')
            return
        bandb.echo = True
        print('=======  Tables to Create:', [t.__name__ for t in tables])
        print('=======  Database:', bandb.url)
        if not skip_exist and exist_tbls:
            del_names = ','.join(tbl.__tablename__ for tbl in exist_tbls)
            print(f'*******  Tables Would Be Deleted: {del_names} ********')
        if require_confirm:
            flag = input('input `yes` to continue:\n')
            if not flag or flag.strip() != 'yes':
                raise Exception('user cancled')
        if not skip_exist and exist_tbls:
            left_tbls = []
            for t in exist_tbls:
                if hasattr(t, 'drop_tbl'):
                    await t.drop_tbl(conn)
                else:
                    left_tbls.append(t)
            if left_tbls:
                BaseDbModel.metadata.drop_all(bandb, [t.__table__ for t in left_tbls])
        BaseDbModel.metadata.create_all(bandb, [t.__table__ for t in tables])
        # 执行表的初始化
        for t in tables:
            if hasattr(t, 'init_tbl'):
                await t.init_tbl(conn)
        await conn.execution_options(isolation_level=bak_iso_level)
    bandb.echo = False
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


async def exec_dbcmd(args: Dict[str, Any]):
    from banbot.config import AppConfig
    AppConfig.init_by_args(args)
    action = args['action']
    async with dba():
        if action == 'rebuild':
            await rebuild_db(_parse_tbls(args), not args['force'], not args.get('yes'))
        elif action == 'correct':
            from banbot.data.toolbox import correct_ohlcvs
            await correct_ohlcvs()
        else:
            raise RuntimeError(f'unsupport db action: {action}')
