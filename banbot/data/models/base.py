#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : base.py
# Author: anyongjin
# Date  : 2023/4/24
import time
import six
import sqlalchemy as sa
from sqlalchemy import create_engine, Column
from sqlalchemy import event as db_event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session as SqlSession
from sqlalchemy.engine import Engine
from sqlalchemy.types import TypeDecorator
from sqlalchemy.orm.session import make_transient
from typing import Optional, List, Union, Type
from banbot.util.common import logger
from banbot.config import AppConfig


_BaseDbModel = declarative_base()
_db_engine: Optional[Engine] = None
_DbSession: Optional[sessionmaker] = None
db_slow_query_timeout = 1


def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault('query_start_time', []).append(time.monotonic())


def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.monotonic() - conn.info['query_start_time'].pop(-1)
    if total > 1:
        logger.error(f'very slow database query found, cost {total:.3f}, {statement}, {parameters}')
    elif total > 0.3:
        logger.warning(f'slow database query found, cost {total:.3f}, {statement}, {parameters}')


def new_db(iso_level: Optional[str] = None, debug: Optional[bool] = None):
    db_url = AppConfig.get()['database']['url']
    logger.info(f'db url:{db_url}')
    # pool_recycle 连接过期时间，根据mysql服务器端连接的存活时间wait_timeout略小些
    create_args = dict(pool_recycle=300)
    if debug is not None:
        create_args['echo'] = debug
    create_args['isolation_level'] = iso_level
    engine = create_engine(db_url, **create_args)
    db_event.listens_for(engine, "before_cursor_execute", before_cursor_execute)
    db_event.listens_for(engine, "after_cursor_execute", after_cursor_execute)
    Session = sessionmaker(bind=engine)
    return engine, Session


def get_db(iso_level: Optional[str] = None, debug: Optional[bool] = None, cache=True):
    global _db_engine, _DbSession
    if _db_engine is not None and (debug is None or _db_engine.echo == debug) \
            and (not iso_level or _db_engine.dialect.isolation_level == iso_level):
        return _db_engine
    engine, Session = new_db(iso_level, debug)
    if cache:
        _db_engine, _DbSession = engine, Session
    return engine


def db_conn(iso_level: Optional[str] = None, debug: Optional[bool] = None, cache=True) -> sa.Connection:
    return get_db(iso_level, debug, cache).connect()


def db_sess(iso_level: Optional[str] = None, debug: Optional[bool] = None):
    get_db(iso_level, debug)
    return _DbSession()


def init_db_session():
    '''
    初始化数据库session
    在使用前必须初始化，可在bot启动时。
    :return:
    '''
    get_db()


class IntEnum(TypeDecorator):
    impl = sa.Integer
    cache_ok = True

    def __init__(self, enumtype, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enumtype = enumtype

    def process_bind_param(self, value, dialect):
        if hasattr(value, 'value'):
            return value.value
        return value

    def process_result_value(self, value, dialect):
        # 暂时禁用int转Enum，其他地方都是使用的int值
        # return self._enumtype(value)
        return value


class BaseDbModel(_BaseDbModel):
    __abstract__ = True

    def __init__(self, *args, **kwargs):
        all_cols = self.__table__.columns
        allow_kwargs = {}
        for k in kwargs:
            if hasattr(all_cols, k):
                allow_kwargs[k] = kwargs[k]
        super().__init__(*args, **allow_kwargs)

    def dict(self, only: List[Union[str, sa.Column]] = None, skips: List[Union[str, sa.Column]] = None):
        all_obj_keys = set(self.__dict__.keys())
        db_keys = set(self.__table__.columns.keys())
        tmp_keys = all_obj_keys - db_keys
        if not hasattr(self, '__table__'):
            return None
        all_cols = self.__table__.columns
        target_cols = []
        if only:
            for c in only:
                if isinstance(c, six.string_types):
                    target_cols.append(all_cols[c])
                else:
                    target_cols.append(c)
        else:
            target_cols.extend(all_cols.values())
        if skips:
            skips = [item.name if hasattr(item, 'name') else item for item in skips]
        data = {}
        for c in target_cols:
            if skips and c.name in skips:
                continue
            value = getattr(self, c.name, None)
            data[c.name] = value
            if isinstance(c.type, IntEnum):
                data[c.name + '_text'] = c.type._enumtype(value).name
            elif isinstance(c.type, sa.DateTime):
                data[c.name] = str(value).replace('T', ' ')
        # 添加临时字段
        for k in tmp_keys:
            if k.startswith('_'):
                continue
            val = getattr(self, k)
            if not val:
                continue
            data[k] = val
        return data

    def update_props(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@db_event.listens_for(Engine, 'before_cursor_execute')
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault('query_start_time', []).append(time.monotonic())


@db_event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.monotonic() - conn.info['query_start_time'].pop(-1)
    if total > db_slow_query_timeout:
        logger.warn(f'Slow Query Found！Cost over {db_slow_query_timeout} secs: {statement}')


def reset_obj(sess: SqlSession, obj: BaseDbModel):
    sess.expunge(obj)
    make_transient(obj)
    obj.id = None
    sess.add(obj)
    return obj


def row_to_dict(row):
    if isinstance(row, BaseDbModel):
        return row.dict()
    data = {}
    for k in row.keys():
        data[k] = getattr(row, k)
    return data


def row_to_object(obj_cls: Type[BaseDbModel], row):
    return obj_cls(**row_to_dict(row))
