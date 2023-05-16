#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : base.py
# Author: anyongjin
# Date  : 2023/4/24
import time
import six
import sqlalchemy as sa
from sqlalchemy import create_engine, pool, Column, orm
from sqlalchemy import event as db_event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session as SqlSession
from sqlalchemy.engine import Engine
from sqlalchemy.types import TypeDecorator
from sqlalchemy.orm.session import make_transient
from typing import Optional, List, Union, Type, Dict
from contextvars import ContextVar
from banbot.util.common import logger
from banbot.config import AppConfig
from banbot.util import btime


_BaseDbModel = declarative_base()
_db_engine: Optional[Engine] = None
_DbSession: Optional[sessionmaker] = None
_db_sess: ContextVar[Optional[SqlSession]] = ContextVar('_db_sess', default=None)
db_slow_query_timeout = 1


def init_db(iso_level: Optional[str] = None, debug: Optional[bool] = None, db_url: str = None):
    '''
    初始化数据库客户端（并未连接到数据库）
    传入的参数将针对所有连接生效。
    如只需暂时生效：
    db.session.connection().execution_options(isolation_level='AUTOCOMMIT')
    或engine.echo=True
    '''
    global _db_engine, _DbSession
    if _db_engine is not None:
        return _db_engine
    try:
        db_cfg = AppConfig.get()['database']
    except Exception:
        assert db_url, '`db_url` is required if config not avaiable'
        db_cfg = dict(url=db_url)
    db_url = db_cfg['url']
    pool_size = db_cfg.get('pool_size', 30) if btime.run_mode in btime.LIVE_MODES else 3
    max_psize = pool_size * 2
    logger.info(f'db url:{db_url}')
    # pool_recycle 连接过期时间，根据mysql服务器端连接的存活时间wait_timeout略小些
    create_args = dict(pool_recycle=3600, poolclass=pool.QueuePool, pool_size=pool_size, max_overflow=max_psize)
    if debug is not None:
        create_args['echo'] = debug
    create_args['isolation_level'] = iso_level
    _db_engine = create_engine(db_url, **create_args)
    _DbSession = sessionmaker(bind=_db_engine)
    return _db_engine


class DBSessionMeta(type):
    # using this metaclass means that we can access db.session as a property at a class level,
    # rather than db().session
    @property
    def session(self) -> SqlSession:
        """Return an instance of Session local to the current async context."""
        session = _db_sess.get()
        if session is None:
            raise RuntimeError('db sess not loaded')
        return session


class DBSession(metaclass=DBSessionMeta):
    _sess = None

    def __new__(cls, *args, **kwargs):
        if not cls._sess:
            cls._sess = super().__new__(cls, *args, **kwargs)
        return cls._sess

    def __init__(self, session_args: Dict = None, commit_on_exit: bool = False):
        self.token = None
        self.session_args = session_args or {}
        self.commit_on_exit = commit_on_exit

    def __enter__(self):
        if _db_sess.get() is None:
            self.token = _db_sess.set(_DbSession(**self.session_args))
            DBSession._sess = self
        return type(self)

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.token:
            return
        sess = _db_sess.get()
        if exc_type is not None:
            sess.rollback()

        if self.commit_on_exit:
            sess.commit()

        sess.close()
        _db_sess.reset(self.token)
        DBSession._sess = None


db: DBSessionMeta = DBSession


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
        if not str(statement).lower().strip().startswith('select'):
            return
        logger.warn(f'Slow Query Found！Cost {total:.2} secs: {statement}')


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
