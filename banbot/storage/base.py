#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : base.py
# Author: anyongjin
# Date  : 2023/4/24
import os
import time
import threading
import traceback
from contextvars import ContextVar, copy_context
from typing import Optional, List, Union, Type, Dict

import six
import sqlalchemy as sa
from sqlalchemy import create_engine, pool, Column, orm  # noqa
from sqlalchemy import event as db_event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session as SqlSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import make_transient
from sqlalchemy.types import TypeDecorator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio.engine import Engine as AsyEngine

from banbot.config import AppConfig
from banbot.util.common import logger
from banbot.util import btime

_BaseDbModel = declarative_base()
_db_engine: Optional[Engine] = None
_DbSession: Optional[sessionmaker] = None
_db_sess: ContextVar[Optional[SqlSession]] = ContextVar('_db_sess', default=None)
_db_engine_asy: Optional[AsyEngine] = None
_DbSessionAsync: Optional[sessionmaker] = None
_db_sess_asy: ContextVar[Optional[AsyncSession]] = ContextVar('_db_sess_asy', default=None)
db_slow_query_timeout = 1


def init_db(iso_level: Optional[str] = None, debug: Optional[bool] = None, db_url: str = None):
    '''
    初始化数据库客户端（并未连接到数据库）
    传入的参数将针对所有连接生效。
    如只需暂时生效：
    db.session.connection().execution_options(isolation_level='AUTOCOMMIT')
    或engine.echo=True
    '''
    global _db_engine, _DbSession, _db_engine_asy, _DbSessionAsync
    if _db_engine is not None:
        return _db_engine
    if not db_url:
        db_url = os.environ.get('ban_db_url')
    try:
        db_cfg = AppConfig.get()['database']
    except Exception:
        assert db_url, '`db_url` is required if config not avaiable'
        db_cfg = dict(url=db_url)
    if not db_url:
        db_url = db_cfg['url']
    pool_size = db_cfg.get('pool_size', 20)
    logger.info(f'db url:{db_url}')
    # pool_recycle 连接过期时间，根据mysql服务器端连接的存活时间wait_timeout略小些
    create_args = dict(pool_recycle=3600, poolclass=pool.QueuePool, pool_size=pool_size, max_overflow=0)
    if debug is not None:
        create_args['echo'] = debug
    create_args['isolation_level'] = iso_level
    _db_engine = create_engine(db_url, **create_args)
    _DbSession = sessionmaker(bind=_db_engine)
    # 实例化异步engine
    db_url = db_url.replace('postgresql:', 'postgresql+asyncpg:')
    _db_engine_asy = create_async_engine(db_url, **create_args)
    _DbSessionAsync = sessionmaker(_db_engine_asy, class_=AsyncSession)
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
    _hold_map = dict()

    def __new__(cls, *args, **kwargs):
        if cls._sess and cls._sess.token and cls._sess.fetch_tid == threading.get_ident():
            return cls._sess
        cls._sess = super().__new__(cls)
        return cls._sess

    def __init__(self, session_args: Dict = None, commit_on_exit: bool = False):
        self.token = None
        self.session_args = session_args or {}
        self.commit_on_exit = commit_on_exit
        self.fetch_tid = 0

    def __enter__(self):
        if _db_sess.get() is None:
            sess = _DbSession(**self.session_args)
            self.token = _db_sess.set(sess)
            self.fetch_tid = threading.get_ident()
            # logger.debug('[%s] set new dbSession: %s, in ctx: %s', self.fetch_tid, sess, copy_context())
            cur_time = btime.time()
            self._hold_map[id(sess)] = (cur_time, traceback.format_stack())
            if round(cur_time) % 60 == 0:
                bad_list = []
                for key, item in self._hold_map.items():
                    if cur_time - item[0] > 300:
                        bad_list.append(f'[{key}] {btime.to_datestr(item[0])} {item[1]}')
                if bad_list:
                    bad_text = "\n".join(bad_list)
                    logger.warning(f'timeout db sess: {bad_text}')
        return type(self)

    def __exit__(self, exc_type, exc_value, traceback):
        sess = _db_sess.get()
        if not sess:
            return

        if exc_type is not None:
            sess.rollback()

        if self.commit_on_exit:
            sess.commit()

        if self.token and self.fetch_tid == threading.get_ident():
            sess.close()
            # logger.debug('[%s] close dbSession: %s, in ctx: %s',self.fetch_tid, sess, copy_context())
            _db_sess.set(None)
            self.token = None
            sess_key = id(sess)
            if sess_key in self._hold_map:
                del self._hold_map[sess_key]


db: DBSessionMeta = DBSession


class DBSessionAsyncMeta(type):
    @property
    def session(cls) -> AsyncSession:
        session = _db_sess_asy.get()
        if session is None:
            raise RuntimeError('db sess not loaded')
        return session


class DBSessionAsync(metaclass=DBSessionAsyncMeta):
    _sess = None

    def __new__(cls, *args, **kwargs):
        if cls._sess and cls._sess.token and cls._sess.fetch_tid == threading.get_ident():
            return cls._sess
        cls._sess = super().__new__(cls)
        return cls._sess

    def __init__(self, session_args: Dict = None, commit_on_exit: bool = False):
        self.token = None
        self.session_args = session_args or {}
        self.commit_on_exit = commit_on_exit
        self.fetch_tid = 0

    async def __aenter__(self):
        if _db_sess_asy.get() is None:
            self.token = _db_sess_asy.set(_DbSessionAsync(**self.session_args))
            self.fetch_tid = threading.get_ident()
        return type(self)

    async def __aexit__(self, exc_type, exc_value, traceback):
        sess = _db_sess_asy.get()
        if not sess:
            return

        if exc_type is not None:
            await sess.rollback()

        if self.commit_on_exit:
            await sess.commit()

        await sess.close()

        if self.token and self.fetch_tid == threading.get_ident():
            _db_sess_asy.set(None)
            self.token = None


dba: DBSessionAsyncMeta = DBSessionAsync


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
        logger.warn(f'Slow Query Found！Cost {total * 1000:.1f} ms: {statement}')


def detach_obj(sess: SqlSession, obj: BaseDbModel):
    sess.expunge(obj)
    make_transient(obj)
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
