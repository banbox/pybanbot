#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : base.py
# Author: anyongjin
# Date  : 2023/4/24
import asyncio
import contextlib
import copy
import os
import threading
import time
import traceback
from contextvars import ContextVar
from typing import Optional, List, Union, Type, Dict, Callable, Any

import six
import sqlalchemy as sa
from sqlalchemy import create_engine, pool, orm, select, update, delete, insert  # noqa
from sqlalchemy import event as db_event
from sqlalchemy.orm.session import make_transient, sessionmaker, Session
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, MappedColumn  # noqa
from sqlalchemy.types import TypeDecorator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker, AsyncEngine, AsyncAttrs
from sqlalchemy.ext.asyncio import AsyncSession as SqlSession

from banbot.config import AppConfig
from banbot.util.common import logger

_db_sess_asy: ContextVar[Optional[AsyncSession]] = ContextVar('_db_sess_asy', default=None)
db_slow_query_timeout = 1

_db_engines: Dict[int, AsyncEngine] = dict()
'维护连接池的字典，每个线程一个连接池'

_DbSessionCls: Dict[int, async_sessionmaker] = dict()
'维护_DbSession的类的字典，每个线程一个类'

_db_engine_ac: Optional[AsyncEngine] = None
'开启了AutoCommit的连接池，且使用NullPoll，可多线程访问'

_DbSessionClsAC: Optional[async_sessionmaker] = None
'_db_engine_ac的Session类，可多线程访问'


def reset_ctx():
    """
    仅用于异步任务重置_db_sess_asy
    """
    _db_sess_asy.set(None)


def get_db_cfg(db_url: str = None, for_async=True) -> dict:
    db_cfg = AppConfig.get().get('database')
    if not db_cfg:
        db_cfg = dict()
    else:
        db_cfg = copy.copy(db_cfg)
        if 'retention' in db_cfg:
            db_cfg.pop('retention')
    if not db_cfg.get('url'):
        if not db_url:
            db_url = os.environ.get('ban_db_url')
            if not db_url:
                raise ValueError('`database` is required in config. or set env: `ban_db_url`')
        db_cfg['url'] = db_url
    if for_async:
        db_cfg['url'] = db_cfg['url'].replace('postgresql:', 'postgresql+asyncpg:')
    if not db_cfg.get('pool_size'):
        db_cfg['pool_size'] = 20
    return db_cfg


def init_db(debug: Optional[bool] = None, db_url: str = None) -> AsyncEngine:
    '''
    初始化数据库客户端（并未连接到数据库）
    传入的参数将针对所有连接生效。
    如只需暂时生效：
    dba.session.connection().execution_options(isolation_level='AUTOCOMMIT')
    或engine.echo=True
    '''
    global _db_engine_ac, _DbSessionClsAC
    thread_id = threading.get_ident()
    if thread_id in _db_engines:
        return _db_engines[thread_id]
    db_cfgs = get_db_cfg(db_url)
    db_url = db_cfgs['url']
    logger.info(f'db url:{db_url}')
    # pool_recycle 连接过期时间，根据mysql服务器端连接的存活时间wait_timeout略小些
    create_args = dict(**db_cfgs, pool_recycle=3600, poolclass=pool.QueuePool, max_overflow=0)
    if debug is not None:
        create_args['echo'] = debug
    # 实例化异步engine
    db_engine = create_async_engine(**create_args)
    SyncSession = sessionmaker(db_engine.sync_engine, class_=Session)
    DbSession = async_sessionmaker(db_engine, class_=AsyncSession,
                                   sync_session_class=SyncSession)
    set_db_events(db_engine.sync_engine, SyncSession)
    _db_engines[thread_id] = db_engine
    _DbSessionCls[thread_id] = DbSession
    if _db_engine_ac is None:
        # 初始化跨进程AUTOCOMMIT连接池
        create_args = dict(
            url=db_url,
            poolclass=pool.QueuePool,
            isolation_level='AUTOCOMMIT'
        )
        _db_engine_ac = create_async_engine(**create_args)
        SyncSession = sessionmaker(_db_engine_ac.sync_engine, class_=Session)
        _DbSessionClsAC = async_sessionmaker(_db_engine_ac, class_=AsyncSession,
                                             sync_session_class=SyncSession)
        set_db_events(_db_engine_ac.sync_engine, SyncSession)
    return db_engine


class DBSessionAsyncMeta(type):
    @property
    def session(cls) -> SqlSession:
        session = _db_sess_asy.get()
        if session is None:
            raise RuntimeError('db sess not loaded')
        return session

    @classmethod
    def new_session(cls, **kwargs):
        thread_id = threading.get_ident()
        if thread_id not in _db_engines:
            init_db()
        DbSession = _DbSessionCls[thread_id]
        return DbSession(**kwargs)

    @contextlib.asynccontextmanager
    async def autocommit(cls, **kwargs):
        if not _DbSessionClsAC:
            init_db()
        async with _DbSessionClsAC(**kwargs) as sess:
            yield sess


class DBSessionAsync(metaclass=DBSessionAsyncMeta):
    _callbacks: Dict[int, List[Callable]] = dict()

    def __init__(self, session_args: Dict = None, commit_on_exit: bool = True):
        self.token = None
        self.session_args = session_args or {}
        self.commit_on_exit = commit_on_exit

    async def __aenter__(self):
        if _db_sess_asy.get() is None:
            thread_id = threading.get_ident()
            if thread_id not in _db_engines:
                init_db()
            DbSession = _DbSessionCls[thread_id]
            sess = DbSession(**self.session_args)
            self.token = _db_sess_asy.set(sess)
            logger.debug('dbsess create %s %s %s', sess, id(sess), traceback.format_stack()[-2])
        return type(self)

    async def __aexit__(self, exc_type, exc_value, traceinfo):
        if not self.token:
            return
        sess = _db_sess_asy.get()
        if not sess:
            return

        success = False
        reason = 'none'
        try:
            if exc_type is not None:
                reason = (exc_type, exc_value)
                await sess.rollback()
            elif self.commit_on_exit:
                await sess.commit()
                success = True
            else:
                reason = f'not commit on exit'
            await asyncio.shield(sess.close())
        except Exception as e:
            stack = "\n".join(traceback.format_stack())
            try:
                sync_sess = sess.sync_session
                conn = await sess.connection()
            except Exception as ex:
                conn = str(ex)
                sync_sess = None
            logger.exception(
                f'dbsess fail: {sess} {sync_sess} {conn} {exc_type} {exc_value} inexc: {traceinfo} {stack} {e}')
        if not success:
            logger.debug('not commit: %s %s', reason, traceback.format_stack()[-2])
        else:
            logger.debug('committed: %s %s', sess, traceback.format_stack()[-2])

        key = id(sess)
        if key in self._callbacks:
            from banbot.util.misc import run_async
            for cb in self._callbacks[key]:
                try:
                    await run_async(cb, success)
                except Exception:
                    logger.exception('run callback after session commit error')
            del self._callbacks[key]

        _db_sess_asy.set(None)
        self.token = None

    @classmethod
    def add_callback(cls, sess: SqlSession, cb: Callable):
        key = id(sess)
        if key not in cls._callbacks:
            cls._callbacks[key] = []
        cls._callbacks[key].append(cb)


dba = DBSessionAsync


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


class BaseDbModel(AsyncAttrs, DeclarativeBase):

    def clone(self):
        all_obj_keys = set(self.__dict__.keys())
        db_keys = set(self.__table__.columns.keys())
        save_keys = db_keys.intersection(all_obj_keys)
        kwargs = dict()
        for k in save_keys:
            val = getattr(self, k)
            if val is None:
                continue
            kwargs[k] = val
        return self.__class__(**kwargs)

    def dict(self, only: List[Union[str, MappedColumn[Any]]] = None, skips: List[Union[str, MappedColumn[Any]]] = None):
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
            try:
                setattr(self, k, v)
            except Exception:
                continue

    @classmethod
    def fields(cls, kwargs):
        all_cols = cls.__table__.columns
        allow_kwargs = {}
        for k in kwargs:
            if hasattr(all_cols, k):
                allow_kwargs[k] = kwargs[k]
        return allow_kwargs


def set_db_events(engine, Session):
    # from banbot.util import btime
    #
    # if btime.debug:
    #     @db_event.listens_for(engine, 'checkin')
    #     def receive_checkin(dbapi_connection, connection_record):
    #         logger.debug('[db] conn return to pool: %s %s', dbapi_connection, connection_record)
    #
    #     @db_event.listens_for(engine, 'checkout')
    #     def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    #         logger.debug('[db] conn retrieve from pool: %s %s %s', dbapi_connection, connection_record, connection_proxy)
    #
    #     @db_event.listens_for(engine, 'close')
    #     def receive_close(dbapi_connection, connection_record):
    #         logger.debug('[db] conn closed: %s %s', dbapi_connection, connection_record)
    #
    #     @db_event.listens_for(engine, 'close_detached')
    #     def receive_close_detached(dbapi_connection):
    #         logger.debug('[db] conn close_detached: %s', dbapi_connection)
    #
    #     @db_event.listens_for(engine, 'connect')
    #     def receive_connect(dbapi_connection, connection_record):
    #         logger.debug('[db] conn connect: %s %s', dbapi_connection, connection_record)
    #
    #     @db_event.listens_for(engine, 'detach')
    #     def receive_detach(dbapi_connection, connection_record):
    #         logger.debug('[db] conn detach: %s %s', dbapi_connection, connection_record)
    #
    #     @db_event.listens_for(engine, 'first_connect')
    #     def receive_first_connect(dbapi_connection, connection_record):
    #         logger.debug('[db] conn first_connect: %s %s', dbapi_connection, connection_record)
    #
    #     @db_event.listens_for(engine, 'invalidate')
    #     def receive_invalidate(dbapi_connection, connection_record, exception):
    #         logger.debug('[db] conn invalidate: %s %s %s', dbapi_connection, connection_record, exception)
    #
    #     @db_event.listens_for(engine, 'reset')
    #     def receive_reset(dbapi_connection, connection_record, reset_state):
    #         logger.debug('[db] conn reset: %s %s %s', dbapi_connection, connection_record, reset_state)
    #
    #     @db_event.listens_for(engine, 'soft_invalidate')
    #     def receive_soft_invalidate(dbapi_connection, connection_record, exception):
    #         logger.debug('[db] conn soft_invalidate: %s %s %s', dbapi_connection, connection_record, exception)
    #
    #     @db_event.listens_for(Session, 'after_begin')
    #     def receive_after_begin(session, transaction, connection):
    #         logger.debug('[db] sess after_begin: %s %s %s', session, transaction, connection)
    #
    #     @db_event.listens_for(Session, 'after_commit')
    #     def receive_after_commit(session):
    #         logger.debug('[db] sess after_commit: %s', session)
    #
    #     @db_event.listens_for(Session, 'after_flush')
    #     def receive_after_flush(session, flush_context):
    #         logger.debug('[db] sess after_flush: %s %s', session, flush_context)
    #
    #     @db_event.listens_for(Session, 'after_rollback')
    #     def receive_after_rollback(session):
    #         logger.debug('[db] sess after_rollback: %s', session)
    #
    #     @db_event.listens_for(Session, 'after_soft_rollback')
    #     def receive_after_soft_rollback(session, previous_transaction):
    #         logger.debug('[db] sess after_soft_rollback: %s %s', session, previous_transaction)
    #
    #     @db_event.listens_for(Session, 'after_transaction_create')
    #     def receive_after_transaction_create(session, transaction):
    #         logger.debug('[db] sess after_transaction_create: %s %s', session, transaction)
    #
    #     @db_event.listens_for(Session, 'after_transaction_end')
    #     def receive_after_transaction_end(session, transaction):
    #         logger.debug('[db] sess after_transaction_end: %s %s', session, transaction)
    #
    #     @db_event.listens_for(Session, 'do_orm_execute')
    #     def receive_do_orm_execute(orm_execute_state):
    #         logger.debug('[db] sess do_orm_execute: %s', orm_execute_state)

    @db_event.listens_for(engine, 'before_cursor_execute')
    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        try:
            # max_len = 500
            # if isinstance(parameters, (list, tuple)) and len(parameters) > 30:
            #     if isinstance(parameters[0], (list, tuple, dict)):
            #         parameters = f'[{parameters[0]}, len: {len(parameters)}]'
            #     elif len(parameters) > max_len:
            #         parameters = f'({parameters[:max_len]}, len: {len(parameters)})'
            # if statement and len(statement) > max_len:
            #     statement = f'{str(statement)[:max_len]}... len: {len(statement)}'
            # args = [conn, conn.connection, cursor, statement, parameters, context, executemany]
            # logger.debug('[db] conn before_cursor_execute %s %s %s %s %s %s %s', *args)
            conn.info['query_start_time'] = time.monotonic()
        except Exception:
            logger.exception(f'log sql execute start time fail: {statement}')

    @db_event.listens_for(engine, "after_cursor_execute")
    def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        try:
            total = time.monotonic() - conn.info['query_start_time']
            if total > db_slow_query_timeout:
                if not str(statement).lower().strip().startswith('select'):
                    return
                logger.warn(f'Slow Query Found！Cost {total * 1000:.1f} ms: {statement}')
        except Exception:
            logger.exception(f'measure execute cost fail: {statement}')


def detach_obj(sess: SqlSession, obj: BaseDbModel, keep_map=False):
    """
    将一个对象从DbSession中解除关联。注意：对此对象的修改将会丢失！如果需要保存修改到数据库，请使用clone
    :param sess:
    :param obj: 要解除关联的对象
    :param keep_map: 默认False，为True时无法访问属性的值
    """
    if obj in sess:
        sess.expunge(obj)
    if not keep_map:
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
