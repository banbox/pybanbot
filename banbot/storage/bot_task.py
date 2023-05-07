#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : bot_task.py
# Author: anyongjin
# Date  : 2023/5/5
from banbot.strategy.base import *
from banbot.storage.common import BotGlobal


class BotTask(BaseDbModel):
    __tablename__ = 'bottask'

    cur_id: ClassVar[int] = 0
    obj: ClassVar['BotTask'] = None

    id = Column(sa.Integer, primary_key=True)
    mode = Column(sa.String(10))  # run mode
    stg_hash = Column(sa.String(64))
    create_at = Column(sa.DateTime)
    start_at = Column(sa.DateTime)
    stop_at = Column(sa.DateTime)

    @classmethod
    def init(cls, start_at: Optional[float] = None):
        if cls.obj is not None:
            return
        sess = db.session
        rmode = btime.run_mode.value
        where_list = [BotTask.mode == rmode, BotTask.stg_hash == BotGlobal.stg_hash]
        task = sess.query(BotTask).filter(*where_list).order_by(BotTask.create_at.desc()).first()
        live_mode = btime.run_mode in btime.TRADING_MODES
        if not task or not live_mode:
            # 非实盘模式下，不可重复使用一个任务，否则可能同一时刻多个完全相同的订单
            ctime = btime.to_datetime(time.time())
            task = BotTask(mode=rmode, create_at=ctime, stg_hash=BotGlobal.stg_hash)
            if live_mode:
                task.start_at = ctime
            elif start_at:
                task.start_at = btime.to_datetime(start_at)
            else:
                raise ValueError('start_at is required for backtesting')
            sess.add(task)
            sess.flush()
        cls.cur_id = task.id
        cls.obj = task
        sess.commit()

    @classmethod
    def update(cls, **kwargs):
        if not cls.obj:
            return
        sess = db.session
        sess.refresh(cls.obj)
        cls.obj.update_props(**kwargs)
        sess.commit()
