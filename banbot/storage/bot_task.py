#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : bot_task.py
# Author: anyongjin
# Date  : 2023/5/5
from banbot.strategy.base import *


class BotTask(BaseDbModel):
    __tablename__ = 'bottask'

    cur_id: ClassVar[int] = 0
    obj: ClassVar['BotTask'] = None

    id = Column(sa.Integer, primary_key=True)
    mode = Column(sa.String(10))  # run mode
    start_at = Column(sa.DateTime)
    stop_at = Column(sa.DateTime)

    @classmethod
    def init(cls):
        if cls.obj is not None:
            return
        sess = db.session
        rmode = btime.run_mode.value
        task = sess.query(BotTask).filter(BotTask.mode == rmode).order_by(BotTask.start_at.desc()).first()
        not_live_mode = btime.run_mode not in btime.TRADING_MODES
        if not task or not_live_mode:
            task = BotTask(mode=rmode, start_at=btime.to_datetime(time.time()))
            sess.add(task)
            sess.flush()
        cls.cur_id = task.id
        cls.obj = task
        sess.commit()
