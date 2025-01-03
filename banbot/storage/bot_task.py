#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : bot_task.py
# Author: anyongjin
# Date  : 2023/5/5

from banbot.util.tf_utils import tfsecs
from banbot.storage.base import *
from banbot.util import btime
from banbot.storage.common import *
from banbot.storage.extension import InfoPart


class BotTask(BaseDbModel, InfoPart):
    __tablename__ = 'bottask'

    cur_id: ClassVar[int] = 0
    obj: ClassVar['BotTask'] = None

    id = mapped_column(sa.Integer, primary_key=True)
    mode = mapped_column(sa.String(10))  # run mode
    stg_hash = mapped_column(sa.String(32))
    create_at = mapped_column(type_=sa.TIMESTAMP(timezone=True))
    start_at = mapped_column(type_=sa.TIMESTAMP(timezone=True))
    stop_at = mapped_column(type_=sa.TIMESTAMP(timezone=True))

    @orm.reconstructor
    def __init__(self, **kwargs):
        super().init_infos(self, kwargs)
        super(BotTask, self).__init__(**kwargs)

    @classmethod
    async def init(cls, start_at: Optional[float] = None):
        '''
        创建交易任务。记录涉及的时间周期
        '''
        if cls.obj is not None:
            return
        live_mode = btime.run_mode in btime.LIVE_MODES
        from banbot.strategy.resolver import StrategyResolver
        config = AppConfig.get()
        # 这里只是为了获取stg_hash:策略哈希特征
        run_jobs = StrategyResolver.load_run_jobs(config, [])
        # 计算任务开始时间
        rmode = btime.run_mode.value
        if not live_mode:
            # 非实时模式，需要设置初始模拟时钟
            warm_secs = max([tfsecs(warm_num, timeframe) for _, timeframe, warm_num, _ in run_jobs])
            start_at = config.get('timerange').startts - warm_secs
            btime.cur_timestamp = start_at
        if config.get('no_db'):
            assert not live_mode, '`no_db` not avaiable in live mode!'
            # 数据不写入到数据库
            ctime = btime.now()
            task = BotTask(id=-1, mode=rmode, create_at=ctime, stg_hash=BotGlobal.stg_hash)
            cls.obj = task
            cls.cur_id = -1
            logger.info(f"init task ok, id: {task.id} hash: {task.stg_hash}")
            return
        sess = dba.session
        where_list = [BotTask.mode == rmode, BotTask.stg_hash == BotGlobal.stg_hash]
        stat = select(BotTask).where(*where_list).order_by(BotTask.create_at.desc()).limit(1)
        task = (await sess.scalars(stat)).first()
        if not task or not live_mode:
            # 非实盘模式下，不可重复使用一个任务，否则可能同一时刻多个完全相同的订单
            ctime = btime.now()
            task = BotTask(mode=rmode, create_at=ctime, stg_hash=BotGlobal.stg_hash)
            if live_mode:
                task.start_at = ctime
            elif start_at:
                task.start_at = btime.to_datetime(start_at)
            else:
                raise ValueError('start_at is required for backtesting')
            sess.add(task)
            await sess.flush()
        cls.cur_id = task.id
        logger.info(f"init task ok, id: {task.id} hash: {task.stg_hash}")
        cls.obj = task

    @classmethod
    async def update(cls, **kwargs):
        if not cls.obj or cls.cur_id < 0:
            return
        sess = dba.session
        await sess.refresh(cls.obj)
        cls.obj.update_props(**kwargs)

    def dict(self, only: List[Union[str, MappedColumn[Any]]] = None, skips: List[Union[str, MappedColumn[Any]]] = None):
        data = super().dict(only, skips)
        data.update(**self.infos)
        del data['info']
        return data
