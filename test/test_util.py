#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_util.py
# Author: anyongjin
# Date  : 2023/5/18
import random
import time
import asyncio

from banbot.util.common import MeasureTime, logger
from banbot.util.misc import BanLock


def test_measure():
    mea = MeasureTime()
    for i in range(30):
        mea.start_for('loop')
        time.sleep(0.02)
        if random.random() < 0.5:
            mea.start_for('random')
            time.sleep(0.01)
        mea.start_for('ano')
        time.sleep(0.01)
    mea.print_all()


def test_for_performance():
    '''
    filter_a cost: 781 ms
    filter_b cost: 672 ms
    filter_c cost: 828 ms
    filter_d cost: 1422 ms
    '''
    data = [random.randrange(1, 10000) for i in range(10000000)]

    def filter_a(big, odd):
        res = data
        if big:
            res = [v for v in res if v > 5000]
        if odd:
            res = [v for v in res if v % 2]
        return res

    def filter_b(big, odd):
        res = [v for v in data if (not big or v > 5000) and (not odd or v % 2)]
        return res

    def filter_c(big, odd):
        res = []
        for v in data:
            if big and v <= 5000:
                continue
            if odd and v % 2 == 0:
                continue
            res.append(v)
        return res

    def filter_d(big, odd):
        def check(v):
            if big and v <= 5000:
                return False
            if odd and v % 2 == 0:
                return False
            return True
        res = [v for v in data if check(v)]
        return res

    fn_list = [filter_a, filter_b, filter_c, filter_d]

    for fn in fn_list:
        name = fn.__name__
        start = time.monotonic()
        res_a = fn(True, True)
        cost_a = time.monotonic() - start
        print(f'{name} cost: {round(cost_a * 1000)} ms, {res_a[:30]}')


async def test_ban_lock():
    tasks = ['a', 'b', 'c', 'd']
    jobs = [only_run_once(n) for n in tasks]
    await asyncio.gather(*jobs)
    logger.info('all complete')


async def only_run_once(name: str):
    async with BanLock('test', 10, force_on_fail=True):
        cost = random.random()
        logger.info(f'{name} runing {round(cost * 1000)} ms')
        await asyncio.sleep(cost)
        logger.info(f'{name} complete')


async def test_db_read():
    """
    测试异步数据库连接，如果不commit，重复执行查询，是否有脏读的问题。
    因异步连接commit后，此session会失效。
    验证结果：不存在脏读
    """
    from banbot.storage import dba, select, TdSignal
    async with dba():
        sess = dba.session
        while True:
            print('======')
            print(sess.in_transaction())
            stmt = select(TdSignal).order_by(TdSignal.id.desc()).limit(1)
            result = (await sess.scalars(stmt)).first()
            print(result.dict())
            print(sess.in_transaction())
            await sess.commit()
            print(sess.in_transaction())
            await asyncio.sleep(3)


async def test_db_flush():
    """
    测试sess.add后，是否自动执行了flush，可以获取到id
    结论：针对sess.add未自动flush
    """
    from banbot.storage import dba, select, TdSignal
    async with dba():
        sess = dba.session
        obj = TdSignal(strategy='fd', symbol_id=12, timeframe='1m', action='dfv', create_at=123, bar_ms=42)
        sess.add(obj)
        print(obj.id)
        await sess.flush()
        print(obj.id)
    print(obj.id)


async def test_detach_commit():
    """
    测试add对象后，再flush，再detach，再commit，是否会被保存
    结论：会保存flush之前的信息。detach后再修改的，不会保存
    """
    from banbot.storage import dba, TdSignal, detach_obj
    async with dba():
        sess = dba.session
        obj = TdSignal(strategy='alimama', symbol_id=1, timeframe='1m', action='dfv', create_at=123, bar_ms=42)
        sess.add(obj)
        print(obj.id)
        await sess.flush()
        print(obj.id)
        detach_obj(sess, obj)
        obj.timeframe = '32m'


async def test_autocommit():
    crt_sql = 'create table temp1 (name varchar(10));'
    from banbot.storage import dba, SqlSession, sa
    async with dba():
        sess: SqlSession = dba.session
        print(sess, id(sess))
        await sess.execute(sa.text(crt_sql))
    crt_sql = 'create table temp2 (name varchar(10));'
    async with dba.autocommit() as sess:
        print(sess, id(sess))
        await sess.execute(sa.text(crt_sql))


if __name__ == '__main__':
    asyncio.run(test_autocommit())
