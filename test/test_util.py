#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_util.py
# Author: anyongjin
# Date  : 2023/5/18
import random
import time

from banbot.util.common import MeasureTime


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


test_for_performance()
