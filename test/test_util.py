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


test_measure()
