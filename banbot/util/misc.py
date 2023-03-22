#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : misc.py
# Author: anyongjin
# Date  : 2023/3/22
import time
import logging
import sys


def utime(secs: int = 0, as_ms: bool = True):
    multipler = 1000 if as_ms else 1
    return round((time.time() + secs) * multipler)

