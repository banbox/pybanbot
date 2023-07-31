#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : __init__.py.py
# Author: anyongjin
# Date  : 2023/2/24
from banbot.worker.sched import start_scheduler
# 启动定时轮训，爬虫，机器人，web进程都可使用，不同进程可注册不同的任务。
# 默认注册的：异常通知
start_scheduler()
