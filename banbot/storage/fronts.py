#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : fronts.py
# Author: anyongjin
# Date  : 2023/7/21
import datetime
from banbot.storage.base import *

min_date_time = datetime.datetime.utcfromtimestamp(0)


class Overlay(BaseDbModel):
    __tablename__ = 'overlay'

    id = Column(sa.Integer, primary_key=True)
    user = Column(sa.Integer, index=True)
    sid = Column(sa.Integer, index=True)
    start_ms = Column(sa.BIGINT, index=True)
    stop_ms = Column(sa.BIGINT, index=True)
    tf_msecs = Column(sa.Integer)
    update_at = Column(sa.DateTime, default=min_date_time)
    data = Column(sa.Text)
