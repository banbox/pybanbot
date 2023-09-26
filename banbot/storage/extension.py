#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : extension.py
# Author: anyongjin
# Date  : 2023/9/26
import json
from banbot.storage.base import *


class InfoPart:
    info = Column(sa.String(4096))

    def __init__(self):
        self.infos = dict()

    def init_infos(self, tbl: BaseDbModel, kwargs: Dict):
        self.infos = dict()
        db_keys = set(tbl.__table__.columns.keys())
        tmp_keys = {k for k in kwargs if k not in db_keys}
        if tbl.id:
            # 从数据库创建映射的值，无需设置，否则会覆盖数据库值
            if self.info:
                # 数据库初始化的必然只包含列名，这里可以直接覆盖
                self.infos: Dict = json.loads(self.info)
            return True
        else:
            self.infos: Dict = {k: kwargs.pop(k) for k in tmp_keys}
            # 仅针对新创建的订单执行下面初始化
            if self.infos:
                # 自行实例化的对象，忽略info参数
                kwargs['info'] = json.dumps(self.infos)
            return False

    def get_info(self, key: str, def_val=None):
        if not self.infos:
            if self.info:
                self.infos: Dict = json.loads(self.info)
            else:
                return def_val
        return self.infos.get(key, def_val)

    def set_info(self, **kwargs):
        self.infos.update(kwargs)
        self.info = json.dumps(self.infos)
