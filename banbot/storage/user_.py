#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : user_.py
# Author: anyongjin
# Date  : 2023/6/12
import datetime
from enum import Enum
from banbot.storage.base import *

min_date_time = datetime.datetime.utcfromtimestamp(0)


class VIPType(Enum):
    normal = 0
    vip = 1


class LoginMedium(Enum):
    MpSubWX = 'sub_wx'
    PcWeb = 'pcweb'


class DbUser(BaseDbModel):
    __tablename__ = 'users'

    __table_args__ = (
        sa.Index('idx_user_mobile', 'mobile'),
    )

    # 插入后更新obj的default值到对应列
    __mapper_args__ = {'eager_defaults': True}

    id = Column(sa.Integer, primary_key=True)
    user_name = Column(sa.String(128))
    avatar = Column(sa.String(256))
    mobile = Column(sa.String(30), nullable=True)
    mobile_verified = Column(sa.Boolean, default=False)
    email = Column(sa.String(120), nullable=True)
    email_verified = Column(sa.Boolean, default=False)
    pwd_salt = Column(sa.String(128))
    last_ip = Column(sa.String(64), nullable=True)
    create_at = Column(sa.DateTime, default=min_date_time)
    last_login = Column(sa.DateTime, default=min_date_time)

    vip_type = Column(IntEnum(VIPType), default=VIPType.normal)
    vip_expire_at = Column(sa.DateTime, default=min_date_time)
    inviter_id = Column(sa.Integer)

    @classmethod
    def init_tbl(cls, conn: sa.Connection):
        ins_cols = "user_name, mobile, mobile_verified"
        places = ":user_name, :mobile, :mobile_verified"
        insert_sql = f"insert into users ({ins_cols}) values ({places})"
        result = [
            dict(user_name='anyongjin', mobile='18932531737', mobile_verified=True),
        ]
        conn.execute(sa.text(insert_sql), result)
        conn.commit()


class ExgUser(BaseDbModel):
    __tablename__ = 'exg_users'

    __table_args__ = (
        sa.Index('idx_exg_users_out_uid', 'out_uid'),
    )

    id = Column(sa.Integer, primary_key=True)
    uid = Column(sa.Integer, index=True)  # 关联的用户ID
    channel = Column(sa.String(20))  # 所属渠道：binance
    out_uid = Column(sa.String(128))  # 交易所的用户ID
    user_name = Column(sa.String(128))  # 交易所的用户昵称

