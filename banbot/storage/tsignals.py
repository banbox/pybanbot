#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : tsignals.py
# Author: anyongjin
# Date  : 2023/6/28

from banbot.storage.base import *


class TdSignal(BaseDbModel):
    '''
    交易信号，记录来自外部的信号
    '''

    __tablename__ = 'tdsignal'

    __table_args__ = (
        sa.Index('idx_sig_symbol_id', 'symbol_id'),
        sa.Index('idx_sig_timeframe', 'timeframe'),
        sa.Index('idx_sig_bar_ms', 'bar_ms'),
        sa.Index('idx_sig_strategy', 'strategy'),
    )

    id = mapped_column(sa.Integer, primary_key=True)
    strategy = mapped_column(sa.String(20))
    symbol_id = mapped_column(sa.Integer)  # ExSymbol对应的ID
    timeframe = mapped_column(sa.String(5))  # 1m, 5m, 15m, ...
    action = mapped_column(sa.String(5))  # buy/sell
    create_at = mapped_column(sa.BIGINT)  # 发出信号时的时间戳，如消息中有取消息中的，否则取当前时间戳
    bar_ms = mapped_column(sa.BIGINT)  # 发出信号时，所属bar的时间戳，此bar应尚未完成
    price = mapped_column(sa.Float)
