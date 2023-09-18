#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : schemas.py
# Author: anyongjin
# Date  : 2023/9/6
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, RootModel


class ExchangeModePayloadMixin(BaseModel):
    trading_mode: Optional[str] = None
    margin_mode: Optional[str] = None
    exchange: Optional[str] = None


class Ping(BaseModel):
    status: str


class AccessToken(BaseModel):
    access_token: str


class AccessAndRefreshToken(AccessToken):
    refresh_token: str


class Version(BaseModel):
    version: str


class StatusMsg(BaseModel):
    status: str


class ResultMsg(BaseModel):
    result: str


class Balance(BaseModel):
    symbol: str
    free: float
    used: float
    total: float
    total_fiat: float


class Balances(BaseModel):
    items: List[Balance]
    total: float


class PerformanceEntry(BaseModel):
    pair: str
    profit_sum: float
    profit_pct: float
    close_num: int


class TagStat(BaseModel):
    tag: str
    wins: int
    losses: int
    draws: int


class Stats(BaseModel):
    exit_reasons: List[TagStat]
    durations: Dict[str, Optional[float]]


class OrderSchema(BaseModel):
    id: int
    task_id: int
    inout_id: int
    symbol: str
    enter: bool
    order_type: str
    order_id: Optional[str] = None
    side: str
    create_at: int
    price: Optional[float] = None
    average: Optional[float] = None
    amount: Optional[float] = None
    filled: Optional[float] = None
    status: int
    fee: Optional[float] = None
    fee_type: Optional[str] = None
    update_at: int


class InoutOrderSchema(BaseModel):
    id: int
    task_id: int
    sid: int
    symbol: str
    short: bool
    quote_cost: float
    strategy: str
    stg_ver: int = 0
    status: int
    enter_tag: Optional[str] = None
    timeframe: str

    enter_at: int
    init_price: float
    current_rate: Optional[float] = None

    exit_at: Optional[int] = None

    profit: Optional[float] = None
    profit_rate: Optional[float] = None
    close_profit: Optional[float] = None

    exit_tag: Optional[str] = None
    leverage: Optional[float] = None
    info: str = None

    enter: OrderSchema
    exit: Optional[OrderSchema] = None


class InoutOrderResponse(BaseModel):
    trades: List[InoutOrderSchema]
    trades_count: int
    offset: int
    total_trades: int


ForceEnterResponse = RootModel[Union[InoutOrderSchema, StatusMsg]]


class SetPairsPayload(BaseModel):
    for_white: bool
    adds: Optional[List[str]] = None
    deletes: Optional[List[str]] = None


class ForceEnterPayload(BaseModel):
    pair: str
    side: str = 'long'
    price: Optional[float] = None
    ordertype: Optional[str] = None
    stakeamount: Optional[float] = None
    entry_tag: Optional[str] = None
    leverage: Optional[float] = None


class ForceExitPayload(BaseModel):
    tradeid: str
    ordertype: Optional[str] = None
    amount: Optional[float] = None

