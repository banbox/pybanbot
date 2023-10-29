#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : schemas.py
# Author: anyongjin
# Date  : 2023/9/6
from typing import List, Optional, Any

from pydantic import BaseModel


class ExchangeModePayloadMixin(BaseModel):
    trading_mode: Optional[str] = None
    margin_mode: Optional[str] = None
    exchange: Optional[str] = None


class Ping(BaseModel):
    status: str


class Version(BaseModel):
    version: str


class StatusMsg(BaseModel):
    status: str


class Balance(BaseModel):
    symbol: str
    free: float
    used: float
    total: float
    total_fiat: float


class Balances(BaseModel):
    items: List[Balance]
    total: float


class SetPairsPayload(BaseModel):
    for_white: bool
    adds: Optional[List[str]] = None
    deletes: Optional[List[str]] = None


class ForceEnterPayload(BaseModel):
    pair: str
    side: str = 'long'
    order_type: Optional[str] = None
    price: Optional[float] = None
    enter_cost: Optional[float] = None
    enter_tag: Optional[str] = None
    leverage: Optional[float] = None
    strategy: Optional[str] = None
    stoploss_price: Optional[float] = None


class ForceExitPayload(BaseModel):
    order_id: str
    order_type: Optional[str] = None
    amount: Optional[float] = None


class ClosePosPayload(BaseModel):
    symbol: str
    amount: float
    side: str
    order_type: str
    price: float = None


class EditJobPayload(BaseModel):
    pair: str
    tf: str
    stgy: str
    key: str
    val: Any
