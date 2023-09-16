#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : schemas.py
# Author: anyongjin
# Date  : 2023/9/6
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union, Iterable

from pydantic import BaseModel, ConfigDict, RootModel, SerializeAsAny

from banbot.config.consts import DATETIME_PRINT_FORMAT
from banbot.types.exg_types import ValidExchangesType


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
    currency: str
    free: float
    balance: float
    used: float
    stake: Iterable[str]
    # Starting with 2.x
    side: str
    leverage: float
    is_position: bool
    position: float


class Balances(BaseModel):
    currencies: List[Balance]
    total: float
    symbol: str
    stake: Iterable[str]


class Count(BaseModel):
    current: int
    max: int
    total_stake: float


class PerformanceEntry(BaseModel):
    pair: str
    profit: float
    profit_ratio: float
    profit_pct: float
    profit_abs: float
    count: int


class Profit(BaseModel):
    profit_closed_percent_mean: float
    profit_closed_ratio_mean: float
    profit_closed_percent_sum: float
    profit_closed_ratio_sum: float
    profit_all_percent_mean: float
    profit_all_ratio_mean: float
    profit_all_percent_sum: float
    profit_all_ratio_sum: float
    trade_count: int
    closed_trade_count: int
    first_trade_date: str
    first_trade_timestamp: Optional[int]
    latest_trade_date: str
    latest_trade_timestamp: Optional[int]
    avg_duration: str
    best_pair: Optional[str]
    best_pair_profit_ratio: float
    winning_trades: int
    losing_trades: int
    profit_factor: float
    winrate: float
    expectancy: float
    expectancy_ratio: float
    max_drawdown: float
    max_drawdown_abs: float
    trading_volume: Optional[float] = None
    bot_start_timestamp: int
    bot_start_date: str


class SellReason(BaseModel):
    wins: int
    losses: int
    draws: int


class Stats(BaseModel):
    exit_reasons: Dict[str, SellReason]
    durations: Dict[str, Optional[float]]


class DailyWeeklyMonthlyRecord(BaseModel):
    date: date
    abs_profit: float
    rel_profit: float
    starting_balance: float
    trade_count: int


class DailyWeeklyMonthly(BaseModel):
    data: List[DailyWeeklyMonthlyRecord]
    stake_currency: List[str]


class ShowConfig(BaseModel):
    name: str
    env: str
    run_mode: str
    leverage: int
    limit_vol_secs: int
    market_type: str
    max_market_rate: float
    odbook_ttl: int
    order_type: str
    prefire: bool
    refill_margin: bool
    take_over_stgy: str
    stake_currency: List[str]
    stake_amount: float
    max_open_orders: Optional[int] = None
    wallet_amounts: Dict[str, float]
    fatal_stop: Dict[str, float]
    fatal_stop_hours: Optional[float] = None
    timerange: Optional[Any]
    run_timeframes: Optional[List[str]]
    watch_jobs: Optional[Dict[str, List[str]]]
    run_policy: List[Dict[str, Any]]
    pairs: List[str]
    paircfg: Dict[str, Any]
    pairlists: List[Dict[str, Any]]
    exchange: Dict[str, Any]
    data_dir: Optional[str]
    database: Optional[Dict[str, Any]] = None
    redis_url: Optional[str] = None
    api_server: Optional[Dict[str, Any]]
    rpc_channels: Optional[Dict[str, Any]]
    webhook: Optional[Dict[str, Any]]


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


class OpenInoutOrderSchema(InoutOrderSchema):
    stoploss_current_dist: Optional[float] = None
    stoploss_current_dist_pct: Optional[float] = None
    stoploss_current_dist_ratio: Optional[float] = None
    stoploss_entry_dist: Optional[float] = None
    stoploss_entry_dist_ratio: Optional[float] = None
    current_rate: float
    total_profit_abs: float
    total_profit_fiat: Optional[float] = None
    total_profit_ratio: Optional[float] = None

    open_order: Optional[str] = None


class InoutOrderResponse(BaseModel):
    trades: List[InoutOrderSchema]
    trades_count: int
    offset: int
    total_trades: int


ForceEnterResponse = RootModel[Union[InoutOrderSchema, StatusMsg]]


class Logs(BaseModel):
    log_count: int
    logs: List[List]


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


class BlacklistPayload(BaseModel):
    blacklist: List[str]


class BlacklistResponse(BaseModel):
    blacklist: List[str]
    errors: Dict
    length: int
    method: List[str]


class WhitelistResponse(BaseModel):
    whitelist: List[str]
    length: int
    method: List[str]


class DeleteTrade(BaseModel):
    cancel_order_count: int
    result: str
    result_msg: str
    trade_id: int


class StrategyListResponse(BaseModel):
    strategies: List[str]


class ExchangeListResponse(BaseModel):
    exchanges: List[ValidExchangesType]


class PairListResponse(BaseModel):
    name: str
    description: str
    is_pairlist_generator: bool
    params: Dict[str, Any]


class PairListsResponse(BaseModel):
    pairlists: List[PairListResponse]


class PairListsPayload(ExchangeModePayloadMixin, BaseModel):
    pairlists: List[Dict[str, Any]]
    blacklist: List[str]
    stake_currency: str


class StrategyResponse(BaseModel):
    strategy: str
    code: str


class AvailablePairs(BaseModel):
    length: int
    pairs: List[str]
    pair_interval: List[List[str]]


class PairHistory(BaseModel):
    strategy: str
    pair: str
    timeframe: str
    timeframe_ms: int
    columns: List[str]
    data: SerializeAsAny[List[Any]]
    length: int
    buy_signals: int
    sell_signals: int
    enter_long_signals: int
    exit_long_signals: int
    enter_short_signals: int
    exit_short_signals: int
    last_analyzed: datetime
    last_analyzed_ts: int
    data_start_ts: int
    data_start: str
    data_stop: str
    data_stop_ts: int
    # TODO[pydantic]: The following keys were removed: `json_encoders`.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-config for more information.
    model_config = ConfigDict(json_encoders={
        datetime: lambda v: v.strftime(DATETIME_PRINT_FORMAT),
    })
