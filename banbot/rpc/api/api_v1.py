#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : api_v1.py
# Author: anyongjin
# Date  : 2023/9/7
import logging

from fastapi import APIRouter, Depends, Query, Body
from fastapi.exceptions import HTTPException

from banbot import __version__
from banbot.rpc import RPC, RPCException
from banbot.config import AppConfig
from banbot.rpc.api.schemas import *
from banbot.storage import db, BotGlobal, ExSymbol, KInfo


logger = logging.getLogger(__name__)

# API version
API_VERSION = 2.33

# Public API, requires no auth.
router_public = APIRouter()
# Private API, protected by authentication
router = APIRouter()


def get_rpc():
    from banbot.rpc.api.webserver import ApiServer
    return ApiServer.rpc


@router_public.get('/ping', response_model=Ping)
def ping():
    """simple ping"""
    return {"status": "pong"}


@router.get('/version', response_model=Version, tags=['info'])
def version():
    """ Bot Version info"""
    return {"version": __version__}


@router.get('/balance', response_model=Balances, tags=['info'])
def balance(rpc: RPC = Depends(get_rpc)):
    """Account Balances"""
    return rpc.balance()


@router.get('/count', response_model=Count, tags=['info'])
def count(rpc: RPC = Depends(get_rpc)):
    return rpc.open_num()


@router.get('/performance', response_model=List[PerformanceEntry], tags=['info'])
def performance(rpc: RPC = Depends(get_rpc)):
    return rpc.performance()


@router.get('/profit', response_model=Profit, tags=['info'])
def profit(rpc: RPC = Depends(get_rpc)):
    config = AppConfig.get()
    return rpc.trade_statistics(config['stake_currency'])


@router.get('/stats', response_model=Stats, tags=['info'])
def stats(rpc: RPC = Depends(get_rpc)):
    return rpc.stats()


@router.get('/profit_by', tags=['info'])
def profit_by(unit: str = Query(...), limit: int = Query(...), rpc: RPC = Depends(get_rpc)):
    return rpc.timeunit_profit(limit, unit)


@router.get('/status', response_model=List[OpenInoutOrderSchema], tags=['info'])
def status(rpc: RPC = Depends(get_rpc)):
    try:
        return rpc.trade_status()
    except RPCException:
        return []


# Using the responsemodel here will cause a ~100% increase in response time (from 1s to 2s)
# on big databases. Correct response model: response_model=TradeResponse,
@router.get('/trades', tags=['info', 'trading'])
def trades(limit: int = 500, offset: int = 0, rpc: RPC = Depends(get_rpc)):
    return rpc.trade_history(limit, offset=offset, order_by_id=True)


@router.get('/trade/{tradeid}', response_model=OpenInoutOrderSchema, tags=['info', 'trading'])
def trade(tradeid: int = 0, rpc: RPC = Depends(get_rpc)):
    try:
        return rpc.trade_status([tradeid])[0]
    except (RPCException, KeyError):
        raise HTTPException(status_code=404, detail='Trade not found.')


@router.get('/show_config', response_model=ShowConfig, tags=['info'])
def show_config(rpc: Optional[RPC] = Depends(get_rpc)):
    return AppConfig.get_pub()


# /forcebuy is deprecated with short addition. use /forceentry instead
@router.post('/forceenter', response_model=ForceEnterResponse, tags=['trading'])
async def force_entry(payload: ForceEnterPayload, rpc: RPC = Depends(get_rpc)):
    od = await rpc.force_entry(payload.pair, payload.price, order_side=payload.side,
                               order_type=payload.ordertype, stake_amount=payload.stakeamount,
                               enter_tag=payload.entry_tag, leverage=payload.leverage)

    if od:
        return ForceEnterResponse.model_validate(od)
    else:
        return ForceEnterResponse.model_validate(
            {"status": f"Error entering {payload.side} trade for pair {payload.pair}."})


# /forcesell is deprecated with short addition. use /forceexit instead
@router.post('/forceexit', response_model=ResultMsg, tags=['trading'])
def forceexit(payload: ForceExitPayload, rpc: RPC = Depends(get_rpc)):
    return rpc.force_exit(payload.tradeid)


@router.get('/blacklist', response_model=BlacklistResponse, tags=['info', 'pairlist'])
def blacklist(rpc: RPC = Depends(get_rpc)):
    return rpc.blacklist()


@router.post('/blacklist', response_model=BlacklistResponse, tags=['info', 'pairlist'])
def blacklist_post(payload: BlacklistPayload, rpc: RPC = Depends(get_rpc)):
    return rpc.blacklist(payload.blacklist)


@router.delete('/blacklist', response_model=BlacklistResponse, tags=['info', 'pairlist'])
def blacklist_delete(pairs_to_delete: List[str] = Query([]), rpc: RPC = Depends(get_rpc)):
    """Provide a list of pairs to delete from the blacklist"""

    return rpc.blacklist_delete(pairs_to_delete)


@router.get('/whitelist', response_model=WhitelistResponse, tags=['info', 'pairlist'])
def whitelist(rpc: RPC = Depends(get_rpc)):
    return rpc.whitelist()


@router.get('/logs', response_model=Logs, tags=['info'])
def logs(limit: Optional[int] = None):
    return RPC.get_logs(limit)


@router.post('/delay_entry', tags=['botcontrol'])
def delay_entry(rpc: RPC = Depends(get_rpc), delay_secs: int = Body(..., embed=True)):
    return rpc.set_allow_trade_after(delay_secs)


@router.post('/reload_config', response_model=StatusMsg, tags=['botcontrol'])
def reload_config(rpc: RPC = Depends(get_rpc)):
    return rpc.reload_config()


@router.get('/pair_stgs', tags=['strategy'])
def pair_stgs():
    from banbot.strategy.resolver import get_strategy
    jobs = BotGlobal.stg_symbol_tfs
    stgy_set = {j[0] for j in jobs}
    stgy_dic = dict()
    for stgy in stgy_set:
        stg_cls = get_strategy(stgy)
        if not stg_cls:
            continue
        stgy_dic[stgy] = stg_cls.version
    jobs = [dict(stgy=j[0], pair=j[1], tf=j[2]) for j in jobs]
    return dict(jobs=jobs, stgy=stgy_dic)


@router.get('/strategy/{strategy}', tags=['strategy'])
def get_stgy_code(strategy: str):
    from banbot.strategy.resolver import get_strategy
    stgy_cls = get_strategy(strategy)
    if not stgy_cls:
        raise HTTPException(status_code=404, detail='Strategy not found')
    return {
        'strategy': stgy_cls.__name__,
        'data': stgy_cls.__source__,
    }


@router.get('/available_pairs', response_model=AvailablePairs, tags=['candle data'])
def list_available_pairs(timeframe: Optional[str] = None, stake_currency: Optional[str] = None,
                         candletype: Optional[str] = None):
    from banbot.util.misc import groupby
    config = AppConfig.get()
    sess = db.session
    info_fts = []
    if timeframe:
        info_fts.append(KInfo.timeframe == timeframe)
    rows = sess.query(KInfo.sid, KInfo.timeframe).filter(*info_fts).all()
    # 记录所有sid可用的时间周期
    sid_tf_rows = [(r.sid, r.timeframe) for r in rows]
    sid_tf_map = groupby(sid_tf_rows, lambda x: x[0])
    sid_tf_map = {k: [t[1] for t in v] for k, v in sid_tf_map.items()}
    allow_sids = {p[0] for p in sid_tf_rows}
    all_pairs = ExSymbol.search('')
    all_pairs = [p for p in all_pairs if p.id in allow_sids]

    if stake_currency:
        all_pairs = [p for p in all_pairs if p.symbol.endswith(stake_currency)]
    if not candletype:
        candletype = config['market_type']
    exg_name = config['exchange']['name']
    all_pairs = [p for p in all_pairs if p.market == candletype and p.exchange == exg_name]

    all_pairs = sorted(all_pairs, key=lambda x: x.symbol)

    pairs = list({x.symbol for x in all_pairs})
    gp_map = {p.symbol: sid_tf_map.get(p.id) for p in all_pairs}
    pair_tflist = [gp_map.get(p, []) for p in pairs]
    result = {
        'length': len(pairs),
        'pairs': pairs,
        'pair_interval': pair_tflist,
    }
    return result


@router.get('/bot_info', tags=['info'])
def bot_info(rpc: RPC = Depends(get_rpc)):
    return rpc.bot_info()

