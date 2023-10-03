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
from banbot.storage import BotGlobal


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


@router.get('/count', tags=['info'])
def count(rpc: RPC = Depends(get_rpc)):
    return rpc.open_num()


@router.get('/statistics', tags=['info'])
def statistics(rpc: RPC = Depends(get_rpc)):
    '''
    整体面板统计信息
    '''
    return rpc.dash_statistics()


@router.get('/stats', response_model=Stats, tags=['info'])
def stats(rpc: RPC = Depends(get_rpc)):
    '''
    显示各个平仓信号的胜率。盈亏时持仓时间。
    '''
    return rpc.stats()


@router.get('/profit_by', tags=['info'])
def profit_by(unit: str = Query(...), limit: int = Query(...), rpc: RPC = Depends(get_rpc)):
    return rpc.timeunit_profit(limit, unit)


@router.get('/orders', tags=['info'])
def orders(status: str = None, limit: int = 0, offset: int = 0, rpc: RPC = Depends(get_rpc)):
    '''
    查询订单列表。status=open表示查询未平仓订单；status=his查询已平仓订单
    '''
    with_total = limit > 0
    return rpc.get_orders(status, limit, offset, with_total, order_by_id=True)


# /forcebuy is deprecated with short addition. use /forceentry instead
@router.post('/forceenter', tags=['trading'])
async def force_entry(payload: ForceEnterPayload, rpc: RPC = Depends(get_rpc)):
    try:
        return await rpc.force_entry(payload.pair, payload.price, side=payload.side,
                                     order_type=payload.order_type, enter_cost=payload.enter_cost,
                                     enter_tag=payload.enter_tag, leverage=payload.leverage,
                                     strategy=payload.strategy, stoploss_price=payload.stoploss_price)
    except Exception as e:
        logger.exception(f'user open order fail: {payload}')
        return dict(code=400, msg=str(e))


# /forcesell is deprecated with short addition. use /forceexit instead
@router.post('/forceexit', tags=['trading'])
def forceexit(payload: ForceExitPayload, rpc: RPC = Depends(get_rpc)):
    return rpc.force_exit(payload.order_id)


@router.get('/pairlist', tags=['info', 'pairlist'])
def pairlist(rpc: RPC = Depends(get_rpc)):
    return rpc.pairlist()


@router.post('/pairlist', tags=['info', 'pairlist'])
async def set_pairlist(payload: SetPairsPayload, rpc: RPC = Depends(get_rpc)):
    return await rpc.set_pairs(payload.for_white, payload.adds or [], payload.deletes or [])


@router.get('/logs', tags=['info'])
def logs(limit: Optional[int] = None):
    return RPC.get_logs(limit)


@router.post('/delay_entry', tags=['botcontrol'])
def delay_entry(rpc: RPC = Depends(get_rpc), delay_secs: int = Body(..., embed=True)):
    return rpc.set_allow_trade_after(delay_secs)


@router.get('/config', tags=['info'])
def show_config():
    import yaml
    config = AppConfig.get_pub()
    content = yaml.safe_dump(config, indent=4, allow_unicode=True)
    return dict(data=config, content=content)


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


@router.get('/performance', response_model=List[PerformanceEntry], tags=['info'])
def performance(rpc: RPC = Depends(get_rpc)):
    '''
    按币种统计大致盈利状态。
    '''
    return rpc.pair_performance()


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


@router.get('/bot_info', tags=['info'])
def bot_info(rpc: RPC = Depends(get_rpc)):
    return rpc.bot_info()

