#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : api_v1.py
# Author: anyongjin
# Date  : 2023/9/7
import logging

from fastapi import APIRouter, Depends, Query, Body
from fastapi.exceptions import HTTPException

from banbot import __version__
from banbot.rpc import RPC
from banbot.config import AppConfig, UserConfig
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
async def count(rpc: RPC = Depends(get_rpc)):
    return await rpc.open_num()


@router.get('/statistics', tags=['info'])
async def statistics(rpc: RPC = Depends(get_rpc)):
    '''
    整体面板统计信息
    '''
    return await rpc.dash_statistics()


@router.get('/incomes', tags=['info'])
async def incomes(intype: str, symbol: str = None, start_time: int = 0, limit: int = 0, rpc: RPC = Depends(get_rpc)):
    """
    获取账户损益资金流水
    """
    return await rpc.incomes(intype, symbol, start_time, limit)


@router.get('/orders', tags=['info'])
async def orders(source: str = 'bot', status: str = None, symbol: str = None,
                 start_time: int = 0, stop_time: int = 0,
                 limit: int = 0, offset: int = 0, rpc: RPC = Depends(get_rpc)):
    '''
    查询订单列表。status=open表示查询未平仓订单；status=his查询已平仓订单
    '''
    with_total = limit > 0
    return await rpc.get_orders(source, status, symbol, start_time, stop_time,
                                limit, offset, with_total)


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
async def forceexit(payload: ForceExitPayload, rpc: RPC = Depends(get_rpc)):
    return await rpc.force_exit(payload.order_id)


@router.post('/close_pos', tags=['trading'])
async def close_pos(payload: ClosePosPayload, rpc: RPC = Depends(get_rpc)):
    return await rpc.close_pos(payload)


@router.post('/calc_profits', tags=['info'])
async def calc_profits(status: str = Body(None, embed=True), rpc: RPC = Depends(get_rpc)):
    return await rpc.calc_profits(status)


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


@router.get('/pair_jobs', tags=['strategy'])
async def get_pair_jobs():
    from banbot.strategy.resolver import get_strategy
    jobs = BotGlobal.stg_symbol_tfs
    jobs = sorted(jobs, key=lambda x: (x[1], x[2], x[0]))
    # 读取所有策略和版本号
    stgy_set = {j[0] for j in jobs}
    stgy_dic = dict()
    for stgy in stgy_set:
        stg_cls = get_strategy(stgy)
        if not stg_cls:
            continue
        stgy_dic[stgy] = stg_cls.version
    # 币对，策略，及其开关配置
    stg_map = dict()
    for key, stg_list in BotGlobal.pairtf_stgs.items():
        pair, tf = key.split('_')
        for stg in stg_list:
            stg_map[(stg.name, pair, tf)] = stg

    items = []
    from banbot.strategy import BaseStrategy
    from banbot.main.addons import MarketPrice
    from banbot.storage import InOutOrder
    open_ods: List[InOutOrder] = await InOutOrder.open_orders()
    for j in jobs:
        stg: BaseStrategy = stg_map.get(j)
        if not stg:
            continue
        price = MarketPrice.get(j[1])
        od_num = len([od for od in open_ods if od.symbol == j[1] and od.strategy == stg.name])
        item = dict(stgy=j[0], pair=j[1], tf=j[2], price=price, od_num=od_num)
        args = []
        for arg in stg.get_job_args_info():
            args.append(dict(**arg, value=getattr(stg, arg['field'])))
        item['args'] = args
        items.append(item)
    return dict(jobs=items, stgy=stgy_dic)


@router.post('/edit_job')
async def edit_pair_job(payload: EditJobPayload, rpc: RPC = Depends(get_rpc)):
    import builtins
    config = UserConfig.get()
    pair_jobs: dict = config.get('pair_jobs')
    if pair_jobs is None:
        pair_jobs = dict()
        config['pair_jobs'] = pair_jobs
    cur_key = f'{payload.pair}_{payload.stgy}'
    job_config: dict = pair_jobs.get(cur_key)
    if job_config is None:
        job_config = dict()
        pair_jobs[cur_key] = job_config
    for arg in payload.args:
        arg_val = arg['value']
        if arg_val is not None:
            val_type = getattr(builtins, arg['val_type'])
            try:
                arg_val = val_type(arg_val)
            except ValueError:
                arg_val = val_type()
            arg['value'] = arg_val
        job_config[arg['field']] = arg['value']
    UserConfig.save()
    await rpc.apply_job_args(payload, job_config)
    return dict(code=200)


@router.get('/task_pairs')
async def task_pairs(start: Optional[int] = Query(None), stop: Optional[int] = Query(None),
                     rpc: RPC = Depends(get_rpc)):
    pairs = await rpc.get_task_pairs(start, stop)
    return dict(pairs=pairs)


@router.get('/performance', tags=['info'])
async def performance(group_by: str = Query(...), pairs: Optional[List[str]] = Query(None),
                      start: Optional[int] = Query(None), stop: Optional[int] = Query(None),
                      limit: int = Query(...), rpc: RPC = Depends(get_rpc)):
    '''
    统计盈利状态。按天，按周，按月，按币种。
    '''
    if group_by == 'symbols':
        from banbot.storage import InOutOrder
        items = await InOutOrder.get_pair_performance(start, stop)
    else:
        items = await rpc.timeunit_profit(group_by, limit, start, stop, pairs)
    tag_res = await rpc.tag_stats()
    return dict(
        items=items,
        **tag_res
    )


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

