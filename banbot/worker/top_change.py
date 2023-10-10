#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : top_change.py
# Author: anyongjin
# Date  : 2023/7/1
from banbot.data.tools import *
from banbot.storage import *
from banbot.storage.base import sa
from banbot.data.cache import BanCache


class TopChange:

    def __init__(self, config: Config):
        self.config = config
        self.exg_name = self.config['exchange']['name']
        self.market = config['market_type']

    async def calculate(self, start_ts: float, stop_ts: float) -> List[Tuple[int, str, float, float, float, float, float]]:
        '''
        计算指定市场的变动率，和成交量。
        :param start_ts: 秒级开始时间。
        :param stop_ts: 秒级结束时间
        :return:
        '''
        sess = dba.session
        # 这是特定于postgresql的语句。查询某一时刻某个市场所有标的的价格。
        market_price_sql = '''
with last_1m as (select distinct on(sid) * from kline_1m
where "time" > to_timestamp(:start) and "time" < to_timestamp(:stop) 
ORDER BY sid, "time" desc)
select symbol.id, symbol.symbol, last_1m."close" from symbol JOIN last_1m on symbol.id = last_1m.sid
where exchange=:exchange and market=:market
ORDER BY sid;'''
        # 查询所有标的最新价格
        where_args = dict(exchange=self.exg_name, market=self.market, start=stop_ts - 1000, stop=stop_ts)
        rows = await sess.execute(sa.text(market_price_sql), where_args)
        latest = {r[1]: r for r in rows}
        # 查询24H前所有标的价格
        where_args.update(start=start_ts - 1000, stop=start_ts)
        rows = await sess.execute(sa.text(market_price_sql), where_args)
        prev = {r[1]: r for r in rows}
        # 查询周期内成交量
        market_vol_sql = '''
with vol_all as (select sid, sum(volume) as volume, sum(volume * "close") as volume_q from kline_1m
where "time" > to_timestamp(:start) and "time" < to_timestamp(:stop) 
GROUP BY sid)
select symbol.symbol, vol_all.volume, vol_all.volume_q from symbol JOIN vol_all on symbol.id = vol_all.sid
where exchange=:exchange and market=:market;'''
        where_args.update(start=start_ts, stop=stop_ts)
        rows = await sess.execute(sa.text(market_vol_sql), where_args)
        vols = {r[0]: r for r in rows}
        # 计算价格变动比率，组合成交量
        car_keys = set(latest.keys()).intersection(prev.keys())
        change_list = []  # [(id, symbol, price, change_rate, volume, volume_q, prc_chg)]
        for symbol in car_keys:
            new_p, old_p = latest[symbol], prev[symbol]
            chg_rate = new_p[2] / old_p[2] - 1
            vol_r = vols.get(symbol) or (None, 0, 0)
            short_name = to_short_symbol(new_p[1])
            change_list.append((new_p[0], short_name, new_p[2], chg_rate, vol_r[1], vol_r[2], 0))
        change_list = sorted(change_list, key=lambda x: x[1])
        return change_list

    async def run(self):
        from banbot.exchange.exchange_utils import secs_day
        from banbot.data.spider import LiveSpider
        interval, delay = 60, 10
        cache_key = f'topchg_{self.exg_name}_{self.market}'
        logger.info(f'run {cache_key}')
        while True:
            cur_time = btime.utcstamp() / 1000
            next_run = (cur_time // interval + 1) * interval + delay
            wait_secs = next_run - cur_time
            if wait_secs > 0:
                await asyncio.sleep(wait_secs)
            # 目前固定对比UTC0点
            start_ts = cur_time // secs_day * secs_day
            try:
                async with dba():
                    data = await self.calculate(start_ts, next_run)
                # 计算价格一分钟是上涨还是下跌
                old_data = BanCache.get(cache_key)
                if old_data:
                    old_map = {r[0]: r for r in old_data}
                    for i, r in enumerate(data):
                        old_r = old_map.get(r[0])
                        if not old_r:
                            continue
                        data[i] = (*r[:-1], r[2] - old_r[2])
                BanCache.set(cache_key, data, interval * 2)
                if LiveSpider.obj:
                    await LiveSpider.obj.broadcast(cache_key, data)
            except Exception:
                logger.exception(f'update {cache_key} error')

    @classmethod
    async def start(cls):
        logger.info('start topchg updator')
        from banbot.storage.base import init_db
        from banbot.config import AppConfig
        init_db()
        worker = TopChange(AppConfig.get())
        asyncio.create_task(worker.run())
