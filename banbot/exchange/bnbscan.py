#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : bnbscan.py
# Author: anyongjin
# Date  : 2023/2/26
import os.path
import pickle

import binance.error
import numpy as np
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient as WsStream
from binance.websocket.spot.websocket_api import SpotWebsocketAPIClient as WsApi
from binance.spot import Spot as RestApi
from typing import List
from collections import OrderedDict
from banbot.config.config import *
from banbot.utils import *


class BnbScan:
    def __init__(self, full_odbook=False, log_his: bool = False):
        '''
        初始化一个币安高频逐秒信息更新端
        :param full_odbook: 是否维护一个完整的1000深度订单簿。False时100ms获取10档深度
        :param log_his: 是否记录到本地磁盘
        '''
        self._is_full_od = full_odbook
        self._log_his = log_his
        self.listen_key = None
        self.listen_key_update = utime(-3600)
        self.base_key, self.quote_key = cfg['pair'].split('/')
        self.pair = cfg['pair'].replace('/', '').upper()
        self.position = 0  # 仓位。提交订单未成交时也算持仓，订单被取消或过期则对应缩减持仓
        auth_args, rest_args, stream_args = get_bnb_client_args()
        handlers = self.init_msg_handler()
        stream_args.update(dict(
            on_message=handlers['on_wsstream_msg'],
            on_error=handlers['on_error']
        ))
        # 有频率限制，仅用于下单，其他请求尽量通过wsapi
        self.rest_api = RestApi(**auth_args, **rest_args)
        self.wsstream = WsStream(**stream_args)
        # # 【握手失败，暂不使用】替代rest-api的websocket调用方式，延迟更低
        # logger.info(f'init wsapi: {auth_args} {stream_args}')
        # self.wsapi = WsApi(**auth_args, **stream_args)
        self.odbook_u = -1 if full_odbook else 0  # -1 尚未初始化，0正在初始化
        # 订单簿
        self.asks, self.bids = OrderedDict(), OrderedDict()
        self.asks_old, self.bids_old = OrderedDict(), OrderedDict()
        self.odbook_his = []
        self.dump_path = os.path.join(os.path.dirname(__file__), 'data.pkl')
        # 开，高，低，关，成交量，笔数，主动买入量
        self.kline_cols = ['o', 'h', 'l', 'c', 'v', 'n', 'V']
        self.max_ksize = 60
        kline_shape = (0, len(self.kline_cols))
        self.klines = np.empty(kline_shape, dtype=np.float64)
        self.klines_bak = np.empty(kline_shape, dtype=np.float64)
        self.init_price = 0
        self.update_od_at = 0
        self.update_kline_at = 0
        self.tick_num = 1
        self._last_log_at = 0
        self.req_cost = 0  # 发送请求耗时（到服务器时间）
        # 交易对象精度，定价对象精度
        self.base_prec, self.quote_prec = 8, 8
        self.price_step = 0.01  # 价格变化的最小单位
        self.quantity_step = 0.00000001  # 交易量变化的最小单位
        self.min_notional = 10  # 最小名义价值（price * quantity）
        # 订阅每秒订单簿深度变化和K线变化
        pair_sm = self.pair.lower()
        self.wsstream.subscribe(f'{pair_sm}@kline_1s', id=15135)
        if self._is_full_od:
            # 完整订单簿模式，订阅逐秒订单簿更新信息
            self.wsstream.subscribe(f"{pair_sm}@depth", id=12345)
        else:
            # 有限深度模式，订阅100ms的10档信息
            self.wsstream.subscribe(f'{pair_sm}@depth10@100ms', id=12345)
        # 监听用户余额和订单变化。
        self.wsstream.user_data(self.get_listen_key(), id=6688560)
        self.wallets = dict()
        # 等待成交的订单
        self.orders = dict()
        # 更新钱包数据
        self.update_account()

    def init_msg_handler(self):

        def on_wsstream_msg(_, msg_text: str):
            try:
                if not msg_text:
                    return
                msg = orjson.loads(msg_text)
                evt = msg.get('e')
                if evt == 'depthUpdate':
                    update_ok = self.update_orderbook(msg['a'], msg['b'], msg['U'], msg['u'])
                    if update_ok:
                        self.update_od_at = round(msg['E'] / 1000)
                        self._try_fire_feed()
                elif evt == 'kline':
                    self.update_klines(msg['k'])
                elif evt == 'outboundAccountPosition':
                    self.update_wallets(msg['B'])
                elif evt == 'balanceUpdate':
                    logger.error(f'     [balanceUpdate] {msg}')
                    self.wallets[msg['a']] = (float(msg['d']), 0)
                elif evt == 'executionReport':
                    self.on_order_update(msg)
                elif msg.get('lastUpdateId') and msg.get('bids'):
                    # 有限深度订单簿100ms， 10档
                    self.update_orderbook(msg['asks'], msg['bids'], 0, 0)
                else:
                    print(f'recv stm:  {msg}')
            except Exception:
                logger.exception(f'on_wsstream_msg fail, msg: {msg_text}')

        def on_wsapi_msg(_, msg: dict):
            print(f'recv api: {type(msg)} {msg}')

        def on_error(_, err):
            print(f'error: {err}')

        return dict(
            on_wsstream_msg=on_wsstream_msg,
            on_wsapi_msg=on_wsapi_msg,
            on_error=on_error
        )

    def update_account(self):
        try:
            # 测试到交易所的延迟
            start_time = utime()
            server_time = self.rest_api.time()['data']['serverTime']
            self.req_cost = round((utime() - start_time) / 2)
            time_diff = start_time + self.req_cost - server_time
            logger.warning(f'server time diff {time_diff} ms, cost: {self.req_cost}')
            # 查询交易规范
            pair_info = self.rest_api.exchange_info(self.pair)['data']['symbols'][0]
            # logger.info(f'pair info: {pair_info}')
            self.base_prec = pair_info['baseAssetPrecision']
            self.quote_prec = pair_info['quoteAssetPrecision']
            pair_filters = pair_info['filters']
            for it in pair_filters:
                ft_type = it['filterType']
                if ft_type == 'PRICE_FILTER':
                    self.price_step = float(it['tickSize'])
                elif ft_type == 'MIN_NOTIONAL':
                    self.min_notional = float(it['minNotional'])
                elif ft_type == 'LOT_SIZE':
                    self.quantity_step = float(it['stepSize'])
            # 取消所有挂单
            open_orders = self.rest_api.get_open_orders(self.pair)['data']
            if len(open_orders):
                self.rest_api.cancel_open_orders(self.pair, timestamp=utime())
            # 更新账户余额
            user_data = self.rest_api.account()['data']
            self.wallets = dict()
            for item in user_data['balances']:
                free, lock = float(item['free']), float(item['locked'])
                self.wallets[item['asset']] = (free, lock)
            # 计算当前仓位
            self.init_price = float(self.rest_api.avg_price(self.pair)['data']['price'])
            quote, base_qt = self.quote_v, self.base_v
            quote_val = quote[0] + base_qt[1]
            base_val = quote[1] + base_qt[0]
            self.position = base_val / (quote_val + base_val)
            # orders = self.rest_api.get_open_orders(self.pair)['data']
            # buy_num = len([o for o in orders if o['side'] == 'BUY'])
            # sell_num = len(orders) - buy_num
            logger.info(f'[account]  {self.quote_key}: {self.quote_v}  {self.base_key}: {base_qt}, '
                        f'pos: {self.position:.3f}')
        except Exception as e:
            logger.exception(f'update user data fail: {e}')

    @property
    def quote_v(self) -> Tuple[float, float]:
        quotes = self.wallets.get(self.quote_key, [0., 0.])
        return round(quotes[0], 2), round(quotes[1], 2)

    @property
    def base_v(self) -> Tuple[float, float]:
        base = self.wallets.get(self.base_key, [0., 0.])
        avg_price = self.klines[0][3] if len(self.klines) else self.init_price
        return round(base[0] * avg_price, 2), round(base[1] * avg_price, 2)

    @property
    def all_assets(self) -> float:
        quote, base = self.quote_v, self.base_v
        return sum(quote) + sum(base)

    def submit_order(self, mtype: int, price: float, quantity: float):
        '''
        提交新订单到交易所（限价单形式）
        :param mtype: 方向：1做多，-1做空
        :param price: 价格
        :param quantity: 交易量
        :return:
        '''
        import uuid
        side = 'BUY' if mtype > 0 else 'SELL'
        price = round(round(price / self.price_step) * self.price_step, self.base_prec)
        if self.quantity_step:
            quantity = round(round(quantity / self.quantity_step) * self.quantity_step, self.base_prec)
        # 检查余额是否充足
        sell_wall_vol = 0
        if mtype > 0:
            wallet_ok = self.quote_v[0] >= price * quantity
        else:
            sell_wall_vol = self.wallets.get(self.base_key, [0.])[0]
            wallet_ok = sell_wall_vol >= quantity
        if not wallet_ok:
            return 0
        order_id = str(uuid.uuid4())
        od_argd = dict(timeInForce='GTC', quantity=quantity, price=price, newClientOrderId=order_id,
                       timestamp=utime(), recvWindow=1000 + self.req_cost)
        try:
            self.rest_api.new_order(self.pair, side, 'LIMIT', **od_argd)
        except binance.error.ClientError as e:
            logger.error(f'submit order fail: [{sell_wall_vol}] {side} {od_argd}  {e}')
            return 0
        equal_money = quantity * price
        self.orders[order_id] = dict(
            symbol=self.pair, side=side, type='LIMIT',
            quantity=quantity, price=price
        )
        self.position += equal_money * mtype / self.all_assets
        return equal_money * mtype

    def on_order_update(self, msg: dict):
        client_id = msg['c']
        state = msg['X']
        if state == 'NEW':
            return
        direc = -1 if msg['S'] == 'SELL' else 1
        # ok_money = msg['Y']  # 订单末次成交金额
        if state in {'CANCELED', 'REJECTED', 'EXPIRED', 'EXPIRED_IN_MATCH'}:
            if client_id in self.orders:
                del self.orders[client_id]
            # 未成交金额=(入场数量-已成交金额)*入场价格
            unfill = (float(msg['q']) - float(msg['z'])) * float(msg['p'])
            self.position -= unfill * direc / self.all_assets
            return
        if state == 'FILLED':
            if client_id in self.orders:
                del self.orders[client_id]
        elif state == 'PARTIALLY_FILLED':
            pass
            # logger.info(f'part filled order: {msg}')
        else:
            logger.error(f'unknown order status: {state}, {msg}')

    def update_wallets(self, arr: List):
        update_log, total_asset = [], 0
        for item in arr:
            key, free, lock = item['a'], float(item['f']), float(item['l'])
            self.wallets[key] = (free, lock)
            if key == self.base_key:
                avg_price = self.klines[0][3] if len(self.klines) else self.init_price
                free_sm, lock_sm = round(free * avg_price, 2), round(lock * avg_price, 2)
                total_asset += free_sm + lock_sm
                update_log.append(f'{key}: {free_sm}--{free_sm}')
            elif key == self.quote_key:
                free_sm, lock_sm = round(free, 2), round(lock, 2)
                total_asset += free_sm + lock_sm
                update_log.append(f'{key}: {free_sm}--{lock_sm}')
        quote, base_qt = self.quote_v, self.base_v
        quote_val = quote[0] + base_qt[1]
        base_val = quote[1] + base_qt[0]
        self.position = base_val / (quote_val + base_val)
        if update_log:
            logger.info(f'wallet: {"    ".join(update_log)}    total: {total_asset:.2f},  pos: {self.position:.3f}')

    def init_orderbook(self):
        if self.odbook_u == 0 or not self._is_full_od:
            # 正在初始化，跳过 || 非完整订单簿模式
            return
        self.odbook_u = 0
        od_res = self.rest_api.depth(self.pair, limit=1000)['data']
        self.update_orderbook(od_res['asks'], od_res['bids'], 0, od_res['lastUpdateId'])

    def update_orderbook(self, asks: List, bids: List, last_id: int, cur_id: int) -> bool:
        if self._is_full_od and cur_id <= self.odbook_u and last_id >= cur_id:
            logger.warning(f'skip invalid od data, end_id: {cur_id}, last: {last_id}, {len(asks)}, {len(bids)}')
            # 丢弃过期和无效的消息
            return False
        if last_id <= 0 or self.odbook_u <= 0:
            self.bids = OrderedDict()
            self.asks = OrderedDict()
        elif last_id != self.odbook_u + 1:
            if not self._is_full_od:
                raise ValueError('pkg loss in limit depth mode!')
            logger.warning(f'odbook pkg loss, last: {self.odbook_u} cur_last: {last_id}, {len(asks)}, {len(bids)}')
            # 发生丢包，重新拉取初始化订单薄
            self.init_orderbook()
            return False
        self.asks.update(OrderedDict(asks))
        self.bids.update(OrderedDict(bids))
        if not self._is_full_od:
            return True
        # 完全订单簿模式，记录更新的ID
        zero_qua = '0.' + '0' * (len(asks[0][1]) - 2)
        del_akeys = [row[0] for row in asks if row[1] == zero_qua]
        del_bkeys = [row[0] for row in bids if row[1] == zero_qua]
        for key in del_akeys:
            self.asks.pop(key, None)
        for key in del_bkeys:
            self.bids.pop(key, None)
        logger.debug(f'odbook update, {self.odbook_u} -> {cur_id} {len(asks)}, {len(bids)}, '
                     f'del: {len(del_akeys)}, {len(del_bkeys)}')
        should_init = self.odbook_u < 0
        if last_id > 0:
            # 只记录来自推送流的更新。因为获取1000档的updateId和推送的updateId格式不匹配，无法比较
            self.odbook_u = cur_id
        if should_init:
            # 为避免过早获取订单簿导致部分信息不全，在初次推送更新时再获取
            self.init_orderbook()
        return not should_init

    def update_klines(self, row: dict):
        # 开，高，低，关，成交量，笔数，主动买入量
        data = [float(row[k]) for k in self.kline_cols]
        self.klines = np.insert(self.klines, 0, np.array(data), 0)
        self.update_kline_at = round(row['t'] / 1000)
        logger.debug(f'update kline, {row["t"]}')
        self._try_fire_feed()

    def _log_his_data(self, cur_secs: int):
        # 记录订单簿历史K线历史，内存中超过max_ksize时保存到文件
        if not self._log_his or cur_secs == self._last_log_at:
            return
        self._last_log_at = cur_secs
        self.odbook_his.append((self.asks_old, self.bids_old))
        if len(self.klines) > self.max_ksize:
            last = self.klines[self.max_ksize, :]
            self.klines = self.klines[:self.max_ksize]
            self.klines_bak = np.append(self.klines_bak, [last], 0)
            if len(self.klines_bak) >= self.max_ksize:
                dsize = len(self.klines_bak)
                logger.info(f'dump kline & odbook, size: {dsize}')
                with open(self.dump_path, 'ab') as fdump:
                    dump_row = [self.odbook_his[:dsize], self.klines_bak]
                    pickle.dump(dump_row, fdump)
                self.klines_bak = self.klines_bak[:0]
                self.odbook_his = self.odbook_his[dsize:]

    def _try_fire_feed(self):
        last_secs = utime(as_ms=False) - 1
        tags = []
        if not self._is_full_od or self.update_od_at >= last_secs:
            tags.append('od')
        if self.update_kline_at >= last_secs:
            tags.append('kline')
        if self.tick_num > 1:
            if len(tags) == 2:
                self._log_his_data(last_secs)
            # 调用外部订阅的函数
            self.on_data_feed(last_secs, tags)
        if len(tags) == 2:
            self.tick_num += 1
            # 上一个K线完成，替换asks_old
            self.asks_old = self.asks.copy()
            self.bids_old = self.bids.copy()
        if last_secs % 300 == 0:
            # 每五分钟刷新listenkey，避免失效
            self.get_listen_key()

    def on_data_feed(self, last_secs, tags: List):
        logger.warning('on_data_feed not implemented')

    def get_listen_key(self):
        cur_time = utime(60)
        if not self.listen_key or self.listen_key_update < cur_time:
            self.listen_key = self.rest_api.new_listen_key()['data']['listenKey']
            self.listen_key_update = utime()
        elif self.listen_key_update < cur_time + 300000:
            logger.warning('listen key is about expired, renew...')
            self.rest_api.renew_listen_key(self.listen_key)
        return self.listen_key

    def close(self):
        self.wsstream.stop()
        # self.wsapi.stop()
