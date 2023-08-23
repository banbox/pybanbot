#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : ccxt_exts.py
# Author: anyongjin
# Date  : 2023/8/22
import ccxt.pro as ccxtpro


def get_asy_overrides():
    return dict()


def get_pro_overrides():
    return dict(binanceusdm=probinanceusdm)


class probinanceusdm(ccxtpro.binanceusdm):

    async def watch_account_config(self, params={}):
        """
        watches account update: leverage and margin
        :param dict params: extra parameters specific to the binance api endpoint
        :returns dict: a account update
        """
        await self.load_markets()
        await self.authenticate(params)
        defaultType = self.safe_string(self.options, 'defaultType', 'spot')
        type = self.safe_string(params, 'type', defaultType)
        subType, params = self.handle_sub_type_and_params('watchMyTrades', None, params)
        if self.isLinear(type, subType):
            type = 'future'
        elif self.isInverse(type, subType):
            type = 'delivery'
        url = self.urls['api']['ws'][type] + '/' + self.options[type]['listenKey']
        client = self.client(url)
        self.set_balance_cache(client, type)
        message = None
        messageHash = type + ':accUpdate'
        return await self.watch(url, messageHash, message, type)

    def handle_account_update(self, client, message):
        '''
        { // 杠杆倍数更新
            "e":"ACCOUNT_CONFIG_UPDATE",       // 事件类型
            "E":1611646737479,                 // 事件时间
            "T":1611646737476,                 // 撮合时间
            "ac":{
            "s":"BTCUSDT",                     // 交易对
            "l":25                             // 杠杆倍数

            }
        }
        { // 保证金状态更新
            "e":"ACCOUNT_CONFIG_UPDATE",       // 事件类型
            "E":1611646737479,                 // 事件时间
            "T":1611646737476,                 // 撮合时间
            "ai":{                             // 用户账户配置
            "j":true                           // 联合保证金状态
            }
        }
        '''
        # each account is connected to a different endpoint
        # and has exactly one subscriptionhash which is the account type
        subscriptions = list(client.subscriptions.keys())
        accountType = subscriptions[0]
        messageHash = accountType + ':accUpdate'
        result = dict(info=message, event='accUpdate')
        if 'ac' in message:
            ac_msg = message['ac']
            index = client.url.find('/stream')
            marketType = 'spot' if (index >= 0) else 'contract'
            marketId = self.safe_string(ac_msg, 's')
            symbol = self.safe_symbol(marketId, None, None, marketType)
            result['symbol'] = symbol
            result['leverage'] = ac_msg['l']
        # 仅期货合约市场有此事件
        client.resolve(result, messageHash)

    def handle_message(self, client, message):
        methods = {
            'depthUpdate': self.handle_order_book,
            'trade': self.handle_trade,
            'aggTrade': self.handle_trade,
            'kline': self.handle_ohlcv,
            'markPrice_kline': self.handle_ohlcv,
            'indexPrice_kline': self.handle_ohlcv,
            '24hrTicker@arr': self.handle_tickers,
            '24hrMiniTicker@arr': self.handle_tickers,
            '24hrTicker': self.handle_ticker,
            '24hrMiniTicker': self.handle_ticker,
            'bookTicker': self.handle_ticker,
            'outboundAccountPosition': self.handle_balance,
            'balanceUpdate': self.handle_balance,
            'ACCOUNT_UPDATE': self.handle_balance,
            'executionReport': self.handle_order_update,
            'ORDER_TRADE_UPDATE': self.handle_order_update,
            'ACCOUNT_CONFIG_UPDATE': self.handle_account_update
        }
        event = self.safe_string(message, 'e')
        if isinstance(message, list):
            data = message[0]
            event = self.safe_string(data, 'e') + '@arr'
        method = self.safe_value(methods, event)
        if method is None:
            requestId = self.safe_string(message, 'id')
            if requestId is not None:
                return self.handle_subscription_status(client, message)
            # special case for the real-time bookTicker, since it comes without an event identifier
            #
            #     {
            #         u: 7488717758,
            #         s: 'BTCUSDT',
            #         b: '28621.74000000',
            #         B: '1.43278800',
            #         a: '28621.75000000',
            #         A: '2.52500800'
            #     }
            #
            if event is None:
                self.handle_ticker(client, message)
                self.handle_tickers(client, message)
        else:
            return method(client, message)

