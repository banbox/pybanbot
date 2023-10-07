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

    async def watch_mark_prices(self, params={}):
        await self.load_markets()
        defaultType = self.safe_string(self.options, 'defaultType', 'spot')
        api_type = self.safe_string(params, 'type', defaultType)
        subType, params = self.handle_sub_type_and_params('watchMarkPrices', None, params)
        if self.isLinear(api_type, subType):
            api_type = 'future'
        elif self.isInverse(api_type, subType):
            api_type = 'delivery'
        messageHash = '!markPrice@arr'
        url = self.urls['api']['ws'][api_type] + '/' + self.stream(api_type, messageHash)
        requestId = self.request_id(url)
        request = {
            'method': 'SUBSCRIBE',
            'params': [
                messageHash,
            ],
            'id': requestId,
        }
        subscribe = {
            'id': requestId,
        }
        query = self.omit(params, 'type')
        message = self.extend(request, query)
        return await self.watch(url, messageHash, message, messageHash, subscribe)

    def handle_mark_prices(self, client, message):
        '''
        [
          {
            "e": "markPriceUpdate",     // 事件类型
            "E": 1562305380000,         // 事件时间
            "s": "BTCUSDT",             // 交易对
            "p": "11185.87786614",      // 标记价格
            "i": "11784.62659091"       // 现货指数价格
            "P": "11784.25641265",      // 预估结算价,仅在结算前最后一小时有参考价值
            "r": "0.00030000",          // 资金费率
            "T": 1562306400000          // 下个资金时间
          }
        ]
        '''
        messageHash = '!markPrice@arr'
        now = self.milliseconds()
        index = client.url.find('/stream')
        marketType = 'spot' if (index >= 0) else 'contract'
        result = []
        for msg in message:
            marketId = self.safe_string(msg, 's')
            symbol = self.safe_symbol(marketId, None, None, marketType)
            result.append(dict(
                info=msg,
                event=self.safe_string(msg, 'e'),
                timestamp=self.safe_integer(msg, 'E', now),
                symbol=symbol,
                markPrice=self.safe_float(msg, 'p'),
                price=self.safe_float(msg, 'i'),
            ))
        client.resolve(result, messageHash)

    async def watch_account_config(self, params={}):
        """
        watches account update: leverage and margin
        :param dict params: extra parameters specific to the binance api endpoint
        :returns dict: a account update
        """
        await self.load_markets()
        await self.authenticate(params)
        defaultType = self.safe_string(self.options, 'defaultType', 'spot')
        api_type = self.safe_string(params, 'type', defaultType)
        subType, params = self.handle_sub_type_and_params('watchAccountConfig', None, params)
        if self.isLinear(api_type, subType):
            api_type = 'future'
        elif self.isInverse(api_type, subType):
            api_type = 'delivery'
        url = self.urls['api']['ws'][api_type] + '/' + self.options[api_type]['listenKey']
        client = self.client(url)
        self.set_balance_cache(client, api_type)
        message = None
        messageHash = api_type + ':accUpdate'
        return await self.watch(url, messageHash, message, api_type)

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
            'ACCOUNT_CONFIG_UPDATE': self.handle_account_update,
            'markPriceUpdate@arr': self.handle_mark_prices
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

