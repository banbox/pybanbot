#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : spread_filter.py
# Author: anyongjin
# Date  : 2023/4/19
from banbot.plugins.pairlist.base import *


class SpreadFilter(PairList):
    '''
    按当下交易活跃度过滤交易对。仅适用于实盘，不可用于回测。（因未记录任意时刻订单簿状态）
    基于tickers中交易对的 1-bid/ask 计算。
    此值过大说明买方和卖方价差过大，流通性较差，限价单不容易成交，滑点较大。
    '''
    need_tickers = True

    def __init__(self, manager, exchange: CryptoExchange,
                 config: Config, handler_cfg: Dict[str, Any]):
        super(SpreadFilter, self).__init__(manager, exchange, config, handler_cfg)

        self.max_ratio = handler_cfg.get('max_ratio', 0.005)
        self.enable = self.enable and self.max_ratio > 0

        if btime.run_mode not in TRADING_MODES:
            self.enable = False
            logger.warning('SpreadFilter not avaiable in backtest, skipping ...')
            return

        if not exchange.get_option('tickers_have_bid_ask'):
            raise RuntimeError(
                f"{self.name} requires exchange to have bid/ask data for tickers, "
                "which is not available for the selected exchange / trading mode."
            )

    def _validate_pair(self, pair: str, ticker: Optional[Ticker]) -> bool:
        if ticker and 'bid' in ticker and 'ask' in ticker and ticker['ask'] and ticker['bid']:
            spread = 1 - ticker['bid'] / ticker['ask']
            if spread > self.max_ratio:
                logger.info(f"Removed {pair} from whitelist, because spread "
                            f"{spread:.3%} > {self.max_ratio:.3%}")
                return False
            else:
                return True
        logger.info(f"Removed {pair} from whitelist due to invalid ticker data: {ticker}")
        return False

