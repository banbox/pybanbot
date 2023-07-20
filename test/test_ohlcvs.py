#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_ohlcvs.py
# Author: anyongjin
# Date  : 2023/5/17
import asyncio
import csv
import random
import time

from banbot.data.tools import *
from banbot.storage.base import init_db, db
from banbot.config import AppConfig


async def test_down():
    from banbot.exchange.crypto_exchange import get_exchange
    from banbot.storage import KLine
    exg = get_exchange('binance', 'future')
    symbol, timeframe = 'BTC/USDT:USDT', '1m'
    tf_msecs = tf_to_secs(timeframe) * 1000
    stop_ms = btime.utcstamp()
    start_ms = stop_ms - tf_msecs * 3
    arr = await fetch_api_ohlcv(exg, symbol, timeframe, start_ms, stop_ms)
    # sid = ExSymbol.get_id(exg.name, symbol, exg.market_type)
    # KLine.insert(sid, timeframe, arr)
    [print(r) for r in arr]
    bar_end = stop_ms // tf_msecs * tf_msecs
    print(bar_end)


def test_kline_insert():
    from banbot.storage import KLine
    KLine.sync_timeframes()
    insert_num = 300
    open_price = 30000
    ohlcvs = []
    cur_stamp = btime.utcstamp() - 1000 * insert_num * 60
    cost_list = []
    for i in range(insert_num):
        oprice = open_price + random.random() - 0.5
        hprice = oprice + 1
        lprice = oprice - 1
        cprice = oprice + random.random()
        bar = (cur_stamp, oprice, hprice, lprice, cprice, random.random() * 1000)
        ohlcvs.append(bar)
        open_price = cprice
        cur_stamp += 60 * 1000
        start = time.monotonic()
        KLine.insert(14, '1m', [bar])
        cost = time.monotonic() - start
        if i > 10:
            cost_list.append(cost)
    print(sum(cost_list) / len(cost_list), max(cost_list), min(cost_list))


async def test_get_kline():
    from banbot.exchange.crypto_exchange import get_exchange
    symbol = 'BTC/USDT:USDT'
    timeframe = '1m'
    exg = get_exchange('binance', 'future')
    # exs = ExSymbol.get('binance', symbol, 'future')
    end_ms = btime.utcstamp()
    # start_ms = end_ms - 30 * 24 * 3600 * 1000
    start_ms = 1689831600000
    arr = await fetch_api_ohlcv(exg, symbol, timeframe, start_ms, end_ms)
    # data = await auto_fetch_ohlcv(exg, exs, '1w', start_ms, end_ms, with_unfinish=True)
    print(arr)


def _compare_ohlcv_arr(arr1: List[tuple], arr2: List[tuple]):
    id1, id2 = 0, 0
    diff_list = []
    while id1 < len(arr1) and id2 < len(arr2):
        bar1, bar2 = arr1[id1], arr2[id2]
        if bar1[0] < bar2[0]:
            id1 += 1
            continue
        if bar2[0] < bar1[0]:
            id2 += 1
            continue
        id1 += 1
        id2 += 1
        price_diff = max(
            abs(bar1[1] - bar2[1]),
            abs(bar1[2] - bar2[2]),
            abs(bar1[3] - bar2[3]),
            abs(bar1[4] - bar2[4]),
        )
        off_rate = price_diff / bar2[4]
        if off_rate < 0.0001:
            continue
        diff_list.append(bar1[0])
    return diff_list


async def test_trade_agg():
    '''
    从原始交易流生成1m的ohlcv，然后从交易所获取1m的ohlcv，都输出到文件，进行比较。
    需要给TradesWatcher的__init__添加下面：
    cln_p = pair.replace('/', '_').replace(':', '_')
    out_path = f'E:/Data/temp/{exg_name}_{market}_{cln_p}.csv'
    import csv
    wt_mode = 'a' if os.path.isfile(out_path) else 'w'
    self.out_file = open(out_path, wt_mode, newline='')
    self.writer = csv.writer(self.out_file)
    self.save_ts = time.time()
    通过watch_trades获取到交易后，用下面代码存储到文件
    csv_rows = [(t['info']['T'], t['info']['E'], t['price'], t['amount'], t['side']) for t in details]
    self.writer.writerows(csv_rows)
    if time.time() - self.save_ts > 10:
        print(f'flush: {self.pair}')
        self.out_file.flush()
        self.save_ts = time.time()
    '''
    data_dir = 'E:/trade/ban_data/temp/'
    names = os.listdir(data_dir)
    for fname in names:
        if not fname.endswith('.csv'):
            continue
        logger.info(f'process: {fname}')
        trade_path = os.path.join(data_dir, fname)
        parts = fname.split('_')
        if len(parts) != 5:
            continue
        coin = parts[2]
        fdata = open(trade_path, 'r')
        trades = []
        for line in fdata:
            time_ts, _, price, amount, side = line.strip().split(',')
            trades.append(dict(timestamp=int(time_ts), price=float(price), amount=float(amount)))
        from banbot.data.tools import build_ohlcvc, trades_to_ohlcv
        ohlcv_arr = trades_to_ohlcv(trades)
        ohlcv_arr, _ = build_ohlcvc(ohlcv_arr, 60, with_count=False)
        start_ms, end_ms = ohlcv_arr[1][0], ohlcv_arr[-1][0]
        ohlcv_arr = ohlcv_arr[1:-1]  # 只保留完整的bar
        agg_path = data_dir + f'{coin}_agg.csv'
        with open(agg_path, 'w', newline='') as fout:
            agg_writer = csv.writer(fout)
            agg_writer.writerows(ohlcv_arr)

        from banbot.exchange.crypto_exchange import get_exchange
        exg = get_exchange('binance', 'future')
        ohlcv_true = await fetch_api_ohlcv(exg, f'{coin}/USDT:USDT', '1m', start_ms, end_ms)
        agg_path = data_dir + f'{coin}_true.csv'
        with open(agg_path, 'w', newline='') as fout:
            agg_writer = csv.writer(fout)
            agg_writer.writerows(ohlcv_true)

        # 对比两个蜡烛数组
        diff_list = _compare_ohlcv_arr(ohlcv_arr, ohlcv_true)
        if not diff_list:
            continue
        diff_text = "\n".join(diff_list)
        print(f'{coin} diff: {diff_text}')


class TradeValidater:
    def __init__(self, price_chg: float, vol: float, price_factor: float = 5, unit_factor: float = 500,
                 cut_when: int = 30, cut_to: int = 10, price_fac: int = 1000):
        '''
        :param price_chg: 价格区间，当价格变化未超出此范围，认为有效
        :param vol: 价格区间对应成交量
        '''
        # 价格区间，及对应的成交量
        self.his_chgs: List[float] = []
        self.his_vols: List[float] = []
        # 单位价格成交量:可使用上一跟蜡烛的vol/(high-low)初始化
        price_chg *= price_fac
        self.unit_vol = vol / price_chg
        self.price_range = price_chg
        # 当前价格
        self.cur_price: Optional[float] = None
        # 当前价格累计成交量
        self.cur_vol: Optional[float] = None
        # 最大允许交易超出的倍数
        self.price_factor = price_factor
        self.unit_factor = unit_factor
        self.cut_when = cut_when
        self.cut_to = cut_to
        self.price_fac = price_fac
        self.prev_filtered = False

    def validate(self, price: float, vol: float) -> Tuple[str, bool]:
        '''
        验证某笔订单是否有效。并更新价格成交量
        '''
        if self.cur_price == price:
            self.cur_vol += vol
            return '', True
        if self.cur_price is None:
            self.cur_price = price
            self.cur_vol = vol
            return '', True
        price_chg = abs(price - self.cur_price) * self.price_fac
        vol_factor = self.unit_vol * price_chg / vol
        price_fac = price_chg / self.price_range
        if price_fac > self.price_factor and vol_factor > self.unit_factor:
            if self.prev_filtered:
                # 出现极值单的概率不足0.01%，为避免连续多个正常订单偶发超过限制被过滤。
                # 这里当前一个被过滤后，强制下一个认为有效
                self.prev_filtered = False
                # logger.error(f'conti trade invalid, price: {price}, vol: {vol}, fac: {td_factor}')
            else:
                self.prev_filtered = True
                return f'{price_fac:.1f},{vol_factor:.1f}', False
        elif self.prev_filtered:
            self.prev_filtered = False
        # 交易有效，将cur_price缓存到chg_vols
        self.his_chgs.append(price_chg)
        self.his_vols.append(self.cur_vol)
        if len(self.his_chgs) > self.cut_when:
            self.his_chgs = self.his_chgs[:self.cut_to]
            self.his_vols = self.his_vols[:self.cut_to]
            # 更新单位价格成交量
            self.unit_vol = sum(self.his_vols) / sum(self.his_chgs)
            self.price_range = sum(self.his_chgs)
        self.cur_price = price
        self.cur_vol = vol
        return f'{price_fac:.1f},{vol_factor:.1f}', True


def analyze_trade_agg(coin: str):
    data_dir = 'E:/trade/ban_data/temp/'
    trade_path = os.path.join(data_dir, f'binance_future_{coin}_USDT_USDT.csv')
    true_reader = csv.reader(open(os.path.join(data_dir, f'true_ohlcv/{coin}_true.csv'), 'r'))
    trade_reader = csv.reader(open(trade_path, 'r'))
    true_ohlcv = [(int(r[0]), *[float(v) for v in r[1:]]) for r in true_reader]
    trades = [(int(r[0]), int(r[1]), float(r[2]), float(r[3]), r[4]) for r in trade_reader]
    agg_ohlcv = [(t[0], t[2], t[2], t[2], t[2], t[3]) for t in trades]
    agg_ohlcv, _ = build_ohlcvc(agg_ohlcv, 60, with_count=False)
    agg_ohlcv = agg_ohlcv[1:-1]  # 只保留完整的bar

    # 对比异常的时间戳
    diff_list = _compare_ohlcv_arr(agg_ohlcv, true_ohlcv)

    bar = agg_ohlcv[0]
    validater = TradeValidater((bar[2] - bar[3]) or 0.001, bar[5])
    agg_id, true_id, trade_id = 0, 0, 0
    bar_intv = true_ohlcv[1][0] - true_ohlcv[0][0]
    logger.info(f'process: {coin}')
    for care_ts in diff_list:
        care_end = care_ts + bar_intv
        while agg_ohlcv[agg_id][0] < care_ts:
            agg_id += 1
        while true_ohlcv[true_id][0] < care_ts:
            true_id += 1
        agg_bar, true_bar = agg_ohlcv[agg_id], true_ohlcv[true_id]
        is_too_big = max(agg_bar[1:-1]) > max(true_bar[1:-1])
        test_price = max(true_bar[1:-1]) if is_too_big else min(true_bar[1:-1])

        bad_ids = []
        del_true, del_bad, skip_bad = [], [], []
        while trade_id < len(trades) and trades[trade_id][0] < care_end:
            time_ts, end_ts, price, amount, side = trades[trade_id]
            trade_id += 1
            fac_text, is_valid = validater.validate(price, amount)
            if time_ts < care_ts:
                if not is_valid:
                    del_true.append(time_ts)
                    logger.warning(f'del true: {time_ts}, fac: {fac_text}')
                continue
            trades.append((time_ts, end_ts, price, amount, side))
            if is_too_big and price > test_price or not is_too_big and price < test_price:
                if is_valid:
                    skip_bad.append(time_ts)
                    logger.warning(f'skip bad: {time_ts}, fac: {fac_text}')
                else:
                    del_bad.append(time_ts)
                    logger.warning(f'del bad: {time_ts}, fac: {fac_text}')
                bad_ids.append(len(trades) - 1)
            else:
                if not is_valid:
                    del_true.append(time_ts)
        print(f'del_true: {del_true}, \ndel_bad: {del_bad}, \nskip_bad: {skip_bad}, \nbad_ids: {bad_ids}')
        # if not bad_ids:
        #     continue
        # bad_name = '_'.join([str(i) for i in bad_ids])
        # out_path = os.path.join(data_dir,  f'bads/{coin}_{care_ts}_{bad_name}.csv')
        # with open(out_path, 'w', newline='') as fout:
        #     writer = csv.writer(fout)
        #     writer.writerow(('start', 'stop', 'price', 'amount', 'side'))
        #     writer.writerows(trades)
        # print(f'write {coin} {care_ts} ok, found bad: {len(bad_ids)}')


def dump_bad_trades():
    bad_dic = {
        '1000LUNC': [1689722640000],
        '1000PEPE': [1689726960000],
        '1000XEC': [1689689160000],
        'AGIX': [1689698880000],
        'ALGO': [1689663480000],
        'ALICE': [1689698880000],
        'ALPHA': [1689682500000, 1689698880000],
        'APE': [1689681300000, 1689732600000],
        'AUDIO': [1689698880000],
        'AVAX': [1689698880000],
        'AXS': [1689684540000],
        'BAT': [1689698880000],
        'BCH': [1689682500000, 1689700200000, 1689712260000],
        'BTC': [1689668820000, 1689677460000, 1689684720000, 1689732600000],
        'C98': [1689694860000],
        'CFX': [1689698880000],
        'COMP': [1689668760000, 1689698880000],
        'CRV': [1689681180000],
        'DOT': [1689731400000],
        'DYDX': [1689698880000],
        'EGLD': [1689698880000],
        'ENJ': [1689741600000],
        'ETC': [1689698880000],
        'ETH': [1689670920000, 1689731220000],
        'FIL': [1689698880000],
        'GALA': [1689681300000, 1689721560000],
        'GMT': [1689661920000, 1689682500000],
        'GRT': [1689716400000],
        'HBAR': [1689698880000],
        'HOOK': [1689736140000],
        'LDO': [1689682380000],
        'LINK': [1689700620000],
        'LTC': [1689735600000],
        'MATIC': [1689698880000],
        'ONE': [1689698880000],
        'QTUM': [1689698880000],
        'RAD': [1689698880000],
        'RNDR': [1689698880000],
        'SOL': [1689682500000, 1689698580000],
        'STX': [1689739260000],
        'SUI': [1689698880000],
        'SUSHI': [1689698760000],
        'XRP': [1689698880000],
        'XVG': [1689658860000],
    }
    data_dir = 'E:/trade/ban_data/temp/'
    out_path = os.path.join(data_dir, 'all_bads.txt')
    fout = open(out_path, 'w', newline='')
    for key, ts_list in bad_dic.items():
        analyze_trade_agg(key, ts_list)
        # if not near_tds:
        #     continue
        # fout.write(f'{key}\n')
        # fout.writelines([(','.join([str(v) for v in t])) + '\n' for t in near_tds])
        # fout.write('\n')
        # fout.flush()
    fout.close()


if __name__ == '__main__':
    AppConfig.init_by_args()
    with db():
        # test_kline_insert()
        # analyze_trade_agg('BCH')
        asyncio.run(test_get_kline())
