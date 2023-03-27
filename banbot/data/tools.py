#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : tools.py
# Author: anyongjin
# Date  : 2023/2/28
import datetime
import os
from banbot.util.common import logger


def convert_bnb_klines_datas(data_dir: str, timeframe: str, group_num: int = 7):
    '''
    将币安下载的K线数据(csv，每个文件是一天的数据)转为feather格式，只保留关键列。
    时间戳，开盘价，最高价，最低价，收盘价，成交量，笔数，主动买入量
    :param data_dir:
    :param timeframe:
    :param group_num:
    :return:
    '''
    import pandas as pd
    names = os.listdir(data_dir)
    fea_tf = f'-{timeframe}-'
    csv_names = [name for name in names if name.endswith('.csv') and name.find(fea_tf) > 0]
    csv_names = sorted(csv_names)
    # tgt_names = [name for name in names if name.endswith('.feather') and name.find(fea_tf) > 0]
    data_list = []
    pair, date_val, start_date = None, None, None

    def merge_and_save():
        nonlocal data_list, start_date
        merge_df = pd.concat(data_list, ignore_index=True, sort=False)
        start_text, end_text = start_date.strftime("%Y-%m-%d"), date_val.strftime("%Y-%m-%d")
        out_name = f'{pair}{fea_tf}{start_text}-{end_text}.feather'
        merge_df.to_feather(os.path.join(data_dir, out_name), compression='lz4')
        logger.info(f'saved: {out_name}')
        data_list = []
        start_date = None
    for i, name in enumerate(csv_names):
        cpair, cdate = (os.path.splitext(name)[0]).split(fea_tf)
        cdata_val = datetime.datetime.strptime(cdate, '%Y-%m-%d').date()
        if date_val and ((cdata_val - date_val).days != 1 or pair != cpair):
            # 和上一个时间不连续
            logger.warning(f'{cpair} date not continus : {date_val} -- {cdata_val}')
            merge_and_save()
        elif len(data_list) >= group_num:
            merge_and_save()
        usecols = ['date', 'open', 'high', 'low', 'close', 'volume', 'count', 'long_vol']
        col_ids = [0, 1, 2, 3, 4, 5, 8, 9]
        max_col_num = max(col_ids) + 1
        df = pd.read_csv(os.path.join(data_dir, name), header=None, usecols=list(range(max_col_num)))
        df.columns = [f'col{i}' if i not in col_ids else usecols[col_ids.index(i)] for i in range(max_col_num)]
        df = df[usecols]
        df['date'] = pd.to_datetime(df['date'], utc=True, unit='ms')
        data_list.append(df)
        pair, date_val = cpair, cdata_val
        if not start_date:
            start_date = cdata_val
    merge_and_save()


if __name__ == '__main__':
    data_dir = r'E:\trade\freqtd_data\user_data\spec_data\bnb1s'
    convert_bnb_klines_datas(data_dir, '1s')
