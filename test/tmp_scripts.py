#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : tmp_scripts.py
# Author: anyongjin
# Date  : 2023/11/4
import os
import shutil
from typing import *


def rn_ws_dir():
    """将下载的ws数据文件名重命名，把时间戳后缀删除，作为文件夹名称"""
    data_dir = r'E:\trade\ban_data\binanceusdm'
    names = os.listdir(data_dir)
    names = [n for n in names if n.endswith('.pkl')]
    groups: Dict[str, List[Tuple[str, str]]] = dict()
    for n in names:
        parts = n.split('.')[0].split('_')
        try:
            int(parts[-1])
        except ValueError:
            continue
        time_ts = parts[-1]
        if len(time_ts) < 10:
            continue
        if time_ts not in groups:
            groups[time_ts] = []
        # 新文件名不含时间戳后缀
        new_name = ('_'.join(parts[:-1])) + '.pkl'
        groups[time_ts].append((n, new_name))
    move_num = 0
    for ts_name, items in groups.items():
        ts_path = os.path.join(data_dir, ts_name)
        if not os.path.isdir(ts_path):
            os.mkdir(ts_path)
        for src_name, tgt_name in items:
            src_path = os.path.join(data_dir, src_name)
            tgt_path = os.path.join(ts_path, tgt_name)
            # 将文件从不带时间戳的路径移动到带时间戳的路径
            shutil.move(src_path, tgt_path)
            move_num += 1
    print(f'moved {move_num} files')


if __name__ == '__main__':
    rn_ws_dir()
