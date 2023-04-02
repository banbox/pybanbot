#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : setup.py
# Author: anyongjin
# Date  : 2023/1/28
from setuptools import setup, find_packages
'''
暂不需要
binance-connector需要3.0.0rc1版本，请勿通过pip install安装。可Git clone然后python setup.py install 安装
'''

setup(
    name='banbot',
    version='0.1',
    author='anyongjin',
    author_email='anyongjin163@163.com',
    description='high freq trade',
    packages=find_packages(),
    install_requires=[
        'TA-Lib',
        'pandas-ta',
        'technical',
        'dash>=2.8.1',
        'jupyter-dash',
        'tabulate',
        'ccxt',
        'orjson'
    ]
)
