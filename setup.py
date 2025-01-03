#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : setup.py
# Author: anyongjin
# Date  : 2023/1/28
from setuptools import setup
'''
暂不需要
binance-connector需要3.0.0rc1版本，请勿通过pip install安装。可Git clone然后python setup.py install 安装
'''
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='banbot',
    version='0.1',
    author='anyongjin',
    author_email='anyongjin163@163.com',
    description='high freq trade',
    packages=['banbot'],
    extras_require={
        "std": requirements,
        "develop": [
            'TA-Lib',
            'pandas-ta',
            'technical',
            'pandas',
            'plotly',
            'scikit_learn'
        ]
    }
)
