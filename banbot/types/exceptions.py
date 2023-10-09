#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : exceptions.py
# Author: anyongjin
# Date  : 2023/9/7

class OperateError(Exception):
    pass


class NetError(Exception):
    pass


class LackOfCash(Exception):
    def __init__(self, amount: float, *args):
        super().__init__(*args)
        self.amount = amount


class AccountBomb(Exception):
    def __init__(self, coin: str, *args):
        super().__init__(*args)
        self.coin = coin
