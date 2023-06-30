#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : test_common.py
# Author: anyongjin
# Date  : 2023/6/30
from typing import Optional


class ShapeBase:
    obj: Optional['ShapeBase'] = None

class Rect(ShapeBase):
    def __init__(self):
        self.name = 'rect'
        Rect.obj = self

class Trianble(ShapeBase):
    def __init__(self):
        self.name = 'trianble'
        Trianble.obj = self


def test_shapes():
    rect1 = Rect()
    triangle = Trianble()
    print(rect1.name)
    print(triangle.name)
    print(Rect.obj.name)
    print(Trianble.obj.name)
    print(ShapeBase.obj)
