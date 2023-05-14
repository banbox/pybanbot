#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : timerange.py
# Author: anyongjin
# Date  : 2023/4/5
import logging
import re
from datetime import datetime, timezone
from typing import Optional

import arrow

from banbot.config.consts import DATETIME_PRINT_FORMAT
from banbot.util.common import logger


class TimeRange:
    """
    object defining timerange inputs.
    [start/stop]type defines if [start/stop]ts shall be used.
    if *type is None, don't use corresponding startvalue.
    """

    def __init__(self, startts: float = 0, stopts: float = 0):
        self.startts = startts
        self.stopts = stopts

    @property
    def startdt(self) -> Optional[datetime]:
        if self.startts:
            return datetime.fromtimestamp(self.startts, tz=timezone.utc)
        return None

    @property
    def stopdt(self) -> Optional[datetime]:
        if self.stopts:
            return datetime.fromtimestamp(self.stopts, tz=timezone.utc)
        return None

    @property
    def timerange_str(self) -> str:
        """
        Returns a string representation of the timerange as used by parse_timerange.
        Follows the format yyyymmdd-yyyymmdd - leaving out the parts that are not set.
        """
        start = ''
        stop = ''
        if startdt := self.startdt:
            start = startdt.strftime('%Y%m%d')
        if stopdt := self.stopdt:
            stop = stopdt.strftime('%Y%m%d')
        return f"{start}-{stop}"

    def __str__(self):
        return f'{self.startts}-{self.stopts}'

    def __eq__(self, other):
        """Override the default Equals behavior"""
        return self.startts == other.startts and self.stopts == other.stopts

    @staticmethod
    def parse_timerange(text: Optional[str]) -> 'TimeRange':
        """
        Parse the value of the argument --timerange to determine what is the range desired
        :param text: value from --timerange
        :return: Start and End range period
        """
        if text is None:
            return TimeRange(0, 0)
        syntax = [(r'^-(\d{8})$', (None, 'date')),
                  (r'^(\d{8})-$', ('date', None)),
                  (r'^(\d{8})-(\d{8})$', ('date', 'date')),
                  (r'^-(\d{10})$', (None, 'date')),
                  (r'^(\d{10})-$', ('date', None)),
                  (r'^(\d{10})-(\d{10})$', ('date', 'date')),
                  (r'^-(\d{13})$', (None, 'date')),
                  (r'^(\d{13})-$', ('date', None)),
                  (r'^(\d{13})-(\d{13})$', ('date', 'date')),
                  ]
        for rex, stype in syntax:
            # Apply the regular expression to text
            match = re.match(rex, text)
            if match:  # Regex has matched
                rvals = match.groups()
                index = 0
                start: int = 0
                stop: int = 0
                if stype[0]:
                    starts = rvals[index]
                    if stype[0] == 'date' and len(starts) == 8:
                        start = arrow.get(starts, 'YYYYMMDD').int_timestamp
                    elif len(starts) == 13:
                        start = int(starts) // 1000
                    else:
                        start = int(starts)
                    index += 1
                if stype[1]:
                    stops = rvals[index]
                    if stype[1] == 'date' and len(stops) == 8:
                        stop = arrow.get(stops, 'YYYYMMDD').int_timestamp
                    elif len(stops) == 13:
                        stop = int(stops) // 1000
                    else:
                        stop = int(stops)
                if start > stop > 0:
                    raise ValueError(
                        f'Start date is after stop date for timerange "{text}"')
                return TimeRange(start, stop)
        raise ValueError(f'Incorrect syntax for timerange "{text}"')
