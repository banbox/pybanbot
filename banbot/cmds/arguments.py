#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : arguments.py
# Author: anyongjin
# Date  : 2023/4/1
import argparse
from typing import Any, Dict, List, Optional, Callable

from banbot.cmds.cli_options import *

ARGS_COMMON = ["config", "logfile", "data_dir", "no_db", "debug"]


class Arguments:

    def __init__(self, args: Optional[List[str]]) -> None:
        self.args = args
        self._parsed_arg: Optional[argparse.Namespace] = None

    def get_parsed_arg(self) -> Dict[str, Any]:
        """
        Return the list of arguments
        :return: List[str] List of arguments
        """
        if self._parsed_arg is None:
            self._build_subcommands()
            self._parsed_arg = self.parser.parse_args(self.args)

        return vars(self._parsed_arg)

    def _build_subcommands(self) -> None:
        """
        Builds and attaches all subcommands.
        :return: None
        """
        # Build shared arguments (as group Common Options)
        com_parser = argparse.ArgumentParser(add_help=False)
        group = com_parser.add_argument_group("Common arguments")
        _build_args(group, ARGS_COMMON)

        self.parser = argparse.ArgumentParser(description='Fast Trading Bot')
        subparsers = self.parser.add_subparsers(dest='command')

        # 注册子命令
        sub_cmds = [_reg_trade, _reg_backtest, _reg_down_data, _reg_dbcmd, _reg_od_compare, _reg_spider]
        for sub_md in sub_cmds:
            sub_md(subparsers, parents=[com_parser])


def _build_args(parser, optionlist):
    for val in optionlist:
        opt = AVAILABLE_CLI_OPTIONS[val]
        parser.add_argument(*opt.cli, dest=val, **opt.kwargs)


def _reg_sub(subparsers, name: str, opts: List[str], run_fn: Callable, **kwargs):
    '''
    注册命令行子命令。
    :paran run_fn: 子命令启动函数，可以是异步函数
    '''
    parser = subparsers.add_parser(name, **kwargs)
    parser.set_defaults(func=run_fn)
    _build_args(parser, opts)


def _reg_trade(subparsers, **kwargs):
    from banbot.cmds.entrys import start_trading
    opts = ["stake_amount", "fee", "pairs", "cluster", "stg_dir"]
    _reg_sub(subparsers, 'trade', opts, start_trading, help='Live Trade', **kwargs)


def _reg_backtest(subparsers, **kwargs):
    from banbot.cmds.entrys import start_backtesting
    opts = ["timerange", "stake_amount", "fee", "pairs", "cprofile", "stg_dir"]
    _reg_sub(subparsers, 'backtest', opts, start_backtesting, help='backtest', **kwargs)


def _reg_down_data(subparsers, **kwargs):
    from banbot.data.spider import run_down_pairs
    opts = ["timerange", "pairs", "timeframes", "medium"]
    _reg_sub(subparsers, 'down_data', opts, run_down_pairs, help='download data', **kwargs)


def _reg_dbcmd(subparsers, **kwargs):
    from banbot.storage.scripts import exec_dbcmd
    opts = ["action", "tables", "force", "yes"]
    _reg_sub(subparsers, 'dbcmd', opts, exec_dbcmd, help='database cmd', **kwargs)


def _reg_spider(subparsers, **kwargs):
    from banbot.data.spider import run_spider_forever
    _reg_sub(subparsers, 'spider', [], run_spider_forever, help='spider cmd', **kwargs)


def _reg_od_compare(subparsers, **kwargs):
    from banbot.optmize.od_compare import run_od_compare
    opts = ["task_hash", "task_id"]
    _reg_sub(subparsers, 'od_compare', opts, run_od_compare, help='compare backtest with live orders', **kwargs)

