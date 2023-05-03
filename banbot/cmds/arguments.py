#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : arguments.py
# Author: anyongjin
# Date  : 2023/4/1
import argparse
from typing import Any, Dict, List, Optional
from banbot.cmds.cli_options import *


ARGS_COMMON = ["config", "logfile", "data_dir", "no_db"]

ARGS_TRADE = ["fee"]

ARGS_WEBSERVER: List[str] = []

ARGS_BACKTEST = ["timerange", "stake_amount", "fee", "pairs", "cprofile"]

ARGS_DOWNDATA = ["timerange", "pairs", "timeframes", "medium"]

ARGS_DBCMD = ["action", "tables", "force"]


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

    def _build_args(self, optionlist, parser):

        for val in optionlist:
            opt = AVAILABLE_CLI_OPTIONS[val]
            parser.add_argument(*opt.cli, dest=val, **opt.kwargs)

    def _build_subcommands(self) -> None:
        """
        Builds and attaches all subcommands.
        :return: None
        """
        # Build shared arguments (as group Common Options)
        _common_parser = argparse.ArgumentParser(add_help=False)
        group = _common_parser.add_argument_group("Common arguments")
        self._build_args(optionlist=ARGS_COMMON, parser=group)

        # Build main command
        self.parser = argparse.ArgumentParser(description='Free, open source crypto trading bot')
        self._build_args(optionlist=[], parser=self.parser)

        from banbot.cmds.entrys import start_trading, start_backtesting

        subparsers = self.parser.add_subparsers(dest='command')

        # LiveTrade
        trade_cmd = subparsers.add_parser('trade', help='Trade module.', parents=[_common_parser])
        trade_cmd.set_defaults(func=start_trading)
        self._build_args(optionlist=ARGS_TRADE, parser=trade_cmd)

        # Backtest
        backtesting_cmd = subparsers.add_parser('backtest', help='Backtesting module.', parents=[_common_parser])
        backtesting_cmd.set_defaults(func=start_backtesting)
        self._build_args(optionlist=ARGS_BACKTEST, parser=backtesting_cmd)

        # Download Data
        downdata_cmd = subparsers.add_parser('down_data', help='download data.', parents=[_common_parser])
        from banbot.data.spider import run_down_pairs
        downdata_cmd.set_defaults(func=run_down_pairs)
        self._build_args(optionlist=ARGS_DOWNDATA, parser=downdata_cmd)

        # DbCmd
        db_cmd = subparsers.add_parser('dbcmd', help='database cmd.', parents=[_common_parser])
        from banbot.data.models.scripts import exec_dbcmd
        db_cmd.set_defaults(func=exec_dbcmd)
        self._build_args(optionlist=ARGS_DBCMD, parser=db_cmd)

        # # Add webserver subcommand
        # webserver_cmd = subparsers.add_parser('webserver', help='Webserver module.',
        #                                       parents=[_common_parser])
        # webserver_cmd.set_defaults(func=start_webserver)
        # self._build_args(optionlist=ARGS_WEBSERVER, parser=webserver_cmd)
