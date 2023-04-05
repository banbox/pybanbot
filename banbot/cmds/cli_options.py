#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : cli_options.py
# Author: anyongjin
# Date  : 2023/4/1


class Arg:
    # Optional CLI arguments
    def __init__(self, *args, **kwargs):
        self.cli = args
        self.kwargs = kwargs


# List of available command line options
AVAILABLE_CLI_OPTIONS = {
    "config": Arg(
        '-c', '--config',
        help=f'Specify configuration file. '
             f'Multiple --config options may be used. '
             f'Can be set to `-` to read config from stdin.',
        action='append',
        metavar='PATH',
    ),
    "logfile": Arg(
        '--logfile', '--log-file',
        help="Log to the file specified. Special values are: 'syslog', 'journald'. "
             "See the documentation for more details.",
        metavar='FILE',
    ),
    "data_dir": Arg(
        '-d', '--datadir', '--data_dir', '--data-dir',
        help='Path to directory with historical backtesting data.',
        metavar='PATH',
    ),
    "dry_run": Arg(
        '--dry-run',
        help='Enforce dry-run for trading (removes Exchange secrets and simulates trades).',
        action='store_true',
    ),
    "dry_run_wallet": Arg(
        '--dry-run-wallet', '--starting-balance',
        help='Starting balance, used for backtesting / hyperopt and dry-runs.',
        type=float,
    ),
    "fee": Arg(
        '--fee',
        help='Specify fee ratio. Will be applied twice (on trade entry and exit).',
        type=float,
        metavar='FLOAT',
    ),
    "timerange": Arg(
        '--timerange',
        help='Specify what timerange of data to use.',
    ),
    "max_open_trades": Arg(
        '--max-open-trades',
        help='Override the value of the `max_open_trades` configuration setting.',
        type=int,
        metavar='INT',
    ),
    "stake_amount": Arg(
        '--stake-amount',
        help='Override the value of the `stake_amount` configuration setting.',
    ),
    "pairs": Arg(
        '-p', '--pairs',
        help='Limit command to these pairs. Pairs are space-separated.',
        nargs='+',
    ),
}
