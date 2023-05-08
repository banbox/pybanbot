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
    "timeframes": Arg(
        '-t', '--timeframes',
        help='Specify which tickers to download. Space-separated list. '
        'Default: `1m`.',
        default=['1m'],
        nargs='+',
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
    "cprofile": Arg('--cprofile', help='perfoamance profile', action='store_true', default=False),
    "action": Arg('--action', help='action name'),
    "tables": Arg('--tables', help='db tables, comma-separated.', nargs='+'),
    "force": Arg('--force', help='force action', action='store_true', default=False),
    "medium": Arg('--medium', help='data medium:db,file', default='db'),
    "no_db": Arg('--no_db', help='not auto load db', action='store_true', default=False),
    "task_hash": Arg('--task_hash', help='stg_hash for tasks'),
    "task_id": Arg('--task_id', help='task ids', type=int, nargs='+'),
}
