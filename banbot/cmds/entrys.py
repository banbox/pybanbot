#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : entrys.py
# Author: anyongjin
# Date  : 2023/4/1
'''
这里只放主要的交易入口逻辑：实盘交易、回测模拟
'''
import asyncio
import signal
from typing import *

from banbot.config import AppConfig
from banbot.storage.common import *
from banbot.util import btime
from banbot.util.common import logger


def term_handler(signum, frame):
    # Raise KeyboardInterrupt - so we can handle it in the same way as Ctrl-C
    raise KeyboardInterrupt()


signal.signal(signal.SIGTERM, term_handler)


def start_trading(args: Dict[str, Any]) -> int:
    """
    Main entry point for trading mode
    """
    from banbot.main.live_trader import LiveTrader
    from banbot.storage.scripts import rebuild_db, get_fail_tables, db

    config = AppConfig.init_by_args(args)
    if get_fail_tables():
        # 有未初始化的表，自动执行脚本创建
        with db():
            rebuild_db(require_confirm=False)
    btime.run_mode = btime.RunMode(config.get('run_mode', 'dry_run'))
    cluster_text = 'Cluster' if config.get('cluster') else 'Stand-Alone'
    logger.warning("Run Mode: %s    Arch: %s", btime.run_mode.value, cluster_text)
    trader = LiveTrader(config)
    try:
        asyncio.run(trader.run())
    except Exception as e:
        logger.error(str(e))
        logger.exception("Fatal exception!")
    except (KeyboardInterrupt):
        logger.info('SIGINT received, aborting ...')
        BotGlobal.state = BotState.STOPPED
        asyncio.run(trader.cleanup())
    finally:
        logger.info("worker found ... calling exit")
    return 0


def start_backtesting(args: Dict[str, Any]) -> None:
    """
    Start Backtesting script
    :param args: Cli args from Arguments()
    :return: None
    """
    # Import here to avoid loading backtesting module when it's not used
    from banbot.optmize.backtest import BackTest

    config = AppConfig.init_by_args(args)
    btime.run_mode = btime.RunMode.BACKTEST
    backtesting = BackTest(config)
    try:
        if args.get('cprofile'):
            cmd_line = 'asyncio.run(backtesting.run())'
            import cProfile
            cProfile.runctx(cmd_line, globals(), locals(), sort='tottime')
        else:
            asyncio.run(backtesting.run())
    except Exception as e:
        logger.error(str(e))
        logger.exception("Fatal exception!")
    except (KeyboardInterrupt):
        logger.info('SIGINT received, aborting ...')
        BotGlobal.state = BotState.STOPPED
    finally:
        logger.info("worker found ... calling exit")

