#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : entrys.py
# Author: anyongjin
# Date  : 2023/4/1
import asyncio
import signal
from typing import *
from banbot.util.common import logger


def start_trading(args: Dict[str, Any]) -> int:
    """
    Main entry point for trading mode
    """
    # Import here to avoid loading worker module when it's not used
    from banbot.main.live_trader import LiveTrader
    from banbot.util import btime
    from banbot.config import Configuration

    def term_handler(signum, frame):
        # Raise KeyboardInterrupt - so we can handle it in the same way as Ctrl-C
        raise KeyboardInterrupt()

    # Create and run worker
    config = Configuration(args, None).get_config()
    signal.signal(signal.SIGTERM, term_handler)
    btime.run_mode = btime.RunMode(config.get('run_mode', 'dry_run'))
    logger.warning(f"Run Mode: {btime.run_mode.value}")
    trader = LiveTrader(config)
    try:
        asyncio.run(trader.run())
    except Exception as e:
        logger.error(str(e))
        logger.exception("Fatal exception!")
    except (KeyboardInterrupt):
        logger.info('SIGINT received, aborting ...')
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
    from banbot.config import Configuration
    from banbot.util import btime
    config = Configuration(args, None).get_config()

    btime.run_mode = btime.RunMode.BACKTEST
    backtesting = BackTest(config, 10000)
    try:
        asyncio.run(backtesting.run())
    except Exception as e:
        logger.error(str(e))
        logger.exception("Fatal exception!")
    except (KeyboardInterrupt):
        logger.info('SIGINT received, aborting ...')
        asyncio.run(backtesting.cleanup())
    finally:
        logger.info("worker found ... calling exit")

