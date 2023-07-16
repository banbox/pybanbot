#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : __main__.py
# Author: anyongjin
# Date  : 2023/4/1
import asyncio
import sys

from banbot.cmds.arguments import *
from banbot.util.common import logger


def main(sysargv: Optional[List[str]] = None) -> None:
    """
    This function will initiate the bot and start the trading loop.
    :return: None
    """

    return_code: Any = 1
    try:
        arguments = Arguments(sysargv)
        args = arguments.get_parsed_arg()

        # Call subcommand.
        if 'func' in args:
            run_func = args['func']
            if asyncio.iscoroutinefunction(run_func):
                return_code = asyncio.run(run_func(args))
            else:
                return_code = run_func(args)
        else:
            # No subcommand was issued.
            raise RuntimeError(
                "Usage of Banbot requires a subcommand to be specified.\n"
            )

    except SystemExit as e:  # pragma: no cover
        return_code = e
    except KeyboardInterrupt:
        logger.info('SIGINT received, aborting ...')
        return_code = 0
    except Exception:
        logger.exception('Fatal exception!')
    finally:
        sys.exit(return_code)


if __name__ == '__main__':
    main()
