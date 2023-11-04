#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : __main__.py
# Author: anyongjin
# Date  : 2023/4/1
import asyncio
import logging
import sys

from banbot.cmds.arguments import *
from banbot.util.common import logger, set_log_level

# logging.basicConfig(level=logging.DEBUG)


async def _run_main(run_func: Callable, nocompress: bool, args):
    from banbot.util.misc import run_async
    from banbot.storage import KLine
    if nocompress:
        async with KLine.decompress():
            await run_async(run_func, args)
    else:
        await run_async(run_func, args)


def main(sysargv: Optional[List[str]] = None) -> None:
    """
    This function will initiate the bot and start the trading loop.
    :return: None
    """

    return_code: Any = 1
    try:
        arguments = Arguments(sysargv)
        args = arguments.get_parsed_arg()
        if args.get('debug'):
            from banbot.util import btime
            btime.debug = True
            set_log_level(logging.DEBUG)
            logger.debug('set logging level to DEBUG')

        # Call subcommand.
        if 'func' in args:
            run_func = args['func']
            nocompress = args.get('nocompress')

            if args.get('cprofile'):
                cmd_line = 'asyncio.run(_run_main(run_func, nocompress, args))'
                import cProfile
                cProfile.runctx(cmd_line, globals(), locals(), sort='tottime')
            else:
                asyncio.run(_run_main(run_func, nocompress, args))
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
