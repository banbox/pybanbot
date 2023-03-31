#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : resolver.py
# Author: anyongjin
# Date  : 2023/3/31
from typing import *

import six

from banbot.strategy.base import BaseStrategy
from pathlib import Path
import sys
import importlib
import importlib.util
from banbot.util.common import logger
import inspect


class PathModifier:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        """Inject path to allow importing with relative imports."""
        sys.path.insert(0, str(self.path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Undo insertion of local path."""
        str_path = str(self.path)
        if str_path in sys.path:
            sys.path.remove(str_path)


def _load_module_strategies(module: Union[str, Any]):
    if isinstance(module, six.string_types):
        module = importlib.import_module(module)
    clsmembers = inspect.getmembers(module, inspect.isclass)
    result = []
    for (name, cls) in clsmembers:
        if not issubclass(cls, BaseStrategy) or name == 'BaseStrategy':
            continue
        result.append(cls)
    return result


def _load_file_strategies(path: str):
    module_path = Path(path)
    with PathModifier(module_path.parent):
        module_name = module_path.stem or ""
        spec = importlib.util.spec_from_file_location(module_name, str(module_path))
        if not spec:
            return []

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore # importlib does not use typehints
        except (AttributeError, ModuleNotFoundError, SyntaxError,
                ImportError, NameError) as err:
            # Catch errors in case a specific module is not installed
            logger.warning(f"Could not import {module_path} due to '{err}'")
            return []

        return _load_module_strategies(module)


def load_strategy_list(config: dict) -> List[Type[BaseStrategy]]:
    result = []
    strategy_paths = config['strategy_paths']
    for path in strategy_paths:
        if path.startswith('banbot.'):
            result.extend(_load_module_strategies(path))
        else:
            result.extend(_load_file_strategies(path))
    return result


def load_run_jobs(config: dict) -> List[Tuple[str, str, List[Type[BaseStrategy]]]]:
    strategy_list = load_strategy_list(config)
    strategy_map = {item.__name__: item for item in strategy_list}
    logger.info(f'found strategy: {list(strategy_map.keys())}')
    result = dict()
    pairlist = config['pairlist']
    for policy in config['run_policy']:
        strategy_cls = strategy_map.get(policy['name'])
        if not strategy_cls:
            raise RuntimeError(f'unknown Strategy: {policy["name"]}')
        for pair, timeframe in pairlist:
            key = f'{pair}_{timeframe}'
            if key not in result:
                result[key] = []
            result[key].append(strategy_cls)
    result_list = []
    for key, slist in result.items():
        pair, timeframe = key.split('_')
        result_list.append((pair, timeframe, slist))
    return result_list
