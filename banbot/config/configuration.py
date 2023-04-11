#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : configuration.py
# Author: anyongjin
# Date  : 2023/4/1
import logging
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing import *
from banbot.config.consts import *
import orjson
from banbot.util.common import logger
from banbot.util.misc import deep_merge_dicts


def load_config_file(path: str) -> Dict[str, Any]:
    """
    Loads a config file from the given path
    :param path: path as str
    :return: configuration as dictionary
    """
    try:
        # Read config from stdin if requested in the options
        with Path(path).open() if path != '-' else sys.stdin as file:
            config = orjson.loads(file.read())
    except FileNotFoundError:
        raise RuntimeError(
            f'Config file "{path}" not found!'
            ' Please create a config file or check whether it exists.')
    except orjson.JSONDecodeError as e:
        raise RuntimeError(
            f'{e}\n'
            f'Please verify the following segment of your configuration:\n{path}'
        )

    return config


def load_from_files(
        files: List[str], base_path: Optional[Path] = None, level: int = 0) -> Dict[str, Any]:
    """
    Recursively load configuration files if specified.
    Sub-files are assumed to be relative to the initial config.
    """
    config: Config = {}
    if level > 5:
        raise RuntimeError("Config loop detected.")

    if not files:
        return dict()
    files_loaded = []
    # We expect here a list of config filenames
    for filename in files:
        logger.info('Using config: %s ...', filename)
        if filename == '-':
            # Immediately load stdin and return
            return load_config_file(filename)
        file = Path(filename)
        if base_path:
            # Prepend basepath to allow for relative assignments
            file = base_path / file

        config_tmp = load_config_file(str(file))
        if 'add_config_files' in config_tmp:
            config_sub = load_from_files(
                config_tmp['add_config_files'], file.resolve().parent, level + 1)
            files_loaded.extend(config_sub.get('config_files', []))
            config_tmp = deep_merge_dicts(config_tmp, config_sub)

        files_loaded.insert(0, str(file))

        # Merge config options, overwriting prior values
        config = deep_merge_dicts(config_tmp, config)

    config['config_files'] = files_loaded

    return config


class Configuration:

    def __init__(self, args: Dict[str, Any], runmode: Optional[RunMode] = None):
        self.args = args
        self.config: Optional[Config] = None
        self.runmode = runmode

    def get_config(self) -> Config:
        """
        Return the config. Use this method to get the bot config
        :return: Dict: Bot config
        """
        if self.config is None:
            self.config = self.load_config()

        return self.config

    def load_config(self) -> Dict[str, Any]:
        """
        Extract information for sys.argv and load the bot configuration
        :return: Configuration dictionary
        """
        # Load all configs
        config: Config = load_from_files(self.args.get("config", []))

        return config
