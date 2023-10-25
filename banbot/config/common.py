#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : common.py
# Author: anyongjin
# Date  : 2023/10/25

import os.path
import sys
from pathlib import Path
import orjson
import yaml
from typing import *


from banbot.config.consts import *
from banbot.util.common import logger
from banbot.util.misc import deep_merge_dicts


def load_config_file(path: str) -> Dict[str, Any]:
    """
    Loads a config file from the given path
    :param path: path as str
    :return: configuration as dictionary
    """
    try:
        is_yml = os.path.splitext(path)[1].lower() in {'.yml', '.yaml'}
        with open(path, 'rb') if path != '-' else sys.stdin as file:
            fdata = file.read()
            if isinstance(fdata, (bytes, bytearray)):
                fdata = fdata.decode('utf-8')
            if is_yml:
                config = yaml.safe_load(fdata)
            else:
                config = orjson.loads(fdata)
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
