#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : configuration.py
# Author: anyongjin
# Date  : 2023/4/1
import copy
import os.path
import sys
from pathlib import Path
from typing import *

import orjson
import yaml

from banbot.config.consts import *
from banbot.util.common import logger, Singleton
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


class AppConfig(metaclass=Singleton):
    '''
    应用最终的启动配置。单例类。
    在应用启动时即初始化完成。
    后续可通过`AppConfig.get()`获取使用
    '''
    obj: Optional['AppConfig'] = None

    def __init__(self, args: Dict[str, Any], runmode: Optional[RunMode] = None):
        self.args = args
        self.config: Optional[Config] = None
        self.runmode = runmode
        AppConfig.obj = self

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
        from banbot.util import get_run_env
        run_env = get_run_env()
        if run_env != 'prod':
            logger.info(f'Running in {run_env}, Please set `ban_run_env=prod` in production running')
        path_list = [] if self.args.get('no_default') else self._get_def_config_paths()
        in_paths = self.args.get("config")
        if in_paths:
            path_list.extend(in_paths)
        config = load_from_files(path_list)
        if not config.get('data_dir'):
            import os
            data_dir = os.environ.get('ban_data_dir')
            if data_dir:
                config['data_dir'] = data_dir
            else:
                raise ValueError('`data_dir` is required in config.json')
        return config

    @property
    def exchange_cfg(self):
        return AppConfig.get_exchange(self.config)

    @classmethod
    def get_exchange(cls, config: Config, exg_name: str = None):
        exchange_all = config['exchange']
        if not exg_name:
            exg_name = exchange_all['name']
        result = exchange_all[exg_name]
        result['name'] = exg_name
        result.update(exchange_all.get('common', dict()))
        return result

    @classmethod
    def get(cls) -> Config:
        if not cls.obj:
            cls.init_by_args()
        return cls.obj.get_config()

    @classmethod
    def get_data_dir(cls):
        import os
        data_dir = os.environ.get('ban_data_dir')
        if data_dir:
            return data_dir
        config = cls.get()
        return config.get('data_dir')

    @classmethod
    async def test_db(cls):
        # 测试数据库连接
        from banbot.storage.base import init_db, dba, sa
        init_db()
        async with dba():
            sess = dba.session
            db_tz = (await sess.execute(sa.text('show timezone;'))).scalar()
            if str(db_tz).lower() != 'utc':
                raise RuntimeError('database timezone must be UTC, please change it in `postgresql.conf`'
                                   'and exec `select pg_reload_conf();` to apply; then re-download all data')
            logger.info(f'Connect DataBase Success')

    @classmethod
    def init_by_args(cls, args: dict = None) -> Config:
        from banbot.util.misc import deep_merge_dicts
        if not args:
            args = dict()
        config = AppConfig(args).get_config()
        deep_merge_dicts(args, config, False)
        if 'func' in config:
            del config['func']
        if 'timerange' in config:
            from banbot.config.timerange import TimeRange
            config['timerange'] = TimeRange.parse_timerange(config['timerange'])
        # 检查是否启用了异常通知，如启用则设置
        from banbot.worker.exc_notify import allow_exc_notify
        from banbot.util.common import set_log_notify
        if allow_exc_notify(config):
            set_log_notify(logger)
        # 更新BotGlobal
        from banbot.storage import BotGlobal
        from banbot.util import btime
        BotGlobal.exg_name = config['exchange']['name']
        BotGlobal.market_type = config['market_type']
        BotGlobal.bot_name = config['name']
        BotGlobal.start_at = btime.time_ms()
        return config

    @classmethod
    def _get_def_config_paths(cls) -> List[str]:
        import os
        data_dir = os.environ.get('ban_data_dir')
        if not data_dir:
            raise ValueError('`ban_data_dir` not configured, config load fail')
        if not os.path.isdir(data_dir):
            raise ValueError(f'`ban_data_dir`:{data_dir} not exits, config load fail')
        result = []
        try_names = ['config.yml', 'config.local.yml']
        for name in try_names:
            full_path = os.path.join(data_dir, name)
            if os.path.isfile(full_path):
                result.append(full_path)
        if not result:
            raise ValueError(f'no config.json found in {data_dir}!')
        return result

    @classmethod
    def get_pub(cls):
        '''返回配置的公开版本，删除敏感信息。'''
        config = copy.deepcopy(cls.get())
        exg_cfg = config.get('exchange')
        if exg_cfg:
            for key, item in exg_cfg.items():
                if not isinstance(item, dict):
                    continue
                credit_keys = [k for k in item if k.startswith('credit_')]
                [item.pop(k) for k in credit_keys]
        if 'database' in config:
            del config['database']
        if 'timerange' in config:
            from banbot.config.timerange import TimeRange
            timerange = config['timerange']
            if isinstance(timerange, TimeRange):
                config['timerange'] = timerange.timerange_str
        chl_cfg = config.get('rpc_channels')
        if chl_cfg:
            keep_keys = {'enabled', 'msg_types', 'type'}
            for key, item in chl_cfg.items():
                if not isinstance(item, dict):
                    continue
                del_keys = [k for k in item if k not in keep_keys]
                [item.pop(k) for k in del_keys]
        return config
