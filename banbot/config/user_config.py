#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : UserConfig.py
# Author: anyongjin
# Date  : 2023/10/25
from banbot.config.common import *
from banbot.util.common import Singleton


class UserConfig(metaclass=Singleton):
    """
    用户在机器人面板保存的设置和修改
    """
    _obj: Optional['UserConfig'] = None

    def __init__(self):
        from banbot.config.appconfig import AppConfig
        config = AppConfig.get()
        data_dir = AppConfig.get_data_dir()
        user_dir = Path(data_dir) / config['name']
        user_dir.mkdir(parents=True, exist_ok=True)
        user_cfg = user_dir / 'config.yml'
        self._config: Config = dict()
        if user_cfg.exists():
            self._config = load_config_file(str(user_cfg))
        UserConfig._obj = self

    @classmethod
    def get(cls) -> Config:
        if not cls._obj:
            UserConfig()
        return cls._obj._config

