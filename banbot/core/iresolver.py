#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : iresolver.py
# Author: anyongjin
# Date  : 2023/4/17
import importlib
import importlib.util
import inspect
import os.path
import sys
from pathlib import Path
from typing import *

from banbot.config.consts import *
from banbot.util.common import logger


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


class IResolver:

    object_type: Type[Any]
    object_type_str: str
    user_subdir: Optional[str] = None
    initial_search_path: Optional[Path]
    extra_path: Optional[str] = None

    @classmethod
    def build_search_paths(cls, config: Config, user_subdir: Optional[str] = None,
                           extra_dirs: List[str] = []):
        abs_paths: List[Path] = []
        if cls.initial_search_path:
            abs_paths.append(cls.initial_search_path)

        if user_subdir:
            abs_paths.insert(0, Path(os.path.join(config['data_dir'], user_subdir)))

        # Add extra directory to the top of the search paths
        for dir in extra_dirs:
            abs_paths.insert(0, Path(dir).resolve())

        if cls.extra_path and (extra := config.get(cls.extra_path)):
            abs_paths.insert(0, Path(extra).resolve())

        return abs_paths

    @classmethod
    def _load_module_objects(cls, module):
        clsmembers = inspect.getmembers(module, inspect.isclass)
        result = []
        for (name, cld_cls) in clsmembers:
            if not issubclass(cld_cls, cls.object_type) or name == cls.object_type_str:
                continue
            result.append(cld_cls)
        return result

    @classmethod
    def _search_objects(cls, dir_or_file: Path):
        result = []
        if dir_or_file.is_file():
            return cls._load_file_objects(dir_or_file)
        for entry in dir_or_file.iterdir():
            if entry.is_symlink():
                continue
            elif entry.is_dir():
                result.extend(cls._search_objects(entry))
                continue
            elif entry.suffix != '.py' or not entry.is_file():
                continue
            result.extend(cls._load_file_objects(entry.resolve()))
        return result

    @classmethod
    def _load_file_objects(cls, module_path: Path):
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
                logger.warning("Could not import %s due to '%s'", module_path, err)
                return []

            return cls._load_module_objects(module)

    @classmethod
    def load_object_list(cls, config: dict, extra_dirs: Optional[List[str]] = None) -> List:
        result = []
        extra_dirs: List[str] = extra_dirs or []
        search_paths = cls.build_search_paths(config, user_subdir=cls.user_subdir, extra_dirs=extra_dirs)
        for path in search_paths:
            if os.path.exists(path):
                result.extend(cls._search_objects(path))
            else:
                logger.warning(f'path not exist to load objects: {path.absolute()}')
        return result
