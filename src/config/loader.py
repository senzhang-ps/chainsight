# -*- coding: utf-8 -*-
"""
配置加载器

从 default_config.yaml 加载默认配置，支持通过环境变量 CHAINSIGHT_CONFIG
指定自定义配置文件进行覆盖。
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


_config_cache: Optional[Dict[str, Any]] = None


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典，override 优先。"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: str) -> dict:
    """加载 YAML 文件。"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_config(force_reload: bool = False) -> Dict[str, Any]:
    """
    加载并缓存配置。

    优先加载 default_config.yaml，然后用 CHAINSIGHT_CONFIG 环境变量
    指定的文件覆盖。

    参数：
        force_reload: 是否强制重新加载（忽略缓存）

    返回：
        合并后的配置字典
    """
    global _config_cache

    if _config_cache is not None and not force_reload:
        return _config_cache

    # 加载默认配置
    default_path = Path(__file__).parent / 'default_config.yaml'
    config = _load_yaml(str(default_path))

    # 加载用户自定义配置（如果存在）
    user_config_path = os.environ.get('CHAINSIGHT_CONFIG')
    if user_config_path and os.path.exists(user_config_path):
        user_config = _load_yaml(user_config_path)
        config = _deep_merge(config, user_config)

    _config_cache = config
    return config


def get_config() -> Dict[str, Any]:
    """获取配置（使用缓存）。"""
    return load_config()


def get_module_config(module_name: str) -> Dict[str, Any]:
    """
    获取指定模块的配置。

    参数：
        module_name: 模块名称（如 'demand_planning', 'shared' 等）

    返回：
        模块配置字典
    """
    config = get_config()
    return config.get(module_name, {})


def get_shared_config() -> Dict[str, Any]:
    """获取跨模块共享配置。"""
    return get_module_config('shared')


def reset_config_cache() -> None:
    """重置配置缓存，下次调用 get_config() 时重新加载。"""
    global _config_cache
    _config_cache = None
