"""
memory_store.py

DuckDB 内存模式懒加载模块。

管理内存数据存储的全局变量与懒加载，避免循环导入和启动延迟。
"""

import logging

logger = logging.getLogger(__name__)

# 懒加载 DuckDB 内存存储函数的全局变量
_memory_store_imported = False
_enable_memory_mode = None
_disable_memory_mode = None
_is_memory_mode_enabled = None
_get_data_store = None


def _ensure_memory_store_imported():
    """懒加载内存存储模块（避免循环导入和启动延迟）"""
    global _memory_store_imported, _enable_memory_mode, _disable_memory_mode
    global _is_memory_mode_enabled, _get_data_store
    if not _memory_store_imported:
        try:
            from ...utils.memory_data_store import (
                enable_memory_mode,
                disable_memory_mode,
                is_memory_mode_enabled,
                get_data_store,
            )
            _enable_memory_mode = enable_memory_mode
            _disable_memory_mode = disable_memory_mode
            _is_memory_mode_enabled = is_memory_mode_enabled
            _get_data_store = get_data_store
        except ImportError as e:
            logger.warning(f"DuckDB内存模块导入失败: {e}")
            _enable_memory_mode = lambda **kwargs: None
            _disable_memory_mode = lambda: None
            _is_memory_mode_enabled = lambda: False
            _get_data_store = lambda: None
        _memory_store_imported = True
