"""
Utility functions and helpers
"""

from .logger_config import setup_logging
from .validation_manager import ValidationManager
from .inventory_balance_checker import InventoryBalanceChecker
from .time_manager import SimulationTimeManager, initialize_time_manager
from .simulation_cache import (
    SimulationCache,
    initialize_simulation_cache,
    get_simulation_cache,
    clear_simulation_cache,
)


def __getattr__(name):
    """延迟导入 config_validator 以避免循环引用。

    config_validator → core.main_integration → core.orchestrator → utils.normalization
    会在包初始化阶段形成循环，因此改用 lazy import。
    """
    if name == 'run_pre_simulation_validation':
        from .config_validator import run_pre_simulation_validation
        return run_pre_simulation_validation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'run_pre_simulation_validation',
    'setup_logging',
    'ValidationManager',
    'InventoryBalanceChecker',
    'SimulationTimeManager',
    'initialize_time_manager',
    'SimulationCache',
    'initialize_simulation_cache',
    'get_simulation_cache',
    'clear_simulation_cache',
]
