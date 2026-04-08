"""Utility namespace with lazy attribute loading.

Keeping the package lightweight avoids importing the full simulation stack when
callers only need small leaf modules such as ``cpu_config``.
"""

from importlib import import_module

_EXPORTS = {
    'run_pre_simulation_validation': ('.config_validator', 'run_pre_simulation_validation'),
    'setup_logging': ('.logger_config', 'setup_logging'),
    'ValidationManager': ('.validation_manager', 'ValidationManager'),
    'InventoryBalanceChecker': ('.inventory_balance_checker', 'InventoryBalanceChecker'),
    'SimulationTimeManager': ('.time_manager', 'SimulationTimeManager'),
    'initialize_time_manager': ('.time_manager', 'initialize_time_manager'),
    'SimulationCache': ('.simulation_cache', 'SimulationCache'),
    'initialize_simulation_cache': ('.simulation_cache', 'initialize_simulation_cache'),
    'get_simulation_cache': ('.simulation_cache', 'get_simulation_cache'),
    'clear_simulation_cache': ('.simulation_cache', 'clear_simulation_cache'),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
