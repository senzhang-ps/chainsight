"""Business module namespace with lazy subpackage loading."""

from importlib import import_module

_SUBMODULES = {
    'demand_planning': '.demand_planning',
    'mrp_planning': '.mrp_planning',
    'production_planning': '.production_planning',
    'deployment_planning': '.deployment_planning',
    'logistics_execution': '.logistics_execution',
}

__all__ = list(_SUBMODULES)


def __getattr__(name):
    if name not in _SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_SUBMODULES[name], __name__)
    globals()[name] = module
    return module
