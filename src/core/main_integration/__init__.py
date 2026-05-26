"""
main_integration package.

Exports the public integration APIs lazily so package initialization does not
pull in simulation_file/simulation_db and create import cycles with orchestrator.
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

__all__ = [
    "run_integrated_simulation",
    "run_integrated_simulation_from_dict",
    "load_configuration",
    "load_configuration_from_dict",
    "check_resume_capability",
    "main",
]


def __getattr__(name: str) -> Any:
    if name == "run_integrated_simulation":
        from .simulation_file import run_integrated_simulation

        return run_integrated_simulation
    if name == "run_integrated_simulation_from_dict":
        from .simulation_db import run_integrated_simulation_from_dict

        return run_integrated_simulation_from_dict
    if name in {"load_configuration", "load_configuration_from_dict"}:
        from .config_loader import load_configuration, load_configuration_from_dict

        return {
            "load_configuration": load_configuration,
            "load_configuration_from_dict": load_configuration_from_dict,
        }[name]
    if name == "check_resume_capability":
        from .resume import check_resume_capability

        return check_resume_capability
    if name == "main":
        from .cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
