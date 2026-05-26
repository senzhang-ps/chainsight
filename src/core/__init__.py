"""
Core execution layer - main orchestration and coordination.

The public API is resolved lazily to avoid package-initialization cycles between
core, main_integration, and orchestrator.
"""

from typing import Any

__all__ = [
    "create_orchestrator",
    "run_integrated_simulation",
    "load_configuration",
    "check_resume_capability",
]


def __getattr__(name: str) -> Any:
    if name == "create_orchestrator":
        from .orchestrator import create_orchestrator

        return create_orchestrator
    if name in {
        "run_integrated_simulation",
        "load_configuration",
        "check_resume_capability",
    }:
        from .main_integration import (
            check_resume_capability,
            load_configuration,
            run_integrated_simulation,
        )

        return {
            "run_integrated_simulation": run_integrated_simulation,
            "load_configuration": load_configuration,
            "check_resume_capability": check_resume_capability,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
