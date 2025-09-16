#!/usr/bin/env python3
"""
Production Integration Runner for Supply Chain Planning System

Purpose
- Read a user-provided configuration file path (.xlsx)
- Create an output directory in the SAME directory as the configuration file
  whose top-level folder name matches the configuration file's stem
- Dispatch a full integrated run and write results to that output directory

Notes
- No test scaffolding, validation stubs, or print statements
- Quiet by default; relies on exit codes and exceptions for failure signaling

CLI
  production_integrator.py \
    --config /path/to/your_config.xlsx \
    --start-date 2024-01-01 \
    --end-date 2024-01-31

The first invocation requires ``--start-date``. Subsequent runs reuse the
start date persisted in ``<config_dir>/<config_stem>/simulation_start.txt``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# External system imports (assumed available in project environment)
from main_integration import run_integrated_simulation, load_configuration  # type: ignore


def _ensure_output_dir(config_path: Path) -> Path:
    """Create the output directory rooted by the config filename stem.

    Structure:
      <config_dir>/<config_stem>/
        └─ run_YYYYMMDD_HHMMSS/  (actual write target to avoid overwrites)

    Returns the leaf path to be used as `output_base_dir`.
    """
    cfg_dir = config_path.parent
    cfg_stem = config_path.stem

    root_dir = cfg_dir / cfg_stem
    # Always ensure the top-level directory exists so its name matches the config
    root_dir.mkdir(parents=True, exist_ok=True)

    # Create a unique run folder under the top-level directory to avoid collisions
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir

def get_or_init_simulation_start(output_root: Path, provided_start: Optional[str]) -> str:
    """Return the persistent simulation start date for this configuration.

    ``output_root`` is the directory that contains all run folders. The start
    date is stored in a ``simulation_start.txt`` file within this directory. If
    the file exists, its contents are returned. Otherwise ``provided_start`` is
    written to the file and returned. ``provided_start`` must be supplied on the
    first run when the file does not yet exist.
    """
    start_file = output_root / "simulation_start.txt"
    if start_file.exists():
        return start_file.read_text(encoding="utf-8").strip()
    if not provided_start:
        raise ValueError("Simulation start date required for first run")
    start_file.write_text(provided_start, encoding="utf-8")
    return provided_start



def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="production_integrator",
        add_help=True,
        description=(
            "Run the integrated planning flow using a given configuration file. "
            "Outputs are written under a folder named after the configuration file."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the configuration .xlsx file",
    )
    parser.add_argument(
        "--start-date",
        required=False,
        help="Simulation start date in YYYY-MM-DD (required for first run)",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Simulation end date in YYYY-MM-DD",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv or sys.argv[1:])

    cfg_path = Path(ns.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    if cfg_path.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
        raise ValueError("Configuration file must be an Excel file (.xlsx/.xlsm/.xls)")

    # Ensure we can load configuration early to fail fast on schema issues
    # (This returns an object usable by your run function or validates the file.)
    _ = load_configuration(str(cfg_path))  # noqa: F841

    output_base_dir = _ensure_output_dir(cfg_path)
    root_dir = output_base_dir.parent
    start_arg = ns["start_date"] if isinstance(ns, dict) else ns.start_date
    simulation_start = get_or_init_simulation_start(root_dir, start_arg)
    # Delegate to the integrated simulation. It is assumed to create its own
    # sub-structure under `output_base_dir` (e.g., orchestrator/module folders).
    _ = run_integrated_simulation(
        config_path=str(cfg_path),
        start_date=simulation_start,
        end_date=str(ns["end_date"]) if isinstance(ns, dict) else ns.end_date,
        output_base_dir=str(output_base_dir),
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        raise e
    except Exception as exc:
        # No prints; signal failure via non-zero exit and exception propagation
        # (Callers can capture stderr/traceback if needed.)
        raise SystemExit(1) from exc

# run example
# First run requires --start-date; subsequent runs may omit it
# python production_integrator.py \
#   --config /path/to/YourConfig.xlsx \
#   --start-date 2024-01-01 \
#   --end-date 2024-01-31