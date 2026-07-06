#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a standalone single-scenario analysis report for each HC AO variant
(baseline-hc-s1 and baseline-hc-s2), using the vetted baseline-hc engine.

The engine ``build_baseline_hc_analysis`` (imported as ``bh``) is hardcoded for
the baseline-hc scenario via module-level globals. Every scenario-dependent
global (run_id, config_name, scenario, scenario/config/analysis paths) is
resolved at call time inside ``bh.main`` and the compute/map helpers, so this
driver simply re-points those globals at each variant and calls ``bh.main()``.

Each variant produces its own self-contained report under
    workspace/<project>/scenarios/<variant>/analysis/<run-tag>/
(extracts/, workbook, analysis.html, run.md) and updates that scenario's
analysis/LATEST.md — identical in structure to the baseline-hc report.

The variants' config folders are identical to baseline-hc except M1_AOConfig, so
the category snapshot fallback (shared file) and the per-scenario SUF / manual
workbooks resolve exactly as they do for baseline-hc.

Read-only against PostgreSQL; safe to re-run (each run writes a new timestamped
folder — delete a prior folder first if you want a single artifact).
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import build_baseline_hc_analysis as bh

# (scenario == config_name, run_id in the results DB)
VARIANTS = [
    ("baseline-hc-s1", "db_baseline-hc-s1_20260706_103801"),
    ("baseline-hc-s2", "db_baseline-hc-s2_20260706_110742"),
]


def point_engine_at(scenario: str, run_id_db: str) -> None:
    """Re-point every scenario-dependent global on the bh module at ``scenario``."""
    scen_dir = bh.PROJECT_DIR / "scenarios" / scenario
    cfg = scen_dir / "config"

    bh.SCENARIO = scenario
    bh.RUN_ID = run_id_db
    bh.CONFIG_NAME = scenario                       # cfg_* tables filter on config_name
    bh.SCEN_DIR = scen_dir
    bh.ANALYSIS_ROOT = scen_dir / "analysis"        # report output root
    bh._CONFIG_DIR = cfg
    bh.SUF_XLSX = cfg / "SUF for XQ HC ChainSight.xlsx"
    bh.MISSING_SUF_XLSX = cfg / "missing SUF manual input.xlsx"
    bh.MISSING_CAT_XLSX = cfg / "missing category manual input.xlsx"


def main() -> None:
    for scenario, run_id_db in VARIANTS:
        print("\n" + "#" * 78)
        print(f"# single-scenario report  |  {scenario}  ({run_id_db})")
        print("#" * 78)
        point_engine_at(scenario, run_id_db)
        bh.main()


if __name__ == "__main__":
    main()
