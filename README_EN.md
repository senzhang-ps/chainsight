# ChainSight

Supply chain planning and simulation system with both file mode and database mode.

[中文版](README_CN.md) | **English**

Last updated: 2026-04-08

## Current Status

This repository has completed phase-3 cleanup. The codebase should now be understood like this:

- Root entry is still `run.py`
- The authoritative runtime dispatch entry is the `src.core.run` package
  - `run.py` imports `from src.core.run import main`
  - that resolves to `src/core/run/__init__.py`
  - the main implementation lives in `src/core/run/run_main.py`
- Main integrated flow lives under `src/core/main_integration/*`
- Shared state lives under `src/core/orchestrator/*`
- Business logic is now package-first and only uses the five real subpackages
  - `src/modules/demand_planning/`
  - `src/modules/mrp_planning/`
  - `src/modules/production_planning/`
  - `src/modules/deployment_planning/`
  - `src/modules/logistics_execution/`
- Cross-module helper consolidation now lives under `src/utils/*`
  - `src/utils/runtime_defaults.py` is the single source for shared tiny defaults reused by M3 and M5
  - `src/utils/normalization_common.py` holds shared identifier normalization helpers while package wrappers preserve historical semantics
  - `src/utils/date_helpers.py` holds shared planning-window, review-day, and lead-time helpers

The following legacy wrappers have already been physically removed:

- `src/modules/module1.py`
- `src/modules/module3.py`
- `src/modules/module4.py`
- `src/modules/module5.py`
- `src/modules/module6.py`
- `src/core/main_integration.py`
- `src/core/orchestrator.py`
- `src/core/parallel_executor.py`
- `src/core/main_integration/module4_runner.py`

## Quick Start

### 1. Create a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows CMD:

```cmd
py -3.12 -m venv .venv
.\.venv\Scripts\activate.bat
```

Linux / macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Verify critical dependencies

```powershell
python -c "import pandas, numpy, duckdb, openpyxl, psycopg; print('deps ok')"
```

## Run Modes

### File mode

In file mode, `--config` takes an Excel path.

```powershell
python run.py --config config/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10
python run.py --config config/BC_S5.xlsx --end-date 2025-10-15 --resume
python run.py --config config/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10 --force-restart
```

### Database mode

In database mode, `--config` takes a config name, not an Excel path.

```powershell
python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-06 --use-db
python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db
```

With explicit DB parameters:

```powershell
python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db --db-host localhost --db-port 5432 --db-name test_db --db-user postgres --db-password 123456
```

Notes:

- `OC_Paste_S1_20251224` should currently be passed without quotes because the config name has no spaces.
- Database mode writes business outputs into PostgreSQL and keeps local log folders.
- Add `--local` if you want local file outputs in addition to DB writes.

## CLI Argument Summary

The current CLI contract is defined by `src/core/run/run_main.py`:

| Argument | Meaning |
|---|---|
| `--config` | Excel path in file mode, config name in DB mode |
| `--start-date` | Required for first run, format `YYYY-MM-DD` |
| `--end-date` | Required, format `YYYY-MM-DD` |
| `--resume` | Automatically resume from interruption point |
| `--resume-from` | Resume from a specific existing run directory |
| `--check-resume` | Check resume status only |
| `--list-runs` | List local run directories |
| `--non-interactive` | Disable interactive run selection |
| `--force-restart` | Ignore resume capability and start fresh |
| `--use-db` | Enable database mode |
| `--db-host` `--db-port` `--db-name` `--db-user` `--db-password` | Database connection parameters |
| `--run-suffix` | Append suffix to local run directory name |
| `--local` | In DB mode, also keep local Excel outputs |

## Current Authoritative Structure

```text
chainsight/
├── run.py
├── requirements.txt
├── config/
├── docs/
├── outputs/
├── pgsql_db/
└── src/
    ├── core/
    │   ├── run/
    │   │   ├── __init__.py
    │   │   ├── run_main.py
    │   │   ├── db_runner.py
    │   │   ├── db_config.py
    │   │   ├── output_dir.py
    │   │   └── local_writer.py
    │   ├── main_integration/
    │   │   ├── simulation_file.py
    │   │   ├── simulation_db.py
    │   │   ├── production_integration.py
    │   │   ├── config_loader.py
    │   │   ├── resume.py
    │   │   └── db_helpers.py
    │   ├── orchestrator/
    │   │   ├── orchestrator_main.py
    │   │   ├── daily_ops.py
    │   │   ├── processors.py
    │   │   └── persistence.py
    │   └── parallel_executor/
    ├── modules/
    │   ├── demand_planning/
    │   ├── mrp_planning/
    │   ├── production_planning/
    │   ├── deployment_planning/
    │   └── logistics_execution/
    ├── services/
    └── utils/
        ├── runtime_defaults.py
        ├── normalization_common.py
        └── date_helpers.py
```

## Recommended Imports

### Business modules

```python
from src.modules import demand_planning as module1
from src.modules import mrp_planning as module3
from src.modules import production_planning as module4
from src.modules import deployment_planning as module5
from src.modules import logistics_execution as module6
```

### Main flow and orchestrator

```python
from src.core.main_integration import run_integrated_simulation
from src.core.main_integration import run_integrated_simulation_from_dict
from src.core.main_integration.production_integration import run_module4_integrated
from src.core.orchestrator import create_orchestrator
```

### Shared utilities for new code

```python
from src.utils.runtime_defaults import DEFAULT_MOQ, DEFAULT_RV
from src.utils.normalization_common import normalize_identifiers_vectorized
from src.utils.date_helpers import compute_planning_window, calculate_transport_lead_time
```

Do not use these removed imports anymore:

```python
from src.modules import module1, module3, module4, module5, module6
from src.core.main_integration.module4_runner import run_module4_integrated
```

## Outputs and Logs

### File mode

Typical output path:

```text
outputs/<config_stem>/run_YYYYMMDD_HHMMSS/
```

Common subdirectories:

- `module1/`
- `module3/`
- `module4/`
- `module5/`
- `module6/`
- `orchestrator/`
- `summary/`

### Database mode

Database mode writes business outputs to PostgreSQL and mainly keeps local log folders:

```text
outputs/db_<config_name>_<timestamp>/
```

Typical file:

- `simulation_log_<timestamp>.txt`

Default DB config file:

- `config/database.json`

## Verified Regression Baseline

### Standard two-day DB regression

```powershell
python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db
```

Verified successful phase-3 run:

- `db_OC_Paste_S1_20251224_20260408_112709`

Key result checkpoints:

- `module1_output_orderlog = 9214`
- `module1_output_shipmentlog = 3094`
- `module4_output_productionplan = 10`
- `module5_output_deploymentplan = 60182`
- `module6_output_deliveryplan = 39`
- `summary_output_ordershipmentcutsummary = 3433`
- `summary_output_fullcapacityexceed = 49`
- `summary_output_fulltruckusage = 1`
- checkpoint `orch_state_json = 186773 bytes`
- checkpoint no longer contains `m1_previous_orders`
- checkpoint `delivery_gr / shipment_log / daily_logs = 0`

### Fixes confirmed not to regress

- PostgreSQL checkpoint `jsonb` size issue fixed
- `m1_previous_orders` removed from checkpoint persistence
- `delivery_gr` dedupe no longer depends on full historical accumulation
- M4 integration adapter is now consolidated under `production_integration.py`

### Zero-drift shared-utility consolidation

The latest consolidation kept the old call surfaces but moved repeated logic to shared utility files:

- shared tiny defaults now come from `src/utils/runtime_defaults.py`
- shared normalization now comes from `src/utils/normalization_common.py`
- shared date / lead-time helpers now come from `src/utils/date_helpers.py`

Verified runs after this consolidation:

- `db_BC_S5_20260408_123150`
- `db_OC_Paste_S1_20251224_20260408_125002`

The OC two-day DB regression remained unchanged after the consolidation:

- `module1_output_orderlog = 9214`
- `module1_output_shipmentlog = 3094`
- `module4_output_productionplan = 10`
- `module5_output_deploymentplan = 60182`
- `module6_output_deliveryplan = 39`
- checkpoint `orch_state_json = 186773 bytes`
- checkpoint still has no `m1_previous_orders`
- checkpoint `delivery_gr / shipment_log / daily_logs = 0`

## Database and Config Notes

### Common configs in this repo

- `config/BC_S5.xlsx`
- `config/BC_S9.xlsx`
- `config/OC_Paste_S1_20251224.xlsx`

### DB-mode config discovery

Database mode resolves by config name. If the config is missing in PostgreSQL, the project can import it from Excel according to the repo search rules. This repo now includes:

- `config/OC_Paste_S1_20251224.xlsx`

That helps guarantee stable DB-mode lookup for `OC_Paste_S1_20251224`.

## Documentation Entry Points

If you are taking over the project, read in this order:

1. [docs/用户使用入口说明.md](docs/用户使用入口说明.md)
2. [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)
3. [docs/交接文档/Refactored/项目交接文档.md](docs/交接文档/Refactored/项目交接文档.md)
4. [docs/OPTIMIZED_CODE_STRUCTURE_DIAGRAM.md](docs/OPTIMIZED_CODE_STRUCTURE_DIAGRAM.md)
5. [docs/交接文档/Refactored/setup.md](docs/交接文档/Refactored/setup.md)
6. [docs/交接文档/Refactored/API.md](docs/交接文档/Refactored/API.md)

Supporting docs:

- [docs/INDEX.md](docs/INDEX.md)
- [docs/README_REFACTORING_MAP.md](docs/README_REFACTORING_MAP.md)
- [docs/PERFORMANCE_OPTIMIZATION_REPORT.md](docs/PERFORMANCE_OPTIMIZATION_REPORT.md)
- [docs/DUCKDB_OPTIMIZATION_GUIDE.md](docs/DUCKDB_OPTIMIZATION_GUIDE.md)
- [docs/CYTHON_OPTIMIZATION_REPORT.md](docs/CYTHON_OPTIMIZATION_REPORT.md)

## Common Questions

### Why do I see many deletions in Git after phase 3

Because the legacy wrappers were physically removed. This is expected and correct.

### Why can DB mode fail with temp-directory permission errors

In restricted environments, temporary directories such as `AppData\\Local\\Temp` may be blocked. That usually indicates an environment write restriction, not a business-logic bug.

### Why do some older docs still mention `module1.py`

Some historical/background docs have not been fully rewritten yet. Prefer this README, `docs/INDEX.md`, `docs/用户使用入口说明.md`, and the current handover docs.

### Why do both `src/core/run.py` and `src/core/run/` exist

The active runtime entry is the `src.core.run` package, meaning `src/core/run/__init__.py` and `src/core/run/run_main.py`. For reading and future maintenance, treat the package directory as authoritative.
