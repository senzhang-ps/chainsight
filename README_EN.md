# ChainSight

Supply-chain planning and simulation system with both **file mode** and **database mode**.

[中文版](README_CN.md) | **English**

**Last updated: 2026-04-15**

## Current Status

- The root entry is still `run.py`, but the real execution entry is `src.core.run.main`
- The integrated flow lives in `src/core/main_integration/`
- Shared state management lives in `src/core/orchestrator/`
- The business modules are the five real subpackages:
  - `src/modules/demand_planning/` (Module 1)
  - `src/modules/mrp_planning/` (Module 3)
  - `src/modules/production_planning/` (Module 4)
  - `src/modules/deployment_planning/` (Module 5)
  - `src/modules/logistics_execution/` (Module 6)
- Shared defaults are centralized in `config/defaults.yaml` + `src/utils/defaults.py`
- Shared identifier normalization is unified in `src/utils/normalization.py`
- Shared resource configuration is unified in `src/utils/resource_config.py`

### Recently Removed Files

These compatibility/duplicate files are gone and should no longer be imported:

- `src/core/main_integration/production_integration.py`
- `src/core/main_integration/normalize.py`
- `src/utils/runtime_defaults.py`
- `src/utils/normalization_common.py`
- `src/utils/cpu_config.py`
- `src/modules/demand_planning/normalization.py`

Module 4 integration now uses `src/core/main_integration/production_runner.py`.

### Current Verification Status

- For the **same date range**, `Module1-6` and `orchestrator` outputs now align with the Dev baseline
- The previous `ori_deployment_uid` sequence mismatch has been fixed
- `summary` comparisons must use the **same run length**, otherwise row counts will naturally differ

## Requirements

- Python 3.12
- A virtual environment is strongly recommended
- PostgreSQL is required only for database mode

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
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Verify core dependencies

```bash
python -c "import pandas, numpy, scipy, duckdb, openpyxl, yaml, tqdm, psycopg, psutil; print('deps ok')"
```

## About requirements.txt

`requirements.txt` now keeps only the **direct install-time dependencies** of the project:

- Data processing: `pandas`, `numpy`, `scipy`, `duckdb`
- Excel I/O: `openpyxl`
- Database mode: `psycopg[binary]`
- Runtime/config: `PyYAML`, `tqdm`, `psutil`

Script-only optional dependencies (for example DOCX generation or alternate Excel writer support) are intentionally not part of the main install set.

## Run Modes

### File mode

In file mode, prefer `--config-dir` and pass a `workspace/<project>/<scenario>/config/` directory. The directory must contain exactly one Excel file; CSV files in the same directory are matched to sheet names case-insensitively and take priority over Excel sheets.

```powershell
python run.py --config-dir D:/PG/chainsight/workspace/SDC/baseline/config --start-date 2025-10-06 --end-date 2025-10-10
python run.py --config-dir SDC/baseline --end-date 2025-12-16 --force-restart --non-interactive
python run.py --config-dir SDC/baseline --end-date 2025-10-15 --resume
```

The short form `<project>/<scenario>` expands to `<workspace_root>/<project>/<scenario>/config`. `workspace_root` is resolved in this order: `CHAINSIGHT_WORKSPACE` environment variable > project-root `.env` with `CHAINSIGHT_WORKSPACE=...` > `config/defaults.yaml::workspace_root` > project-root `workspace/`. The old `--config` Excel path remains available during the transition, but new file-mode calls should move to `--config-dir`.

### Database mode

In database mode, `--config` can be a config name, an Excel file path, or a config directory containing exactly one Excel file. Add `--use-db`. For a directory input, ChainSight uses the only Excel file in that directory and derives `config_name` from the Excel filename:

```powershell
python run.py --config D:/PG/chainsight/config/sdc --start-date 2026-06-29 --end-date 2026-07-01 --use-db --non-interactive
python run.py --config D:/PG/chainsight/config/sdc/sdc.xlsx --start-date 2026-06-29 --end-date 2026-07-01 --use-db --non-interactive
python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-06 --use-db
python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db
```

With explicit DB parameters:

```powershell
python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db --db-host localhost --db-port 5432 --db-name test_db --db-user postgres --db-password 123456
```

Notes:

- File mode writes local output folders
- Database mode primarily writes business results to PostgreSQL and keeps local run logs
- Add `--local` in DB mode if you also want local file outputs

## CLI Summary

The CLI contract is defined by `src/core/run/run_main.py`:

| Argument | Meaning |
|---|---|
| `--config-dir` | Scenario `config/` directory in file mode, either absolute path or `<project>/<scenario>` short form |
| `--config` | Deprecated Excel path in file mode; config name, Excel path, or single-Excel directory path in DB mode |
| `--start-date` | Required for the first run, format `YYYY-MM-DD` |
| `--end-date` | Required, format `YYYY-MM-DD` |
| `--resume` | Resume automatically |
| `--resume-from` | Resume from a specific run directory |
| `--check-resume` | Check resume state only |
| `--list-runs` | List local run directories |
| `--non-interactive` | Disable interactive selection |
| `--force-restart` | Ignore resume capability and restart from scratch |
| `--use-db` | Enable database mode |
| `--db-host` `--db-port` `--db-name` `--db-user` `--db-password` | Database connection parameters |
| `--run-suffix` | Append a suffix to the run directory |
| `--local` | Keep local file outputs in DB mode |

## Current Authoritative Structure

```text
chainsight/
├── run.py
├── requirements.txt
├── config/
│   ├── defaults.yaml
│   └── *.xlsx
├── docs/
├── outputs/
├── workspace/
│   └── <project>/<scenario>/config/
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
    │   │   ├── __init__.py
    │   │   ├── simulation_file.py
    │   │   ├── simulation_db.py
    │   │   ├── production_runner.py
    │   │   ├── config_loader.py
    │   │   ├── resume.py
    │   │   ├── seed.py
    │   │   └── db_helpers.py
    │   ├── orchestrator/
    │   └── parallel_executor/
    ├── modules/
    │   ├── demand_planning/
    │   ├── mrp_planning/
    │   ├── production_planning/
    │   ├── deployment_planning/
    │   └── logistics_execution/
    ├── services/
    └── utils/
        ├── defaults.py
        ├── normalization.py
        ├── resource_config.py
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
from src.core.main_integration.production_runner import run_daily_production_planning_integrated
from src.core.main_integration.production_runner import load_current_date_production_gr
from src.core.orchestrator import create_orchestrator
```

### Shared config and utilities

```python
from src.utils.defaults import (
    RESOURCE_UTILIZATION,
    M1_FUTURE_CUTOFF_DAYS,
    M1_DEFAULT_MAX_ADVANCE_DAYS,
    M6_MAX_WAIT_DAYS,
    M6_RANDOM_SEED,
)

from src.utils.normalization import normalize_identifiers, normalize_material
from src.utils.resource_config import get_optimal_threads, get_optimal_memory
from src.utils.date_helpers import compute_planning_window, calculate_transport_lead_time
```

### Deprecated imports you should not use

```python
from src.core.main_integration.production_integration import ...  # removed; use production_runner instead
from src.utils.runtime_defaults import ...  # removed
from src.utils.normalization_common import ...  # removed
from src.utils.cpu_config import ...  # removed
```

## Outputs and Logs

### File mode output

```text
outputs/<config_stem>/run_YYYYMMDD_HHMMSS/
```

Typical subdirectories:

- `module1/`
- `module3/`
- `module4/`
- `module5/`
- `module6/`
- `orchestrator/`
- `summary/`

### Database mode output

```text
outputs/db_<config_name>_<timestamp>/
```

Typically this mainly contains:

- `simulation_log_<timestamp>.txt`

## Recommended Verification Commands

Two-day file-mode regression:

```powershell
python run.py --config config/OC_Paste_S1_20251224.xlsx --end-date 2025-12-16 --force-restart --non-interactive
```

Important: if you do regression comparison, use your own comparison script/tool and make sure both `summary` directories cover the same date range.

## Documentation Entry Points

- [docs/INDEX.md](docs/INDEX.md)
- [docs/handover/refactored/00-overview/handover.md](docs/handover/refactored/00-overview/handover.md)
- [docs/handover/refactored/00-overview/setup.md](docs/handover/refactored/00-overview/setup.md)
- [docs/handover/refactored/02-api/api.md](docs/handover/refactored/02-api/api.md)
- [docs/_archive/](docs/_archive/)

---

For ongoing regression work, deployment, or database initialization, prefer this README and `src/core/run/run_main.py` over older historical docs.
