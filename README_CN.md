# ChainSight

供应链规划与仿真系统，支持**文件模式**和**数据库模式**两种运行方式。

**中文** | [English](README_EN.md)

**更新时间：2026-04-15**

## 当前状态

- 根入口仍是 `run.py`，实际执行入口是 `src.core.run.main`
- 主集成流程位于 `src/core/main_integration/`
- 共享状态管理位于 `src/core/orchestrator/`
- 业务模块保留 5 个真实子包：
  - `src/modules/demand_planning/`（Module 1）
  - `src/modules/mrp_planning/`（Module 3）
  - `src/modules/production_planning/`（Module 4）
  - `src/modules/deployment_planning/`（Module 5）
  - `src/modules/logistics_execution/`（Module 6）
- 共享默认参数已集中到 `config/defaults.yaml` + `src/utils/defaults.py`
- 共享标识符归一化已统一到 `src/utils/normalization.py`
- 资源配置统一到 `src/utils/resource_config.py`

### 最近完成的清理

以下兼容/重复文件已删除，不应再引用：

- `src/core/main_integration/production_integration.py`
- `src/core/main_integration/normalize.py`
- `src/utils/runtime_defaults.py`
- `src/utils/normalization_common.py`
- `src/utils/cpu_config.py`
- `src/modules/demand_planning/normalization.py`

Module 4 集成调用当前以 `src/core/main_integration/production_runner.py` 为准。

### 当前验证结论

- 在**相同日期范围**下，`Module1-6` 与 `orchestrator` 输出已和 Dev 基线对齐
- 之前的 `ori_deployment_uid` 序号偏差已修复
- `summary` 对比必须使用**相同运行天数**，否则行数天然不同

## 环境要求

- Python 3.12
- 建议使用独立虚拟环境
- 如需数据库模式，需可访问 PostgreSQL

## 快速开始

### 1. 创建虚拟环境

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

### 2. 安装依赖

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. 验证关键依赖

```bash
python -c "import pandas, numpy, scipy, duckdb, openpyxl, yaml, tqdm, psycopg, psutil; print('deps ok')"
```

## requirements.txt 说明

`requirements.txt` 当前只保留**项目直接安装依赖**：

- 数据处理：`pandas`、`numpy`、`scipy`、`duckdb`
- Excel：`openpyxl`
- 数据库：`psycopg[binary]`
- 配置与运行：`PyYAML`、`tqdm`、`psutil`

脚本型可选依赖（如 DOCX 生成、备用 Excel writer）默认不放入主安装清单；如有需要可按脚本再补装。

## 运行方式

### 文件模式

文件模式首选 `--config-dir`，传入 `workspace/<project>/<scenario>/config/` 目录。该目录内必须有且只有一个 Excel 文件；同目录 CSV 会按 sheet 名大小写不敏感匹配，并优先于 Excel sheet 读取。

```powershell
python run.py --config-dir D:/PG/chainsight/workspace/SDC/baseline/config --start-date 2025-10-06 --end-date 2025-10-10
python run.py --config-dir SDC/baseline --end-date 2025-12-16 --force-restart --non-interactive
python run.py --config-dir SDC/baseline --end-date 2025-10-15 --resume
```

短格式 `<project>/<scenario>` 会展开为 `<workspace_root>/<project>/<scenario>/config`。`workspace_root` 优先级为：环境变量 `CHAINSIGHT_WORKSPACE` > 项目根 `.env` 中的 `CHAINSIGHT_WORKSPACE=...` > `config/defaults.yaml` 的 `workspace_root` > 项目根 `workspace/`。旧 `--config` Excel 路径仍在过渡期内可用，但文件模式新调用应迁移到 `--config-dir`。

### 数据库模式

数据库模式下，`--config` 可传配置名、Excel 文件路径，或包含唯一 Excel 的配置目录路径，并加 `--use-db`。传目录时会自动选中目录内唯一 Excel，`config_name` 使用 Excel 文件名：

```powershell
python run.py --config D:/PG/chainsight/config/sdc --start-date 2026-06-29 --end-date 2026-07-01 --use-db --non-interactive
python run.py --config D:/PG/chainsight/config/sdc/sdc.xlsx --start-date 2026-06-29 --end-date 2026-07-01 --use-db --non-interactive
python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-06 --use-db
python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db
```

显式指定数据库参数：

```powershell
python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db --db-host localhost --db-port 5432 --db-name test_db --db-user postgres --db-password 123456
```

说明：

- 文件模式会写本地输出目录
- 数据库模式默认主要写数据库，并保留本地运行日志
- 数据库模式如需同时落本地文件，可加 `--local`

## CLI 参数速览

以 `src/core/run/run_main.py` 为准：

| 参数 | 说明 |
|---|---|
| `--config-dir` | 文件模式传场景 `config/` 目录，支持绝对路径或 `<project>/<scenario>` 短格式 |
| `--config` | 已废弃的文件模式 Excel 路径；数据库模式传配置名、Excel 路径或含唯一 Excel 的目录路径 |
| `--start-date` | 首次运行必填，格式 `YYYY-MM-DD` |
| `--end-date` | 必填，格式 `YYYY-MM-DD` |
| `--resume` | 自动续跑 |
| `--resume-from` | 从指定运行目录继续 |
| `--check-resume` | 只检查续跑状态 |
| `--list-runs` | 列出已有运行目录 |
| `--non-interactive` | 关闭交互式选择 |
| `--force-restart` | 忽略续跑能力，强制重跑 |
| `--use-db` | 启用数据库模式 |
| `--db-host` `--db-port` `--db-name` `--db-user` `--db-password` | 数据库连接参数 |
| `--run-suffix` | 给运行目录追加后缀 |
| `--local` | 数据库模式下同时保留本地文件输出 |

## 当前权威目录结构

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

## 推荐导入方式

### 业务模块

```python
from src.modules import demand_planning as module1
from src.modules import mrp_planning as module3
from src.modules import production_planning as module4
from src.modules import deployment_planning as module5
from src.modules import logistics_execution as module6
```

### 主流程与 Orchestrator

```python
from src.core.main_integration import run_integrated_simulation
from src.core.main_integration import run_integrated_simulation_from_dict
from src.core.main_integration.production_runner import run_daily_production_planning_integrated
from src.core.main_integration.production_runner import load_current_date_production_gr
from src.core.orchestrator import create_orchestrator
```

### 共享配置与工具

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

### 不要再使用的旧导入

```python
from src.core.main_integration.production_integration import ...  # 已删除，请改用 production_runner
from src.utils.runtime_defaults import ...  # 已删除
from src.utils.normalization_common import ...  # 已删除
from src.utils.cpu_config import ...  # 已删除
```

## 输出与日志

### 文件模式输出

```text
outputs/<config_stem>/run_YYYYMMDD_HHMMSS/
```

常见子目录：

- `module1/`
- `module3/`
- `module4/`
- `module5/`
- `module6/`
- `orchestrator/`
- `summary/`

### 数据库模式输出

```text
outputs/db_<config_name>_<timestamp>/
```

通常主要保留：

- `simulation_log_<timestamp>.txt`

## 当前推荐验证命令

两天文件模式回归：

```powershell
python run.py --config config/OC_Paste_S1_20251224.xlsx --end-date 2025-12-16 --force-restart --non-interactive
```

注意：如需做结果回归，请使用你自己的对比脚本或外部工具，并确保 `summary` 目录两边运行天数一致。

## 文档入口

- [docs/INDEX.md](docs/INDEX.md)
- [docs/handover/refactored/00-overview/handover.md](docs/handover/refactored/00-overview/handover.md)
- [docs/handover/refactored/00-overview/setup.md](docs/handover/refactored/00-overview/setup.md)
- [docs/handover/refactored/02-api/api.md](docs/handover/refactored/02-api/api.md)
- [docs/_archive/](docs/_archive/)

---

如需继续做回归、部署或数据库初始化，优先以本 README 和 `src/core/run/run_main.py` 为准。
