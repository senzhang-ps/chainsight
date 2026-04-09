# ChainSight

供应链规划仿真系统，支持文件模式与数据库模式两种运行方式。

**中文** | [English](README_EN.md)

更新时间：2026-04-08

## 当前状态

当前仓库已经完成第三阶段结构收口，代码应按下面的现实理解：

- 根入口仍是 `run.py`
- 运行分发的权威入口是 `src.core.run` 包
  - `run.py` 中的 `from src.core.run import main` 实际会走 `src/core/run/__init__.py`
  - 主要实现位于 `src/core/run/run_main.py`
- 主集成流程位于 `src/core/main_integration/*`
- 共享状态位于 `src/core/orchestrator/*`
- 业务模块只保留五个真实子包
  - `src/modules/demand_planning/`
  - `src/modules/mrp_planning/`
  - `src/modules/production_planning/`
  - `src/modules/deployment_planning/`
  - `src/modules/logistics_execution/`
- 跨模块共享 helper 现在统一收口到 `src/utils/*`
  - `src/utils/runtime_defaults.py` 是 M3 / M5 共享小默认值的单一真源
  - `src/utils/normalization_common.py` 提供共享标识符标准化实现，各包 wrapper 继续保留历史语义
  - `src/utils/date_helpers.py` 提供共享窗口、review day 与 lead time helper

以下旧 wrapper 已经从代码树中移除，不应再作为当前入口理解：

- `src/modules/module1.py`
- `src/modules/module3.py`
- `src/modules/module4.py`
- `src/modules/module5.py`
- `src/modules/module6.py`
- `src/core/main_integration.py`
- `src/core/orchestrator.py`
- `src/core/parallel_executor.py`
- `src/core/main_integration/module4_runner.py`

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
python -m pip install -r requirements.txt
```

### 3. 验证关键依赖

```powershell
python -c "import pandas, numpy, duckdb, openpyxl, psycopg; print('deps ok')"
```

## 运行方式

### 文件模式

文件模式下，`--config` 传 Excel 路径。

```powershell
python run.py --config config/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10
python run.py --config config/BC_S5.xlsx --end-date 2025-10-15 --resume
python run.py --config config/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10 --force-restart
```

### 数据库模式

数据库模式下，`--config` 传配置名，不传 Excel 路径。

```powershell
python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-06 --use-db
python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db
```

显式指定数据库参数：

```powershell
python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db --db-host localhost --db-port 5432 --db-name test_db --db-user postgres --db-password 123456
```

说明：

- `OC_Paste_S1_20251224` 当前不需要引号，因为配置名没有空格。
- 数据库模式会把业务结果写入 PostgreSQL，本地主要保留日志目录。
- 如需同时写本地文件，可加 `--local`。

## CLI 参数速览

当前主参数以 `src/core/run/run_main.py` 为准：

| 参数 | 说明 |
|---|---|
| `--config` | 文件模式传 Excel 路径；数据库模式传配置名 |
| `--start-date` | 首次运行必填，格式 `YYYY-MM-DD` |
| `--end-date` | 必填，格式 `YYYY-MM-DD` |
| `--resume` | 自动续跑 |
| `--resume-from` | 指定已有运行目录继续 |
| `--check-resume` | 只检查续跑状态，不执行仿真 |
| `--list-runs` | 列出已有运行目录 |
| `--non-interactive` | 续跑时关闭交互选择 |
| `--force-restart` | 忽略续跑能力，强制重跑 |
| `--use-db` | 启用数据库模式 |
| `--db-host` `--db-port` `--db-name` `--db-user` `--db-password` | 数据库连接参数 |
| `--run-suffix` | 给本地运行目录追加后缀 |
| `--local` | 数据库模式下同时保留本地 Excel 输出 |

## 当前权威目录结构

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
from src.core.main_integration.production_integration import run_module4_integrated
from src.core.orchestrator import create_orchestrator
```

### 新代码推荐直接导入的共享工具

```python
from src.utils.runtime_defaults import DEFAULT_MOQ, DEFAULT_RV
from src.utils.normalization_common import normalize_identifiers_vectorized
from src.utils.date_helpers import compute_planning_window, calculate_transport_lead_time
```

不要再使用：

```python
from src.modules import module1, module3, module4, module5, module6
from src.core.main_integration.module4_runner import run_module4_integrated
```

## 运行输出与日志

### 文件模式

输出通常位于：

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

### 数据库模式

数据库模式的业务结果写入 PostgreSQL，本地主要保留日志目录：

```text
outputs/db_<config_name>_<timestamp>/
```

常见文件：

- `simulation_log_<timestamp>.txt`

默认数据库参数文件：

- `config/database.json`

## 当前已验证的回归基线

### 标准两天 DB 回归

```powershell
python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db
```

当前已确认通过的第三阶段删壳后批次：

- `db_OC_Paste_S1_20251224_20260408_112709`

关键结果：

- `module1_output_orderlog = 9214`
- `module1_output_shipmentlog = 3094`
- `module4_output_productionplan = 10`
- `module5_output_deploymentplan = 60182`
- `module6_output_deliveryplan = 39`
- `summary_output_ordershipmentcutsummary = 3433`
- `summary_output_fullcapacityexceed = 49`
- `summary_output_fulltruckusage = 1`
- checkpoint `orch_state_json = 186773 bytes`
- checkpoint 中不再包含 `m1_previous_orders`
- checkpoint 中 `delivery_gr / shipment_log / daily_logs = 0`

### 当前确认未回归的修复点

- PostgreSQL checkpoint `jsonb` 过大问题已修复
- `m1_previous_orders` 不再写入 checkpoint
- `delivery_gr` 去重不再依赖全历史大列表
- M4 集成适配已经从 `module4_runner.py` 收口到 `production_integration.py`

### 本轮零结果漂移共享工具收口

这次新增的共享模块没有改变业务调用面，只是把重复逻辑收成单一真源：

- 共享小默认值统一到 `src/utils/runtime_defaults.py`
- 共享标识符标准化统一到 `src/utils/normalization_common.py`
- 共享日期 / lead time helper 统一到 `src/utils/date_helpers.py`

收口后已验证通过的运行包括：

- `db_BC_S5_20260408_123150`
- `db_OC_Paste_S1_20251224_20260408_125002`

其中 `OC_Paste_S1_20251224` 两天 DB 回归结果保持不变：

- `module1_output_orderlog = 9214`
- `module1_output_shipmentlog = 3094`
- `module4_output_productionplan = 10`
- `module5_output_deploymentplan = 60182`
- `module6_output_deliveryplan = 39`
- checkpoint `orch_state_json = 186773 bytes`
- checkpoint 中仍不包含 `m1_previous_orders`
- checkpoint 中 `delivery_gr / shipment_log / daily_logs = 0`

## 数据库与配置说明

### 当前常用配置

- `config/BC_S5.xlsx`
- `config/BC_S9.xlsx`
- `config/OC_Paste_S1_20251224.xlsx`

### 数据库模式配置来源

数据库模式会根据配置名查找数据库中的配置；若缺失，会按项目配置搜索规则从 Excel 导入。当前仓库已放置：

- `config/OC_Paste_S1_20251224.xlsx`

这样可以保证 `OC_Paste_S1_20251224` 在数据库模式下稳定识别。

## 文档入口

如果你是第一次接手，建议按这个顺序看：

1. [docs/用户使用入口说明.md](docs/用户使用入口说明.md)
2. [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)
3. [docs/交接文档/Refactored/项目交接文档.md](docs/交接文档/Refactored/项目交接文档.md)
4. [docs/OPTIMIZED_CODE_STRUCTURE_DIAGRAM.md](docs/OPTIMIZED_CODE_STRUCTURE_DIAGRAM.md)
5. [docs/交接文档/Refactored/setup.md](docs/交接文档/Refactored/setup.md)
6. [docs/交接文档/Refactored/API.md](docs/交接文档/Refactored/API.md)

补充材料：

- [docs/INDEX.md](docs/INDEX.md)
- [docs/README_REFACTORING_MAP.md](docs/README_REFACTORING_MAP.md)
- [docs/PERFORMANCE_OPTIMIZATION_REPORT.md](docs/PERFORMANCE_OPTIMIZATION_REPORT.md)
- [docs/DUCKDB_OPTIMIZATION_GUIDE.md](docs/DUCKDB_OPTIMIZATION_GUIDE.md)
- [docs/CYTHON_OPTIMIZATION_REPORT.md](docs/CYTHON_OPTIMIZATION_REPORT.md)

## 常见问题

### 为什么 VSCode 里旧文件不见了，但 Git 里出现很多删除

这是第三阶段收口的预期结果。旧 wrapper 是被物理删除的，不是“隐藏”。

### 为什么数据库模式有时会报临时目录权限错误

在受限环境中，`AppData\\Local\\Temp` 之类的临时目录可能不可写。这类失败通常不是业务逻辑错误，需要在可写环境重跑。

### 为什么有些旧文档还在提 `module1.py`

有一部分历史背景文档还没完全收口。当前应优先以本 README、`docs/INDEX.md`、`docs/用户使用入口说明.md` 和交接文档为准。

### 项目里为什么同时存在 `src/core/run.py` 和 `src/core/run/`

当前实际入口是 `src.core.run` 包，也就是 `src/core/run/__init__.py` 和 `src/core/run/run_main.py`。阅读和后续开发应优先看包目录实现。
