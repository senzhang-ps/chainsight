# ChainSight 快速参考

更新时间：2026-04-08

## 1. 常用命令

### 1.1 安装依赖

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 1.2 文件模式

```powershell
python run.py --config config/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10
python run.py --config config/BC_S5.xlsx --end-date 2025-10-25 --resume
python run.py --config config/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10 --force-restart
```

### 1.3 数据库模式

```powershell
python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-06 --use-db
python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db
```

### 1.4 显式指定数据库参数

```powershell
python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db --db-host localhost --db-port 5432 --db-name test_db --db-user postgres --db-password 123456
```

## 2. 当前推荐导入路径

### 2.1 业务模块

```python
from src.modules import demand_planning as module1
from src.modules import mrp_planning as module3
from src.modules import production_planning as module4
from src.modules import deployment_planning as module5
from src.modules import logistics_execution as module6
```

### 2.2 主流程

```python
from src.core.main_integration import run_integrated_simulation
from src.core.main_integration import run_integrated_simulation_from_dict
from src.core.main_integration.production_integration import run_module4_integrated
from src.core.orchestrator import create_orchestrator
```

## 3. 已移除的旧导入

不要再使用：

```python
from src.modules import module1, module3, module4, module5, module6
from src.core.main_integration.module4_runner import run_module4_integrated
```

原因：

- 这些旧文件已经被物理删除
- 当前代码树以包和子包作为唯一权威入口

## 4. 当前目录结构速记

```text
run.py
src/
  core/
    run.py
    run/
    main_integration/
    orchestrator/
    parallel_executor/
  modules/
    demand_planning/
    mrp_planning/
    production_planning/
    deployment_planning/
    logistics_execution/
  services/
  utils/
pgsql_db/
config/
outputs/
docs/
```

## 5. 当前最关键的验证基线

### 5.1 第二阶段收口后 DB 回归

- `db_OC_Paste_S1_20251224_20260408_111526`

### 5.2 第三阶段删壳后 DB 回归

- `db_OC_Paste_S1_20251224_20260408_112709`

两次结果一致的关键指标：

- `module1_output_orderlog = 9214`
- `module1_output_shipmentlog = 3094`
- `module4_output_productionplan = 10`
- `module5_output_deploymentplan = 60182`
- `module6_output_deliveryplan = 39`
- `summary_output_ordershipmentcutsummary = 3433`
- `summary_output_fullcapacityexceed = 49`
- `summary_output_fulltruckusage = 1`
- checkpoint `orch_state_json = 186773 bytes`
- checkpoint 中 `m1_previous_orders = False`
- checkpoint 中 `delivery_gr / shipment_log / daily_logs = 0`

## 6. 常看路径

### 6.1 日志

```text
outputs/db_<config_name>_<timestamp>/simulation_log_<timestamp>.txt
```

### 6.2 数据库配置

```text
config/database.json
```

### 6.3 结构图和交接文档

```text
docs/OPTIMIZED_CODE_STRUCTURE_DIAGRAM.md
docs/交接文档/Refactored/项目交接文档.md
```

## 7. 一句话建议

- 结构改动后先跑 `BC_S5` 短窗，再跑 `OC_Paste_S1_20251224` 两天 DB 回归。
- 数据库模式的 `--config` 优先传配置名，不传路径。
- `OC_Paste_S1_20251224` 当前不需要加引号。
