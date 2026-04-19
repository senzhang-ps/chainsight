# ChainSight 项目交接文档

**编写人**：陈显跃  
**更新时间**：2026-04-15  
**适用范围**：当前 `ChainSight` 主仓库（Phase-3 complete）

## 1. 交接结论

当前代码已经完成从“保留 legacy wrapper 的过渡态”到“只保留包入口和真实子包实现”的第三阶段收口。

本次交接应以以下结论为准：

- 当前主开发基础应视为 `src/` 与 `pgsql_db/`
- `ChainSight_Dev` 仅作为历史结果和业务口径对照基线
- `src/modules/module1.py` 到 `module6.py` 已删除
- `src/core/main_integration.py`、`src/core/orchestrator.py`、`src/core/run.py` 等单体文件已删除（package 已存在，module 为死代码）
- `src/core/main_integration/module4_runner.py` 已删除，M4 集成适配收口到 `production_runner.py`
- 7 个未使用的 utils 模块已删除（duckdb_sql_wrapper, high_perf_executor 等）
- 数据库 checkpoint 过大问题已修复，且在第三阶段删壳后回归未退化
- 新增 YAML 驱动的参数集中化（`config/defaults.yaml` → `src/utils/defaults.py`）
- 新增统一标识符归一化实现（`src/utils/normalization.py`）

## 2. 当前权威目录

### 2.1 运行与编排

- `run.py`（仓库 CLI 入口）
- `src/core/run/`（7 个文件：run_main, db_runner, db_config, local_writer, output_dir, utils）
- `src/core/main_integration/`（13 个文件：simulation_file/db, production_runner, config_loader, resume, seed 等）

### 2.2 共享状态

- `src/core/orchestrator/`（9 个文件：orchestrator_main, daily_ops, processors, persistence, inventory_log, views, models, normalize）
- `src/core/parallel_executor/`（4 个文件：parallel_executor_main, convenience, models）

### 2.3 业务模块

- `src/modules/demand_planning/`（12 个文件）
- `src/modules/mrp_planning/`（12 个文件）
- `src/modules/production_planning/`（12 个文件）
- `src/modules/deployment_planning/`（16 个文件）
- `src/modules/logistics_execution/`（12 个文件）

### 2.4 数据库边界

- `pgsql_db/`（19 个文件）

### 2.5 共享工具层（src/utils/ — 15 个文件）

- `defaults.py` — YAML 配置加载器，从 `config/defaults.yaml` 读取跨模块共享默认参数
- `normalization.py` — 统一标识符归一化实现（单一真源，5 处重复 → 1 处）
- `date_helpers.py` — 共享窗口、review day 与 lead time helper
- `config_validator.py` / `validation_manager.py` — 配置校验
- `logger_config.py` / `time_manager.py` — 日志与时间管理
- `simulation_cache.py` / `memory_data_store.py` — 缓存与内存存储
- `inventory_balance_checker.py` — 库存平衡检查
- `duckdb_accelerator.py` / `duckdb_optimizer.py` — DuckDB 加速
- `resource_config.py` — 资源配置

### 2.6 配置与工具

- `config/defaults.yaml` — YAML 形式的跨模块共享默认参数（单一真源）。数据库连接配置位于其 `database:` 节点，由 `pgsql_db/settings.py` 读取
- `config/*.xlsx` — 仿真输入配置（BC_S5, BC_S9, OC_Paste_S1_20251224, PDS1）

## 3. 这次收口具体做了什么

## 3.1 第一阶段

- 修复数据库 checkpoint `jsonb` 过大问题
- 将 M6 真实实现迁入 `src/modules/logistics_execution/main.py`
- 将主链路导入切到模块子包
- 对 `src.modules` 与 `src.utils` 做懒加载，降低循环导入风险

## 3.2 第二阶段

- 将 `module1.py`、`module4.py` 压成薄兼容层
- 将 M4 集成适配层规范化为 `src/core/main_integration/production_runner.py`
- 让 `simulation_file.py` 与 `simulation_db.py` 直接使用新命名入口

## 3.3 第三阶段

- 物理删除所有 `src/modules/module*.py` wrapper（5 个文件，~1,690 行）
- 物理删除 `src/core/main_integration.py`、`src/core/orchestrator.py`、`src/core/run.py`（3 个文件，~3,500 行）
- 物理删除 `src/core/main_integration/module4_runner.py`（重命名为 `production_runner.py`）
- 物理删除 7 个未使用的 utils 模块（~2,300 行）：duckdb_sql_wrapper, high_perf_executor, multiprocess_executor, parallel_optimizer, performance, process_pool_executor, optimization_config
- `src/modules/__init__.py` 改为包别名导入：`from . import demand_planning as module1` 等

## 3.4 零结果漂移共享工具收口

- 新增 `config/defaults.yaml` + `src/utils/defaults.py`，YAML 驱动的参数集中化
- 新增 `src/utils/normalization.py`，统一 5 处重复的标识符归一化实现
- 删除 `src/utils/runtime_defaults.py`、`src/utils/normalization_common.py`、`src/utils/cpu_config.py`，避免兼容层继续分叉
- `src/utils/date_helpers.py` 统一窗口、review day、lead time 的底层 helper
- M3 / M5 constants 改为从 `src.utils.defaults` 导入共享默认值
- M1 / M3 / M5 / Orchestrator / main_integration 的 normalization 包装改为从 `src.utils.normalization` 导入

## 4. 当前推荐导入规则

### 4.1 业务模块

```python
from src.modules import demand_planning as module1
from src.modules import mrp_planning as module3
from src.modules import production_planning as module4
from src.modules import deployment_planning as module5
from src.modules import logistics_execution as module6
```

### 4.2 M4 集成适配

```python
from src.core.main_integration.production_runner import run_module4_integrated
from src.core.main_integration.production_runner import load_current_date_production_gr
```

### 4.3 主流程与 Orchestrator

```python
from src.core.main_integration import run_integrated_simulation
from src.core.main_integration import run_integrated_simulation_from_dict
from src.core.orchestrator import create_orchestrator
```

### 4.4 共享工具层（推荐新导入方式）

```python
# 跨模块共享参数（从 YAML 加载）
from src.utils.defaults import DEFAULT_MOQ, DEFAULT_RV, DEFAULT_HORIZON
from src.utils.defaults import DEFAULT_CHANGEOVER_TIME, DEFAULT_PUSH_LEVELS

# 统一的标识符归一化
from src.utils.normalization import normalize_identifiers, normalize_material

# 共享日期与前置期 helper
from src.utils.date_helpers import compute_planning_window, calculate_transport_lead_time
```

说明：

- 新代码优先使用 `src.utils.defaults` 和 `src.utils.normalization`（新的单一真源）
- `runtime_defaults.py`、`normalization_common.py`、`cpu_config.py` 已删除，不应再继续引用
- 模块内部若已有包装函数，优先继续走模块包装，避免直接跳过历史语义适配

## 5. 已确认删除的旧入口

以下文件已不再存在：

- `src/modules/module1.py`
- `src/modules/module3.py`
- `src/modules/module4.py`
- `src/modules/module5.py`
- `src/modules/module6.py`
- `src/core/main_integration.py`（package 已存在）
- `src/core/orchestrator.py`（package 已存在）
- `src/core/run.py`（package 已存在）
- `src/core/main_integration/module4_runner.py`（重命名为 production_runner.py）
- `src/utils/duckdb_sql_wrapper.py`
- `src/utils/high_perf_executor.py`
- `src/utils/multiprocess_executor.py`
- `src/utils/parallel_optimizer.py`
- `src/utils/performance.py`
- `src/utils/process_pool_executor.py`
- `src/utils/optimization_config.py`

因此，任何继续依赖这些文件名的脚本都需要迁移导入路径。

## 6. 当前最重要的运行口径

### 6.1 文件模式

`--config` 传 Excel 路径。

示例：

```powershell
python run.py --config config/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10
```

### 6.2 数据库模式

`--config` 传配置名。

示例：

```powershell
python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-06 --use-db
python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db
```

注意：

- `OC_Paste_S1_20251224` 当前不需要加引号
- 当前仓库里已放置 `config/OC_Paste_S1_20251224.xlsx`

## 7. 本次交接的关键验证证据

## 7.1 第二阶段收口验证

- 运行命令：

```powershell
.\.venv\Scripts\python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db
```

- 成功 `run_id`：`db_OC_Paste_S1_20251224_20260408_111526`

## 7.2 第三阶段删壳后验证

- 运行命令：

```powershell
.\.venv\Scripts\python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db
```

- 成功 `run_id`：`db_OC_Paste_S1_20251224_20260408_112709`

## 7.3 第三阶段删壳后关键结果

关键输出表行数：

- `module1_output_orderlog = 9214`
- `module1_output_shipmentlog = 3094`
- `module4_output_productionplan = 10`
- `module5_output_deploymentplan = 60182`
- `module6_output_deliveryplan = 39`
- `summary_output_ordershipmentcutsummary = 3433`
- `summary_output_fullcapacityexceed = 49`
- `summary_output_fulltruckusage = 1`

checkpoint 关键指标：

- `status = completed`
- `last_batch_end = 2025-12-16`
- `orch_state_json = 186773 bytes`
- `m1_previous_orders = False`
- `delivery_gr = 0`
- `shipment_log = 0`
- `daily_logs = 0`

## 7.4 共享工具收口后的回归验证

- 运行命令：

```powershell
.\.venv\Scripts\python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-06 --use-db
.\.venv\Scripts\python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db
```

- 成功 `run_id`：
  - `db_BC_S5_20260408_123150`
  - `db_OC_Paste_S1_20251224_20260408_125002`

- 关键结果保持不变：
  - `BC_S5`：`738 / 282 / 0 / 10125 / 0`
  - `OC_Paste_S1_20251224`：`9214 / 3094 / 10 / 60182 / 39`
  - `OC` summary：`3433 / 49 / 1`
  - `OC` checkpoint：`186773 bytes`，`m1_previous_orders = False`，`delivery_gr / shipment_log / daily_logs = 0`

## 8. 当前确认未回归的修复点

### 8.1 checkpoint 过大问题

本次交接前已经修复 PostgreSQL `jsonb` 超限问题，当前第三阶段回归后仍保持正确：

- checkpoint 不再把大历史对象写回 `orch_state_json`
- `m1_previous_orders` 不再落入 checkpoint
- `delivery_gr` 去重逻辑不再依赖全历史列表

### 8.2 结构性收口不改变运行结果

经过第二阶段和第三阶段两轮 DB 回归，关键结果一致，说明：

- 模块入口迁移没有改变运行结果
- wrapper 删除没有改变运行结果
- M4 集成适配层重命名没有改变运行结果

### 8.3 共享工具收口不改变业务语义

经过本轮共享工具收口后的 BC / OC 两套 DB 回归，当前可以认为：

- 共享默认值集中化没有改变 M3 / M5 的运行结果
- normalization 抽取没有改变 Orchestrator 与 main_integration 之间原本不同的物料语义
- 日期 / lead time helper 抽取没有改变 M4 / M5 / M3 的窗口与前置期口径

## 9. 当前仍需注意的事项

### 9.1 代码工作树仍未提交

截至本文档更新时，这批改动仍在工作树中，尚未整理成提交。

### 9.2 `config/OC_Paste_S1_20251224.xlsx` 是本次补齐的运行配置

这个文件是为了保证数据库模式能稳定识别该配置而放到 `config/` 根目录的。  
后续是否保留为正式配置文件，需要由分支维护者决定。

### 9.3 数据库模式在受限环境中可能被 `Temp` 目录权限阻断

在沙箱环境中，数据库模式可能因为无法写 `AppData\Local\Temp` 而失败。  
这类失败不代表业务逻辑错误，需要在允许写临时目录的环境下重跑。

### 9.4 Summary 落库过程中存在 Pandas `Boolean Series key will be reindexed` warning

当前 warning 不影响这次回归成功，但仍值得在后续单独清理。

## 10. 接手建议

1. 以后新增结构改动时，默认使用包路径，不要重新引入 `module1.py` 之类的平铺入口。
2. 结构改动后先跑 `BC_S5` 短窗，再跑 `OC_Paste_S1_20251224` 两天数据库回归。
3. 若涉及 checkpoint、Orchestrator 或 M4/M6 适配逻辑改动，必须复查 `orch_state_json` 大小和关键表行数。
4. 若有人依据旧文档继续寻找 `src/modules/module4.py`，应直接引导其改看 `src/modules/production_planning/`。
5. 若要修改默认值，编辑 `config/defaults.yaml`（无需改 Python 代码）；若要修改归一化逻辑，编辑 `src/utils/normalization.py`（所有模块自动受益）；若要修改日期 helper，编辑 `src/utils/date_helpers.py`。

## 11. 推荐的后续动作

短期建议：

- 把当前工作树整理为清晰提交
- 决定是否正式纳入 `config/OC_Paste_S1_20251224.xlsx`
- 清理 `module_data_writer.py` 中的 Pandas warning

中期建议：

- 为结构变更建立自动化回归脚本
- 将当前两天 OC 回归和一天 BC 回归固化为标准 smoke
- 持续补齐文档中仍引用旧 wrapper 的历史材料

## 12. 与本交接文档配套阅读

- `docs/handover/refactored/00-overview/cleanup_and_improvements.md` — 第三阶段清理详细记录
- `docs/handover/refactored/00-overview/setup.md`
- `docs/handover/refactored/02-api/api.md`
- `docs/handover/refactored/03-modules/modules_compat.md`
- `docs/handover/refactored/04-services-utils/utils.md`
