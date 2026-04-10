# ChainSight Refactoring Map

更新时间：2026-04-08

## 1. 这份文档的定位

本文档用于回答两个问题：

1. `ChainSight_Dev` 的平铺代码，当前在 `src/` 里分别落到了哪里。
2. 第三阶段删壳以后，哪些旧入口已经彻底删除，应该如何迁移导入。

如果你要实际运行项目或接手维护，优先看：

1. [INDEX.md](INDEX.md)
2. [用户使用入口说明.md](用户使用入口说明.md)
3. [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
4. [交接文档/Refactored/项目交接文档.md](交接文档/Refactored/项目交接文档.md)

## 2. 当前总判断

当前代码结构已经从“平铺单体 + 子包并存”收口到“包优先 + 子包优先”：

- 业务模块以五个真实子包为准
- `src/core/main_integration.py`、`src/core/orchestrator.py` 等旧单体文件已删除
- `src/modules/module1.py` 到 `src/modules/module6.py` 已删除
- `src/core/main_integration/module4_runner.py` 已删除
- M4 集成适配的当前位置是 `src/core/main_integration/production_integration.py`
- 跨模块共享的小默认值、normalize helper、date helper 已收口到 `src/utils/`

## 3. 顶层映射

| Dev / 历史平铺入口 | 当前对应位置 | 当前状态 |
|---|---|---|
| `run.py` | `run.py` + `src/core/run/__init__.py` + `src/core/run/run_main.py` | 仍存在，但已按包拆分 |
| `main_integration.py` | `src/core/main_integration/*` | 旧单体文件已删除 |
| `orchestrator.py` | `src/core/orchestrator/*` | 旧单体文件已删除 |
| `parallel_executor.py` | `src/core/parallel_executor/*` | 旧单体文件已删除 |
| `summary_report_generator.py` | `src/services/summary_report_generator.py` | 已迁入 services |
| `performance_profiler.py` | `src/services/performance_profiler.py` | 已迁入 services |
| `config_validator.py` | `src/utils/config_validator.py` | 已迁入 utils |
| `logger_config.py` | `src/utils/logger_config.py` | 已迁入 utils |
| `time_manager.py` | `src/utils/time_manager.py` | 已迁入 utils |

## 3.1 新增的共享基础模块

这些文件不是直接从单个 Dev 平铺文件一比一搬迁而来，而是把多个模块中的重复底层逻辑零结果漂移地收口到了 `src/utils/`：

| 当前共享文件 | 用途 | 主要服务对象 |
|---|---|---|
| `src/utils/runtime_defaults.py` | 统一共享小默认值 | M3 / M5 |
| `src/utils/normalization_common.py` | 统一标识符标准化底层实现 | M1 / M3 / M5 / M4 / Orchestrator / main_integration |
| `src/utils/date_helpers.py` | 统一窗口、review day、lead time 底层 helper | M3 / M4 / M5 |

## 4. 五大业务模块映射

| 历史文件 | 当前权威子包 | 说明 |
|---|---|---|
| `module1.py` | `src/modules/demand_planning/*` | 旧 facade 已删除 |
| `module3.py` | `src/modules/mrp_planning/*` | 旧 facade 已删除 |
| `module4.py` | `src/modules/production_planning/*` | 旧 facade 已删除 |
| `module5.py` | `src/modules/deployment_planning/*` | 旧 facade 已删除 |
| `module6.py` | `src/modules/logistics_execution/*` | 旧 facade 已删除，真实入口在 `main.py` |

### 4.1 M1 Demand Planning

当前主要文件：

- `src/modules/demand_planning/config.py`
- `src/modules/demand_planning/dps.py`
- `src/modules/demand_planning/forecast.py`
- `src/modules/demand_planning/order.py`
- `src/modules/demand_planning/consume.py`
- `src/modules/demand_planning/shipment.py`
- `src/modules/demand_planning/integration.py`

当前包级公开入口：

- `run_daily_order_generation`
- `generate_supply_demand_log_for_integration`
- `load_config`

### 4.2 M3 MRP Planning

当前主要文件：

- `src/modules/mrp_planning/config_loader.py`
- `src/modules/mrp_planning/layer_assignment.py`
- `src/modules/mrp_planning/lead_time.py`
- `src/modules/mrp_planning/net_demand.py`
- `src/modules/mrp_planning/mrp_simulation.py`
- `src/modules/mrp_planning/integration.py`

### 4.3 M4 Production Planning

当前主要文件：

- `src/modules/production_planning/config_loader.py`
- `src/modules/production_planning/demand_loader.py`
- `src/modules/production_planning/plan_builder.py`
- `src/modules/production_planning/capacity_allocator.py`
- `src/modules/production_planning/state_manager.py`
- `src/modules/production_planning/output_writer.py`
- `src/modules/production_planning/main.py`

当前跨层集成适配位置：

- `src/core/main_integration/production_integration.py`

### 4.4 M5 Deployment Planning

当前主要文件：

- `src/modules/deployment_planning/data_loader.py`
- `src/modules/deployment_planning/demand_collector.py`
- `src/modules/deployment_planning/allocation.py`
- `src/modules/deployment_planning/push_allocation.py`
- `src/modules/deployment_planning/inventory.py`
- `src/modules/deployment_planning/main.py`

### 4.5 M6 Logistics Execution

当前主要文件：

- `src/modules/logistics_execution/config_loader.py`
- `src/modules/logistics_execution/expression_evaluator.py`
- `src/modules/logistics_execution/capacity_manager.py`
- `src/modules/logistics_execution/vehicle_packer.py`
- `src/modules/logistics_execution/delivery_processor.py`
- `src/modules/logistics_execution/inventory_manager.py`
- `src/modules/logistics_execution/main.py`

关键说明：

- 第三阶段后，真实实现已经集中到 `src/modules/logistics_execution/main.py`
- 不应再回头找 `src/modules/module6.py`

## 5. 集成与状态层映射

### 5.1 运行分发层

当前应这样理解：

- 仓库根入口：`run.py`
- Python 导入入口：`src.core.run`
- 包级导出：`src/core/run/__init__.py`
- 主实现：`src/core/run/run_main.py`
- 辅助实现：`src/core/run/db_runner.py`、`db_config.py`、`output_dir.py`、`local_writer.py`

### 5.2 主集成流程层

当前主要文件：

- `src/core/main_integration/simulation_file.py`
- `src/core/main_integration/simulation_db.py`
- `src/core/main_integration/config_loader.py`
- `src/core/main_integration/resume.py`
- `src/core/main_integration/db_helpers.py`
- `src/core/main_integration/production_integration.py`

### 5.3 Orchestrator 共享状态层

当前主要文件：

- `src/core/orchestrator/orchestrator_main.py`
- `src/core/orchestrator/daily_ops.py`
- `src/core/orchestrator/processors.py`
- `src/core/orchestrator/persistence.py`
- `src/core/orchestrator/models.py`

## 6. 导入迁移规则

### 6.1 当前推荐写法

```python
from src.modules import demand_planning as module1
from src.modules import mrp_planning as module3
from src.modules import production_planning as module4
from src.modules import deployment_planning as module5
from src.modules import logistics_execution as module6

from src.core.main_integration import run_integrated_simulation
from src.core.main_integration import run_integrated_simulation_from_dict
from src.core.main_integration.production_integration import run_module4_integrated
from src.core.orchestrator import create_orchestrator
```

### 6.2 需要迁移掉的旧写法

```python
from src.modules import module1, module3, module4, module5, module6
from src.core.main_integration.module4_runner import run_module4_integrated
```

### 6.3 旧路径到新路径

| 旧导入 | 新导入 |
|---|---|
| `src.modules.module1` | `src.modules.demand_planning` |
| `src.modules.module3` | `src.modules.mrp_planning` |
| `src.modules.module4` | `src.modules.production_planning` |
| `src.modules.module5` | `src.modules.deployment_planning` |
| `src.modules.module6` | `src.modules.logistics_execution` |
| `src.core.main_integration.module4_runner` | `src.core.main_integration.production_integration` |

## 7. 与这次修复一起落下来的结构性变化

这一轮除了删壳，还保留了今天修掉的运行时问题：

- checkpoint 已轻量化，不再把大历史对象塞进 `orch_state_json`
- `m1_previous_orders` 不再持久化到 checkpoint
- `delivery_gr` 去重不再依赖全历史列表
- 这些变更已经在删壳后回归中验证未退化

## 8. 验证证据

当前最关键的结构回归命令：

```powershell
python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db
```

已验证成功批次：

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

## 9. 如何使用这份映射文档

如果你看到旧材料里仍提到 `module1.py`、`main_integration.py`、`module4_runner.py`：

1. 先不要按旧文件名去找实现。
2. 先回到本文件第 3 到第 6 节看映射。
3. 再进入当前包目录阅读真实代码。

## 10. 一句话结论

`ChainSight_Dev` 的平铺结构已经迁移为 `src/` 下的包结构；第三阶段以后，旧 facade 与旧单体文件都应视为历史概念，而不是当前代码入口。
