# `src/modules` 模块导入与兼容性文档

> **文档版本**: v2.0
> **最后更新**: 2026-04-06
> **变更说明**: v1.0 中记录的 `module1.py`~`module6.py` 薄包装层文件已在合并重构中全部删除。本版本描述当前的包导入结构和向后兼容方案。

---

## 目录

1. [概述](#概述)
2. [当前包结构](#当前包结构)
3. [向后兼容方案](#向后兼容方案)
4. [各子包导出清单](#各子包导出清单)
5. [迁移指南](#迁移指南)
6. [与 v1.0 的变更对照](#与-v10-的变更对照)

---

## 概述

在 v2.0 合并重构之前，`src/modules/` 目录下存在 5 个薄包装层文件（`module1.py`、`module3.py`、`module4.py`、`module5.py`）和 1 个 1547 行的核心物流执行模块（`module6.py`）。这些文件的职责是从对应子包中 re-export 公开接口，以支持 `from modules.module1 import xxx` 形式的旧式调用。

**在合并重构中，这些文件已全部删除。** 原因如下：

1. **包优先原则**：Python 在同级目录中，包（子目录）比同名 `.py` 文件优先级更高，薄包装层在实际运行时不被导入
2. **消除重复**：shim 文件的导出列表与子包 `__init__.py` 完全重复，增加了维护负担
3. **module6.py 拆分**：1547 行的核心逻辑已拆分为 `logistics_execution/` 子包下的 8 个子模块

当前所有导入统一通过子包进行，向后兼容通过 `src/modules/__init__.py` 中的别名实现。

---

## 当前包结构

```
src/modules/
├── __init__.py                     # 统一导出 + 向后兼容别名
├── demand_planning/                # M1 需求规划
│   ├── __init__.py
│   ├── constants.py
│   ├── normalization.py            # → re-export from src.utils.normalization
│   ├── config_loader.py
│   ├── forecast_splitter.py
│   ├── order_generator.py
│   ├── order_consumer.py
│   ├── shipment_calculator.py
│   ├── integration.py
│   └── io_handler.py
├── mrp_planning/                   # M3 MRP 计划
│   ├── __init__.py
│   ├── constants.py                # 从 config.yaml 读取共享参数
│   ├── utils.py                    # normalize 函数已替换为 shared import
│   ├── config_loader.py
│   ├── lead_time.py
│   ├── layer_assignment.py
│   ├── node_processor.py
│   ├── net_demand.py
│   ├── mrp_simulation.py
│   └── integration.py
├── production_planning/            # M4 生产计划
│   ├── __init__.py
│   ├── constants.py                # 从 config.yaml 读取共享参数
│   ├── types.py
│   ├── utils.py                    # normalize_location 已替换为 shared import
│   ├── state_manager.py
│   ├── config_loader.py
│   ├── demand_loader.py
│   ├── plan_builder.py
│   ├── capacity_allocator.py
│   ├── output_writer.py
│   └── main.py
├── deployment_planning/            # M5 部署规划
│   ├── __init__.py
│   ├── constants.py                # 从 config.yaml 读取共享参数
│   ├── normalizer.py               # → re-export from src.utils.normalization
│   ├── config_loader.py
│   ├── demand_collector.py
│   ├── allocation.py
│   ├── batch_optimizer.py
│   ├── deployment_generator.py
│   └── main.py
└── logistics_execution/            # M6 物流执行（原 module6.py 拆分）
    ├── __init__.py
    ├── constants.py                # ALLOWED_EXPRESSION_VARS
    ├── initializer.py              # 运行参数初始化
    ├── data_preparer.py            # 数据准备与映射构建
    ├── simulation_engine.py        # 仿真循环引擎
    ├── route_processor.py          # 路线处理与装载
    ├── constraint_enforcer.py      # 约束验证
    ├── output_builder.py           # 输出构建与 Excel 写入
    ├── main.py                     # 公开入口函数
    ├── vehicle_packer.py           # 车辆装箱（原有）
    └── delivery_utils.py           # 交付工具函数（原有）
```

---

## 向后兼容方案

### `src/modules/__init__.py` 别名机制

```python
from . import demand_planning
from . import mrp_planning
from . import production_planning
from . import deployment_planning
from . import logistics_execution

# 向后兼容别名
module1 = demand_planning
module3 = mrp_planning
module4 = production_planning
module5 = deployment_planning
module6 = logistics_execution
```

这意味着以下两种导入方式等价：

```python
# 新方式（推荐）
from src.modules import demand_planning
demand_planning.run_daily_order_generation(...)

# 旧方式（仍可用，不推荐用于新代码）
from src.modules import module1
module1.run_daily_order_generation(...)
```

### core 层的过渡导入

`src/core/main_integration/` 中的 3 个调用文件使用了别名模式以最小化改动：

```python
# simulation_file.py, simulation_db.py
from ...modules import (
    demand_planning as module1,
    mrp_planning as module3,
    production_planning as module4,
    deployment_planning as module5,
    logistics_execution as module6,
)

# production_planning_runner.py
from ...modules import production_planning as module4
```

文件内部仍使用 `module1.xxx`、`module4.xxx` 等调用形式，但实际导入的是子包。

### 子包内部的 re-export 兼容

各子包 `__init__.py` 中提供了常用的向后兼容别名：

| 子包 | 别名 | 指向 |
|------|------|------|
| `mrp_planning` | `_normalize_location` | `normalize_location` |
| `mrp_planning` | `_normalize_material` | `normalize_material` |
| `mrp_planning` | `_normalize_identifiers` | `normalize_identifiers` |
| `production_planning` | `analyze_end_of_day_changeover_state` | `_analyze_end_of_day_changeover` |

---

## 各子包导出清单

### demand_planning

| 类别 | 导出名称 |
|------|---------|
| 规范化函数 | `normalize_material`, `normalize_location`, `normalize_identifiers` |
| 配置加载 | `load_module1_config`, `validate_module1_config` |
| DPS / 供应选择 | `run_dps`, `select_supply_source` |
| 预测拆分 | `split_forecast` |
| 订单生成 | `generate_orders` |
| 订单消耗 | `consume_orders` |
| 发货计算 | `calculate_shipments` |
| 集成入口 | `run_daily_demand_planning`, `run_demand_planning_module` |
| IO 工具 | `read_module1_inputs`, `write_module1_outputs` |

### mrp_planning

| 类别 | 导出名称 |
|------|---------|
| 常量 | `DEFAULT_MOQ`, `DEFAULT_RV`, `DEFAULT_HORIZON` |
| 工具函数 | `apply_moq_rv`, `normalize_location`, `normalize_material`, `normalize_identifiers`, `apportion_largest_remainder`, `lookup_moq_rv_three_keys`, `build_ptf_lsk_cache`, `get_ptf_lsk` |
| 配置加载 | `load_config`, `load_module1_daily_outputs`, `load_excel_with_sheets` |
| 提前期 | `compute_root_horizon`, `determine_lead_time`, `infer_sending_location_type` |
| 层级分配 | `assign_location_layers` |
| 核心功能 | `calculate_daily_net_demand`, `run_mrp_layered_simulation_daily`, `run_integrated_mode` |

### production_planning

| 类别 | 导出名称 |
|------|---------|
| 常量 | `IDENTIFIER_COLS`, `DEFAULT_CHANGEOVER_TIME`, `PLAN_COLUMNS`, `EXCEED_COLUMNS`, `VALIDATION_COLUMNS`, `CHANGEOVER_LOG_COLUMNS`, `UNCONSTRAINED_PLAN_COLUMNS`, `REQUIRED_CONFIG_SHEETS`, `SHEET_KEY_MAPPING` |
| 类型 | `LineState`, `ChangeoverInfo`, `PlanRecord`, `ExceedRecord`, `ValidationIssue` |
| 工具函数 | `normalize_location`, `cast_identifiers_to_str`, `validate_merge_keys`, `compute_planning_window`, `is_review_day`, `dedup_issues`, `round_up_to_batch`, `safe_float_conversion`, `ensure_dataframe_columns` |
| 状态管理 | `get_or_init_simulation_start`, `save_line_state`, `load_line_state`, `save_allocated_capacity`, `load_allocated_capacity`, `load_all_previous_capacity` |
| 配置 | `load_config`, `validate_config` |
| 需求 | `load_daily_net_demand` |
| 计划构建 | `build_unconstrained_plan_for_single_day`, `optimal_changeover_sequence` |
| 产能分配 | `centralized_capacity_allocation_with_changeover`, `extract_allocated_capacity_from_plan`, `validate_capacity_allocation`, `extract_line_states_from_plan`, `calculate_changeover_metrics`, `simulate_production`, `_analyze_end_of_day_changeover` |
| 输出 | `write_output`, `generate_consolidated_output` |
| 主入口 | `run_daily_production_planning`, `main`, `DailyProductionPlanner` |

### deployment_planning

| 类别 | 导出名称 |
|------|---------|
| 规范化 | `normalize_location`, `normalize_material`, `normalize_identifiers` |
| 配置与数据加载 | `load_config`, `load_deployment_inputs` |
| 需求收集 | `collect_demands` |
| 分配 | `allocate_inventory`, `apply_push_pull` |
| 部署生成 | `generate_deployment_plan` |
| 集成入口 | `main` |

### logistics_execution

| 类别 | 导出名称 |
|------|---------|
| 常量 | `ALLOWED_EXPRESSION_VARS` |
| 初始化 | `initialize_run_params` |
| 数据准备 | `prepare_data` |
| 仿真引擎 | `run_simulation_loop` |
| 路线处理 | `process_routes` |
| 约束验证 | `enforce_shipment_constraint`, `validate_shipment_delivery_constraint` |
| 输出构建 | `generate_outputs` |
| 公开入口 | `run_daily_physical_flow`, `run_physical_flow_module`, `main` |
| 原有工具 | `VehiclePacker`, `sample_delivery_delay`, `batch_sample_delivery_delays_duckdb` |

---

## 迁移指南

### 如果你的代码使用 `from modules.module1 import xxx`

无需修改。`modules.__init__.py` 中的 `module1 = demand_planning` 别名保证了此导入仍然有效。但建议在新代码中改用：

```python
from src.modules.demand_planning import run_daily_order_generation
```

### 如果你的代码使用 `from modules.module6 import run_physical_flow_module`

这仍然有效，因为 `module6 = logistics_execution` 别名存在。但 module6.py 文件已不存在，所有逻辑已拆入 `logistics_execution/` 子包的 8 个子模块中。

### normalize 函数的统一

所有 normalize 函数现在统一由 `src/utils/normalization.py` 提供。各模块保留的 re-export 层：

| 原始调用路径 | 当前状态 |
|------------|---------|
| `demand_planning.normalization.normalize_location` | re-export from `src.utils.normalization` |
| `deployment_planning.normalizer.normalize_location` | re-export from `src.utils.normalization` |
| `mrp_planning.utils.normalize_location` | import from `src.utils.normalization` |
| `production_planning.utils.normalize_location` | import from `src.utils.normalization` |
| `core.orchestrator.normalize._normalize_location` | re-export with private name |
| `core.main_integration.normalize._normalize_location` | re-export with private name |

---

## 与 v1.0 的变更对照

| 项目 | v1.0 (重构前) | v2.0 (合并重构后) |
|------|-------------|-----------------|
| module1.py | 167 行薄包装层，存在 | **已删除** |
| module3.py | 94 行薄包装层，存在 | **已删除** |
| module4.py | 229 行薄包装层，存在 | **已删除** |
| module5.py | 157 行薄包装层，存在 | **已删除** |
| module6.py | 1547 行核心逻辑 | **已删除**，拆分为 logistics_execution/ 下 8 个子模块 |
| 向后兼容 | 通过 shim 文件 re-export | 通过 `__init__.py` 别名 (`module1 = demand_planning`) |
| normalize | 每个模块独立实现（6-7 处重复） | 统一由 `src/utils/normalization.py` 提供，各处 re-export |
| 配置参数 | 硬编码在各模块 constants.py | 从 `src/config/default_config.yaml` 集中读取 |
| CLI 支持 | module5.py 内嵌 `__main__` | 统一通过 `run.py` 入口 |
