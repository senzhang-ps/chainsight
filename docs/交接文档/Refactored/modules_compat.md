# `src/modules` 兼容层模块文档

本文档覆盖 `src/modules/` 目录下的 5 个兼容层包装器文件：`module1.py`、`module3.py`、`module4.py`、`module5.py`，以及核心物流执行模块 `module6.py`。

---

## 目录

1. [概述](#概述)
2. [module1.py — 需求规划兼容层](#module1py--需求规划兼容层)
3. [module3.py — MRP 规划兼容层](#module3py--mrp-规划兼容层)
4. [module4.py — 生产规划兼容层](#module4py--生产规划兼容层)
5. [module5.py — 部署规划兼容层](#module5py--部署规划兼容层)
6. [module6.py — 物流执行模块](#module6py--物流执行模块)

---

## 概述

`module1.py`、`module3.py`、`module4.py`、`module5.py` 均为**向后兼容轻量包装层**，其职责是：

- 从对应子包（`demand_planning`、`mrp_planning`、`production_planning`、`deployment_planning`）导入所有公开接口
- 通过 `from <subpackage> import *` 或显式重导出，保证调用方无需修改即可使用
- 部分文件提供额外的向后兼容别名

`module6.py` 是真正的业务实现模块（1547 行），负责每日物流执行仿真。

---

## module1.py — 需求规划兼容层

**文件路径**：`src/modules/module1.py`  
**行数**：167 行  
**角色**：`demand_planning` 子包的薄包装层

### 导出内容

| 类别 | 导出名称 |
|------|---------|
| 常量（6 个） | `DEMAND_ELEMENT_PRIORITY`、`DEFAULT_SAFETY_STOCK_DAYS`、`MAX_FORECAST_HORIZON`、`MIN_ORDER_QTY`、`SHIPMENT_LEAD_TIME`、`SPLIT_TOLERANCE` |
| 规范化函数（公开） | `normalize_material`、`normalize_location`、`normalize_identifiers` |
| 规范化函数（私有，兼容别名） | `_normalize_material`、`_normalize_location`、`_normalize_identifiers` |
| 配置加载 | `load_module1_config`、`validate_module1_config` |
| DPS / 供应选择 | `run_dps`、`select_supply_source` |
| 预测拆分 | `split_forecast` |
| 订单生成 | `generate_orders` |
| 订单消耗 | `consume_orders` |
| 发货计算 | `calculate_shipments` |
| 集成模式 | `run_daily_demand_planning`、`run_demand_planning_module` |
| IO 工具 | `read_module1_inputs`、`write_module1_outputs` |

### 设计说明

- 不包含任何独立业务逻辑；所有实现均在 `demand_planning` 子包中
- 通过重导出模式支持 `from modules.module1 import run_demand_planning_module` 等调用形式
- 详细函数说明请参阅 [modules_demand_planning.md](modules_demand_planning.md)

---

## module3.py — MRP 规划兼容层

**文件路径**：`src/modules/module3.py`  
**行数**：94 行  
**角色**：`mrp_planning` 子包的薄包装层

### 导出内容

所有 `mrp_planning` 子包公开接口，以及以下向后兼容别名：

| 别名 | 指向 |
|------|------|
| `_normalize_location` | `mrp_planning._normalize_location` |
| `_normalize_material` | `mrp_planning._normalize_material` |
| `_normalize_identifiers` | `mrp_planning._normalize_identifiers` |
| `_lookup_moq_rv_three_keys` | `mrp_planning._lookup_moq_rv_three_keys` |
| `_apportion_largest_remainder` | `mrp_planning._apportion_largest_remainder` |
| `_build_ptf_lsk_cache_m3` | `mrp_planning._build_ptf_lsk_cache_m3` |
| `_get_ptf_lsk` | `mrp_planning._get_ptf_lsk` |
| `_compute_root_horizon` | `mrp_planning._compute_root_horizon` |

### 设计说明

- 向后兼容别名主要用于测试代码和遗留脚本，不建议在新代码中直接使用以下划线开头的别名
- 详细函数说明请参阅 [modules_mrp_planning.md](modules_mrp_planning.md)

---

## module4.py — 生产规划兼容层

**文件路径**：`src/modules/module4.py`  
**行数**：229 行  
**角色**：`production_planning` 子包的薄包装层，额外暴露一个公开 API

### 导出内容

#### 子包公开接口（全量重导出）

所有 `production_planning` 子包公开接口。

#### 数据类类型（5 个）

| 类型名 | 说明 |
|--------|------|
| `LineState` | 生产线当前状态快照 |
| `ChangeoverInfo` | 换线事件信息 |
| `PlanRecord` | 生产计划记录 |
| `ExceedRecord` | 产能超出记录 |
| `ValidationIssue` | 验证问题 |

#### 额外公开函数

```python
def analyze_end_of_day_changeover_state(
    plan_df: pd.DataFrame,
    cap_df: pd.DataFrame,
    co_def: dict,
    simulation_date: str,
    rate_map: dict
) -> dict:
```

**功能**：分析仿真日末的换线状态。是内部函数 `_analyze_end_of_day_changeover()` 的公开包装。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `plan_df` | `pd.DataFrame` | 当日生产计划 DataFrame |
| `cap_df` | `pd.DataFrame` | 产能配置 DataFrame |
| `co_def` | `dict` | 换线定义字典 |
| `simulation_date` | `str` | 仿真日期（`YYYY-MM-DD`） |
| `rate_map` | `dict` | 生产速率映射 |

**返回值**：`dict`，包含换线状态分析结果。

### 设计说明

- `analyze_end_of_day_changeover_state()` 是该文件中唯一超出简单重导出的额外 API，提供向后兼容的公开访问入口
- 详细函数说明请参阅 [modules_production_planning.md](modules_production_planning.md)

---

## module5.py — 部署规划兼容层

**文件路径**：`src/modules/module5.py`  
**行数**：157 行  
**角色**：`deployment_planning` 子包的薄包装层，支持 CLI 模式

### 导出内容

所有 `deployment_planning` 子包公开接口。

### CLI 支持

当以 `python -m modules.module5` 或直接执行时，支持以下 4 个命令行参数：

| 参数 | 说明 | 示例 |
|------|------|------|
| `--input` | 输入文件路径 | `--input data/input.xlsx` |
| `--output` | 输出文件路径 | `--output data/output.xlsx` |
| `--sim_start` | 仿真开始日期（`YYYY-MM-DD`） | `--sim_start 2024-01-01` |
| `--sim_end` | 仿真结束日期（`YYYY-MM-DD`） | `--sim_end 2024-12-31` |

**示例调用**：

```bash
python -m modules.module5 \
    --input data/deployment_input.xlsx \
    --output data/deployment_output.xlsx \
    --sim_start 2024-01-01 \
    --sim_end 2024-03-31
```

### 设计说明

- 详细函数说明请参阅 [modules_deployment_planning.md](modules_deployment_planning.md)

---

## module6.py — 物流执行模块

**文件路径**：`src/modules/module6.py`  
**行数**：1547 行  
**角色**：核心物流执行仿真，负责每日装载、发运、记录

### 模块概述

module6 模拟物理货运流程：对每个仿真日期，将 `deployment_planning` 输出的部署计划转换为实际发运记录。核心逻辑为：按路线、按车型对待发货需求进行两轮装载，触发 MDQ/WFR/VFR 规则判断，并记录所有发运、未满足需求、异常等日志。

### 数据流

```
DeploymentPlan (输入)
        │
        ▼
_prepare_data()          ← 验证、去重、构建映射
        │
        ▼
_run_simulation_loop()   ← 按日期迭代
        │
        ▼
_process_daily_demands() ← 收集 pending 需求
        │
        ▼
_process_routes()        ← 按路线遍历（processed_routes set 去重）
        │
        ▼
_process_single_route()  ← 单路线处理，遍历车型
        │
        ▼
_process_truck_type()    ← 两轮装载
   ┌────┴────┐
first pass  second pass（仅在触发条件满足后）
        │
        ▼
_generate_shipment_records()  ← 生成发运记录，更新状态
        │
        ▼
_handle_remaining_demands()   ← 超期需求 → unsat_log
        │
        ▼
_generate_outputs()           ← 写入 6 个 Sheet 到 Excel
```

---

### 主入口函数

#### `run_physical_flow_module()`

```python
def run_physical_flow_module(
    config: dict,
    mode: str = "standalone",
    deployment_plan_df: Optional[pd.DataFrame] = None,
    ...
) -> dict:
```

**功能**：物流执行主入口，支持独立（standalone）和集成（integrated）两种模式。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | `dict` | 全局配置字典 |
| `mode` | `str` | `"standalone"` 或 `"integrated"` |
| `deployment_plan_df` | `Optional[pd.DataFrame]` | 集成模式下直接传入的部署计划 |

**返回值**：`dict`，包含 7 个字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `delivery_plan` | `pd.DataFrame` | 交付计划 |
| `vehicle_log` | `pd.DataFrame` | 车辆日志（带列名的空表，结构保留） |
| `truck_usage` | `pd.DataFrame` | 车辆使用统计（groupby 汇总） |
| `unsatisfied_log` | `pd.DataFrame` | 未满足需求日志 |
| `validation_log` | `pd.DataFrame` | 验证日志 |
| `bypass_log` | `pd.DataFrame` | MDQ bypass 规则命中日志 |
| `statistics` | `dict` | 汇总统计信息 |

**别名**：`main = run_physical_flow_module`（文件末尾第 1535 行）

---

#### `run_daily_physical_flow()`

```python
def run_daily_physical_flow(
    simulation_date: str,
    config: dict,
    shared_state: dict
) -> dict:
```

**功能**：集成模式的日度入口，由 Orchestrator 每日调用。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `simulation_date` | `str` | 当日日期（`YYYY-MM-DD`） |
| `config` | `dict` | 全局配置 |
| `shared_state` | `dict` | 跨模块共享状态（包含 deployment plan 等） |

**返回值**：`dict`，同 `run_physical_flow_module()` 的返回结构。

---

### 内部函数详解

#### `_initialize_run_params(config)` → `dict`

初始化运行参数，合并全局配置默认值。

**行范围**：216–256

---

#### `_init_integrated_params(config, shared_state)` → `dict`

集成模式下从 `shared_state` 提取参数（部署计划、库存状态等）。

**行范围**：259–287

---

#### `_init_standalone_params(config)` → `dict`

独立模式下从文件读取输入数据。

**行范围**：290–331

---

#### `_prepare_data(params)` → `dict`

验证输入数据、去重、构建各类查找映射。

**行范围**：334–410

**构建的映射**：
- `priority_map`：`demand_element → priority`（由 `_build_priority_map()` 生成）
- `material_map`：`material → {weight, volume}`（由 `_build_material_map()` 生成）
- 路线配置、车型配置等

---

#### `_build_priority_map(demand_config)` → `dict`

```python
priority_map: Dict[str, int]  # demand_element → priority
```

**行范围**：451–459

---

#### `_build_material_map(material_config)` → `dict`

```python
material_map: Dict[str, dict]  # material → {"weight": float, "volume": float}
```

**行范围**：462–468

---

#### `_prepare_deployment_plan(df)` → `pd.DataFrame`

对部署计划进行稳定排序并生成唯一标识符（UID）。

**行范围**：542–583

**UID 格式**：`{material}_{location}_{demand_element}_{date}_{index}`

---

#### `_run_simulation_loop(params, state)` → `None`

主仿真循环，按日期迭代调用每日处理流程。

**行范围**：626–673

---

#### `_init_aggregation_status(deployment_plan_df)` → `dict`

初始化每个 UID 的聚合状态，用于追踪装载进度。

**行范围**：676–695

**状态结构**：

```python
agg_status[uid] = {
    "remaining_qty": float,    # 剩余待发数量
    "deployed_qty": float,     # 已部署数量
    "waiting_days": int,       # 已等待天数
    "status": str              # "pending" | "shipped" | "unsatisfied"
}
```

---

#### `_process_daily_demands(simulation_date, agg_status, deployment_plan_df)` → `list`

收集当日所有 pending 需求，过滤出跨节点（cross_node）的需求记录。

**行范围**：698–746

---

#### `_collect_pending_demands(agg_status, deployment_plan_df)` → `list`

遍历 `agg_status`，构建 pending 需求列表（含 UID、数量、优先级等）。

**行范围**：749–791

---

#### `_process_routes(pending_demands, params, state)` → `None`

按路线处理 pending 需求，使用 `processed_routes` set 去重，确保每条路线只处理一次。按 `cross_node_sorted`（首次出现顺序）遍历。

**行范围**：800–836

---

#### `_process_single_route(route, truck_types, demands, params, state)` → `None`

对单条路线，遍历所有可用车型进行装载。

**行范围**：839–884

---

#### `_process_truck_type(route, truck_type, demands, params, state)` → `None`

核心装载函数，实现两轮装载机制。

**行范围**：887–990

**两轮装载机制**：

| 轮次 | 目标 | 触发条件 |
|------|------|---------|
| 第一轮（first pass） | 尽量装载，不超容量上限 | 始终执行 |
| 第二轮（second pass） | 贴近 1.0 满载率，跳过已装入的索引 | WFR/VFR 超阈值、MDQ bypass 规则命中、等待超过 `max_wait_days` |

---

#### `_first_pass_loading(demands, capacity, params)` → `Tuple[list, float]`

第一轮装载：按优先级顺序装入，直到不超容量为止。

**行范围**：993–1027

**返回值**：`(loaded_indices, total_weight_or_volume)`

---

#### `_second_pass_loading(demands, capacity, loaded_indices, params)` → `list`

第二轮装载：跳过已装入的索引，尝试填满至接近满载。

**行范围**：1030–1069

---

#### `_build_context(demand, agg_status, params)` → `dict`

构建 MDQ bypass 规则评估所需的上下文变量。

**行范围**：1072–1090

**上下文变量**：

| 变量 | 类型 | 说明 |
|------|------|------|
| `sending` | `str` | 发送节点 |
| `receiving` | `str` | 接收节点 |
| `truck_type` | `str` | 车型 |
| `demand_element` | `str` | 需求元素 |
| `waiting_days` | `int` | 已等待天数 |
| `deployed_qty_ratio` | `float` | `qty_units / mdq`（若 mdq=0 则为 0.0） |
| `exception_MDQ` | `int` | `1` 表示 mdq=0（异常情况） |

---

#### `determine_trigger_cause(context, params)` → `str`

判断本次装载的触发原因。

**触发条件**：
- WFR（重量填充率）超阈值
- VFR（体积填充率）超阈值
- MDQ bypass 规则命中
- 等待天数超过 `max_wait_days`

**返回值**：`str`，触发原因描述，如 `"WFR"`, `"VFR"`, `"MDQ_BYPASS"`, `"MAX_WAIT"`

---

#### `_generate_shipment_records(loaded_demands, truck_type, route, simulation_date, params, state)` → `list`

生成发运记录，使用批量/单条采样交付延迟（≥10 条用 `batch_sample_delivery_delays_duckdb()`，否则用 `sample_delivery_delay()`），并更新 `agg_status`。

**行范围**：1093–1194

---

#### `_handle_remaining_demands(simulation_date, agg_status, params, state)` → `None`

处理超过 `max_wait_days` 的需求：记录到 `unsat_log`，并将状态清零。

**行范围**：1197–1230

---

#### `_enforce_shipment_constraint()` *(已禁用)*

**行范围**：1233–1312

按比例裁剪出货量使其不超订单量。**已注释禁用**，不参与实际执行。

---

#### `_validate_shipment_delivery_constraint()` *(仅验证)*

**行范围**：1315–1381

验证出货量是否超出订单约束，仅报告不裁剪。

---

#### `_generate_outputs(state, params)` → `dict`

构建 6 个 DataFrame 并写入 Excel 文件。

**行范围**：1384–1457

**6 个输出 Sheet**：

| Sheet 名 | 来源函数 | 说明 |
|---------|---------|------|
| `DeliveryPlan` | `_build_delivery_plan_df()` | 交付计划，含发运日期、到达日期、数量 |
| `VehicleLog` | `_build_vehicle_df()` | 车辆日志（结构保留，实际为空表带列名） |
| `TruckUsageLog` | `_build_usage_df()` | 按路线、车型的 groupby 使用统计 |
| `UnsatisfiedMDQLog` | 直接从 `unsat_log` 构建 | 未满足 MDQ 需求记录 |
| `ValidationLog` | 直接从 `validation_log` 构建 | 约束验证报告 |
| `BypassRuleHitLog` | 直接从 `bypass_log` 构建 | MDQ bypass 规则命中记录 |

---

#### `_write_excel_output(output_path, sheets_dict)` → `None`

将 6 个 Sheet 写入同一个 Excel 文件。

**行范围**：1515–1531

---

### DeliveryPlan 输出列结构

| 列名 | 类型 | 说明 |
|------|------|------|
| `date` | `str` | 发运日期 |
| `material` | `str` | 物料编号 |
| `sending` | `str` | 发送节点 |
| `receiving` | `str` | 接收节点 |
| `truck_type` | `str` | 车型 |
| `demand_element` | `str` | 需求元素 |
| `qty_units` | `float` | 发运数量（单位） |
| `weight` | `float` | 重量 |
| `volume` | `float` | 体积 |
| `delivery_date` | `str` | 预计到达日期 |
| `trigger_cause` | `str` | 触发原因 |
| `uid` | `str` | 唯一标识符 |

---

### UnsatisfiedMDQLog 输出列结构

| 列名 | 类型 | 说明 |
|------|------|------|
| `date` | `str` | 超期日期 |
| `material` | `str` | 物料编号 |
| `sending` | `str` | 发送节点 |
| `receiving` | `str` | 接收节点 |
| `demand_element` | `str` | 需求元素 |
| `remaining_qty` | `float` | 未发出的剩余数量 |
| `waiting_days` | `int` | 已等待天数 |
| `uid` | `str` | 唯一标识符 |

---

### 配置参数说明

| 配置键 | 类型 | 说明 |
|--------|------|------|
| `max_wait_days` | `int` | 最大等待天数，超出则记录未满足需求 |
| `wfr_threshold` | `float` | 重量填充率触发阈值 |
| `vfr_threshold` | `float` | 体积填充率触发阈值 |
| `mdq_bypass_rules` | `list[dict]` | MDQ bypass 规则列表 |
| `truck_types` | `dict` | 车型配置（载重、容积等） |
| `routes` | `list[dict]` | 路线配置（发送节点、接收节点、车型列表） |

---

### 依赖关系

```
module6.py
├── src/modules/logistics_execution/      ← 实际实现（若存在子包结构）
├── src/utils/logger_config.py            ← 日志
├── src/utils/time_manager.py             ← 日期管理
├── src/utils/memory_data_store.py        ← 集成模式数据共享
└── src/utils/simulation_cache.py         ← 缓存
```

---

### 已知设计问题

1. **`_enforce_shipment_constraint()` 已注释禁用**：裁剪逻辑存在但不生效，仅有验证版本 `_validate_shipment_delivery_constraint()` 运行
2. **`VehicleLog` 为空表**：`_build_vehicle_df()` 返回带列名的空 DataFrame，实际车辆级别日志未被填充
