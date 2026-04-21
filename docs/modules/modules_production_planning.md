# `src/modules/production_planning` — 生产计划模块文档

## 文档信息

| 项 | 内容 |
|---|---|
| 文档版本 | v1.1 |
| 最后更新 | 2026-04-10 |
| 编写人 | 陈显跃 |
| 适用范围 | `src/modules/production_planning/` 目录（共 12 个文件） |
| 目标读者 | 算法工程师、测试工程师、业务分析师 |

---

## 模块概述

`src/modules/production_planning/` 是 ChainSight 的 **约束产能生产计划仿真引擎**（Module 4），负责：

1. 读取 Module3 输出的净需求（仅 layer=0）
2. 生成**无约束计划**（审查日才重新生成）
3. 对无约束计划执行**约束产能分配**（含跨天换产延续）
4. **蒙特卡洛生产可靠性仿真**（Binomial 分布）
5. 将产线状态、已分配产能持久化，支持跨日状态恢复

---

## 文件结构

```
src/modules/production_planning/
├── __init__.py               # 公开 API 导出（run_daily_production_planning）
├── constants.py              # 列名常量、工作表映射
├── types.py                  # 核心 dataclass 类型定义
├── config_loader.py          # 配置加载与校验
├── demand_loader.py          # 净需求数据加载
├── utils.py                  # 通用工具函数
├── state_manager.py          # 状态持久化（产线状态、已分配产能）
├── plan_builder.py           # 无约束计划构建 + 换产序列优化
├── capacity_allocator.py     # 约束产能分配（核心，1086行）
├── main.py                   # 主流程编排 + 命令行入口（600行）
├── output_writer.py          # Excel 输出管理
└── duckdb_batch_calculator.py # DuckDB 生产仿真加速
```

---

## 主要文件说明

### `constants.py`

定义所有列名常量和配置映射。

| 常量 | 类型 | 说明 |
|---|---|---|
| `IDENTIFIER_COLS` | `List[str]` | 标识符列：material/location/line/delegate_line/from_material/to_material |
| `DEFAULT_CHANGEOVER_TIME` | `float` | 默认换产时间：24.0 小时 |
| `PLAN_COLUMNS` | `List[str]` | 生产计划表 13 列 |
| `EXCEED_COLUMNS` | `List[str]` | 产能超额表 6 列 |
| `VALIDATION_COLUMNS` | `List[str]` | 校验结果表 12 列 |
| `CHANGEOVER_LOG_COLUMNS` | `List[str]` | 换产日志表 8 列 |
| `UNCONSTRAINED_PLAN_COLUMNS` | `List[str]` | 无约束计划表 7 列 |
| `REQUIRED_CONFIG_SHEETS` | `List[str]` | 必需的配置工作表名称列表（6 个） |
| `SHEET_KEY_MAPPING` | `dict` | 工作表名 → 内部字典 key 映射 |

**生产计划表 (`PLAN_COLUMNS`) 完整列名：**
`material, location, line, simulation_date, production_plan_date, available_date, uncon_planned_qty, con_planned_qty, produced_qty, changeover_id, changeover_time, changeover_time_remaining, is_first_changeover_day`

**必需配置工作表：**
`M4_MaterialLocationLineCfg, M4_LineCapacity, M4_ChangeoverMatrix, M4_ChangeoverDefinition, M4_ProductionReliability, Global_DemandPriority`

---

### `types.py`

定义模块内使用的所有核心数据类。

#### `@dataclass LineState`

产线当日结束后的持久化状态。

| 字段 | 类型 | 说明 |
|---|---|---|
| `last_material` | `str` | 最后生产的物料编号 |
| `last_location` | `str` | 地点 |
| `last_production_date` | `str` | 最后生产日期（`YYYY-MM-DD`） |
| `last_activity` | `str` | 最后活动类型：`"production"` 或 `"changeover"` |
| `changeover_info` | `Optional[ChangeoverInfo]` | 若最后活动为换产，记录换产详情 |

方法：`to_dict() -> dict`，`from_dict(d: dict) -> LineState`（类方法）

#### `@dataclass ChangeoverInfo`

换产过程的详细信息。

| 字段 | 类型 | 说明 |
|---|---|---|
| `changeover_id` | `str` | 换产类型 ID |
| `from_material` | `str` | 换产前物料 |
| `to_material` | `str` | 换产后物料 |
| `total_time` | `float` | 换产总时长（小时） |
| `completed_time` | `float` | 已完成时长 |
| `remaining_time` | `float` | 剩余时长（跨天延续用） |

#### `@dataclass PlanRecord`

单条生产计划记录。

| 字段 | 类型 | 说明 |
|---|---|---|
| `material` | `str` | 物料 |
| `location` | `str` | 地点 |
| `line` | `str` | 产线 |
| `simulation_date` | `pd.Timestamp` | 仿真日期 |
| `production_plan_date` | `pd.Timestamp` | 计划生产日期 |
| `available_date` | `pd.Timestamp` | 可用日期（生产完+MCT） |
| `uncon_planned_qty` | `int` | 无约束计划量 |
| `con_planned_qty` | `int` | 约束后计划量 |
| `produced_qty` | `Optional[int]` | 仿真实际产出量 |
| `changeover_id` | `Optional[str]` | 关联换产 ID |

#### `@dataclass ExceedRecord`

产能超额记录。

| 字段 | 类型 | 说明 |
|---|---|---|
| `material` | `str` | 物料 |
| `location` | `str` | 地点 |
| `line` | `str` | 产线 |
| `simulation_date` | `pd.Timestamp` | 仿真日期 |
| `production_plan_date` | `pd.Timestamp` | 超额发生日期 |
| `unmet_uncon_planned_qty` | `int` | 未满足的无约束计划量 |

#### `@dataclass ValidationIssue`

配置或计划校验问题记录。

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | `str` | 问题类型（如 `"config"`, `"capacity_validation"`） |
| `issue` | `str` | 问题描述 |
| `sheet` | `Optional[str]` | 相关工作表 |
| `row` | `Optional[int]` | 相关行号 |

---

### `config_loader.py`

#### `load_config(config_file: str) -> dict`

加载 Module4 配置 Excel 文件，返回工作表字典。

| 参数 | 类型 | 说明 |
|---|---|---|
| `config_file` | `str` | 配置 Excel 文件路径 |

**返回值：** `dict`，key 为内部键（`SHEET_KEY_MAPPING` 映射后），value 为 `pd.DataFrame`。

主要 key：`MaterialLocationLineCfg, LineCapacity, ChangeoverMatrix, ChangeoverDefinition, ProductionReliability, NetDemandTypePriority`

#### `validate_config(cfg: dict) -> List[dict]`

校验配置完整性，检查：
- 必需工作表是否存在
- 关键列是否缺失
- 产率、产能数值是否合理

**返回值：** `List[dict]`，每条为一个 `ValidationIssue` 对应的字典。

---

### `demand_loader.py`

#### `load_daily_net_demand(module3_output_dir: str, simulation_date: pd.Timestamp) -> pd.DataFrame`

从 Module3 输出目录加载**前一天**的净需求数据，**仅筛选 layer=0** 的记录。

| 参数 | 类型 | 说明 |
|---|---|---|
| `module3_output_dir` | `str` | Module3 日度输出目录 |
| `simulation_date` | `pd.Timestamp` | 当前仿真日期（读取的是 `simulation_date - 1 day` 的文件） |

**返回值：** `pd.DataFrame`，列包含 `material, location, requirement_date, total_gap, ao_gap, fc_gap, ss_gap, layer`。

**文件命名规则：** `Module3Output_{YYYYMMDD}.xlsx`（读取前一天日期）

---

### `utils.py`

#### `normalize_location(location: Any) -> str`

标准化地点编号。

#### `cast_identifiers_to_str(df: pd.DataFrame, cols: list) -> pd.DataFrame`

将 DataFrame 中指定列强制转换为字符串类型。

#### `compute_planning_window(simulation_date: pd.Timestamp, simulation_start: pd.Timestamp, lsk: int) -> Tuple[pd.Timestamp, pd.Timestamp]`

计算生产计划窗口的起始/结束日期。

| 参数 | 类型 | 说明 |
|---|---|---|
| `simulation_date` | `pd.Timestamp` | 当前仿真日期 |
| `simulation_start` | `pd.Timestamp` | 仿真开始日期 |
| `lsk` | `int` | LSK 周期数 |

**返回值：** `Tuple[pd.Timestamp, pd.Timestamp]`，`(window_start, window_end)`

#### `is_review_day(simulation_date: pd.Timestamp, simulation_start: pd.Timestamp, lsk: int) -> bool`

判断当前仿真日期是否为**审查日**。

**公式：** `(simulation_date - simulation_start).days % lsk == 0`

无约束计划仅在审查日重新生成，其他日期使用上次审查日生成的计划。

#### `round_up_to_batch(quantity: float, batch_size: int) -> int`

将数量向上舍入到批次大小的整数倍。

#### `ensure_dataframe_columns(df: pd.DataFrame, required_cols: list) -> pd.DataFrame`

确保 DataFrame 包含所有必需列，缺失列填充为 NaN。

#### `dedup_issues(issues: List[dict]) -> List[dict]`

对问题列表去重（按 `issue` 字段）。

---

### `state_manager.py`

跨日状态持久化管理，所有状态文件保存于 `output_dir/`。

**文件命名规则：**
- 产线状态：`line_states_{YYYYMMDD}.json`
- 已分配产能：`allocated_capacity_{YYYYMMDD}.json`
- 仿真起始日期：`simulation_start.txt`

#### `get_or_init_simulation_start(output_dir: str, requested_start: Optional[pd.Timestamp]) -> pd.Timestamp`

获取或初始化仿真起始日期。若 `simulation_start.txt` 已存在则读取，否则使用 `requested_start` 并保存。

#### `save_line_state(output_dir: str, simulation_date: pd.Timestamp, line_states: dict) -> None`

将产线状态字典序列化为 JSON 并保存。

#### `load_line_state(output_dir: str, simulation_date: pd.Timestamp) -> dict`

加载**前一天**（`simulation_date - 1 day`）的产线状态。若文件不存在返回空字典。

#### `save_allocated_capacity(output_dir: str, simulation_date: pd.Timestamp, capacity: dict) -> None`

保存当日已分配产能记录（key 格式：`"{location}|{line}|{date}"`）。

#### `load_allocated_capacity(output_dir: str, date: pd.Timestamp) -> dict`

加载指定日期的已分配产能记录。

#### `load_all_previous_capacity(output_dir: str, simulation_date: pd.Timestamp) -> dict`

加载 simulation_date 之前**所有日期**的已分配产能，合并为单一字典（用于跨日产能校验）。

---

### `plan_builder.py`

无约束计划构建与换产序列优化。

#### `build_unconstrained_plan_for_single_day(net_demand: pd.DataFrame, mlcfg: pd.DataFrame, simulation_date: pd.Timestamp, simulation_start: pd.Timestamp, issues: list) -> pd.DataFrame`

为仿真日期构建无约束生产计划。

| 参数 | 类型 | 说明 |
|---|---|---|
| `net_demand` | `pd.DataFrame` | Module3 输出净需求（layer=0） |
| `mlcfg` | `pd.DataFrame` | 物料-地点-产线配置 |
| `simulation_date` | `pd.Timestamp` | 当前仿真日期 |
| `simulation_start` | `pd.Timestamp` | 仿真起始日期 |
| `issues` | `list` | 问题收集列表（会被修改） |

**关键逻辑：**
- 仅在 `is_review_day()` 返回 True 时重新生成计划
- 非审查日：从上次审查日的输出文件中读取计划直接使用
- 为每个 (material, location) 对找匹配产线，调用 `optimal_changeover_sequence()` 排序

**返回值：** `pd.DataFrame`，列：`material, location, line, planned_date, uncon_planned_qty, simulation_date, original_quantity`

#### `optimal_changeover_sequence(batches: List[dict], co_matrix: pd.Series) -> List[dict]`

用**贪心算法**计算最小换产时间的生产序列。

| 参数 | 类型 | 说明 |
|---|---|---|
| `batches` | `List[dict]` | 待排序的生产批次列表，每个含 `material, quantity` |
| `co_matrix` | `pd.Series` | 换产矩阵，index=`(from_material, to_material)`，value=`changeover_id` |

**排序算法：**
1. 首件选**最大量**的物料（最优先生产量最大的）
2. 后续每件选**最小换产时间**的物料（并列时选数量最大者打破平局）
3. 时间复杂度：O(n²)

**返回值：** 重排序后的批次列表。

---

### `capacity_allocator.py`

约束产能分配核心模块（1086行）。

#### 函数 `centralized_capacity_allocation_with_changeover(...) -> Tuple[pd.DataFrame, pd.DataFrame]`

主入口：对无约束计划执行约束产能分配（含换产处理）。

| 参数 | 类型 | 说明 |
|---|---|---|
| `uncon_plan` | `pd.DataFrame` | 无约束计划 |
| `cap_df` | `pd.DataFrame` | 产线产能数据（含 date/line/capacity 列） |
| `rate_map` | `pd.Series` | 产率映射，index=`(material, line)`，value=hours/unit |
| `co_mat` | `pd.Series` | 换产矩阵，index=`(from, to)`，value=changeover_id |
| `co_def` | `dict` | 换产定义，key=`(changeover_id, line)`，value=time(h) |
| `mlcfg` | `pd.DataFrame` | 物料-地点-产线配置 |
| `previous_line_states` | `dict` | 前日产线状态（从 state_manager 加载） |
| `simulation_date` | `pd.Timestamp` | 仿真日期 |
| `previously_allocated_capacity` | `dict` | 历史已分配产能 |
| `issues` | `list` | 问题收集列表 |

**返回值：** `Tuple[pd.DataFrame, pd.DataFrame]`，`(plan_log, exceed_log)`

#### 类 `CapacityAllocator`

产能分配的核心状态机。

```python
class CapacityAllocator:
    cap_map: Dict[tuple, float]          # (location, line, date) → 剩余产能
    rate_map: dict                        # (material, line) → 产率
    mct_map: dict                         # (material, location) → MCT
    previously_allocated: dict            # 历史已分配产能
    has_location: bool                    # 产能数据是否含 location 列
```

##### `__init__(cap_df, rate_map, mct_map, previously_allocated)`

初始化并构建 `cap_map`（从产能 DataFrame 构建快速查找字典）。

##### `allocate_batch(line, sim_date, batch, location, material, window_start, window_end, changeover) -> Tuple[List[Dict], Optional[Dict]]`

对单个批次在给定时间窗口内逐日分配产能。

| 参数 | 说明 |
|---|---|
| `batch` | 包含 `uncon_planned_qty` 的批次字典 |
| `changeover` | 包含 `remain, coid, is_first` 的换产状态字典 |
| `window_start/end` | 生产窗口起止日期 |

**逐日处理逻辑：**
1. 查 `cap_map` 获取当日剩余产能
2. 扣减跨天换产剩余时间（`_consume_changeover()`）
3. 换产期间无法生产（仅消耗产能时间）
4. 换产完成后按产率计算可生产量（`min(prod_remain, today_cap * rate)`）
5. 窗口结束仍有剩余 → 生成 `exceed` 记录

**返回值：** `(plans_list, exceed_dict_or_None)`

##### `_allocate_day(...) -> Dict`

单日产能分配的内部实现，返回包含 `plan, prod_remain, co_remain, coid_to_log, is_first_co_day` 的状态字典。

##### `_consume_changeover(co_remain, today_cap) -> Tuple[float, float, float, bool]`

从当日产能中消耗换产时间。

**返回值：** `(co_used, co_remain, today_cap_after, is_completed)`

##### `_adjust_for_previous_allocation(current_cap, location, line, day_dt) -> float`

从当日名义产能中扣减历史已分配产能，返回实际可用产能。

---

#### 模块级函数

##### `extract_allocated_capacity_from_plan(plan_df, rate_map, changeover_def) -> Dict[str, float]`

从生产计划反算各（地点、产线、日期）组合已消耗的产能小时数，用于持久化。

**key 格式：** `"{location}|{line}|{YYYY-MM-DD}"`

##### `validate_capacity_allocation(plan_log, previously_allocated, simulation_date, rate_map, changeover_def) -> List[dict]`

校验当日计划与历史产能分配是否冲突，返回校验问题列表。

##### `extract_line_states_from_plan(plan_df, cap_df, co_def, simulation_date, rate_map) -> Dict[str, Any]`

从计划日志提取当日末的各产线状态，调用 `_analyze_end_of_day_changeover()` 推断是否有未完成换产。

##### `_analyze_end_of_day_changeover(plan_df, cap_df, co_def, simulation_date, rate_map) -> Dict[str, Any]`

分析日末换产状态：若日末剩余产能 ≈ 典型换产时间（±0.1h），则推断该产线有未完成换产（`changeover_id="INFERRED_INCOMPLETE"`）。

##### `calculate_changeover_metrics(production_plan, changeover_def) -> pd.DataFrame`

从生产计划计算换产汇总指标（count、time、cost、mu_loss）。

**返回列：** `date, location, line, changeover_type, count, time, cost, mu_loss`

##### `simulate_production(plan, pr_cfg, seed) -> pd.DataFrame`

**蒙特卡洛生产可靠性仿真**：对每条计划记录用 Binomial 分布模拟实际产出。

| 参数 | 类型 | 说明 |
|---|---|---|
| `plan` | `pd.DataFrame` | 含 `con_planned_qty` 的计划表 |
| `pr_cfg` | `pd.DataFrame` | 生产可靠性配置（含 `location, line, pr` 列） |
| `seed` | `Optional[int]` | 随机种子（默认来自配置，通常为 42） |

**公式：** `produced_qty = Binomial(con_planned_qty, pr)`，其中 `pr` 为产线可靠率（0~1）

**重要限制：** 不能对 plan 排序后再仿真，排序会改变随机数分配顺序导致结果不可复现。

优先使用 DuckDB 加速版（`simulate_production_batch_duckdb()`），失败时 fallback。

---

### `main.py`

主流程编排类和命令行入口。

#### 函数 `run_daily_production_planning(config_file, module3_output_dir, simulation_date, simulation_start, output_dir) -> str`

模块对外公开的主调用函数（在 `__init__.py` 中导出）。

| 参数 | 类型 | 说明 |
|---|---|---|
| `config_file` | `str` | 配置 Excel 文件路径 |
| `module3_output_dir` | `str` | Module3 日度输出目录 |
| `simulation_date` | `pd.Timestamp` | 当前仿真日期 |
| `simulation_start` | `pd.Timestamp` | 仿真起始日期 |
| `output_dir` | `str` | 输出目录 |

**返回值：** `str`，输出文件路径。

#### 类 `DailyProductionPlanner`

日度计划流程的编排类，封装完整执行管道。

```python
class DailyProductionPlanner:
    config_file: str
    module3_output_dir: str
    simulation_date: pd.Timestamp
    simulation_start: pd.Timestamp
    output_dir: str
    issues: List[dict]     # 校验问题累积列表
```

##### `run() -> str`

执行完整生产计划流程，返回输出文件路径。

**执行管道：**
```
1. _load_and_validate_config()   → 加载并校验配置
2. _load_net_demand()            → 加载前一天 layer=0 净需求
3. _prepare_config()             → 标准化 mlcfg 标识符
4. _build_unconstrained_plan()   → 构建无约束计划（审查日才重新生成）
5. _allocate_capacity()          → 约束产能分配（含跨天换产）
6. _simulate_and_finalize()      → 生产可靠性仿真 + 产能校验
7. calculate_changeover_metrics()→ 计算换产指标
8. _save_states()                → 持久化产线状态 + 已分配产能
9. _write_output()               → 写出 Excel 输出文件
```

##### `_build_changeover_matrix(cfg) -> pd.Series`

从配置构建换产矩阵 Series，index=`(from_material, to_material)`，排序后返回。

#### 命令行入口 `main()`

支持两种执行模式：

**Daily 模式（推荐）：**
```bash
python -m src.modules.production_planning.main \
    --config config.xlsx \
    --mode daily \
    --module3_output_dir ./output/module3/ \
    --simulation_date 2024-01-15 \
    --output_dir ./output/module4/
```

**Legacy 模式（兼容旧版）：**
```bash
python -m src.modules.production_planning.main \
    --config config.xlsx \
    --mode legacy \
    --input input.xlsx \
    --sim_start 2024-01-01 \
    --sim_end 2024-03-31 \
    --output output.xlsx
```

Legacy 模式直接读取配置中的 `NetDemand` 工作表，不依赖 Module3 输出。

---

### `output_writer.py`

#### `write_output(plan, exc, issues, changeover_log, out_path, simulation_date, skip_file_output) -> str`

写出 Module4 的 Excel 输出文件，支持 DuckDB 内存模式。

| 参数 | 类型 | 说明 |
|---|---|---|
| `plan` | `pd.DataFrame` | 生产计划 |
| `exc` | `pd.DataFrame` | 产能超额记录 |
| `issues` | `List[dict]` | 校验问题列表 |
| `changeover_log` | `pd.DataFrame` | 换产日志 |
| `out_path` | `str` | 基础输出路径 |
| `simulation_date` | `Optional[pd.Timestamp]` | 仿真日期（有则自动加日期后缀） |
| `skip_file_output` | `bool` | True = 仅写内存存储，跳过磁盘 |

**输出文件路径规则：** `{dir}/{base}_{YYYYMMDD}.xlsx`（有 simulation_date 时）

**输出 Excel 工作表：**

| 工作表 | 内容 |
|---|---|
| `ProductionPlan` | 生产计划（含 produced_qty） |
| `CapacityExceed` | 产能超额记录 |
| `Validation` | 配置校验问题 |
| `ChangeoverLog` | 换产指标汇总 |

#### `generate_consolidated_output(daily_output_files, output_path) -> None`

合并多个每日输出文件为单一汇总 Excel（用于多日仿真结果聚合）。

---

### `duckdb_batch_calculator.py`

#### `simulate_production_batch_duckdb(plan, pr_cfg, seed, run_id) -> pd.DataFrame`

DuckDB 加速版生产仿真，实际使用 NumPy 向量化 Binomial 采样（替代逐行 apply）。

| 参数 | 类型 | 说明 |
|---|---|---|
| `plan` | `pd.DataFrame` | 生产计划 |
| `pr_cfg` | `pd.DataFrame` | 生产可靠性配置 |
| `seed` | `Optional[int]` | 随机种子 |
| `run_id` | `Optional[str]` | 运行 ID（用于性能统计） |

**前提条件（任一不满足则降级）：**
- `DUCKDB_INTEGRATION_AVAILABLE = True`（duckdb_integration 可导入）
- `DuckDBConfig.enabled = True`
- `len(plan) >= DuckDBConfig.min_rows_threshold`

**核心实现：**
```python
rng = np.random.RandomState(seed)
produced = np.array([rng.binomial(int(n), float(p)) for n, p in zip(n_vals, p_vals)])
```

#### `_simulate_production_pandas(plan, pr_cfg, seed, run_id) -> pd.DataFrame`

Pandas fallback 版本，与主模块中的 `simulate_production()` 逻辑相同。

#### `is_duckdb_available() -> bool`

检查 DuckDB 集成是否可用且已启用。

#### `get_duckdb_config() -> dict`

返回 DuckDB 配置状态（`available, enabled`）。

---

## 数据结构

### 产线状态 JSON 格式

```json
{
    "LINE_A": {
        "last_material": "MAT001",
        "last_location": "PLANT01",
        "last_production_date": "2024-01-15",
        "last_activity": "changeover",
        "changeover_info": {
            "changeover_id": "CO_TYPE_1",
            "from_material": "MAT001",
            "to_material": "MAT002",
            "total_time": 4.0,
            "completed_time": 2.5,
            "remaining_time": 1.5
        }
    }
}
```

### 已分配产能 JSON 格式

```json
{
    "PLANT01|LINE_A|2024-01-15": 7.5,
    "PLANT01|LINE_A|2024-01-16": 3.2,
    "PLANT01|LINE_B|2024-01-15": 8.0
}
```

### `MaterialLocationLineCfg` 数据表关键列

| 列名 | 类型 | 说明 |
|---|---|---|
| `material` | `str` | 物料编号 |
| `location` | `str` | 地点 |
| `delegate_line` | `str` | 代理产线（实际产线） |
| `prd_rate` | `float` | 产率（unit/hour） |
| `lsk` | `int` | 周期数（review cycle） |
| `mct` | `int` | 最小周期时间（天） |

---

## 依赖关系

```
main.py (DailyProductionPlanner)
    ├── config_loader.py         (load_config, validate_config)
    ├── demand_loader.py         (load_daily_net_demand)
    ├── utils.py                 (cast_identifiers_to_str, is_review_day)
    ├── plan_builder.py          (build_unconstrained_plan_for_single_day)
    │       └── utils.py        (compute_planning_window, is_review_day)
    ├── capacity_allocator.py    (centralized_capacity_allocation_with_changeover)
    │       ├── types.py        (LineState, ChangeoverInfo)
    │       └── duckdb_batch_calculator.py (simulate_production_batch_duckdb)
    ├── state_manager.py         (save/load_line_state, save/load_allocated_capacity)
    └── output_writer.py         (write_output)
            └── utils/memory_data_store  (DuckDB 内存模式，延迟导入)
```

---

## 数据流图

```
Module3 输出 (前一天 layer=0 净需求)
        │
        ▼
demand_loader.load_daily_net_demand()
        │
        ▼
state_manager.load_line_state()          ← line_states_{昨天}.json
state_manager.load_all_previous_capacity() ← allocated_capacity_*.json（所有历史）
        │
        ▼
plan_builder.build_unconstrained_plan_for_single_day()
    └── [仅审查日] optimal_changeover_sequence() → 换产序列优化
        │
        ▼
capacity_allocator.centralized_capacity_allocation_with_changeover()
    ├── 恢复跨天换产剩余时间（from LineState.changeover_info.remaining_time）
    ├── 按换产序列逐批分配产能
    │   └── CapacityAllocator._allocate_day()（逐日消耗产能）
    └── 产能不足 → ExceedRecord
        │
        ▼
capacity_allocator.simulate_production()  (Binomial 仿真)
    └── [有 DuckDB] simulate_production_batch_duckdb()
        │
        ▼
calculate_changeover_metrics()    → 换产指标（count/time/cost/mu_loss）
validate_capacity_allocation()    → 产能校验（检测历史冲突）
        │
        ▼
state_manager.save_line_state()          → line_states_{今天}.json
state_manager.save_allocated_capacity() → allocated_capacity_{今天}.json
        │
        ▼
output_writer.write_output()
        │
        ▼
Module4Output_{YYYYMMDD}.xlsx
    ├── ProductionPlan     (con_planned_qty + produced_qty)
    ├── CapacityExceed     (未满足需求)
    ├── Validation         (配置/产能校验问题)
    └── ChangeoverLog      (换产汇总指标)
```

---

## 关键设计决策

| 决策 | 原因 |
|---|---|
| 仅读取前一天 Module3 输出 | Module4 是日度驱动，当天运行时 Module3 还未产出当天结果 |
| 无约束计划仅在审查日生成 | LSK 周期内需求变化小，频繁重计划会造成排产震荡 |
| 换产序列首件选最大量 | 最大量物料往往是关键路径，优先排产减少整体延误 |
| 产线状态持久化 JSON | 换产可能跨越多天（如 24h 换产），必须在日度边界记录未完成状态 |
| Binomial 仿真不排序 | 随机数分配顺序与原始顺序绑定，排序会破坏结果可复现性 |
| already_allocated 跨日累积 | 防止同一产线产能在多次仿真运行中被重复分配 |
| DuckDB 延迟导入 | 避免循环导入，内存模式为可选功能不影响主流程 |
