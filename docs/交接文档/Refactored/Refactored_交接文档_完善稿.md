# project_cs_pg Refactored 交接文档
**文档版本**：v3.0  
**更新日期**：2026-04-14  
**适用范围**：`ChainSight / project_cs_pg` 当前 `src/` 本地模式与 `--use-db` 数据库模式  
**整理依据**：`项目交接文档.md`、`ARCHITECTURE.md`、`API.md`、`core.md`、`module.md`、`services.md`、`setup.md`、`utils.md`、`modules_compat.md`、`函数上下游依赖矩阵.md`、`函数级接口与文件格式总表.md`、`模块级时序图文档.md`、`CLEANUP_AND_IMPROVEMENTS.md`、`BC算法优化测试报告.md`、`OC算法优化测试报告.md`  
**目标读者**：后续研发、测试、运维、交付评审、维护负责人  

---

## 1. 文档目的

这份文档不是对旧版交接稿做措辞修补，而是基于 `Refactored` 目录里的实际交接材料，重新整理一份能够直接交付、能够直接接手、能够直接指导维护的正式版本。

本文件重点回答以下问题：

1. 当前项目的权威结构是什么，哪些旧路径已经失效。
2. 主流程从入口到输出按什么顺序执行，状态在什么位置沉淀。
3. 本轮重构到底优化了什么，为什么要这样优化，不优化会有什么风险。
4. M1、M3、M4、M5、M6 各模块的职责、输入、输出、子文件结构、关键函数分别是什么。
5. 本地模式与数据库模式的边界在哪里，输出表和汇总表怎么理解。
6. BC 与 OC 两类长周期场景的验证结果如何，是否真正实现了零漂移和性能收益。
7. 接手后第一步该看什么、改什么、怎么验证，哪些地方最容易踩坑。

因此，这份文档的定位不是"说明改过哪些代码"，而是提供一个从架构、链路、模块、输出、验证到维护动作的完整交接视角。

---

## 2. 实际交接资料来源

当前 `docs/交接文档/Refactored/` 目录中的资料分工已经比较清楚，可以按以下方式理解：

| 文档 | 作用 |
|---|---|
| `项目交接文档.md` | 总体交接背景、重构方向与交接目标 |
| `ARCHITECTURE.md` | 五层结构、本地版与数据库版整体架构 |
| `API.md` | 外部入口、核心 API、DB 接口与调用协议 |
| `core.md` | `src/core/` 各包职责、主入口、续跑、编排器 |
| `module.md` | M1、M3、M4、M5、M6 模块规格说明 |
| `services.md` | 性能分析与汇总报告生成器 |
| `setup.md` | 环境准备、运行命令、部署和验证步骤 |
| `utils.md` | 默认参数、归一化、缓存、DuckDB、校验工具 |
| `modules_compat.md` | 旧导入路径到当前路径的迁移关系 |
| `函数上下游依赖矩阵.md` | 函数级上下游依赖、排障定位依据 |
| `函数级接口与文件格式总表.md` | 函数签名、输入输出文件格式、运行顺序 |
| `模块级时序图文档.md` | 模块间时序与主要调用关系 |
| `CLEANUP_AND_IMPROVEMENTS.md` | 清理项、收口项、净代码变化与零漂移说明 |
| `BC算法优化测试报告.md` | BC 场景性能与一致性验证 |
| `OC算法优化测试报告.md` | OC 场景性能与一致性验证 |

本文件是在上述资料基础上整合形成，不再重复逐字搬运，而是从交接视角重新编排为"结构 -> 链路 -> 模块 -> 优化 -> 验证 -> 接手动作"的顺序。

---

## 3. 当前项目的权威结构

### 3.1 五层结构

项目建议按五层结构理解：

| 层级 | 权威位置 | 核心职责 |
|---|---|---|
| CLI / 入口层 | `run.py`、`src/core/run/` | 参数解析、模式分发、输出目录选择 |
| Core 编排层 | `src/core/main_integration/`、`src/core/orchestrator/`、`src/core/parallel_executor/` | 主循环、共享状态、续跑、并行执行 |
| Modules 业务层 | `src/modules/demand_planning/`、`mrp_planning/`、`production_planning/`、`deployment_planning/`、`logistics_execution/` | M1、M3、M4、M5、M6 核心业务算法 |
| Services 服务层 | `src/services/` | 性能分析、汇总报表、辅助服务 |
| Utils / Storage 基础层 | `src/utils/`、`config/`、`pgsql_db/` | 默认参数、归一化、缓存、验证、DuckDB、数据库落库 |

### 3.2 当前权威入口

| 领域 | 权威位置 | 说明 |
|---|---|---|
| CLI 总入口 | `run.py` -> `src/core/run/run_main.py::main()` | 仓库根 `run.py` 只做转发 |
| 本地模式主入口 | `src/core/main_integration/simulation_file.py::run_integrated_simulation()` | 从 Excel 配置启动 |
| 数据库模式主入口 | `src/core/main_integration/simulation_db.py::run_integrated_simulation_from_dict()` | 从 `cfg_*` 表恢复 `config_dict` 后启动 |
| M4 集成桥接 | `src/core/main_integration/production_integration.py::run_module4_integrated()` | 把 M4 接入主流程 |
| Orchestrator 构造入口 | `src/core/orchestrator/orchestrator_main.py::create_orchestrator()` | 生成共享状态控制器 |
| 并行框架入口 | `src/core/parallel_executor/parallel_executor_main.py::ParallelExecutor` | 承载无共享写冲突的并行任务 |

### 3.3 已失效的旧入口

以下路径已经不应再作为维护入口：

| 旧路径 | 当前替代路径 |
|---|---|
| `src/core/main_integration.py` | `src/core/main_integration/` |
| `src/core/orchestrator.py` | `src/core/orchestrator/` |
| `src/core/run.py` | `src/core/run/` |
| `src/modules/module1.py` | `src/modules/demand_planning/` |
| `src/modules/module3.py` | `src/modules/mrp_planning/` |
| `src/modules/module4.py` | `src/modules/production_planning/` |
| `src/modules/module5.py` | `src/modules/deployment_planning/` |
| `src/modules/module6.py` | `src/modules/logistics_execution/` |
| `src/core/main_integration/module4_runner.py` | `src/core/main_integration/production_integration.py` |

如果后续维护还继续围绕这些旧路径展开，就会重新回到"目录看起来能改，实际改不到真正主链路"的状态。

---

## 4. 端到端主链路

### 4.1 本地模式主流程

本地模式按以下顺序执行：

1. `run.py`
2. `src/core/run/run_main.py::main()`
3. `src/core/main_integration/simulation_file.py::run_integrated_simulation()`
4. 加载配置、校验、读取随机种子、初始化输出目录
5. `create_orchestrator()` 构造共享状态
6. 按日循环执行：`save_beginning_inventory()` → `cleanup_past_due_open_deployments()` → `_process_delivery_arrivals()` → `load_current_date_production_gr()` → `module1.run_daily_order_generation()` → `process_module1_shipments()` → `run_module4_integrated()` → `process_module4_production()` → `module5.main()` → `process_module5_deployment()` → `module6.run_daily_physical_flow()` → `process_module6_delivery()` → `module3.run_integrated_mode()` → `save_ending_inventory()` → `output_daily_inventory_summary()` → `save_daily_state()`
7. `InventoryBalanceChecker` 做账平验证
8. `SummaryReportGenerator.generate_all_reports()` 生成汇总结果

### 4.2 数据库模式主流程

数据库模式按以下顺序执行：

1. `run.py`
2. `src/core/run/run_main.py::main()`
3. `src/core/run/db_runner.py::_run_with_database()`
4. `DatabaseInitializer.initialize()`
5. `_load_config_from_database()`
6. `run_integrated_simulation_from_dict()`
7. `ModuleDataWriter.write_module_results_from_dict()`
8. `ModuleDataWriter.write_orchestrator_data()`
9. `ModuleDataWriter.generate_summary_reports_from_db()`

### 4.3 主链路的输入、状态与落点

| 阶段 | 核心函数 | 读什么 | 写什么 | 落点 |
|---|---|---|---|---|
| 运行分发 | `main()` | CLI 参数、配置路径、DB 参数 | 输出目录、模式选择 | `outputs/<config>/run_*` |
| 主循环启动 | `run_integrated_simulation()` / `run_integrated_simulation_from_dict()` | Excel / `config_dict` | 初始化 `Orchestrator`，按天驱动模块 | 本地输出或 DB 输出 |
| M1 | `run_daily_order_generation()` | M1 配置、库存、订单日历、预测 | `orders_df`、`shipment_df`、`cut_df`、`supply_demand_df` | `module1_output_YYYYMMDD.xlsx` |
| M1 回写 | `process_module1_shipments()` | M1 发货结果 | 扣减库存、记录发货 | `shipment_log_YYYYMMDD.csv` |
| M4 | `run_module4_integrated()` | M4 配置、M3 输出 | `production_df`、`exceed_log`、`issues_df`、`changeover_log` | `Module4Output_YYYYMMDD.xlsx` |
| M4 回写 | `process_module4_production()` | 生产计划结果 | 生产 GR、backlog、库存增加 | `production_gr_YYYYMMDD.csv` |
| M5 | `module5.main()` | M1、M4 输出，Orchestrator 视图，M5 配置 | `deployment_plan` 等 | `Module5Output_YYYYMMDD.xlsx` |
| M5 回写 | `process_module5_deployment()` | 调拨计划 | `open_deployment` | `open_deployment_YYYYMMDD.csv` |
| M6 | `run_daily_physical_flow()` | open deployment、运力/路线配置、库存 | `delivery_plan`、`vehicle_log`、`truck_usage` 等 | `Module6Output_YYYYMMDD.xlsx` |
| M6 回写 | `process_module6_delivery()` | `DeliveryPlan` | 扣库存、在途、到货、发运日志 | `planning_intransit_YYYYMMDD.csv`、`delivery_gr_YYYYMMDD.csv` |
| M3 | `run_integrated_mode()` | M1 输出、Orchestrator 状态、M3/M4/M5 配置 | `net_demand_df` | `Module3Output_YYYYMMDD.xlsx` |
| 状态固化 | `save_daily_state()` | 当日全部状态 | CSV 快照 | `orchestrator/*.csv` |
| 汇总输出 | `generate_all_reports()` | 每日模块输出与状态 CSV | 汇总报表 | `summary/*.xlsx` / `.csv` |

主链路的核心原则是：

1. 模块只负责返回业务结果，不直接写全局共享状态。
2. 共享状态统一由 `Orchestrator` 的 `process_module*()` 方法写入。
3. 每日结束都固化快照，保证续跑、排障、追踪都能按日恢复。

---

## 5. 本轮优化的方向、原因与具体落点

### 5.1 结构收口

#### 为什么要优化

优化前存在典型的"平铺 wrapper + 包目录并存"问题。`module1.py` 到 `module6.py` 看起来像主模块，但实际只是重导出；`main_integration.py`、`orchestrator.py`、`run.py` 等单体文件已经被同名包目录覆盖，保留它们只会误导接手人。

#### 改前状态

- `src/modules/module1.py`（约 320 行）至 `module6.py`（约 340 行）共 5 个平铺文件，每个文件的内容只是 `from src.modules.demand_planning import *` 等纯重导出
- `src/core/main_integration.py`（约 1500 行）、`src/core/orchestrator.py`（约 1400 行）、`src/core/run.py`（约 600 行）三个单体文件与同名包目录并存，在 Python 中包目录优先级高于同名模块文件，因此这些单体文件实际已成为死代码
- `src/core/main_integration/module4_runner.py`（约 150 行）命名不规范

#### 改后状态

1. 删除 `src/modules/module1.py` 至 `module6.py`（共 5 个文件，约 1690 行）
2. 删除 `src/core/main_integration.py`、`src/core/orchestrator.py`、`src/core/run.py`（共 3 个文件，约 3500 行）
3. 规范 `module4_runner.py` 为 `production_integration.py`
4. 保留 `src/modules/__init__.py` 中的别名导出，兼顾渐进迁移

#### 为什么必须改

1. **维护者改到死代码**：新人看到 `module1.py` 会认为那是主模块而去修改它，实际运行时 Python 加载的是 `demand_planning/` 包，修改完全不生效
2. **Python 包优先级**：当 `src/core/main_integration/` 目录和 `src/core/main_integration.py` 文件同时存在时，`import src.core.main_integration` 始终导入包目录，单体文件被完全忽略
3. **搜索噪音**：IDE 全文搜索时大量重复定义干扰定位

#### 具体收益

1. 文件树与实际导入关系对齐
2. 维护者不再容易改到死代码
3. 搜索与影响分析的噪音显著减少

### 5.2 默认参数集中化

#### 为什么要优化

MOQ、RV、Horizon、Lead Time、Push Levels、Changeover Time 这类参数原本分散在多个模块常量文件中。这样做的问题不是"风格不统一"，而是任何一次参数调整都可能变成跨模块同步修改，极易造成口径漂移。

#### 改前状态

```python
# src/modules/mrp_planning/constants.py
DEFAULT_MOQ = 1   # 硬编码
DEFAULT_RV = 1    # 硬编码

# src/modules/deployment_planning/constants.py
DEFAULT_MOQ = 1   # 又一份硬编码
DEFAULT_RV = 1    # 又一份硬编码
```

同一个参数在 mrp_planning 和 deployment_planning 各定义一次，改一处漏另一处就会产生口径漂移。

#### 改后状态

```yaml
# config/defaults.yaml
shared:
  default_moq: 1
  default_rv: 1
  default_ptf: 0
  default_lsk: 1
  default_lead_time: 1
  default_horizon: 1
production:
  default_changeover_time: 24.0
deployment:
  default_push_levels: [1.2, 1.5, 2.0, 2.5, 3.0]
```

所有模块统一从 `src/utils/defaults.py` 导入：

```python
from src.utils.defaults import DEFAULT_MOQ, DEFAULT_RV
```

#### 当前共享参数

| 常量 | 默认值 | 用途 | 使用方 |
|---|---|---|---|
| `DEFAULT_MOQ` | 1 | 最小起订量 | M3、M5 |
| `DEFAULT_RV` | 1 | Round value | M3、M5 |
| `DEFAULT_PTF` | 0 | 共享前置窗口 | M4、M5 |
| `DEFAULT_LSK` | 1 | 批次规模参数 | M5 |
| `DEFAULT_LEAD_TIME` | 1 | 调拨回退 lead time | M3、M5 |
| `DEFAULT_HORIZON` | 1 | MRP 时间窗口 | M3 |
| `DEFAULT_CHANGEOVER_TIME` | 24.0 | M4 默认换型时长 | M4 |
| `DEFAULT_PUSH_LEVELS` | [1.2,1.5,2.0,2.5,3.0] | M5 push 层级阈值 | M5 |

#### 为什么必须改

修改一个参数只需改一个 YAML 文件，不需要搜索多个 Python 常量文件并逐一同步。YAML 文件还天然适配版本控制和配置管理流程。

#### 具体收益

参数治理从"改多个 Python 常量文件"变成"改一个 YAML 权威数据源（Single Source of Truth）"，更适合持续维护和版本追踪。

### 5.3 标识符归一化统一

#### 为什么要优化

`material`、`location`、`sending`、`receiving` 是项目跨模块联动的关键字段。优化前这些字段有多套类似但不完全一致的归一化逻辑，风险在于：

1. 某些地方去 `.0`，某些地方不去
2. 某些地方补零到四位，某些地方保留原值
3. 空值转字符串时可能被污染成 `'nan'`

这种问题不一定立刻报错，但会让 `merge`、`join`、`groupby` 出现隐蔽偏差。

#### 改前状态

5 处独立的归一化实现分别在：

1. `src/modules/demand_planning/normalization.py`
2. `src/modules/mrp_planning/normalizer.py`
3. `src/modules/deployment_planning/normalizer.py`
4. `src/modules/production_planning/` 内部
5. `src/core/orchestrator/normalize.py`

每处实现的细节略有不同，例如有的对 NaN 值做 `fillna('')` 再 `astype(str)`，有的直接 `astype(str)` 导致 NaN 变成字符串 `'nan'`。

#### 改后状态

新增 `src/utils/normalization.py` 作为权威数据源（Single Source of Truth），所有模块统一导入：

```python
from src.utils.normalization import normalize_identifiers
```

#### 关键函数

| 函数 | 参数 | 作用 |
|---|---|---|
| `normalize_material(val)` | 任意标量值 | 去掉 `.0` 伪小数，如 `'12345.0'` -> `'12345'` |
| `normalize_location(val)` | 任意标量值 | 地点编码补零到四位，如 `'50'` -> `'0050'` |
| `normalize_identifiers(df, extra_columns=None)` | DataFrame + 可选额外列名 | DataFrame 级统一归一化，先 `fillna('')` 再 `astype(str)` |

#### 为什么必须改

关键修复在于 `fillna('')` 的顺序：必须在 `astype(str)` 之前调用，否则 `None` 和 `NaN` 会被转成字符串 `'nan'`，导致后续 join 失败。统一到一处后，修复一次全局生效。

#### 具体收益

后续任何跨模块主键异常，都能优先回到权威数据源排查，不再需要逐个模块寻找历史分叉。

### 5.4 死代码清理

#### 为什么要优化

重构过程中积累了大量"优化尝试"、"设计阶段方案"、"废弃工具模块"，无当前代码引用其中的任何函数，但保留在代码库中会误导接手人认为这些是活跃代码。

#### 删除的文件清单

| 文件 | 约行数 | 原用途 | 删除原因 |
|---|---|---|---|
| `src/utils/duckdb_sql_wrapper.py` | ~330 | 早期 DuckDB SQL 包装器 | 已迭代，当前使用底层 API |
| `src/utils/high_perf_executor.py` | ~330 | 高性能执行器 | 未使用，无反向引用 |
| `src/utils/multiprocess_executor.py` | ~330 | 多进程执行器 | 改为 DuckDB 方案 |
| `src/utils/parallel_optimizer.py` | ~330 | 并行优化器 | 未实现，无反向引用 |
| `src/utils/performance.py` | ~330 | 性能监测 | 迁移至 `services/` |
| `src/utils/process_pool_executor.py` | ~330 | 进程池执行器 | 方案已变 |
| `src/utils/optimization_config.py` | ~320 | 优化配置 | 仅被已删模块引用 |

合计删除约 7 个未使用工具模块，约 2300 行。

#### 代码行数变化总结

| 分类 | 文件数 | 约行数 |
|---|---|---|
| 删除：模块级平铺文件 | 5 | 1,690 |
| 删除：核心层单体文件 | 3 | 3,500 |
| 删除：适配器文件 | 1 | 150 |
| 删除：未使用工具模块 | 7 | 2,300 |
| **小计删除** | **16** | **~7,640** |
| 创建：YAML 配置与加载器 | 2 | 100 |
| 创建：归一化统一实现 | 1 | 250 |
| 创建：Module 6 拆分 | 2 | 680 |
| 创建：回归对比工具 | 1 | 300 |
| **小计创建** | **6** | **~1,330** |
| **净减少** | — | **~6,310** |

### 5.5 Bug 修复详情

本轮重构过程中发现并修复了以下影响结果正确性的 Bug：

#### 5.5.1 CSV 输出覆盖问题

- **现象**：多天仿真时，同名 CSV 文件被后续天覆盖，只保留最后一天的数据
- **根因**：文件写入模式使用 `mode='w'`（覆盖）而非 `mode='a'`（追加）
- **修复**：改为追加写入，确保每天的数据累积

#### 5.5.2 uid_sequence 重置问题

- **现象**：每日调拨 UID 从 1 开始计数，跨天后 UID 重复
- **根因**：`DeploymentUID` 的 sequence 计数器在每日开始时被重置
- **修复**：将 uid_sequence 提升到 Orchestrator 级别，跨天保持连续递增

#### 5.5.3 库存浮点精度问题

- **现象**：库存量出现如 `99.99999999` 之类的浮点误差
- **根因**：多步运算中浮点累积误差
- **修复**：在关键汇总节点做 `int()` 或 `round()` 转换，统一为整数精度

#### 5.5.4 排序稳定性问题

- **现象**：相同数据多次运行输出顺序不一致，导致回归对比困难
- **根因**：`pandas.DataFrame.sort_values()` 默认使用不稳定排序（quicksort）
- **修复**：所有排序统一加 `kind='mergesort'` 保证稳定排序；关键输出添加 `_stable_sort_output()` 函数

#### 5.5.5 数据库精度问题

- **现象**：DB 模式下部分数值字段精度与本地模式不一致
- **根因**：PostgreSQL 写入时浮点数未做精度控制
- **修复**：`create_table_from_df()` 添加 `round_float_values` 参数控制精度

### 5.6 Module6 子包化与职责拆分

#### 为什么要优化

物流执行模块天然复杂，既涉及路线、车型、MDQ、延迟、装载，也涉及输出工作表。如果继续把所有逻辑堆在一个大文件里，后续任何一个运输规则修改都会连带影响输出与调试。

#### 改前状态

Module6 为单体文件 `src/modules/module6.py`，所有逻辑（仿真循环、路线处理、车型装载、输出生成）混在一个文件中。

#### 改后状态

当前 Module6 拆为 11 个子文件，职责清晰分离（详见 6.6 节）。

#### 具体收益

运输规则改动、装载逻辑改动、输出格式改动三类问题可以分离处理，后续维护成本明显降低。

### 5.7 性能优化

#### 为什么要优化

BC 和 OC 两个长周期场景原始耗时过高，影响开发、回归和交付节奏。性能问题主要集中在 M3 和 M5 这些数据量大、规则复杂、循环密集的模块。

#### 具体优化手段

| 优化手段 | 应用位置 | 效果 |
|---|---|---|
| DuckDB 批量计算 | M3 `duckdb_batch_calculator.py`、M5 `duckdb_batch_calculator.py`、M6 `duckdb_batch_calculator.py` | SQL 查询替代 Python 逐行循环 |
| 向量化需求收集 | M5 `demand_collector_vectorized.py` | 批量处理替代逐节点收集 |
| SimulationCache | `src/utils/simulation_cache.py` | 层级分配、网络、lead time 等中间结果复用 |
| DuckDB Accelerator | `src/utils/duckdb_accelerator.py` | 通用批量过滤和分组加速 |
| DuckDB Optimizer | `src/utils/duckdb_optimizer.py` | 索引构建和净需求批量计算 |
| ThreadPoolExecutor | M5 `_process_layer_demands()` | 多线程并行处理层内节点 |
| ProcessPoolExecutor | M5 `multiprocess_optimizer.py` | 多进程突破 GIL 限制 |
| 静态配置缓存 | M5 `data_loader.py` | 避免每天重复加载和归一化静态数据 |
| 批量预过滤 | M5 `batch_optimizer.py` | DuckDB 批量预过滤层数据 |
| Horizon 批量计算 | M5 `horizon_batch_calculator.py` | 一次性计算全部节点的 horizon |

#### 具体收益

性能提升并不是抽象概念，而是在 BC / OC 报告中以全周期数据体现出来，详见第 9 章。

### 5.8 数据库模式工程化

#### 为什么要优化

本地模式适合开发调试，但对于长周期回归、结果对比、生产化跑批与审计并不够。数据库模式如果只是"能写入几张表"，依然无法满足正式交付。

#### 做了什么

当前数据库模式已形成完整链路：

| 组件 | 文件位置 | 职责 |
|---|---|---|
| `DatabaseInitializer` | `pgsql_db/db_initializer.py` | 建库、补结构、触发配置导入 |
| `ExcelImporter` | `pgsql_db/excel_importer.py` | Excel 配置导入 `cfg_*` 表 |
| `DatabaseConnection` | `pgsql_db/db_connection.py` | 连接管理、表操作、数据读写 |
| `ModuleDataWriter` | `pgsql_db/module_data_writer.py` | 模块结果和状态写入输出表 |
| `table_mapping.py` | `pgsql_db/table_mapping.py` | 输出表名映射规则 |
| `table_schemas.py` | `pgsql_db/table_schemas.py` | 输出表列定义 |

#### 输出表映射

| 模块结果 | 输出表 |
|---|---|
| `module1.orders_df` | `module1_output_orderlog` |
| `module1.shipment_df` | `module1_output_shipmentlog` |
| `module3.net_demand_df` | `module3_output_netdemand` |
| `module4.production_df` | `module4_output_productionplan` |
| `module5.deployment_plan` | `module5_output_deploymentplan` |
| `module6.delivery_plan` | `module6_output_deliveryplan` |

#### 具体收益

数据库模式已经不是试验性旁路，而是完整的配置导入、结果落库、汇总生成和按 `run_id` 隔离的正式运行模式。

### 5.9 续跑、快照与可观察性

#### 为什么要优化

长周期仿真如果没有续跑和状态快照，一旦中断就必须整体重跑；如果某一天结果异常，也很难定位到底是哪一层状态出错。

#### 做了什么

1. `resume.py` 提供续跑判断与恢复能力
2. 每天固化 `orchestrator/*.csv`
3. 汇总层集中生成全周期报表
4. 性能层提供 `performance_profiler.py`

#### 续跑依赖的核心状态文件

| 文件模式 | 用途 |
|---|---|
| `unrestricted_inventory_YYYYMMDD.csv` | 可用库存快照 |
| `open_deployment_YYYYMMDD.csv` | 开放调拨池 |
| `planning_intransit_YYYYMMDD.csv` | 在途记录 |
| `space_quota_YYYYMMDD.csv` | 空间约束状态 |
| `delivery_gr_YYYYMMDD.csv` | 到货历史 |
| `production_gr_YYYYMMDD.csv` | 生产入库历史 |
| `shipment_log_YYYYMMDD.csv` | 客户发货历史 |
| `delivery_shipment_log_YYYYMMDD.csv` | 调拨发运历史 |
| `inventory_change_log_YYYYMMDD.csv` | 库存变动流水 |
| `daily_logs_YYYYMMDD.csv` | 日志摘要 |

---

## 6. 模块详细说明

### 6.1 Core 层

#### 6.1.1 `src/core/run/`

CLI 和运行分发层，负责解析命令行参数、选择运行模式、确定输出目录。

| 函数 | 参数 | 返回值 | 作用 |
|---|---|---|---|
| `main()` | CLI 参数 | None | CLI 总入口，选择本地或 DB 模式 |
| `_ensure_output_dir()` | 配置路径 | str | 确定本次 run 输出目录 |
| `_run_with_database()` | DB 连接参数 | None | DB 模式总控 |
| `_load_config_from_database()` | DB 连接 | dict | 从 `cfg_*` 表恢复 `config_dict` |

#### 6.1.2 `src/core/main_integration/`

主集成层，包含 14 个子文件，负责本地/DB 两种模式的主循环、续跑判断、M4 桥接、配置加载等。

| 文件 | 职责 |
|---|---|
| `simulation_file.py` | 本地模式主循环 |
| `simulation_db.py` | DB 模式主循环 |
| `resume.py` | 续跑检测、状态恢复 |
| `production_integration.py` | M4 集成桥接 |
| `config_loader.py` | 配置加载和标准化 |
| `db_helpers.py` | DB 模式的批量写入、检查点 |
| `memory_store.py` | 内存存储管理 |
| `seed.py` | 随机种子管理 |

关键函数：

| 函数 | 参数 | 返回值 | 作用 |
|---|---|---|---|
| `run_integrated_simulation()` | config_path, start_date, end_date, output_base_dir, force_restart | dict | 本地模式主入口，包含校验、续跑、每日处理全流程 |
| `run_integrated_simulation_from_dict()` | config_data, config_name, start_date, end_date, output_base_dir, skip_validation, resume, db, run_key, batch_size, run_id | dict | DB 模式主入口 |
| `detect_last_complete_date()` | output_base_dir, start_date, end_date | str | 找到最后完整日期 |
| `check_resume_capability()` | output_base_dir, start_date, end_date | dict | 生成续跑信息 |
| `restore_orchestrator_state()` | orchestrator, restore_date, output_base_dir | None | 从状态 CSV 恢复 Orchestrator |
| `run_module4_integrated()` | config_dict, module3_output_dir, simulation_date, ... | dict | M4 桥接层 |
| `load_current_date_production_gr()` | module4_output_dir, current_date, start_date | DataFrame | 读取当日可入库生产 |
| `load_configuration()` | config_path | dict | 从 Excel 加载配置 |
| `load_configuration_from_dict()` | config_data, config_name | dict | 从 DataFrame 字典加载配置 |
| `load_global_seed()` | config_dict | int | 读取随机种子 |
| `set_module_seeds()` | config_dict, global_seed | int | 设置各模块统一随机种子 |

#### 6.1.3 `src/core/orchestrator/`

Orchestrator 是全局共享状态的单一事实源。采用 Mixin 架构，拆分为 10 个子文件，每个 Mixin 负责一类功能。

| 文件 | Mixin 类 | 职责 |
|---|---|---|
| `orchestrator_main.py` | `Orchestrator` + `create_orchestrator()` | 主类定义和构造入口 |
| `processors.py` | `OrchestratorProcessorsMixin` | 模块结果回写（M1/M4/M5/M6） |
| `views.py` | `OrchestratorViewsMixin` | 状态查询视图（14 个 get_ 方法） |
| `inventory_log.py` | `OrchestratorInventoryLogMixin` | 库存变动日志生成 |
| `daily_ops.py` | `OrchestratorDailyOpsMixin` | 每日操作（到货、清理过期调拨） |
| `persistence.py` | `OrchestratorPersistenceMixin` | 状态持久化（CSV 快照保存） |
| `models.py` | `DeploymentUID` | 调拨 UID 模型 |
| `normalize.py` | — | 编排器内部归一化 |

Orchestrator 维护的核心状态：

1. `unrestricted_inventory` — 可用库存
2. `open_deployment` — 开放调拨池
3. `in_transit` — 在途记录
4. `production_gr` — 生产入库记录
5. `delivery_gr` — 调拨到货记录
6. `shipment_log` — 客户发货历史
7. `delivery_shipment_log` — 调拨发运历史
8. `space_quota` — 空间配额

**Processors Mixin 关键函数**：

| 函数 | 参数 | 作用 |
|---|---|---|
| `process_module1_shipments()` | shipment_df, date | 应用 M1 发货结果，扣减库存、记录发货 |
| `process_module4_production()` | production_df, date | 缓存生产到 backlog、执行 GR 增加库存 |
| `process_module5_deployment()` | deployment_df, date | 把调拨计划写入开放调拨池，生成 UID |
| `process_module6_delivery()` | delivery_df, date | 处理发运、更新在途、到货入库 |

**Views Mixin 关键函数**：

| 函数 | 参数 | 返回值 | 作用 |
|---|---|---|---|
| `get_unrestricted_inventory_view()` | date | DataFrame | 可用库存快照 |
| `get_current_unrestricted_inventory()` | — | Dict[(mat,loc), int] | 当前库存字典 |
| `get_planning_intransit_view()` | date | DataFrame | 在途记录视图 |
| `get_open_deployment_view()` | date | DataFrame | 开放调拨视图 |
| `get_space_quota_view()` | date | DataFrame | 空间配额视图 |
| `get_production_plan_backlog_view()` | date | DataFrame | 生产 backlog 视图 |
| `get_all_production_view()` | date | DataFrame | GR + 未来生产合并视图 |
| `get_production_gr_view()` | date | DataFrame | 生产收货记录 |
| `get_delivery_gr_view()` | date | DataFrame | 调拨到货记录 |
| `get_shipment_log_view()` | date | DataFrame | 客户发货记录 |
| `get_delivery_shipment_log_view()` | date | DataFrame | 调拨发运记录 |
| `get_beginning_inventory_view()` | date | DataFrame | 期初库存快照 |
| `get_summary_statistics()` | date | dict | 统计摘要 |

**Daily Ops Mixin 关键函数**：

| 函数 | 作用 |
|---|---|
| `run_daily_processing()` | 执行每日处理序列：清理、到货、模块处理 |
| `_process_delivery_arrivals()` | 处理当日到货的在途调拨 |
| `cleanup_past_due_open_deployments()` | 清理超期的开放调拨 |

**Persistence Mixin 关键函数**：

| 函数 | 作用 |
|---|---|
| `save_daily_state()` | 固化每日状态 CSV |
| `save_beginning_inventory()` | 保存期初库存快照 |
| `save_ending_inventory()` | 保存期末库存快照 |
| `output_daily_inventory_summary()` | 输出详细库存变动摘要 |

### 6.2 M1：需求规划（demand_planning）

#### 模块概览

| 项目 | 内容 |
|---|---|
| 模块路径 | `src/modules/demand_planning/` |
| 核心职责 | 周预测转日订单、发货与缺货计算、供需日志输出 |
| 主要输入 | `M1_DemandForecast`、`M1_ForecastError`、`M1_OrderCalendar`、`M1_AOConfig`、`M1_DPSConfig`、`M1_SupplyChoiceConfig` |
| 主要输出 | `orders_df`、`shipment_df`、`cut_df`、`supply_demand_df`、`summary_df` |
| 主入口 | `run_daily_order_generation()` |

#### 子文件结构

| 文件 | 职责 |
|---|---|
| `integration.py` | 模块对外入口 `run_daily_order_generation()`，串联预测、订单、消耗、发货全流程 |
| `forecast.py` | 预测展开：周预测→日订单（整除+余数分配）、DPS 分裂、供应选择调整 |
| `order.py` | 日订单生成：AO+Normal 两类订单创建、日期展开 |
| `consume.py` | 消耗计算基础版 |
| `consume_optimized.py` | 消耗计算优化版：`consume_orders()` 按优先级匹配库存 |
| `shipment.py` | 发货仿真：`simulate_shipment_for_single_day()` 按 min(订单,库存) 分配 |
| `dps.py` | DPS 处理：`apply_dps()` 按比例分裂需求 |
| `normalization.py` | M1 归一化入口（委托给 `src/utils/normalization.py`） |
| `io_utils.py` | 输入输出工具：Excel 读写、数据格式转换 |
| `config.py` | M1 模块配置常量 |
| `constants.py` | M1 常量定义 |

#### 核心数据流

```
demand_forecast → DPS 分裂 → 供应选择调整 → 周→日展开(整除+余数)
→ AO 订单 + Normal 订单 → 消耗(优先级匹配库存) → 发货(min(订单,库存))
→ 削减计算 → 供需日志输出
```

#### 关键函数

| 函数 | 位置 | 参数 | 作用 |
|---|---|---|---|
| `run_daily_order_generation()` | `integration.py` | config_dict, orchestrator, current_date, output_dir, ... | M1 单日对外入口，串联全流程 |
| `expand_forecast_to_days_integer_split()` | `forecast.py` | forecast_df, date | 周预测整数拆分为日订单（整除+余数分配给最后一天） |
| `generate_daily_orders()` | `order.py` | config, date, forecast | 从展开后的预测生成 AO+Normal 订单 |
| `consume_orders()` | `consume_optimized.py` | orders_df, inventory, priority_map | 按优先级匹配库存消耗订单 |
| `simulate_shipment_for_single_day()` | `shipment.py` | orders_df, inventory | 按 min(订单量,可用库存) 计算发货 |
| `apply_dps()` | `dps.py` | forecast_df, dps_config | 按 DPS 配置比例分裂预测需求 |
| `apply_supply_choice()` | `forecast.py` | forecast_df, supply_choice_config | 供应选择调整预测量 |

### 6.3 M3：MRP 规划（mrp_planning）

#### 模块概览

| 项目 | 内容 |
|---|---|
| 模块路径 | `src/modules/mrp_planning/` |
| 核心职责 | 分层净需求计算，将缺口沿网络向上游传导 |
| 主要输入 | M1 输出、库存、在途、GR、开放调拨、BOM/网络/LeadTime |
| 主要输出 | `net_demand_df` |
| 主入口 | `run_integrated_mode()` |

#### 子文件结构

| 文件 | 职责 |
|---|---|
| `integration.py` | 模块对外入口 `run_integrated_mode()`，串联 MRP 全流程 |
| `mrp_simulation.py` | MRP 分层仿真主循环 `run_mrp_layered_simulation_daily()` |
| `net_demand.py` | 净需求计算 `calculate_daily_net_demand()`，AO>Forecast>SafetyStock 优先级 |
| `layer_assignment.py` | 层级分配 `assign_location_layers()`，BFS 算法 |
| `lead_time.py` | 提前期确定 `determine_lead_time()`，Plant/DC 区分 |
| `node_processor.py` | 单节点处理逻辑 |
| `data_indexer.py` | 数据索引构建，加速查找 |
| `duckdb_batch_calculator.py` | DuckDB 批量计算（>50 节点自动切换） |
| `config_loader.py` | M3 配置加载 |
| `constants.py` | M3 常量（从 defaults 导入共享参数） |
| `utils.py` | M3 内部工具函数 |

#### 核心算法

1. **层级分配（BFS）**：从最终需求节点（叶子节点）开始，通过 BFS 为每个 material-location 对分配层级编号，最底层（叶子）层级最低
2. **净需求计算**：`净需求 = max(0, 毛需求 - 可用库存 - 在途 - 未来生产 + 安全库存缺口)`，按 AO > Forecast > SafetyStock 优先级处理
3. **分层仿真**：从最低层（需求端）到最高层（供应端）逐层计算，每层的缺口向上游传导

#### 关键函数

| 函数 | 位置 | 作用 |
|---|---|---|
| `run_integrated_mode()` | `integration.py` | M3 集成模式主入口 |
| `run_mrp_layered_simulation_daily()` | `mrp_simulation.py` | 每日分层 MRP 仿真循环 |
| `calculate_daily_net_demand()` | `net_demand.py` | 计算单节点净需求 |
| `assign_location_layers()` | `layer_assignment.py` | BFS 层级分配 |
| `determine_lead_time()` | `lead_time.py` | 确定 Plant/DC 提前期 |

#### 性能优化

当层内节点数超过 50 个时，自动切换到 DuckDB 批量计算模式，使用 SQL 查询替代 Python 逐行循环。同时利用 `ThreadPoolExecutor` 对层内独立节点进行并行处理。

### 6.4 M4：生产计划（production_planning）

#### 模块概览

| 项目 | 内容 |
|---|---|
| 模块路径 | `src/modules/production_planning/` |
| 核心职责 | 按产能、换型、可靠率约束生成生产计划 |
| 主要输入 | `M4_MaterialLocationLineCfg`、`M4_LineCapacity`、`M4_ChangeoverMatrix`、`M4_ChangeoverDefinition`、`M4_ProductionReliability`、M3 净需求 |
| 主要输出 | `production_df`、`exceed_log`、`issues_df`、`changeover_log` |
| 主入口 | `run_daily_production_planning()` |

#### 子文件结构

| 文件 | 职责 |
|---|---|
| `main.py` | 模块对外入口 `run_daily_production_planning()` |
| `plan_builder.py` | 无约束计划构建 `build_unconstrained_plan_for_single_day()` |
| `capacity_allocator.py` | 产能分配+换型处理 `centralized_capacity_allocation_with_changeover()` |
| `demand_loader.py` | 从 M3 输出加载需求 |
| `config_loader.py` | M4 配置加载 |
| `state_manager.py` | 产线状态管理（backlog、上次物料记录） |
| `output_writer.py` | 输出 Excel 文件写入 |
| `duckdb_batch_calculator.py` | DuckDB 批量计算 |
| `constants.py` | M4 常量（从 defaults 导入 CHANGEOVER_TIME） |
| `types.py` | M4 类型定义 |
| `utils.py` | M4 内部工具函数 |

#### 核心算法

1. **无约束计划**：根据 M3 净需求直接生成不考虑产能限制的理想生产计划
2. **产能分配+换型**：`centralized_capacity_allocation_with_changeover()` 考虑产线产能限制、换型时间扣除，按优先级分配产能
3. **最优换型序列**：`optimal_changeover_sequence()` 贪心算法最小化换型次数
4. **可靠性仿真**：`simulate_production()` 使用二项分布仿真实际产出（考虑产线可靠率）
5. **Backlog 跟踪**：未完成的生产需求自动滚入下一天

#### 关键函数

| 函数 | 位置 | 作用 |
|---|---|---|
| `run_daily_production_planning()` | `main.py` | M4 单日入口 |
| `build_unconstrained_plan_for_single_day()` | `plan_builder.py` | 构建无约束计划 |
| `centralized_capacity_allocation_with_changeover()` | `capacity_allocator.py` | 产能分配+换型处理 |
| `optimal_changeover_sequence()` | `capacity_allocator.py` | 最优换型序列（贪心） |
| `simulate_production()` | `main.py` | 可靠性仿真（二项分布） |

### 6.5 M5：调拨规划（deployment_planning）

#### 模块概览

| 项目 | 内容 |
|---|---|
| 模块路径 | `src/modules/deployment_planning/` |
| 核心职责 | 多层网络下的调拨计划、push/pull 分配、库存平衡 |
| 主要输入 | M1 输出、M4 输出、库存、在途、开放调拨、`M5_DeployConfig`、`M5_PushPullModel` |
| 主要输出 | `deployment_plan`、`unfulfilled_log`、`stock_on_hand_log`、`validation_log` |
| 主入口 | `main()` |

#### 子文件结构

| 文件 | 职责 |
|---|---|
| `main.py` | 模块主入口 `main()`，多层处理编排 |
| `allocation.py` | MOQ/RV 约束、优先级库存分配、空间配额 |
| `demand_collector.py` | 逐节点需求收集（SDL、安全库存、订单、缺口） |
| `demand_collector_vectorized.py` | 向量化批量需求收集 |
| `inventory.py` | 库存状态计算（投影库存、可用库存） |
| `push_allocation.py` | Push/SoftPush 分配逻辑 |
| `data_loader.py` | 配置加载（静态缓存+动态并行加载） |
| `cache_utils.py` | PTF/LSK、lead time、网络索引缓存 |
| `batch_optimizer.py` | DuckDB 批量预过滤 |
| `horizon_batch_calculator.py` | 批量 horizon 计算 |
| `duckdb_batch_calculator.py` | DuckDB 批量 MOQ/RV 和优先级分配 |
| `multiprocess_optimizer.py` | 多进程并行处理 |
| `normalizer.py` | 标识符归一化（委托给 utils） |
| `validation.py` | 配置校验和输出排序 |
| `constants.py` | M5 常量 |

#### 核心算法

1. **分层处理（上游→下游）**：按 BFS 层级从最底层到最高层逐层处理，每层独立收集需求并分配库存
2. **需求收集**：从 SupplyDemandLog、SafetyStock、OrderLog、上游缺口缓冲四个来源聚合需求
3. **MOQ/RV 应用**：`apply_grouped_moq_rv()` 按路线分组，使用最大余数法（Largest Remainder Method）比例分配
4. **优先级库存分配**：`apply_priority_allocation_vectorized()` 使用 numpy 向量化，按优先级从高到低填满
5. **管道供应分配**：`allocate_pipeline_supply()` 依次扣减在途、入站调拨、未来生产
6. **缺口传播**：下游未满足的需求通过 `up_gap_buffer` 向上游传导
7. **Push/SoftPush**：`push_softpush_allocation()` 在常规分配后，将上游多余库存按安全库存覆盖率推向下游
8. **空间配额**：`apply_receiving_space_quota()` 按接收方空间限制裁剪部署计划

#### 关键函数

| 函数 | 位置 | 参数 | 作用 |
|---|---|---|---|
| `main()` | `main.py` | input_path, config_dict, orchestrator, current_date, ... | M5 主入口 |
| `_process_layer_demands()` | `main.py` | layer, all_pairs, sim_date, config, ... | 处理单层所有节点需求 |
| `_allocate_pipeline_sources()` | `main.py` | demand_rows, adjusted_qtys, loc, mat, ... | 管道供应分配 |
| `_process_gaps_and_create_plans()` | `main.py` | demand_rows, adjusted_qtys, mat, loc, ... | 处理缺口、创建部署计划 |
| `_update_soh_dict()` | `main.py` | soh_dict, deployment_plan_rows, ... | 更新库存字典 |
| `apply_moq_rv()` | `allocation.py` | qty, moq, rv, is_cross_node, max_qty | 单行 MOQ/RV 约束 |
| `apply_grouped_moq_rv()` | `allocation.py` | demand_rows, location, shipment_qty_limit | 分组 MOQ/RV（最大余数法） |
| `apply_priority_allocation_vectorized()` | `allocation.py` | demand_rows, adjusted_qtys, current_stock, priority_map | 向量化优先级分配 |
| `allocate_pipeline_supply()` | `allocation.py` | demand_rows, adjusted_qtys, location, ... | 管道供应分配 |
| `apply_receiving_space_quota()` | `allocation.py` | deployment_plan_rows, receiving_space, ... | 空间配额裁剪 |
| `collect_node_demands()` | `demand_collector.py` | material, location, sim_date, config, ... | 逐节点需求收集 |
| `collect_demands_batch_vectorized()` | `demand_collector_vectorized.py` | pairs, sim_date, config, ... | 向量化批量收集 |
| `push_softpush_allocation()` | `push_allocation.py` | deployment_plan_rows, config, dynamic_soh, ... | Push/SoftPush 分配 |
| `build_horizon_cache()` | `horizon_batch_calculator.py` | all_pairs, sim_date, ... | 批量构建 horizon 缓存 |

### 6.6 M6：物流执行（logistics_execution）

#### 模块概览

| 项目 | 内容 |
|---|---|
| 模块路径 | `src/modules/logistics_execution/` |
| 核心职责 | 发运、装载、路线、延迟、在途与到货 |
| 主要输入 | open deployment、`M6_MaterialMD`、`M6_TruckTypeSpecs`、`M6_TruckReleaseCon`、`M6_DeliveryDelayDistribution`、`M6_MDQBypassRules` |
| 主要输出 | `delivery_plan`、`vehicle_log`、`truck_usage`、`unsatisfied_log`、`validation_log`、`bypass_log` |
| 主入口 | `run_daily_physical_flow()` |

#### 子文件结构

| 文件 | 职责 |
|---|---|
| `main.py` | 模块主入口，支持独立和集成两种模式 |
| `simulation.py` | 核心仿真循环、路线处理、需求收集 |
| `output_writer.py` | 输出 DataFrame 构建、约束校验、Excel 写入 |
| `capacity_manager.py` | 运力管理：容量归一化、查询、最优车序 |
| `config_loader.py` | 配置加载（独立/集成模式） |
| `delivery_processor.py` | 延迟抽样、MDQ 绕过、lead time、记录创建 |
| `duckdb_batch_calculator.py` | DuckDB 批量延迟抽样 |
| `expression_evaluator.py` | 安全布尔表达式求值（AST 解析） |
| `inventory_manager.py` | 物理库存跟踪与更新 |
| `validators.py` | 数据校验、去重、报告生成 |
| `vehicle_packer.py` | 车辆装载优化（重量/体积/库存约束） |

#### 核心算法

1. **仿真循环**：`run_simulation_loop()` 逐日处理，收集待发需求、按路线分组、按车型迭代装载
2. **两遍装载**：先贪心装载（first pass），再补充装载（second pass，按阈值触发）
3. **延迟抽样**：`sample_delivery_delay()` 从概率分布中抽样延迟天数，优先精确路线匹配
4. **MDQ 绕过**：`should_bypass_mdq()` 通过安全表达式求值判断是否绕过最小发货量限制
5. **车辆装载优化**：`VehiclePacker` 考虑重量、体积、库存三重约束的贪心装载
6. **发运触发**：根据 WFR/VFR 阈值、超时、绕过规则三种条件触发发运

#### 关键函数

| 函数 | 位置 | 参数 | 作用 |
|---|---|---|---|
| `run_daily_physical_flow()` | `main.py` | config_dict, orchestrator, current_date, output_dir, ... | M6 单日入口 |
| `run_physical_flow_module()` | `main.py` | 多参数 | M6 总入口（支持独立/集成） |
| `run_simulation_loop()` | `simulation.py` | run_params, prepared_data | 核心仿真循环 |
| `handle_remaining_demands()` | `simulation.py` | route_demands, agg_status, ... | 处理未满足需求 |
| `enforce_shipment_constraint()` | `output_writer.py` | delivery_plan_df, orchestrator, validation_log | 发货约束校验 |
| `generate_outputs()` | `output_writer.py` | run_params, results, validation_log, skip_file_output | 输出生成 |
| `sample_delivery_delay()` | `delivery_processor.py` | sending, receiving, dist_df | 延迟抽样 |
| `should_bypass_mdq()` | `delivery_processor.py` | context, rules, evaluator | MDQ 绕过判定 |
| `normalize_capacity_plan()` | `capacity_manager.py` | truck_cap_df, sim_start, sim_end | 容量归一化为日粒度 |
| `VehiclePacker.add_demand()` | `vehicle_packer.py` | idx, demand_row, inventory_limit | 装载需求（重量/体积/库存约束） |
| `SafeExpressionEvaluator.eval()` | `expression_evaluator.py` | expr, context | 安全布尔表达式求值 |

---

## 7. Utils 与 Storage 基础层详细说明

### 7.1 `src/utils/` 工具层

#### 7.1.1 `defaults.py` — 默认参数加载器

从 `config/defaults.yaml` 加载共享默认参数，以 Python 常量形式暴露。所有模块通过 `from src.utils.defaults import DEFAULT_MOQ` 导入。

#### 7.1.2 `normalization.py` — 统一归一化

| 函数 | 参数 | 作用 |
|---|---|---|
| `normalize_material(val)` | 任意标量 | 去除 `.0` 后缀 |
| `normalize_location(val)` | 任意标量 | 补零到 4 位 |
| `normalize_identifiers(df, extra_columns)` | DataFrame | 批量归一化所有标识符列 |

预定义的列名常量：
- `LOCATION_COLUMNS`: location, sending, receiving, sourcing 等
- `MATERIAL_COLUMNS`: material 等
- `STRING_ONLY_COLUMNS`: 仅需 str 转换的列
- `ALL_IDENTIFIER_COLUMNS`: 所有需要归一化的列

#### 7.1.3 `simulation_cache.py` — 仿真缓存

`SimulationCache` 类缓存跨天不变的中间计算结果，避免每天重复计算：

| 方法 | 作用 |
|---|---|
| `get_layer_assignment()` | 获取预计算的层级分配 |
| `get_active_network(sim_date)` | 获取活跃网络数据 |
| `get_ptf_lsk(material, location)` | 获取 PTF/LSK 值 |
| `get_lead_time(sending, receiving)` | 获取 lead time |
| `get_upstream(material, location)` | 获取上游节点 |
| `get_moq_rv(material, sending)` | 获取 MOQ/RV |
| `get_safety_stock(material, location, date)` | 获取安全库存 |
| `get_stats()` / `print_stats()` | 缓存命中率统计 |

全局实例管理：`initialize_simulation_cache()`, `get_simulation_cache()`, `clear_simulation_cache()`

#### 7.1.4 `duckdb_accelerator.py` — 通用加速器

`DuckDBAccelerator` 单例类，提供通用的 DuckDB 加速操作：

| 方法 | 作用 |
|---|---|
| `batch_filter_by_material_location()` | 批量过滤 material-location 对 |
| `aggregate_by_groups()` | 分组聚合 |
| `filter_date_range()` | 日期范围过滤 |
| `build_material_location_index()` | 构建分组索引 |

#### 7.1.5 `duckdb_optimizer.py` — 高级优化器

`DuckDBOptimizer` 单例类，提供高级 DuckDB 优化操作：

| 方法 | 作用 |
|---|---|
| `register_df()` / `unregister_df()` | 注册/注销 DataFrame 为 DuckDB 表 |
| `query()` | 执行 SQL 查询 |
| `batch_build_sdl_index()` | 批量构建供需日志索引 |
| `batch_build_ss_index()` | 批量构建安全库存索引 |
| `batch_build_order_index()` | 批量构建订单索引 |
| `batch_calculate_net_demand()` | 批量计算净需求 |

#### 7.1.6 `inventory_balance_checker.py` — 账平校验

`InventoryBalanceChecker` 类验证库存收支平衡：

| 方法 | 作用 |
|---|---|
| `check_daily_balance(date)` | 检查单日库存平衡 |
| `check_period_balance(start, end)` | 检查区间库存平衡 |
| `check_negative_inventory(date)` | 检查是否存在负库存 |
| `validate_inventory_consistency()` | 综合校验（平衡+负值） |

### 7.2 `pgsql_db/` 数据库层

#### 7.2.1 `db_connection.py` — 数据库连接

`DatabaseConnection` 类封装 PostgreSQL 连接管理：

| 方法 | 作用 |
|---|---|
| `connect()` / `close()` | 建立/关闭连接 |
| `create_database_if_not_exists()` | 自动建库 |
| `create_table_from_df()` | 从 DataFrame 建表并写入 |
| `read_table()` | 读表为 DataFrame |
| `execute_query()` / `execute_non_query()` | 执行 SQL |
| `table_exists()` / `get_all_tables()` | 表存在性检查 |

#### 7.2.2 `excel_importer.py` — Excel 导入

`ExcelImporter` 类将 Excel 配置文件导入数据库 `cfg_*` 表：

| 方法 | 作用 |
|---|---|
| `import_excel_file()` | 导入单个 Excel 文件的所有 sheet |
| `import_multiple_files()` | 批量导入多个 Excel 文件 |

#### 7.2.3 `module_data_writer.py` — 模块数据写入

`ModuleDataWriter` 类将模块运行结果写入数据库：

| 方法 | 作用 |
|---|---|
| `write_module_results_from_dict()` | 从内存字典写入模块结果 |
| `write_orchestrator_data()` | 写入 Orchestrator 状态 |
| `ensure_output_tables_exist()` | 预创建输出表结构 |
| `truncate_output_tables()` | 按 run_id 删除旧数据 |
| `delete_batch_data()` | 删除指定日期起的批量数据 |

#### 7.2.4 `db_initializer.py` — 数据库初始化

`DatabaseInitializer` 类管理数据库生命周期：

| 方法 | 作用 |
|---|---|
| `initialize()` | 完整初始化（建库、建表、导入配置） |
| `import_config_from_excel()` | 从 Excel 导入配置 |
| `get_available_configs()` | 查询已有配置 |
| `get_status_report()` | 生成状态报告 |

#### 7.2.5 `table_mapping.py` 与 `table_schemas.py`

- `get_config_table_name(sheet, prefix)` — 获取配置表名
- `get_output_table_name(module, file_pattern)` — 获取输出表名
- `get_columns(module, sheet)` — 获取列定义
- `get_all_module_tables()` — 获取所有模块表 schema

---

## 8. 输入、输出与标准文件格式

### 8.1 核心配置表

| 表名 / Sheet | 主要用途 | 主要使用方 |
|---|---|---|
| `Global_Network` | 节点网络关系 | M3、M5 |
| `Global_LeadTime` | 提前期 | M3、M5、M6 |
| `Global_DemandPriority` | 优先级 | M4、M5、M6 |
| `Global_seed` | 随机种子 | 主流程、M4、M6 |
| `Global_SpaceCapacity` | 空间约束 | Orchestrator、M5 |
| `M1_DemandForecast` | 周预测 | M1 |
| `M1_InitialInventory` | 期初库存 | Orchestrator、M1 |
| `M3_SafetyStock` | 安全库存（独立配置文件，与其他配置文件无关） | M3、M5 |
| `M4_LineCapacity` | 产能 | M4 |
| `M5_DeployConfig` | 调拨规则 | M3、M5 |
| `M6_TruckTypeSpecs` | 车型参数 | M6 |

### 8.2 每日模块输出

| 模块 | 文件名模式 | 核心工作表 |
|---|---|---|
| M1 | `module1_output_YYYYMMDD.xlsx` | `OrderLog`、`ShipmentLog`、`CutLog`、`SupplyDemandLog`、`Summary` |
| M3 | `Module3Output_YYYYMMDD.xlsx` | `NetDemand` |
| M4 | `Module4Output_YYYYMMDD.xlsx` | `ProductionPlan`、`CapacityExceed`、`Validation`、`ChangeoverLog` |
| M5 | `Module5Output_YYYYMMDD.xlsx` | `DeploymentPlan`、`UnfulfilledLog`、`StockOnHandLog`、`Validation` |
| M6 | `Module6Output_YYYYMMDD.xlsx` | `DeliveryPlan`、`VehicleLog`、`TruckUsageLog`、`UnsatisfiedMDQLog`、`ValidationLog`、`BypassRuleHitLog` |

### 8.3 Orchestrator 状态文件

| 文件名模式 | 含义 |
|---|---|
| `unrestricted_inventory_YYYYMMDD.csv` | 可用库存快照 |
| `open_deployment_YYYYMMDD.csv` | 开放调拨池 |
| `planning_intransit_YYYYMMDD.csv` | 在途记录 |
| `space_quota_YYYYMMDD.csv` | 空间状态 |
| `production_plan_backlog_YYYYMMDD.csv` | 生产 backlog |
| `production_gr_YYYYMMDD.csv` | 生产收货 |
| `delivery_gr_YYYYMMDD.csv` | 调拨到货 |
| `shipment_log_YYYYMMDD.csv` | 客户发货 |
| `delivery_shipment_log_YYYYMMDD.csv` | 调拨发运 |
| `inventory_change_log_YYYYMMDD.csv` | 库存变化流水 |
| `daily_logs_YYYYMMDD.csv` | 日志摘要 |

### 8.4 汇总输出

| 文件 | 说明 |
|---|---|
| `full_order_shipment_cut_report.xlsx` | 订单 / 发货 / 削减汇总 |
| `full_delivery_plan_report.xlsx` | 交付汇总 |
| `full_truck_usage_report.xlsx` | 车辆使用汇总 |
| `full_exceed_capacity_report.xlsx` | 超产能汇总 |
| `full_changeover_report.xlsx` | 换型汇总 |
| `full_deployment_plan_report.xlsx` | 调拨汇总 |
| `full_production_plan_report.xlsx` | 生产汇总 |
| `historical_inventory_record.csv` | 历史库存轨迹 |

---

## 9. 本地模式与数据库模式的边界

| 维度 | 本地模式 | 数据库模式 |
|---|---|---|
| 配置来源 | Excel 工作簿 | `cfg_*` 表 |
| 主入口 | `run_integrated_simulation()` | `run_integrated_simulation_from_dict()` |
| 中间结果 | 每日 Excel / CSV | 内存结果 + 输出表 |
| 结果落点 | `outputs/<config>/run_*` | `module*_output_*`、`orchestrator_*`、`summary_output_*` |
| 适用场景 | 开发调试、短周期验证 | 长周期回归、沉淀查询、生产化跑批 |
| 优势 | 可读性高、问题好定位 | 回归方便、对比方便、审计方便 |

建议使用方式如下：

1. 开发新规则、查单日问题，优先用本地模式。
2. 做长周期性能回归或交付验证，优先用数据库模式。
3. 需要跨批次对比、长期沉淀结果时，优先使用数据库模式。

---

## 10. BC / OC 验证结果

### 10.1 BC 场景

BC 场景覆盖 87 天，区间为 `2025-10-05` 至 `2025-12-30`。全周期耗时如下：

| 模式 | 全周期耗时 | 平均每天 | 相对提升 |
|---|---|---|---|
| Dev | 9540 秒（159.0 分钟） | 109.7 秒/天 | 基线 |
| Src | 2897 秒（48.3 分钟） | 33.3 秒/天 | 相比 Dev 提升 3.29x |
| DB | 1563 秒（26.1 分钟） | 18.0 秒/天 | 相比 Dev 提升 6.10x；相比 Src 提升 1.85x |

模块级提升如下：

| 模块 | Dev | Src | DB | DB vs Dev |
|---|---|---|---|---|
| M1 订单生成 | ~18.3 秒/天 | ~5.7 秒/天 | ~5.6 秒/天 | 3.29x |
| M3 净需求计算 | ~24.1 秒/天 | ~5.8 秒/天 | ~2.3 秒/天 | 10.70x |
| M5 调拨规划 | ~60.6 秒/天 | ~18.4 秒/天 | ~7.1 秒/天 | 8.54x |

一致性结论：

1. 16/16 张表全部 PASS
2. `SupplyDemandLog` 2,084,295 行保持一致
3. `DeploymentPlan` 1,149,959 行保持一致
4. `DeliveryPlan` 43,055 行保持一致

### 10.2 OC 场景

OC 场景覆盖 76 天，区间为 `2025-12-15` 至 `2026-02-28`。全周期耗时如下：

| 模式 | 全周期耗时 | 平均每天 | 相对提升 |
|---|---|---|---|
| Dev | 97611 秒（1626.8 分钟） | 1284.4 秒/天 | 基线 |
| Src | 14854 秒（247.6 分钟） | 195.4 秒/天 | 相比 Dev 提升 6.57x |
| DB | 7850 秒（130.8 分钟） | 103.3 秒/天 | 相比 Dev 提升 12.43x；相比 Src 提升 1.89x |

模块级提升如下：

| 模块 | Dev | Src | DB | DB vs Dev |
|---|---|---|---|---|
| M1 订单生成 | ~377.2 秒/天 | ~38.0 秒/天 | ~36.0 秒/天 | 10.48x |
| M3 净需求计算 | ~342.8 秒/天 | ~44.9 秒/天 | ~20.4 秒/天 | 16.84x |
| M5 调拨规划 | ~544.6 秒/天 | ~103.9 秒/天 | ~39.0 秒/天 | 13.96x |

一致性结论：

1. 16/16 张表全部 PASS
2. `SupplyDemandLog` 16,251,840 行保持一致
3. `DeploymentPlan` 3,336,330 行保持一致
4. `DeliveryPlan` 71,145 行保持一致

### 10.3 总结

这轮优化的结论可以明确概括为一句话：

结构发生了明显收口，性能获得了实质提升，但 BC / OC 长周期结果口径未发生漂移。

---

## 11. 三版本一致性验证

本轮交付要求 Dev（单体开发版）、Src（重构模块化版）、DB（数据库模式）三个版本的输出完全一致，以 Dev 版本输出为基准。

### 11.1 验证结果

| 对比维度 | 比对项数 | 结果 |
|---|---|---|
| Src vs Dev（模块+汇总） | 42 项 | 42/42 OK |
| DB vs Dev（Orchestrator） | 14 项 | 14/14 OK |
| DB vs Dev（模块+汇总） | 47 项 | 47/47 OK |
| DB vs Src（全量） | 61 项 | 61/61 OK |

### 11.2 验证方法

使用 `tools/regression_compare.py` 工具进行逐表、逐列、逐行对比：

```
python tools/regression_compare.py <基线输出目录> <目标输出目录> --tolerance 1e-6
```

测试时间区间：`2025-12-15` 至 `2025-12-16`。

---

## 12. 接手建议

### 12.1 阅读顺序建议

建议按下面的顺序建立上下文：

1. 先看本文件，建立整体结构认知。
2. 再看 `ARCHITECTURE.md`，理解五层结构与双模式边界。
3. 再看 `core.md`，理解主流程、续跑与 Orchestrator。
4. 再看 `module.md`，理解 M1 / M3 / M4 / M5 / M6。
5. 涉及改动或排障时，再查 `函数上下游依赖矩阵.md` 与 `函数级接口与文件格式总表.md`。

### 12.2 常见任务应落在哪里

| 任务类型 | 首选落点 |
|---|---|
| 新增运行参数 | `src/core/run/` |
| 调整执行顺序或续跑逻辑 | `src/core/main_integration/` |
| 新增共享状态或状态字段 | `src/core/orchestrator/` |
| 调整某个业务规则 | 对应 `src/modules/*/` |
| 调整默认参数 | `config/defaults.yaml` |
| 调整统一归一化 | `src/utils/normalization.py` |
| 调整汇总报表 | `src/services/summary_report_generator.py` |
| 调整数据库落库与汇总 | `pgsql_db/` |

### 12.3 接手后的第一轮动作

1. 先跑一次短周期本地模式，确认模块输出和状态 CSV 都能落出来。
2. 再跑一次数据库模式，确认 `cfg_*`、`module*_output_*`、`summary_output_*` 链路完整。
3. 再选 BC 或 OC 做一次标准回归，确认本地环境与当前基线一致。

---

## 13. 最容易踩坑的地方

| 风险点 | 典型现象 | 优先检查位置 |
|---|---|---|
| 改了旧 wrapper 路径 | 代码改了但运行结果不变 | 是否改到了已删除或兼容路径 |
| 标识符不一致 | join 行数异常、部分结果突然为空 | `src/utils/normalization.py` |
| 调拨与物流对不上 | `DeploymentPlan` 与 `DeliveryPlan` 无法闭环 | M5 输出、M6 UID 与路线处理 |
| 续跑恢复异常 | 恢复后某些状态错位 | `resume.py` 与 `orchestrator/*.csv` |
| 结果空表 | 模块未报错，但输出为空 | 上游输入是否为空、日期是否错位 |
| 性能突然变差 | 单场景可跑，但耗时明显增加 | 是否重新引入逐行循环、缓存失效 |
| 默认参数分叉 | 不同模块用不同默认值 | 是否绕过 `config/defaults.yaml` 直接硬编码 |
| DB 模式结果不一致 | 本地和 DB 输出有差异 | 浮点精度、排序稳定性、列类型转换 |

推荐排障顺序：

1. 先定位是哪一个模块的输出出现问题。
2. 再看该模块的上游输入是否正常。
3. 涉及库存、在途、开放调拨时优先看 Orchestrator 视图。
4. 只在数据库模式出问题时，再核对写库与表映射。
5. 如果结果正常但耗时异常，再做性能剖析。

---

## 14. 后续建议

### 14.1 建议继续坚持的方向

1. 新代码继续遵守"入口薄、编排稳、算法纯、写入专"的边界。
2. 共享默认参数继续只在 `defaults.yaml` 维护。
3. 标识符归一化继续只保留权威数据源。
4. 性能优化继续和零漂移验证绑定推进。

### 14.2 不建议再回退的做法

1. 不建议重新引入平铺 wrapper。
2. 不建议把共享参数散回模块内部。
3. 不建议复制新的私有归一化实现。
4. 不建议绕过 Orchestrator 直接在模块里写共享状态。

---

## 15. 结论

基于 `Refactored` 目录中的实际交接文档，可以明确得出以下结论：

1. 当前项目结构已经完成收口，权威入口明确。
2. 本地模式与数据库模式已经形成清晰边界。
3. M1 / M3 / M4 / M5 / M6 的输入输出契约和主链路已经稳定。
4. 默认参数、标识符规则、Module6 结构、数据库模式能力都已经形成长期维护基础。
5. BC / OC 两类场景已经验证了性能收益和结果一致性。
6. 三版本（Dev / Src / DB）输出完全一致性已通过验证。

因此，这一版不是"临时能跑的整理版"，而是可以作为正式交接基线继续维护的版本。后续团队只要沿着当前已经明确的边界继续扩展，维护成本会稳定得多；反之，如果重新回到旧路径、旧写法或分散配置的模式，项目很快又会失去可维护性。
