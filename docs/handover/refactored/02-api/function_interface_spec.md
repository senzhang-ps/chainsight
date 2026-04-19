# ChainSight 函数级接口与文件格式总表

## 封面信息

- 文档名称：ChainSight 函数级接口与文件格式总表
- 文档编号：CS-API-20260306-001
- 文档版本：v1.2
- 编写日期：2026-03-06
- 最后更新：2026-04-10
- 编写人：陈显跃
- 文档状态：正式交付版

## 审批栏

| 角色 | 姓名 | 状态 | 日期 | 备注 |
|---|---|---|---|---|
| 编写 | 陈显跃 | 已完成 | 2026-03-06 | 初版交付 |
| 修订 | 陈显跃 | 已完成 | 2026-04-10 | 同步第三阶段包化重构 |
| 复核 | 待填写 | 待复核 | 待填写 |  |
| 审批 | 待填写 | 待审批 | 待填写 |  |

## 修订记录

| 版本 | 日期 | 修订人 | 修订说明 |
|---|---|---|---|
| v1.0 | 2026-03-06 | 陈显跃 | 首次形成函数级接口、文件格式、关键复杂文件深度说明 |
| v1.2 | 2026-04-10 | 陈显跃 | 第三阶段重构：单体 `run.py` / `main_integration.py` / `orchestrator.py` / `module1~6.py` 全部拆分为包目录。文中所有老 `.py` 路径请按下方"路径迁移指引"映射理解。 |

## 路径迁移指引（第三阶段，2026-04-10）

文档正文中可能仍保留旧写法，以下为权威映射：

| 旧路径（文中仍可能出现） | 新实际位置 |
|---|---|
| `src/core/run.py` | 包 `src/core/run/`，主入口文件：`run_main.py` |
| `src/core/main_integration.py` | 包 `src/core/main_integration/`，主要入口：`simulation_file.py`（本地模式）、`simulation_db.py`（DB 模式）、`production_integration.py`（M4 集成）、`resume.py`（断点续跑）、`config_loader.py`（配置加载） |
| `src/core/orchestrator.py` | 包 `src/core/orchestrator/`，主类在 `orchestrator_main.py`；按模块的 `process_moduleX_*` 在 `processors.py`；视图在 `views.py`；快照持久化在 `persistence.py`；日常操作在 `daily_ops.py` |
| `src/core/parallel_executor.py` | 包 `src/core/parallel_executor/`，主入口：`parallel_executor_main.py` |
| `src/modules/module1.py` | `src/modules/demand_planning/`（入口 `integration.py`） |
| `src/modules/module3.py` | `src/modules/mrp_planning/`（入口 `integration.py` 与 `mrp_simulation.py`） |
| `src/modules/module4.py` | `src/modules/production_planning/`（入口 `main.py`） |
| `src/modules/module5.py` | `src/modules/deployment_planning/`（入口 `main.py`） |
| `src/modules/module6.py` | `src/modules/logistics_execution/`（入口 `main.py`） |
| `src/utils/optimization_config.py` | **已删除**（参数全部迁移至 `config/defaults.yaml`，通过 `src/utils/defaults.py` 读取） |
| `src/utils/parallel_optimizer.py` | **已删除**（第三阶段清理，未使用） |
| `src/utils/process_pool_executor.py` | **已删除**（第三阶段清理，未使用） |
| `src/utils/multiprocess_executor.py` | **已删除**（第三阶段清理，未使用） |
| `src/utils/high_perf_executor.py` | **已删除**（第三阶段清理，未使用） |
| `src/utils/performance.py` | **已删除**（并入 `src/services/performance_profiler.py`） |
| `src/utils/duckdb_sql_wrapper.py` | **已删除**（功能合并进 `src/utils/duckdb_accelerator.py`） |

## 文档定位

- 本文是维护者使用频率最高的函数索引手册。
- 它强调函数输入、输出、调用位置、文件格式和高风险修改点。

## 1. 用途说明

- 本文档专门回答三个问题：函数按什么顺序执行、每个关键函数吃什么数据、会吐出什么文件或内存结果。
- 由于项目规模较大，本文以“主链路函数 + 核心公共函数 + 关键内部函数 + 所有标准输出格式”为主，覆盖交接和排障最常用的函数集合。
- 更细的逐文件文档可继续参考：`docs/core.md`、`docs/modules_demand_planning.md`、`docs/modules_mrp_planning.md`、`docs/modules_production_planning.md`、`docs/modules_deployment_planning.md`、`docs/modules_logistics_execution.md`、`docs/services.md`、`docs/utils.md`。

## 2. 程序总运行顺序

### 2.1 本地模式总顺序

1. `run.py`（仓库根入口，转发到 `src/core/run/`）
2. `src/core/run/run_main.py::main()`
3. `src/core/main_integration/simulation_file.py::run_integrated_simulation(...)`
4. 初始化时间管理器、日志、配置、Orchestrator、输出目录
5. 按天循环执行：
   - `save_beginning_inventory(...)`
   - `cleanup_past_due_open_deployments(...)`
   - `_process_delivery_arrivals(...)`
   - `load_current_date_production_gr(...)`
   - `module1.run_daily_order_generation(...)`
   - `orchestrator.process_module1_shipments(...)`
   - `run_module4_integrated(...)`
   - `orchestrator.process_module4_production(...)`
   - `module5.main(...)`
   - `orchestrator.process_module5_deployment(...)`
   - `module6.run_daily_physical_flow(...)`
   - `orchestrator.process_module6_delivery(...)`
   - `module3.run_integrated_mode(...)`
   - `save_ending_inventory(...)`
   - `save_daily_state(...)`
6. `InventoryBalanceChecker` 做库存平衡校验
7. `SummaryReportGenerator.generate_all_reports()` 生成汇总报告

### 2.2 数据库模式总顺序

1. `run.py`（仓库根入口，转发到 `src/core/run/`）
2. `src/core/run/run_main.py::main()`
3. `src/core/run/db_runner.py::_run_with_database(...)`
4. `DatabaseInitializer.initialize(...)`
5. `_load_config_from_database(...)`
6. `run_integrated_simulation_from_dict(...)`
7. `ModuleDataWriter.write_module_results_from_dict(...)`
8. `ModuleDataWriter.write_orchestrator_data(...)`
9. `ModuleDataWriter.generate_summary_reports_from_db(...)`

## 3. 标准输入文件格式

### 3.1 配置文件来源

- 本地模式：Excel 工作簿，典型路径如 `config/BC_S5.xlsx`
- 数据库模式：统一 `cfg_*` 表，通过 `config_name` 区分配置

### 3.2 关键配置表与用途

| 表名/Sheet | 主要用途 | 主要被谁使用 |
|---|---|---|
| `Global_Network` | 网络关系、上下游节点 | M3、M5 |
| `Global_LeadTime` | 运输与计划提前期 | M3、M5、M6 |
| `Global_DemandPriority` | 需求优先级 | M4、M5、M6 |
| `Global_seed` | 随机种子 | 主流程、M4、M6 |
| `Global_SpaceCapacity` | 空间约束 | Orchestrator、M5 |
| `M1_DemandForecast` | 周度需求预测 | M1 |
| `M1_InitialInventory` | 初始库存 | Orchestrator、M1 |
| `M1_OrderCalendar` | 下单日历 | M1 |
| `M1_AOConfig` | AO 订单参数 | M1 |
| `M1_DPSConfig` | DPS 拆分规则 | M1 |
| `M1_SupplyChoiceConfig` | 供给选择规则 | M1 |
| `M3_SafetyStock` | 安全库存 | M3、M5 |
| `M4_MaterialLocationLineCfg` | 物料-地点-产线映射、PTF/LSK | M3、M4、M5 |
| `M4_LineCapacity` | 产能数据 | M4 |
| `M4_ChangeoverMatrix` | 换产关系 | M4 |
| `M4_ChangeoverDefinition` | 换产耗时定义 | M4 |
| `M4_ProductionReliability` | 生产可靠率 | M4 |
| `M5_DeployConfig` | MOQ/RV 等调拨规则 | M3、M5 |
| `M5_PushPullModel` | Push/Pull/Soft Push 策略 | M5 |
| `M6_MaterialMD` | MDQ 相关参数 | M6 |
| `M6_TruckTypeSpecs` | 车型规格 | M6 |
| `M6_TruckReleaseCon` | 发车条件表达式 | M6 |
| `M6_DeliveryDelayDistribution` | 延迟分布 | M6 |
| `M6_MDQBypassRules` | MDQ 绕过规则 | M6 |
| `M6_TruckCapacityPlan` | 运力计划 | M6 |

## 4. 标准输出文件格式

### 4.1 每日模块输出命名规则

| 模块 | 文件名模式 | 主要 Sheet |
|---|---|---|
| Module1 | `module1_output_YYYYMMDD.xlsx` | `OrderLog`, `ShipmentLog`, `CutLog`, `SupplyDemandLog`, `Summary` |
| Module3 | `Module3Output_YYYYMMDD.xlsx` | `NetDemand` |
| Module4 | `Module4Output_YYYYMMDD.xlsx` | `ProductionPlan`, `CapacityExceed`, `Validation`, `ChangeoverLog` |
| Module5 | `Module5Output_YYYYMMDD.xlsx` | `DeploymentPlan`, `UnfulfilledLog`, `StockOnHandLog`, `Validation` |
| Module6 | `Module6Output_YYYYMMDD.xlsx` | `DeliveryPlan`, `VehicleLog`, `TruckUsageLog`, `UnsatisfiedMDQLog`, `ValidationLog`, `BypassRuleHitLog` |

### 4.2 Orchestrator 状态文件命名规则

| 文件名模式 | 含义 |
|---|---|
| `unrestricted_inventory_YYYYMMDD.csv` | 当天可用库存快照 |
| `open_deployment_YYYYMMDD.csv` | 当天开放调拨池 |
| `planning_intransit_YYYYMMDD.csv` | 在途记录 |
| `space_quota_YYYYMMDD.csv` | 空间额度快照 |
| `production_plan_backlog_YYYYMMDD.csv` | 已确认生产 backlog |
| `production_gr_YYYYMMDD.csv` | 生产收货记录 |
| `delivery_gr_YYYYMMDD.csv` | 交付收货记录 |
| `shipment_log_YYYYMMDD.csv` | 对客发货日志 |
| `delivery_shipment_log_YYYYMMDD.csv` | 调拨发运日志 |
| `inventory_change_log_YYYYMMDD.csv` | 库存变化流水 |
| `daily_logs_YYYYMMDD.csv` | 日志摘要 |

### 4.3 汇总输出文件

| 输出文件 | 来源 | 说明 |
|---|---|---|
| `full_order_shipment_cut_report.xlsx` | M1 | 订单/发货/缺货汇总 |
| `full_delivery_plan_report.xlsx` | M6 | 物流交付汇总 |
| `full_truck_usage_report.xlsx` | M6 | 车辆使用汇总 |
| `full_exceed_capacity_report.xlsx` | M4 | 超产能汇总 |
| `full_changeover_report.xlsx` | M4 | 换产汇总 |
| `full_deployment_plan_report.xlsx` | M5 | 调拨汇总 |
| `full_production_plan_report.xlsx` | M4 | 生产计划汇总 |
| `historical_inventory_record.csv` | Orchestrator + M1 | 历史库存口径汇总 |

## 5. 核心层函数总表

### 5.1 `src/core/run/`（入口：`run_main.py`）

#### `main(argv=None) -> int`
- 功能：命令行总入口，解析参数、选择本地或数据库模式、初始化输出目录和日志。
- 输入：命令行参数，例如 `--config`、`--start-date`、`--end-date`、`--resume`、`--use-db`。
- 输出：返回码，成功为 `0`。
- 运行顺序：整个程序第一层入口。

#### `_ensure_output_dir(config_path, resume_mode, resume_from, start_date, end_date, interactive, run_suffix) -> Path`
- 功能：决定本次运行到底写到哪个 `run_*` 目录。
- 输入：配置文件路径、续跑参数。
- 输出：一个具体的输出目录路径。
- 文件影响：创建 `outputs/<config>/run_YYYYMMDD_HHMMSS/`。

#### `_run_with_database(ns) -> int`
- 功能：数据库模式总控。
- 输入：数据库连接参数、配置名、日期范围。
- 输出：返回码，成功为 `0`。
- 运行顺序：仅在 `--use-db` 时执行。

#### `_load_config_from_database(db, config_name) -> dict`
- 功能：从统一 `cfg_*` 表读取配置，还原成仿真核心能接受的 `config_dict`。
- 输入：数据库连接对象、配置名。
- 输出：按 Sheet 语义组织的配置字典。
- 表读取：`cfg_global_network`、`cfg_m1_demandforecast` 等。

### 5.2 `src/core/main_integration/`（入口：`simulation_file.py` / `simulation_db.py` / `production_integration.py` / `resume.py`）

#### `run_integrated_simulation(config_path, start_date, end_date, output_base_dir, force_restart) -> dict`
- 功能：本地模式主仿真入口。
- 输入：配置 Excel 路径、日期范围、输出目录。
- 输出：结构化结果字典，常见 key：`results`、`summary_reports`、`balance_report`、`output_directory`。
- 运行顺序：本地模式核心主入口。

#### `run_integrated_simulation_from_dict(config_data, config_name, start_date, end_date, output_base_dir, skip_validation, skip_summary_report) -> dict`
- 功能：数据库模式或内存模式主仿真入口。
- 输入：内存配置字典而不是 Excel 文件。
- 输出：同样返回结构化结果字典。
- 运行顺序：DB 模式在配置读库之后调用。

#### `detect_last_complete_date(output_base_dir, start_date, end_date) -> str | None`
- 功能：检查已有 run 是否具备续跑条件。
- 输入：输出目录、日期范围。
- 输出：最后完整日期；找不到则返回 `None`。
- 文件读取：检查 10 个核心 `orchestrator/*.csv` 文件集合。

#### `check_resume_capability(output_base_dir, start_date, end_date) -> dict`
- 功能：把续跑信息整理成结构化结果。
- 输出字段：`can_resume`、`last_complete_date`、`resume_from_date`、`days_completed`、`days_remaining`、`already_completed`。

#### `restore_orchestrator_state(orchestrator, restore_date, output_base_dir) -> None`
- 功能：从 CSV 恢复 Orchestrator 状态。
- 输入文件：`unrestricted_inventory_*`、`open_deployment_*`、`planning_intransit_*`、`production_gr_*`、`delivery_gr_*`、`shipment_log_*`、`delivery_shipment_log_*`、`daily_logs_*` 等。
- 输出：不返回值，直接修改内存中的 Orchestrator。

#### `run_module4_integrated(config_dict, module3_output_dir, simulation_date, simulation_start, output_dir, skip_file_output, module3_result) -> dict`
- 功能：桥接 M4，使主流程能用统一方式调用生产计划。
- 输入：M4 配置、Module3 输出目录或 `module3_result` 内存结果。
- 输出：`production_df`、`exceed_log`、`issues_df`、`changeover_log`。
- 输出文件：`Module4Output_YYYYMMDD.xlsx`。

#### `load_current_date_production_gr(module4_output_dir, current_date, start_date) -> pd.DataFrame`
- 功能：从历史 M4 输出中找出“今天真正可入库”的生产记录。
- 输入文件：`Module4Output_YYYYMMDD.xlsx` 中的 `ProductionPlan`。
- 输出列：`material`, `location`, `line`, `simulation_date`, `available_date`, `produced_qty`。

#### `load_global_seed(config_dict) -> int`
- 功能：统一读取随机种子。
- 输出：整数种子，默认 `42`。

### 5.3 `src/core/orchestrator/`（入口：`orchestrator_main.py`；辅助：`processors.py`、`views.py`、`persistence.py`、`daily_ops.py`、`inventory_log.py`、`normalize.py`、`models.py`）

#### `initialize_inventory(initial_inventory_df) -> None`
- 输入格式：DataFrame，列至少含 `material`, `location`, `quantity`。
- 功能：初始化 `unrestricted_inventory` 和 `initial_inventory`。

#### `set_space_capacity(space_capacity_df) -> None`
- 输入格式：DataFrame，列至少含 `location`, `eff_from`, `eff_to`, `capacity`。
- 功能：加载空间约束配置。

#### `process_module1_shipments(shipment_df, date) -> None`
- 输入格式：DataFrame，常见列 `date`, `material`, `location`, `quantity`, `demand_type`, `order_id`。
- 功能：扣减库存并写入对客发货日志。
- 运行顺序：M1 执行后立刻调用。

#### `process_module4_production(production_df, date) -> None`
- 输入格式：DataFrame，至少含 `material`, `location`, `available_date`, `produced_qty`。
- 功能：把生产结果写入 backlog，并对当天可用生产做入库。
- 运行顺序：M4 后调用，也可能在每天开头先处理历史可用生产。

#### `process_module5_deployment(deployment_df, date) -> None`
- 输入格式：DataFrame，至少含 `material`, `sending`, `receiving`, `planned_deployment_date`, `deployed_qty`, `demand_element`。
- 功能：生成稳定 `ori_deployment_uid`，写入开放调拨池。
- 运行顺序：M5 后调用。

#### `process_module6_delivery(delivery_df, date) -> None`
- 输入格式：DataFrame，至少含 `material`, `sending`, `receiving`, `actual_ship_date`, `actual_delivery_date`, `delivery_qty`, `ori_deployment_uid`, `vehicle_uid`。
- 功能：发货地扣库存、更新开放调拨、生成在途或直接 GR。
- 运行顺序：M6 后调用。

#### `_process_delivery_arrivals(date) -> None`
- 功能：把当天到达的在途订单转成收货入库。
- 运行顺序：每天正式跑模块前先执行。

#### `cleanup_past_due_open_deployments(date, grace_days, write_audit) -> pd.DataFrame`
- 功能：清理过期的开放调拨。
- 输出文件：审计 CSV。
- 返回：被清理记录明细表。

#### `save_daily_state(date) -> None`
- 功能：把恢复续跑需要的全部状态写盘。
- 输出文件：见第 4.2 节的所有 `orchestrator_*.csv` 规则。

## 6. Module1 函数与文件格式

### 6.1 入口函数

#### `run_daily_order_generation(config_dict, simulation_date, output_dir, orchestrator, skip_file_output, previous_orders_df) -> dict`
- 运行顺序：每天最先执行的业务模块。
- 输入：
  - `config_dict`：M1 所需配置表字典。
  - `simulation_date`：当天日期。
  - `orchestrator`：用于读取可用库存。
  - `previous_orders_df`：DB 模式下历史订单池。
- 输出：
  - `orders_df`
  - `shipment_df`
  - `cut_df`
  - `supply_demand_df`
  - `summary_df`
  - `output_file`
  - `all_orders_for_next_day`
- 输出文件：`module1_output_YYYYMMDD.xlsx`

### 6.2 关键子函数

#### `expand_forecast_to_days_integer_split(demand_weekly, start_date, num_weeks, simulation_end_date) -> pd.DataFrame`
- 输入表列：`material`, `location`, `week`, `quantity`
- 输出表列：`date`, `material`, `location`, `week`, `demand_type`, `quantity`, `original_quantity`
- 功能：把周度预测均分到日度。

#### `generate_daily_orders(sim_date, original_forecast, current_forecast, ao_config, order_calendar, forecast_error) -> pd.DataFrame`
- 输入：日预测、AO 配置、订单日历、预测误差。
- 输出：订单表。
- 功能：生成 AO 和 Normal 订单。

#### `simulate_shipment_for_single_day(simulation_date, order_log, current_inventory, material_list, location_list, production_plan, delivery_plan) -> tuple`
- 输入：订单池、当前库存字典。
- 输出：`shipment_df`, `cut_df`, `updated_inventory_dict`
- 功能：决定今天哪些订单能发、哪些缺货。

#### `generate_supply_demand_log_for_integration(...) -> pd.DataFrame`
- 功能：生成供需日志供下游模块使用。

#### `save_module1_output_with_supply_demand(orders_df, shipment_df, supply_demand_df, output_file, cut_df) -> None`
- 输出 Excel Sheet：
  - `OrderLog`
  - `ShipmentLog`
  - `CutLog`
  - `SupplyDemandLog`
  - `Summary`
- 各 Sheet 标准列：
  - `OrderLog`：`date`, `material`, `location`, `demand_type`, `quantity`, `simulation_date`, `advance_days`
  - `ShipmentLog`：`date`, `material`, `location`, `quantity`, `demand_type`, `order_id`
  - `CutLog`：`date`, `material`, `location`, `quantity`
  - `SupplyDemandLog`：`date`, `material`, `location`, `quantity`, `demand_element`

## 7. Module3 函数与文件格式

#### `load_module1_daily_outputs(module1_output_dir, simulation_date) -> dict`
- 输入文件：`module1_output_YYYYMMDD.xlsx`
- 读取来源：M1 输出目录。
- 输出：包含 AO、FC、SS 等需求信息的字典或 DataFrame 集合。

#### `assign_location_layers(bom_df, demand_locations) -> dict`
- 输出：`(material, location) -> layer` 映射。
- 功能：给网络分层，决定 MRP 的传递顺序。

#### `determine_lead_time(material, sending, receiving, config, indexer) -> int`
- 输出：计划窗口用的提前期天数。

#### `calculate_daily_net_demand(node, date, config) -> dict`
- 输出字段：`ao_gap`, `fc_gap`, `ss_gap`, `total_gap`, `inventory`, `supply`, `demand_ao`, `demand_fc`, `demand_ss` 等。

#### `run_mrp_layered_simulation_daily(config, simulation_date, indexer) -> pd.DataFrame`
- 运行顺序：每天最后一个业务模块执行。
- 输出文件：由集成函数写成 `Module3Output_YYYYMMDD.xlsx`。
- 输出 Sheet：`NetDemand`
- 标准列：`material`, `location`, `layer`, `date`, `ao_gap`, `fc_gap`, `ss_gap`, `total_gap`, `inventory`, `supply`, `demand_ao`, `demand_fc`, `demand_ss`

#### `run_integrated_mode(config, simulation_date, memory_store) -> pd.DataFrame`
- 功能：集成模式入口，供主流程调用。
- 输入：当前配置、日期、可选内存存储对象。
- 输出：净需求 DataFrame。

## 8. Module4 函数与文件格式

#### `load_daily_net_demand(module3_output_dir, simulation_date) -> pd.DataFrame`
- 输入文件：前一天的 `Module3Output_YYYYMMDD.xlsx`
- 读取 Sheet：`NetDemand`
- 过滤：只取 `layer=0` 记录。

#### `build_unconstrained_plan_for_single_day(net_demand, mlcfg, simulation_date, simulation_start, issues) -> pd.DataFrame`
- 输出列：`material`, `location`, `line`, `planned_date`, `uncon_planned_qty`, `simulation_date`, `original_quantity`
- 功能：构建理想生产计划。

#### `optimal_changeover_sequence(batches, co_matrix) -> list`
- 功能：对批次排序，减少换产损耗。

#### `centralized_capacity_allocation_with_changeover(...) -> tuple`
- 输出：`plan_log`, `exceed_log`
- 功能：按产能和换产约束对生产进行实际分配。

#### `extract_line_states_from_plan(plan_df, cap_df, co_def, simulation_date, rate_map) -> dict`
- 输出：各产线日末状态字典。

#### `simulate_production(plan, pr_cfg, seed) -> pd.DataFrame`
- 功能：按照可靠率模拟实际产出。
- 输出：在计划表上补 `produced_qty`。

#### `write_output(plan, exc, issues, changeover_log, out_path, simulation_date, skip_file_output) -> str`
- 输出文件：`Module4Output_YYYYMMDD.xlsx`
- Sheet：`ProductionPlan`, `CapacityExceed`, `Validation`, `ChangeoverLog`

## 9. Module5 函数与文件格式

#### `main(input_path, output_path, sim_start, sim_end, config_dict, module1_output_dir, module4_output_path, orchestrator, current_date, skip_file_output, module1_result, module4_result) -> dict`
- 运行顺序：每天 M4 之后、M6 之前。
- 输出：
  - `deployment_plan`
  - `stock_on_hand_log`
  - `unfulfilled_log`
  - `validation_log`
- 输出文件：`Module5Output_YYYYMMDD.xlsx`

#### `load_integrated_config(...) -> dict`
- 功能：从 `config_dict`、M1、M4 和 Orchestrator 组合出 M5 所需输入。
- 输入可能来自：
  - `module1_output_YYYYMMDD.xlsx`
  - `Module4Output_YYYYMMDD.xlsx`
  - Orchestrator 当前状态视图

#### `collect_node_demands(material, location, config, ...) -> list | DataFrame`
- 功能：汇总节点需求源，包括 SDL、安全库存、订单和上游 gap。

#### `calculate_projected_inventory(material, location, current_inventory, ...) -> float`
- 功能：计算窗口内预测库存。

#### `calculate_available_inventory(material, location, projected_inventory, ...) -> float`
- 功能：计算真正可分配库存。

#### `apply_grouped_moq_rv(demand_rows, location) -> DataFrame`
- 功能：按路径分组后应用 MOQ/RV。

#### `apply_priority_allocation_vectorized(demand_rows, adjusted_qtys, current_stock, demand_priority_map) -> DataFrame | tuple`
- 功能：按优先级分配库存。

#### `apply_receiving_space_quota(...) -> DataFrame`
- 功能：应用接收地空间上限。

#### `push_softpush_allocation(...) -> DataFrame`
- 功能：处理 Push / Soft Push 调拨。

#### `log_outputs(output_path, outputs) -> None`
- 输出 Excel Sheet：`DeploymentPlan`, `UnfulfilledLog`, `StockOnHandLog`, `Validation`
- 排序规则：输出前按固定字段排序，保证结果稳定可比。

## 10. Module6 函数与文件格式

#### `run_daily_physical_flow(config_dict, orchestrator, current_date, output_dir, max_wait_days, random_seed, skip_file_output) -> dict`
- 运行顺序：每天 M5 之后、M3 之前。
- 输出：
  - `delivery_plan`
  - `vehicle_log`
  - `truck_usage`
  - `unsatisfied_log`
  - `validation_log`
  - `bypass_log`
  - `statistics`
- 输出文件：`Module6Output_YYYYMMDD.xlsx`

#### `run_physical_flow_module(...) -> dict`
- 功能：M6 统一主入口，兼容独立模式和集成模式。
- 输入：独立模式用 Excel；集成模式用 `config_dict + orchestrator + current_date`。

#### `load_integrated_config(config_dict, orchestrator, current_date) -> dict`
- 功能：读取开放调拨和 M6 配置，构造物流执行输入。

#### `normalize_capacity_plan(truck_cap_df, current_date) -> pd.DataFrame`
- 功能：把运力计划整理成当前日期可用的容量视图。

#### `sample_delivery_delay(sending, receiving, dist_df) -> int`
- 功能：按延迟分布采样运输延迟天数。

#### `calculate_actual_delivery_date(...) -> pd.Timestamp`
- 功能：给发运记录推算实际到货日。

#### `should_bypass_mdq(material, sending, receiving, bypass_rules) -> bool`
- 功能：判断某条线路是否可绕过 MDQ 规则。

#### `VehiclePacker`
- 功能：装车优化对象。
- 输入：待装货订单、车辆规格和容量。
- 输出：车辆装载结果、剩余容量、车辆日志。

#### 标准输出 Sheet
- `DeliveryPlan`
- `VehicleLog`
- `TruckUsageLog`
- `UnsatisfiedMDQLog`
- `ValidationLog`
- `BypassRuleHitLog`

## 11. Services 层函数与文件格式

### 11.1 `SummaryReportGenerator`

#### `generate_all_reports() -> None`
- 功能：统一生成 8 份 summary 汇总文件。
- 输入来源：每日模块输出 Excel 和 Orchestrator CSV。
- 输出：`summary/*.xlsx` 或 `.csv`。

#### 各报告与来源
- `_generate_order_shipment_cut_report()`：读 M1 的 `OrderLog`、`ShipmentLog`、`CutLog`
- `_generate_delivery_report()`：读 M6 的 `DeliveryPlan`
- `_generate_truck_usage_report()`：读 M6 的 `TruckUsageLog`
- `_generate_capacity_exceed_report()`：读 M4 的 `CapacityExceed`
- `_generate_changeover_report()`：读 M4 的 `ChangeoverLog`
- `_generate_deployment_report()`：读 M5 的 `DeploymentPlan`
- `_generate_production_report()`：读 M4 的 `ProductionPlan`
- `_generate_historical_inventory_report()`：读 Orchestrator CSV + M1 输出

### 11.2 `PerformanceProfiler`
- 功能：对模块或函数做性能采样。
- 输出文件：`logs/profiles/performance_profile_<module>_<timestamp>.txt`

## 12. Utils 层关键函数

#### `setup_logging(log_dir, module_name, level) -> logging.Logger`
- 功能：创建文件 + 控制台双输出日志器。
- 输出文件：`<module_name>_YYYYMMDD_HHMMSS.log`

#### `run_pre_simulation_validation(config) -> ValidationResult`
- 功能：仿真前做统一校验。
- 主要校验范围：Global、M1、M3、M4、M5、M6、跨模块一致性。

#### `initialize_time_manager(sim_start, sim_end, calendar_config) -> SimulationTimeManager`
- 功能：初始化全局时间管理器。

#### `InventoryBalanceChecker.validate_inventory_consistency(...)`
- 功能：检查期初、收发货、期末库存口径是否守恒。

## 13. 数据库函数与表格式

### 13.1 `DatabaseInitializer.initialize(config_name, auto_import_config, verbose) -> dict`
- 功能：数据库准备总入口。
- 输入：配置名、是否自动导入配置。
- 输出：初始化结果字典，含 `success`、`database_created`、`config_imported` 等字段。

### 13.2 `DatabaseConnection.create_table_from_df(df, table_name, if_exists, add_write_time, config_name, config_type) -> bool`
- 功能：根据 DataFrame 自动建表和写入。
- 自动元数据列：`config_name`, `config_type`, `db_write_time`。

### 13.3 `ModuleDataWriter.write_module_results_from_dict(all_results, run_id, if_exists, truncate_first) -> dict`
- 功能：把内存中的模块结果直接写入标准输出表。
- 输入：主流程返回的 `all_results`。
- 输出表映射：
  - M1：`module1_output_orderlog`, `module1_output_shipmentlog`, `module1_output_cutlog`, `module1_output_supplydemandlog`, `module1_output_summary`
  - M3：`module3_output_netdemand`
  - M4：`module4_output_productionplan`, `module4_output_capacityexceed`, `module4_output_validation`, `module4_output_changeoverlog`
  - M5：`module5_output_deploymentplan`, `module5_output_unfulfilledlog`, `module5_output_stockonhandlog`, `module5_output_validation`
  - M6：`module6_output_deliveryplan`, `module6_output_vehiclelog`, `module6_output_truckusagelog`, `module6_output_unsatisfiedmdqlog`, `module6_output_validationlog`, `module6_output_bypassrulehitlog`

### 13.4 `ModuleDataWriter.write_orchestrator_data(orch_dir, run_id, if_exists) -> dict`
- 功能：把 `orchestrator/*.csv` 聚合写入数据库。

### 13.5 `ModuleDataWriter.generate_summary_reports_from_db(run_id, start_date, end_date, if_exists) -> dict`
- 功能：直接从数据库输出表再生成 summary 表。

### 13.6 统一表名规则

#### 配置表
- 统一前缀：`cfg_`
- 示例：`cfg_global_network`, `cfg_m1_demandforecast`, `cfg_m4_linecapacity`

#### 输出表
- 统一前缀：`module*_output_*`, `orchestrator_*`, `summary_output_*`

#### 运行隔离字段
- `run_id`：区分每次运行
- `sim_date`：区分仿真日

## 14. 交接建议

- 要排查“今天为什么结果不对”，先看第 2 节执行顺序，再看对应模块入口函数。
- 要排查“文件为什么没生成”或“续跑为什么失败”，先看第 4 节和第 5 节的 Core 文件规则。
- 要排查“DB 版为什么表里没数据”，先看第 13 节，重点检查 `config_name` 和 `run_id`。
- 要让新人快速入门，建议先按本文件顺序读完 Core、M1、M5、M6，再回头看 M3、M4。

## 15. 按文件的函数手册

本章把前面的主链路说明展开成“按文件查函数”的手册形式。为了保证交接可用性，这里覆盖：

- 对外公开函数
- 主流程一定会走到的关键内部函数
- 重要的数据装配函数、写文件函数、落库函数
- 典型优化路径和验证路径函数

说明：

- 这里优先讲“维护最需要知道的函数”，不强行把每一个极小工具函数都逐个展开。
- 如果某个文件本身只是 facade 兼容层，会重点说明它 re-export 了哪些真实实现。

## 16. `src/core` 文件手册

### 16.1 `src/core/main_integration/` 包

#### `run_integrated_simulation(...)`
- 路径：主运行路径
- 作用：本地模式总控函数，负责把预校验、续跑、状态恢复、模块串联、库存校验、汇总报告整合到一个统一流程里。
- 主要输入：
  - `config_path`：Excel 配置文件路径
  - `start_date`, `end_date`
  - `output_base_dir`
  - `force_restart`
- 主要依赖：`load_configuration(...)`、`run_pre_simulation_validation(...)`、`create_orchestrator(...)`、M1/M4/M5/M6/M3 各模块入口。
- 主要输出：结果字典，含 `results`、`summary_reports`、`balance_report`、`output_directory` 等；同时会写每日 Excel/CSV 文件。
- 运行顺序：整个本地仿真的绝对核心入口。

#### `run_integrated_simulation_from_dict(...)`
- 路径：DB 路径
- 作用：数据库模式和内存模式总控函数，逻辑与 `run_integrated_simulation(...)` 接近，但输入来自内存配置而不是 Excel 文件。
- 主要输入：
  - `config_data`
  - `config_name`
  - `start_date`, `end_date`
  - `skip_validation`, `skip_summary_report`
- 主要输出：结果字典，通常交给 `ModuleDataWriter` 继续写入数据库。

#### `load_configuration(...)`
- 路径：主运行路径
- 作用：从 Excel 中加载全局与各模块配置，并统一做标识符标准化。
- 输入：Excel 文件路径。
- 输出：`config_dict`，按配置表名组织的 DataFrame 字典。

#### `load_configuration_from_dict(...)`
- 路径：DB 路径
- 作用：把数据库读取出来的 `config_data` 恢复成与本地 Excel 加载后结构一致的 `config_dict`。
- 主要输入：数据库读取结果。
- 主要输出：标准化配置字典。

#### `detect_last_complete_date(...)`
- 路径：主运行路径
- 作用：扫描 `orchestrator/` 目录，判断当前 run 最后一个完整仿真日。
- 输入文件：
  - `unrestricted_inventory_YYYYMMDD.csv`
  - `open_deployment_YYYYMMDD.csv`
  - `planning_intransit_YYYYMMDD.csv`
  - `space_quota_YYYYMMDD.csv`
  - `delivery_gr_YYYYMMDD.csv`
  - `production_gr_YYYYMMDD.csv`
  - `shipment_log_YYYYMMDD.csv`
  - `delivery_shipment_log_YYYYMMDD.csv`
  - `inventory_change_log_YYYYMMDD.csv`
  - `daily_logs_YYYYMMDD.csv`
- 输出：最后完整日期字符串或 `None`。

#### `check_resume_capability(...)`
- 路径：主运行路径
- 作用：给续跑功能提供结构化判断结果。
- 输出字段：`can_resume`, `last_complete_date`, `resume_from_date`, `days_completed`, `days_remaining`, `already_completed`。

#### `restore_orchestrator_state(...)`
- 路径：主运行路径
- 作用：把磁盘上的状态 CSV 恢复成内存对象。
- 输入文件：库存、开放调拨、在途、空间配额、backlog、历史日志等 CSV。
- 输出：不返回值，直接修改 `orchestrator` 实例。
- 重点恢复对象：
  - `unrestricted_inventory`
  - `open_deployment`
  - `in_transit`
  - `space_quota`
  - `production_plan_backlog`
  - `shipment_log`
  - `production_gr`
  - `delivery_gr`
  - `delivery_shipment_log`

#### `run_module4_integrated(...)`
- 路径：主运行路径
- 作用：统一桥接 M4 生产计划，优先从内存中的 `module3_result` 取净需求，必要时再回读文件。
- 输入：
  - `config_dict`
  - `module3_output_dir`
  - `simulation_date`, `simulation_start`
  - `output_dir`
  - `module3_result`
- 输出：
  - `production_df`
  - `exceed_log`
  - `issues_df`
  - `changeover_log`
- 输出文件：`Module4Output_YYYYMMDD.xlsx`

#### `load_current_date_production_gr(...)`
- 路径：主运行路径
- 作用：从历史 M4 输出中收集“今天真正应入库”的生产记录。
- 输入文件：`Module4Output_YYYYMMDD.xlsx` 的 `ProductionPlan`。
- 输出列：`material`, `location`, `line`, `simulation_date`, `available_date`, `produced_qty`。

#### `load_global_seed(...)`
- 路径：主运行路径
- 作用：统一读取随机种子，默认回退到 `42`。

#### `set_module_seeds(...)`
- 路径：主运行路径
- 作用：给全局随机过程设种子，保证仿真复现。

#### `_normalize_location / _normalize_material / _normalize_identifiers`
- 路径：主运行路径
- 作用：统一物料、地点、发送地、接收地的格式，避免 join 错配。

### 16.2 `src/core/orchestrator/` 包

#### `DeploymentUID`
- 路径：主运行路径
- 作用：为调拨和在途记录提供稳定唯一键。
- 输入字段：`material`, `sending`, `receiving`, `planned_deploy_date`, `demand_element`, `sequence`。
- 输出：形如 `material|sending|receiving|date|demand_element|000001` 的 UID 字符串。

#### `Orchestrator.__init__(...)`
- 路径：主运行路径
- 作用：初始化全局状态中枢。
- 初始化的核心状态：
  - `unrestricted_inventory`
  - `open_deployment`
  - `in_transit`
  - `production_gr`
  - `delivery_gr`
  - `shipment_log`
  - `delivery_shipment_log`
  - `production_plan_backlog`
  - `daily_beginning_inventory`
  - `daily_ending_inventory`
  - `daily_logs`

#### `initialize_inventory(...)`
- 路径：主运行路径
- 输入表格式：`material`, `location`, `quantity`
- 作用：初始化库存账本和初始库存副本。

#### `set_space_capacity(...)`
- 路径：主运行路径
- 输入表格式：`location`, `eff_from`, `eff_to`, `capacity`
- 作用：设置空间/库容约束。

#### `get_unrestricted_inventory_view(date)`
- 路径：主运行路径
- 输出格式：`date`, `material`, `location`, `quantity`
- 作用：把当前库存字典整理成 DataFrame 供模块消费或落盘。

#### `get_current_unrestricted_inventory()`
- 路径：主运行路径
- 输出：`{(material, location): quantity}`
- 作用：供 M1、M5、M6 快速取库存。

#### `get_planning_intransit_view(date)`
- 路径：主运行路径
- 输出列：`transit_uid`, `date`, `material`, `sending`, `receiving`, `actual_ship_date`, `actual_delivery_date`, `quantity`, `ori_deployment_uid`, `vehicle_uid`
- 作用：提供计划中的在途视图。

#### `get_open_deployment_view(date)`
- 路径：主运行路径
- 输出列：`material`, `sending`, `receiving`, `planned_deployment_date`, `deployed_qty`, `demand_element`, `ori_deployment_uid`
- 作用：给 M6 提供待发运调拨池。

#### `get_space_quota_view(date)`
- 路径：主运行路径
- 输出列：`receiving`, `date`, `max_qty`
- 作用：为 M5 提供接收端空间约束。

#### `process_module1_shipments(shipment_df, date)`
- 路径：主运行路径
- 输入表：`ShipmentLog` DataFrame
- 作用：扣减库存并记录对客发货。
- 运行顺序：M1 执行后立即调用。

#### `process_module4_production(production_df, date)`
- 路径：主运行路径
- 输入表：生产计划结果 DataFrame
- 作用：先把未来生产缓存进 backlog，再把当天可用生产入库。
- 运行顺序：M4 后调用，也会在每天开始时处理历史已成熟生产。

#### `process_module5_deployment(deployment_df, date)`
- 路径：主运行路径
- 输入表：`DeploymentPlan`
- 作用：生成稳定 `ori_deployment_uid` 并加入开放调拨池。
- 关键点：写入前稳定排序，保证 UID 可复现。

#### `process_module6_delivery(delivery_df, date)`
- 路径：主运行路径
- 输入表：`DeliveryPlan`
- 作用：扣减发送地库存、更新开放调拨、生成在途或当天 GR。

#### `_process_delivery_arrivals(date)`
- 路径：主运行路径
- 作用：把当天到达的在途记录转成 `delivery_gr` 并入库。

#### `cleanup_past_due_open_deployments(date, grace_days, write_audit)`
- 路径：主运行路径
- 作用：清理已经过期太久的开放调拨。
- 输出：被清理记录 DataFrame 和审计文件。

#### `save_beginning_inventory(date)` / `save_ending_inventory(date)`
- 路径：主运行路径
- 作用：保存期初、期末库存快照。

#### `save_daily_state(date)`
- 路径：主运行路径
- 作用：把续跑和审计需要的所有状态写入 CSV。
- 输出文件：见本手册第 4 节 `orchestrator_*.csv` 规则。

#### `get_summary_statistics()`
- 路径：主运行路径
- 作用：收集库存、调拨、在途、日志等统计指标，供主流程和汇总使用。

### 16.3 `src/core/run/` 包

#### `main(argv=None)`
- 路径：主运行路径
- 作用：CLI 总入口，决定走本地还是 DB 模式。

#### `_list_existing_runs(...)`
- 路径：主运行路径
- 作用：列出某个配置对应的所有 `run_*` 目录，并给出续跑能力分析。

#### `_prompt_user_run_selection(...)`
- 路径：主运行路径
- 作用：当存在多个历史运行目录时，让用户交互选择要续跑哪一个。

#### `_ensure_output_dir(...)`
- 路径：主运行路径
- 作用：统一输出目录管理和 resume 目录选择。

#### `_write_results_to_local(...)`
- 路径：可选路径
- 作用：把内存中的模块结果再写回本地 Excel，常用于 DB 模式附带导出本地结果。

#### `_run_with_database(ns)`
- 路径：DB 路径
- 作用：数据库模式总入口，负责建库、导入配置、启动仿真、结果落库。

#### `_load_config_from_database(db, config_name)`
- 路径：DB 路径
- 作用：从 `cfg_*` 表读取指定配置。

## 17. `src/modules/demand_planning` 文件手册

### 17.1 `integration.py`

#### `run_daily_order_generation(...)`
- 路径：主运行路径
- 作用：M1 日主入口。
- 主要输入：
  - `config_dict`
  - `simulation_date`
  - `output_dir`
  - `orchestrator`
  - `previous_orders_df`
- 主要输出：
  - `orders_df`
  - `shipment_df`
  - `cut_df`
  - `supply_demand_df`
  - `summary_df`
  - `output_file`
  - `all_orders_for_next_day`
- 输出文件：`module1_output_YYYYMMDD.xlsx`

#### `_validate_config(...)`
- 路径：主运行路径
- 作用：确认 M1 的预测、AO、订单日历等输入完整。

#### `_prepare_forecasts(...)`
- 路径：主运行路径
- 作用：先做 DPS 和 supply choice，再把周预测拆成订单用和 SDL 用两套日预测。

#### `_merge_with_history(...)`
- 路径：主运行路径
- 作用：把历史未执行完的订单和今天新订单合成累积订单池。

#### `_generate_shipments(...)`
- 路径：主运行路径
- 作用：用可用库存去满足当天订单，生成发货和缺货。

#### `generate_supply_demand_log_for_integration(...)`
- 路径：主运行路径
- 作用：生成供需日志，供后续模块使用。

### 17.2 `order.py`

#### `generate_daily_orders(...)`
- 路径：主运行路径
- 作用：生成 AO 和 normal 订单。

#### `_compute_ml_avg_demand(...)`
- 路径：主运行路径
- 作用：计算物料-地点平均需求，为 AO/normal 订单提供均值基础。

#### `_generate_ao_orders(...)`
- 路径：主运行路径
- 作用：按照 AO 配置和提前天数生成 AO 订单。

#### `_generate_normal_orders(...)`
- 路径：主运行路径
- 作用：生成常规订单。

#### `_aggregate_orders(...)`
- 路径：主运行路径
- 作用：把订单标准化并聚合到统一粒度。

#### `generate_quantity_with_percent_error(...)`
- 路径：可选路径
- 作用：给订单量施加误差，保证非负。

#### `consume_forecast_ao_logic(...)` / `consume_forecast_normal_logic(...)`
- 路径：兼容/理解路径
- 作用：说明 AO 和 normal 订单如何消耗预测。

### 17.3 `consume.py`

#### `consume_orders(...)`
- 路径：主运行路径
- 作用：订单消耗总入口。

#### `consume_ao_orders_serial(...)` / `consume_normal_orders_serial(...)`
- 路径：可选路径
- 作用：串行版订单消耗。

#### `_consume_single_order(...)`
- 路径：可选路径
- 作用：单订单扣减 forecast。

#### `_consume_ao_orders_parallel(...)` / `_consume_normal_orders_parallel(...)`
- 路径：可选路径
- 作用：多进程并行消耗。

### 17.4 `consume_optimized.py`

#### `consume_orders_vectorized(...)`
- 路径：主运行路径
- 作用：M1 当前默认优化消耗实现。

#### `consume_orders_duckdb(...)`
- 路径：可选路径
- 作用：DuckDB 版订单消耗。

#### `_consume_orders_fast(...)`
- 路径：主运行路径
- 作用：直接对数量数组做原地扣减，是当前热点优化点之一。

### 17.5 `shipment.py`

#### `generate_shipment_with_inventory_check(...)`
- 路径：主运行路径
- 作用：M1 发货总入口。

#### `simulate_shipment_for_single_day(...)`
- 路径：主运行路径
- 作用：发货/缺货核心算法。

#### `_build_available_inventory_from_orchestrator(...)`
- 路径：主运行路径
- 作用：从 Orchestrator 拼装真实可用库存。

### 17.6 `forecast.py`

#### `expand_forecast_to_days_integer_split(...)`
- 路径：主运行路径
- 作用：周预测拆分到日度。

#### `prepare_daily_forecasts(...)`
- 路径：主运行路径
- 作用：生成订单预测和 SDL 预测。

### 17.7 `dps.py`

#### `apply_dps(...)`
- 路径：主运行路径
- 作用：把部分需求拆去 `dps_location`。

#### `apply_supply_choice(...)`
- 路径：主运行路径
- 作用：把 supply choice 调整量加到预测上。

### 17.8 `io_utils.py`

#### `load_previous_orders(...)`
- 路径：主运行路径
- 作用：读取最近相关天数的历史 OrderLog。
- 输入文件：`module1_output_*.xlsx`

#### `save_module1_output_with_supply_demand(...)`
- 路径：主运行路径
- 作用：把 M1 结果写成标准 Excel。
- Sheet：`OrderLog`, `ShipmentLog`, `CutLog`, `SupplyDemandLog`, `Summary`

#### `_build_summary(...)`
- 路径：主运行路径
- 作用：生成 M1 的 Summary sheet。

## 18. `src/modules/mrp_planning` 文件手册

### 18.1 `integration.py`

#### `run_integrated_mode(...)`
- 路径：主运行路径
- 作用：M3 集成模式总入口。
- 主要输入：M1 输出、Orchestrator 视图、M3 配置、M4/M5 部分配置。
- 主要输出：`net_demand_df` 和 `Module3Output_YYYYMMDD.xlsx`。

#### `_load_static_configs(...)`
- 路径：主运行路径
- 作用：预装载 SafetyStock、Network、LeadTime、DeployConfig 等静态配置。

#### `_load_module1_data(...)`
- 路径：主运行路径
- 作用：从内存或 M1 文件读取 SDL / Shipment / Order。

#### `_load_orchestrator_data(...)`
- 路径：主运行路径
- 作用：拉取期初库存、在途、交付收货、生产收货、开放调拨等。

#### `_calculate_net_demand(...)`
- 路径：主运行路径
- 作用：调用 MRP 核心计算。

#### `_save_daily_output(...)`
- 路径：主运行路径
- 作用：写出 `Module3Output_YYYYMMDD.xlsx`。

### 18.2 `mrp_simulation.py`

#### `run_mrp_layered_simulation_daily(...)`
- 路径：主运行路径
- 作用：M3 的真正核心算法入口，按层逐步向上游传 gap。

#### `_init_simulation_context(...)`
- 路径：主运行路径
- 作用：预建活动网络、层级、缓存和上下文对象。

#### `_prepare_production_df(...)`
- 路径：主运行路径
- 作用：清洗生产数据，区分今天和未来的供应。

#### `_process_layer(...)`
- 路径：主运行路径
- 作用：处理一整层节点，自动选择并行或 DuckDB 批量方式。

#### `_process_layer_parallel(...)`
- 路径：主运行路径
- 作用：线程池逐节点并行处理。

#### `_process_layer_batch(...)`
- 路径：可选路径
- 作用：DuckDB 批量处理层内节点。

### 18.3 `node_processor.py`

#### `NodeProcessor.process(...)`
- 路径：主运行路径
- 作用：单节点净需求计算 + 向父节点传 gap 的完整封装。

#### `_compute_parent_gaps(...)`
- 路径：主运行路径
- 作用：对 gap 应 MOQ/RV 后向上游分摊。

### 18.4 `net_demand.py`

#### `calculate_daily_net_demand(...)`
- 路径：主运行路径
- 作用：标准版净需求计算。

#### `calculate_daily_net_demand_indexed(...)`
- 路径：主运行路径
- 作用：索引加速版净需求计算。

#### `_calculate_supply_side(...)`
- 路径：主运行路径
- 作用：汇总库存、在途、GR、生产等供给。

#### `_calculate_demand_side(...)`
- 路径：主运行路径
- 作用：汇总 AO、forecast 和 safety stock 需求。

#### `_calculate_gaps(...)`
- 路径：主运行路径
- 作用：按 AO -> FC -> SS 顺序消耗供给得到缺口。

### 18.5 其他重要文件

- `data_indexer.py`：给日模拟建立 O(1) 索引。
- `lead_time.py`：根节点 horizon 与普通节点 lead time 计算。
- `layer_assignment.py`：按 `(material, location)` 分层。
- `config_loader.py`：装载 M3 配置和 M1 每日输出。
- `duckdb_batch_calculator.py`：批量净需求和回退实现。
- `utils.py`：MOQ/RV、PTF/LSK、最大余数法、标识规范化。

## 19. `src/modules/production_planning` 文件手册

### 19.1 `main.py`

#### `run_daily_production_planning(...)`
- 路径：主运行路径
- 作用：M4 对外主入口。

#### `DailyProductionPlanner.run()`
- 路径：主运行路径
- 作用：执行完整 M4 流程：加载配置 -> 读取净需求 -> 无约束计划 -> 产能分配 -> 生产仿真 -> 输出。

#### `_load_net_demand(...)`
- 路径：主运行路径
- 作用：读取前一日 `Module3Output_*.xlsx`。

#### `_allocate_capacity(...)`
- 路径：主运行路径
- 作用：调用集中产能分配核心函数。

#### `_simulate_and_finalize(...)`
- 路径：主运行路径
- 作用：按可靠率产生 `produced_qty` 并追加校验。

#### `_save_states(...)`
- 路径：主运行路径
- 作用：保存 `line_states_*.json` 和 `allocated_capacity_*.json`。

### 19.2 `plan_builder.py`

#### `build_unconstrained_plan_for_single_day(...)`
- 路径：主运行路径
- 作用：构建当日理想生产需求。

#### `_build_plan_for_material(...)`
- 路径：主运行路径
- 作用：按物料地点和审查日逻辑生成计划。

#### `_create_plan_record(...)`
- 路径：主运行路径
- 作用：把需求数量转换成批量化生产计划记录。

#### `optimal_changeover_sequence(...)`
- 路径：主运行路径
- 作用：尽量减少换产时间。

### 19.3 `capacity_allocator.py`

#### `centralized_capacity_allocation_with_changeover(...)`
- 路径：主运行路径
- 作用：M4 最核心函数，负责约束产能分配和换产处理。

#### `CapacityAllocator.allocate(...)`
- 路径：主运行路径
- 作用：逐线逐批执行产能分配。

#### `_allocate_batch(...)`
- 路径：主运行路径
- 作用：在单个窗口里分配单批次生产。

#### `_lookup_changeover(...)`
- 路径：主运行路径
- 作用：查换产定义和耗时。

#### `simulate_production(...)`
- 路径：主运行路径
- 作用：根据可靠率模拟实际产出。

#### `validate_capacity_allocation(...)`
- 路径：校验路径
- 作用：检查产能是否冲突或超配。

#### `calculate_changeover_metrics(...)`
- 路径：主运行路径
- 作用：汇总换产指标。

#### `extract_allocated_capacity_from_plan(...)` / `extract_line_states_from_plan(...)`
- 路径：主运行路径
- 作用：把计划转换为可持久化的状态。

### 19.4 其他重要文件

- `output_writer.py`：写 `ProductionPlan`, `CapacityExceed`, `Validation`, `ChangeoverLog`。
- `config_loader.py`：装载 M4 配置并校验。
- `demand_loader.py`：读取 layer=0 净需求。
- `state_manager.py`：跨日状态持久化。
- `duckdb_batch_calculator.py`：生产仿真加速。
- `utils.py`：审查日、窗口、批量取整、补列等工具。
- `types.py`：M4 状态和输出结构。

## 20. `src/modules/deployment_planning` 文件手册

### 20.1 `main.py`

#### `main(...)`
- 路径：主运行路径
- 作用：M5 主入口。
- 主要输出：
  - `deployment_plan`
  - `stock_on_hand_log`
  - `unfulfilled_log`
  - `validation_log`

#### `_initialize_soh_dict(...)`
- 路径：主运行路径
- 作用：初始化库存字典。

#### `_process_layer_demands(...)`
- 路径：主运行路径
- 作用：对某层节点统一收集需求，支持向量化和并行。

#### `_allocate_pipeline_sources(...)`
- 路径：主运行路径
- 作用：用 pipeline 供给覆盖缺口。

#### `_process_gaps_and_create_plans(...)`
- 路径：主运行路径
- 作用：生成调拨计划行和未满足日志。

#### `_update_soh_dict(...)`
- 路径：主运行路径
- 作用：把当天部署结果写回库存视图。

### 20.2 `data_loader.py`

#### `load_integrated_config(...)`
- 路径：主运行路径
- 作用：集成模式装配 M5 所需所有输入。

#### `load_config(...)`
- 路径：主运行路径
- 作用：文件模式读取配置。

#### `StaticConfigCache`
- 路径：可选路径
- 作用：缓存静态配置，减少重复读和重复标准化。

#### `load_module1_daily_shipment(...)` / `load_module1_daily_orders(...)`
- 路径：主运行路径
- 输入文件：`module1_output_YYYYMMDD.xlsx`

#### `load_orchestrator_delivery_gr(...)` / `load_orchestrator_open_deployment(...)`
- 路径：主运行路径
- 输入：Orchestrator 视图。

### 20.3 `allocation.py`

#### `apply_grouped_moq_rv(...)`
- 路径：主运行路径
- 作用：按路径分组应用 MOQ/RV。

#### `apply_priority_allocation_vectorized(...)`
- 路径：主运行路径
- 作用：优先级分配库存，是 M5 的关键热点函数。

#### `allocate_pipeline_supply(...)`
- 路径：主运行路径
- 作用：把未来供给分配给剩余缺口。

#### `apply_receiving_space_quota(...)`
- 路径：主运行路径
- 作用：应用接收空间上限。

### 20.4 `demand_collector.py`

#### `collect_node_demands(...)`
- 路径：主运行路径
- 作用：单节点需求收集总入口。

#### `_collect_sdl_demands(...)`
- 路径：主运行路径
- 作用：从 SDL 收集窗口内需求。

#### `_collect_safety_stock_demands(...)`
- 路径：主运行路径
- 作用：收集安全库存补货需求。

#### `_collect_order_demands(...)`
- 路径：主运行路径
- 作用：收集 AO / normal 订单需求。

#### `_collect_gap_demands(...)`
- 路径：主运行路径
- 作用：收集上下游 gap 传播需求。

### 20.5 `inventory.py`

#### `calculate_available_inventory(...)`
- 路径：主运行路径
- 作用：计算当前真正可分配库存。

#### `calculate_projected_inventory(...)`
- 路径：主运行路径
- 作用：计算预测库存，考虑未来在途和未来生产。

#### `build_open_deployment_inbound(...)`
- 路径：主运行路径
- 作用：把开放调拨转换成接收端未来入库视图。

### 20.6 `push_allocation.py`

#### `push_softpush_allocation(...)`
- 路径：主运行路径
- 作用：执行 push / soft-push 补货逻辑。

#### `_calculate_receiving_ss_data(...)`
- 路径：主运行路径
- 作用：准备接收端安全库存目标。

#### `_calculate_commitments(...)`
- 路径：主运行路径
- 作用：估算未来承诺消耗。

#### `_select_push_level(...)`
- 路径：主运行路径
- 作用：选出 push 级别。

#### `_allocate_push_quantities(...)`
- 路径：主运行路径
- 作用：把 push 量分摊到多个接收端。

### 20.7 `validation.py`

#### `validate_config_before_run(...)`
- 路径：校验路径
- 作用：M5 配置运行前校验。

#### `log_outputs(...)`
- 路径：主运行路径
- 作用：把 M5 四张表写到 `Module5Output_YYYYMMDD.xlsx`。

### 20.8 其他重要文件

- `cache_utils.py`：索引、PTF/LSK、lead time、活动网络缓存。
- `demand_collector_vectorized.py`：向量化批量需求收集。
- `duckdb_batch_calculator.py`：DuckDB 批量 MOQ/RV 与优先级分配。
- `batch_optimizer.py`：分层预过滤和索引预热。
- `horizon_batch_calculator.py`：批量 horizon 预计算。
- `multiprocess_optimizer.py`：多进程层处理。
- `normalizer.py`：标识符标准化。

## 21. `src/modules/logistics_execution` 文件手册

### 21.1 `main.py`（原 `module6.py` 主流程）

#### `run_daily_physical_flow(...)`
- 路径：主运行路径
- 作用：M6 单日集成入口。
- 输出：`delivery_plan`, `vehicle_log`, `truck_usage`, `unsatisfied_log`, `validation_log`, `bypass_log`, `statistics`
- 输出文件：`Module6Output_YYYYMMDD.xlsx`

#### `run_physical_flow_module(...)`
- 路径：主运行路径
- 作用：M6 总入口，兼容独立模式和集成模式。

#### `_initialize_run_params(...)`
- 路径：主运行路径
- 作用：统一独立模式和集成模式参数。

#### `_prepare_data(...)`
- 路径：主运行路径
- 作用：加载配置、部署计划、库存、校验结果。

#### `_run_simulation_loop(...)`
- 路径：主运行路径
- 作用：真正执行当天物流仿真。

#### `_generate_outputs(...)`
- 路径：主运行路径
- 作用：整理多张输出表并决定是否写文件。

### 21.2 `config_loader.py`

#### `load_integrated_config(...)`
- 路径：主运行路径
- 作用：集成模式装配 M6 输入。

#### `load_standalone_config(...)`
- 路径：主运行路径
- 作用：文件模式读取 M6 配置。

### 21.3 `delivery_processor.py`

#### `sample_delivery_delay(...)`
- 路径：主运行路径
- 作用：采样运输延迟。

#### `should_bypass_mdq(...)`
- 路径：主运行路径
- 作用：判断是否绕过 MDQ。

#### `calculate_lead_time(...)`
- 路径：主运行路径
- 作用：读取路线时效参数。

#### `calculate_actual_delivery_date(...)`
- 路径：主运行路径
- 作用：推算实际交货日期。

#### `create_delivery_record(...)`
- 路径：主运行路径
- 作用：生成 DeliveryPlan 行。

#### `create_bypass_record(...)`
- 路径：主运行路径
- 作用：生成 BypassRuleHitLog 行。

#### `create_unsatisfied_record(...)`
- 路径：主运行路径
- 作用：生成 UnsatisfiedMDQLog 行。

### 21.4 `inventory_manager.py`

#### `calculate_physical_inventory(...)`
- 路径：主运行路径
- 作用：计算 M6 视角下可见库存。

#### `update_inventory_after_load(...)`
- 路径：主运行路径
- 作用：装车后扣减库存。

#### `calculate_inventory_limit(...)`
- 路径：主运行路径
- 作用：决定单个需求行的最大可装数量。

### 21.5 `capacity_manager.py`

#### `normalize_capacity_plan(...)`
- 路径：主运行路径
- 作用：把运力计划标准化成按天容量。

#### `build_capacity_map(...)`
- 路径：主运行路径
- 作用：预建容量查询字典。

#### `get_truck_capacity(...)`
- 路径：主运行路径
- 作用：查某日某路线某车型可用台数。

#### `get_optimal_truck_sequence(...)`
- 路径：主运行路径
- 作用：生成推荐车型顺序。

### 21.6 `vehicle_packer.py`

#### `VehiclePacker`
- 路径：主运行路径
- 作用：核心装车器。

#### `VehiclePacker.add_demand(...)`
- 路径：主运行路径
- 作用：尝试把一条需求装进车里。

#### `create_vehicle_log_entry(...)`
- 路径：主运行路径
- 作用：生成 VehicleLog 行。

#### `determine_trigger_cause(...)`
- 路径：主运行路径
- 作用：判断车辆触发发运的原因。

### 21.7 `validators.py`

#### `check_and_deduplicate(...)`
- 路径：校验路径
- 作用：检查并去重 M6 配置表。

#### `validate_deployment_plan(...)`
- 路径：校验路径
- 作用：校验调拨计划输入是否合法。

#### `generate_validation_report(...)`
- 路径：校验路径
- 作用：输出验证报告。

### 21.8 `expression_evaluator.py`

#### `SafeExpressionEvaluator`
- 路径：主运行路径
- 作用：安全执行规则表达式，防止直接 `eval` 带来的风险。

### 21.9 `duckdb_batch_calculator.py`

#### `batch_sample_delivery_delays_duckdb(...)`
- 路径：可选路径
- 作用：批量采样路线延迟。

## 22. `src/services` 文件手册

### 22.1 `summary_report_generator.py`

#### `SummaryReportGenerator`
- 路径：可选路径
- 作用：全周期汇总生成器。

#### `generate_all_reports()`
- 路径：可选路径
- 作用：统一生成 8 种 summary 报告。

#### `_generate_order_shipment_cut_report()`
- 输入：M1 的 `OrderLog`, `ShipmentLog`, `CutLog`
- 输出：订单/发货/缺货汇总。

#### `_generate_delivery_report()`
- 输入：M6 的 `DeliveryPlan`
- 输出：交付汇总。

#### `_generate_truck_usage_report()`
- 输入：M6 的 `TruckUsageLog`
- 输出：卡车使用汇总。

#### `_generate_capacity_exceed_report()`
- 输入：M4 的 `CapacityExceed`
- 输出：超产能汇总。

#### `_generate_changeover_report()`
- 输入：M4 的 `ChangeoverLog`
- 输出：换产汇总。

#### `_generate_deployment_report()`
- 输入：M5 的 `DeploymentPlan`
- 输出：部署汇总。

#### `_generate_production_report()`
- 输入：M4 的 `ProductionPlan`
- 输出：生产汇总。

#### `_generate_historical_inventory_report()`
- 输入：Orchestrator CSV + M1 输出
- 输出：历史库存口径汇总。

### 22.2 `performance_profiler.py`

#### `PerformanceProfiler`
- 路径：可选路径
- 作用：通过 cProfile 收集性能热点。

#### `profile_function(...)`
- 路径：可选路径
- 作用：装饰器式性能分析。

## 23. `src/utils` 文件手册

### 23.1 校验和基础设施

#### `config_validator.py`
- `ConfigValidator.validate_all_configurations`：全量配置预校验总控。
- `_validate_global_configs`：检查 Global 表。
- `_validate_module1_configs` 到 `_validate_module6_configs`：逐模块检查。
- `_validate_cross_module_consistency`：做跨模块一致性校验。
- `run_pre_simulation_validation`：对外统一入口。

#### `validation_manager.py`
- `ValidationManager`：统一错误、警告、信息收集器。
- `add_error/add_warning/add_info`：记录校验消息。
- `write_report`：输出文本验证报告。

#### `logger_config.py`
- `DualLogger`：双通道日志器。
- `PrintRedirector`：把 `print` 重定向进日志。
- `setup_logging`：初始化日志。

#### `time_manager.py`
- `SimulationTimeManager`：统一仿真日期管理。
- `initialize_time_manager/get_time_manager`：全局实例入口。

#### `inventory_balance_checker.py`
- `InventoryBalanceChecker`：库存守恒检查器。
- `check_daily_balance`：单日检查。
- `check_period_balance`：全周期检查。

### 23.2 资源和并行相关

#### `resource_config.py`
- `get_cpu_count/get_optimal_threads/get_optimal_workers`：资源建议。
- `get_total_memory_gb/get_available_memory_gb/get_optimal_memory`：内存建议。
- `get_duckdb_config`：DuckDB 资源参数建议。

#### `cpu_config.py`
- `get_cpu_count/get_max_workers/get_optimal_workers`：CPU worker 建议。

#### ~~`optimization_config.py`~~（**第三阶段已删除**）
参数改由 `config/defaults.yaml` 承载，通过 `src/utils/defaults.py` 的 `get_default(...)` 读取。

#### ~~`parallel_optimizer.py`~~（**第三阶段已删除，未使用**）

#### ~~`process_pool_executor.py`~~（**第三阶段已删除，未使用**）

#### ~~`multiprocess_executor.py`~~（**第三阶段已删除，未使用**）

#### ~~`high_perf_executor.py`~~（**第三阶段已删除，未使用**）

### 23.3 缓存、内存和 DuckDB

#### `memory_data_store.py`
- `MemoryDataStore`：DuckDB 内存表中转站。
- `enable_memory_mode/disable_memory_mode/is_memory_mode_enabled`：开关控制。
- `write_module1_output/write_module3_output/write_module4_output/write_module5_output/write_module6_output`：按模块写内存表。
- `read_module3_net_demand/read_module4_production_plan`：回读关键结果。

#### `simulation_cache.py`
- `SimulationCache`：通用仿真缓存。
- `initialize_simulation_cache/get_simulation_cache/clear_simulation_cache`：全局缓存入口。

#### ~~`performance.py`~~（**第三阶段已删除**）
- 原 `normalize_*_vectorized` 已统一迁入 `src/utils/normalization.py`
- 其余辅助函数并入 `src/services/performance_profiler.py`

#### ~~`duckdb_sql_wrapper.py`~~（**第三阶段已删除**）
- 原 `DuckDBSQL` 能力合并进 `src/utils/duckdb_accelerator.py`，对外接口改为 `DuckDBAccelerator.*`。

#### `duckdb_optimizer.py`
- `DuckDBOptimizer`：DuckDB 优化器与索引管理。
- `build_indexes_with_duckdb`：用 DuckDB 预建索引。

#### `duckdb_accelerator.py`
- `DuckDBAccelerator`：DuckDB 加速入口。
- `get_accelerator`：获取全局实例。

## 24. `pgsql_db` 文件手册

### 24.1 活跃主链路文件

#### `db_connection.py`
- `DatabaseConnection`：PostgreSQL 底层连接和建表写表封装。
- `database_exists/create_database_if_not_exists/test_connection`：数据库存在性和连通性管理。
- `get_all_tables/read_table/execute_query`：读表和查询。
- `create_table_from_df`：核心写表函数，自动加元数据列和建表。

#### `db_initializer.py`
- `DatabaseInitializer`：建库、检查配置、自动导入配置的统一入口。
- `initialize`：DB 模式最重要的准备函数。
- `check_config_data_exists/find_config_file/import_config_from_excel`：配置检测与导入。
- `initialize_database`：对外便捷封装。

#### `excel_importer.py`
- `ExcelImporter`：把 Excel 配置导入 `cfg_*` 表。
- `import_excel_file`：单配置导入主入口。
- `import_config_files`：批量导入便捷函数。

#### `module_data_writer.py`
- `ModuleDataWriter`：结果落库中心类。
- `truncate_output_tables`：按 `run_id` 删除旧结果。
- `write_module_results_from_dict`：把主流程内存结果直接写库。
- `write_orchestrator_data`：把 Orchestrator CSV 落库。
- `generate_summary_reports_from_db`：从数据库表再生成 summary 表。

#### `table_mapping.py`
- `get_config_table_name`：把 Excel Sheet 映射为 `cfg_*` 表名。
- `get_output_table_name`：把输出文件或 key 映射成标准输出表。
- `get_all_output_tables`：列出全部输出表名。

#### `table_schemas.py`
- `get_columns`：为空表提供标准列。
- `get_all_module_tables`：列出各模块空表 schema。

### 24.2 当前重要的优化/实验文件

#### `optimized_processor.py`
- `OptimizedDataProcessor`：高性能数据处理基础设施。

#### `module_engine.py`
- `ModuleCalculationEngine`：批量净需求、订单消耗、层需求收集等优化入口。

#### `module_optimizers.py`
- `Module5Optimizer`, `Module6Optimizer`：模块级 DuckDB 优化器。

#### `optimized_simulation.py`
- `run_optimized_simulation_from_dict`：优化版数据库仿真入口。
- 说明：当前生产 DB 主链未强依赖它。

#### `duckdb_integration.py`
- `calculate_net_demand_batch_duckdb`、`apply_moq_rv_batch_duckdb`、`priority_allocation_batch_duckdb`：关键批量算法桥接函数。

#### `high_performance_engine.py`
- `DuckDBCalculator`、`HybridQueryEngine`、`IncrementalComputeManager`：更高阶的性能实验基础设施。

#### `performance_dashboard.py`
- `PerformanceDashboard`：采集和输出性能指标。
- `RealTimeMonitor`：实时进度监控。

## 25. 五个业务子包的入口（第三阶段后）

> **第三阶段更新（2026-04-10）**：原 `src/modules/module1.py`～`module6.py` 5 个单文件门面已全部删除。  
> 外部代码现通过 `from src.modules.demand_planning import run_daily_order_generation` 等直接使用子包；  
> 为兼容历史写法，`src/modules/__init__.py` 中保留 `module1 = demand_planning` 等别名，`import src.modules.module1` 仍可工作。

### `src/modules/demand_planning/`（原 `module1.py`）
- 入口文件：`integration.py`
- 重点函数：`run_daily_order_generation`, `generate_daily_orders`, `generate_shipment_with_inventory_check`, `expand_forecast_to_days_integer_split`, `apply_dps`, `apply_supply_choice`, `load_config`

### `src/modules/mrp_planning/`（原 `module3.py`）
- 入口文件：`integration.py` + `mrp_simulation.py`
- 重点函数：`run_integrated_mode`, `run_mrp_layered_simulation_daily`, `calculate_daily_net_demand`, `assign_location_layers`, `determine_lead_time`

### `src/modules/production_planning/`（原 `module4.py`）
- 入口文件：`main.py`
- 重点函数：`run_daily_production_planning`, `DailyProductionPlanner`, `build_unconstrained_plan_for_single_day`, `centralized_capacity_allocation_with_changeover`, `simulate_production`

### `src/modules/deployment_planning/`（原 `module5.py`）
- 入口文件：`main.py`
- 重点函数：`main`, `load_integrated_config`, `collect_node_demands`, `apply_grouped_moq_rv`, `apply_priority_allocation_vectorized`, `push_softpush_allocation`

### `src/modules/logistics_execution/`（原 `module6.py`）
- 入口文件：`main.py`（与 `simulation.py`、`output_writer.py` 协作）
- 重点入口：`run_daily_physical_flow`, `run_physical_flow_module`

## 26. 建议的阅读顺序

如果你想按"真正能接手项目"的顺序看函数，建议这样读：

1. `src/core/run/run_main.py`
2. `src/core/main_integration/simulation_file.py`
3. `src/core/orchestrator/orchestrator_main.py` + `processors.py`
4. `src/modules/demand_planning/integration.py`
5. `src/modules/deployment_planning/main.py`
6. `src/modules/logistics_execution/main.py` + `simulation.py`
7. `src/modules/mrp_planning/mrp_simulation.py`
8. `src/modules/production_planning/capacity_allocator.py`
9. `pgsql_db/db_initializer.py`
10. `pgsql_db/module_data_writer.py`

## 27. 交接结论

- 第三阶段已完成单体文件全部拆分：`module1.py`～`module6.py` 及 `run.py` / `main_integration.py` / `orchestrator.py` / `parallel_executor.py` 均被删除，转为包目录结构。
- 业务逻辑统一存放于 `src/modules/<业务子包>/`；兼容别名仅保留在 `src/modules/__init__.py`。
- 如果要进一步把这份手册升级成"逐文件逐函数完全展开版"，下一步最值得继续深挖的是：
  1. `src/core/orchestrator/`（重点 `orchestrator_main.py` + `processors.py`）
  2. `src/modules/deployment_planning/main.py`
  3. `src/modules/production_planning/capacity_allocator.py`
  4. `src/modules/logistics_execution/main.py` + `simulation.py`
  5. `pgsql_db/module_data_writer.py`

## 28. 五个核心复杂文件深度手册

这一章单独把 5 个最复杂、最容易改坏结果的文件拆开讲。阅读时建议把它当成“高风险修改说明书”。

### 28.1 `src/core/orchestrator/` 包深度说明（原 `orchestrator.py`）

#### 标识标准化函数组

##### `_normalize_material(...)`
- 类型：重要内部
- 运行顺序：几乎所有状态写入和状态恢复前都会走到。
- 上游输入：字符串、整数、浮点形式的物料编码。
- 下游输出：无 `.0` 后缀的标准物料编码。
- 业务意义：保证同一个物料在库存、调拨、在途、GR 里都指向同一把钥匙。
- 修改风险：一旦这里和别处规则不一致，会出现库存分桶、M3/M5 查不到同一物料的问题。

##### `_normalize_location(...)` / `_normalize_sending(...)` / `_normalize_receiving(...)`
- 类型：重要内部
- 运行顺序：状态读写、视图输出、UID 相关逻辑都会用到。
- 上游输入：地点字符串、纯数字地点、带前导零地点。
- 下游输出：数字地点补到 4 位，非数字地点保持原样。
- 业务意义：统一地点主键，避免 `816` 和 `0816` 被当成不同节点。
- 修改风险：这是跨模块键匹配的底层契约，不建议随意改规则。

##### `_normalize_identifiers(df)`
- 类型：重要内部
- 运行顺序：保存 CSV、恢复 CSV、生成视图时反复使用。
- 上游输入：任意包含 `material/location/sending/receiving/sourcing` 的 DataFrame。
- 下游输出：标准化后的 DataFrame。
- 业务意义：把状态和输出变成统一口径，便于续跑和数据库落库。

#### `DeploymentUID`

##### `to_string()`
- 类型：重要内部
- 运行顺序：M5 结果进入 Orchestrator 时调用。
- 上游输入：`material`, `sending`, `receiving`, `planned_deploy_date`, `demand_element`, `sequence`
- 下游输出：`ori_deployment_uid`
- 业务意义：给每一条部署需求一个稳定可追踪主键。
- 修改风险：分隔符、字段顺序、sequence 格式都不能轻改，否则历史 run、M6 发运、DB 对账都会断链。

##### `from_string(uid_str)`
- 类型：重要内部
- 运行顺序：解析 UID 或做排障时使用。
- 输出：还原后的结构化 UID 对象。

#### `Orchestrator.__init__(...)`
- 类型：公开
- 运行顺序：主流程启动早期。
- 上游输入：`start_date`, `output_dir`
- 下游输出：初始化所有内存状态，创建输出目录。
- 业务意义：建立仿真的全局状态中枢。
- 修改风险：这里定义的状态字段决定哪些数据能跨天、哪些数据能续跑恢复。

#### `initialize_inventory(initial_inventory_df)`
- 类型：公开
- 运行顺序：仿真初始化阶段。
- 输入格式：`material`, `location`, `quantity`
- 下游输出：`unrestricted_inventory`, `initial_inventory`
- 业务意义：建立期初库存。
- 修改风险：会清空旧库存；如果在续跑路径误调用，会把恢复好的库存覆盖掉。

#### `set_space_capacity(space_capacity_df)`
- 类型：公开
- 运行顺序：初始化阶段。
- 输入格式：`location`, `eff_from`, `eff_to`, `capacity`
- 下游输出：`space_capacity`
- 业务意义：提供 M5 的接收空间约束。

#### 视图函数组

##### `get_unrestricted_inventory_view(date)`
- 类型：公开
- 输出列：`date`, `material`, `location`, `quantity`
- 下游用途：保存 `unrestricted_inventory_YYYYMMDD.csv`、供排障和 DB 落库使用。

##### `get_current_unrestricted_inventory()`
- 类型：公开
- 输出：`{(material, location): quantity}`
- 下游用途：M1、M5、M6 直接取库存。

##### `get_planning_intransit_view(date)`
- 类型：公开
- 输出列：`transit_uid`, `date`, `material`, `sending`, `receiving`, `actual_ship_date`, `actual_delivery_date`, `quantity`, `ori_deployment_uid`, `vehicle_uid`
- 下游用途：续跑恢复、DB 落库、M5 pipeline supply 计算。

##### `get_open_deployment_view(date)`
- 类型：公开
- 输出列：`material`, `sending`, `receiving`, `planned_deployment_date`, `deployed_qty`, `demand_element`, `ori_deployment_uid`
- 下游用途：M6 待发运输入。
- 修改风险：函数本身明确不做清理，清理只在日处理入口统一执行；不要把副作用塞进 getter。

##### `get_space_quota_view(date)`
- 类型：公开
- 输出列：`receiving`, `date`, `max_qty`
- 计算公式：`capacity - unrestricted_inventory`
- 下游用途：M5 接收端空间裁剪。

##### `get_production_plan_backlog_view(date)`
- 类型：公开
- 输出列：`material`, `location`, `available_date`, `quantity`
- 下游用途：落盘、续跑恢复、DB 落库。

#### 模块处理函数组

##### `process_module1_shipments(shipment_df, date)`
- 类型：公开
- 运行顺序：每日模块顺序中的第一步。
- 输入来源：M1 的 `shipment_df`。
- 下游效果：
  - 扣减 `unrestricted_inventory`
  - 追加 `shipment_log`
  - 写入 `shipment_log_by_date`
- 业务意义：处理对客户真实出库。
- 修改风险：M6 和 summary 都依赖 shipment log 口径。

##### `process_module4_production(production_df, date)`
- 类型：公开
- 运行顺序：M4 后。
- 输入来源：M4 的 `production_df`。
- 下游效果：
  - 更新 `production_plan_backlog`
  - 当天 `available_date == date` 的记录写入 `production_gr`
  - 增加库存
- 业务意义：区分“未来会到货的生产”与“今天真正入库的生产”。
- 修改风险：backlog 去重和汇总逻辑是 M3 future production 口径的重要前提。

##### `process_module5_deployment(deployment_df, date)`
- 类型：公开
- 运行顺序：M5 后。
- 输入来源：M5 的 `DeploymentPlan`。
- 下游效果：新增 `open_deployment`。
- 业务意义：把部署计划变成待执行状态对象。
- 修改风险：UID 生成前的稳定排序是结果可复现的关键。

##### `process_module6_delivery(delivery_df, date)`
- 类型：公开
- 运行顺序：M6 后。
- 输入来源：M6 的 `delivery_df`。
- 下游效果：
  - 扣减 `open_deployment`
  - 扣减发送地库存
  - 追加 `delivery_shipment_log`
  - 未来到货写 `in_transit`
  - 当天到货写 `delivery_gr` 并入库
- 业务意义：把物流计划变成真实库存变化。
- 修改风险：
  - 只处理 `actual_ship_date == 当前日`
  - 同一 deployment 拆多车时必须依赖 `vehicle_uid`

#### 日处理与恢复函数组

##### `_process_delivery_arrivals(date)`
- 类型：重要内部
- 运行顺序：每天一开始，早于 M1/M4/M5/M6。
- 输入来源：`self.in_transit`
- 下游效果：到货入库、写 `delivery_gr`、删除完成的 transit。
- 业务意义：把历史发出的货在今天收回库存。
- 修改风险：如果放到日末，会改变当天 M5/M6 可见库存。

##### `run_daily_processing(...)`
- 类型：公开
- 运行顺序：Orchestrator 的每日总入口。
- 下游顺序：
  1. `cleanup_past_due_open_deployments`
  2. `_process_delivery_arrivals`
  3. `process_module1_shipments`
  4. `process_module4_production`
  5. `process_module5_deployment`
  6. `process_module6_delivery`
  7. `save_daily_state`
- 修改风险：不能轻易改变这个顺序。

##### `_safe_convert_to_int(value)`
- 类型：重要内部
- 作用：把 pandas Series、标量、空值安全转成整数。
- 业务意义：避免脏数据导致全流程报错。

##### `set_past_due_cleanup_grace_days(days)`
- 类型：公开
- 作用：设置 open deployment 过期清理宽限天数。

##### `cleanup_past_due_open_deployments(date, grace_days, write_audit)`
- 类型：公开
- 运行顺序：每天开头一次。
- 输出文件：`open_deployment_pastdue_cleanup_YYYYMMDD.csv`
- 业务意义：清理长期未执行的调拨挂账。
- 修改风险：严格规则是 `planned_deployment_date < threshold_date`，不是 `<=`。

##### `save_daily_state(date)`
- 类型：公开
- 运行顺序：每天最后。
- 输出文件：
  - `unrestricted_inventory_YYYYMMDD.csv`
  - `open_deployment_YYYYMMDD.csv`
  - `planning_intransit_YYYYMMDD.csv`
  - `space_quota_YYYYMMDD.csv`
  - `production_plan_backlog_YYYYMMDD.csv`
  - `delivery_gr_YYYYMMDD.csv`
  - `production_gr_YYYYMMDD.csv`
  - `shipment_log_YYYYMMDD.csv`
  - `delivery_shipment_log_YYYYMMDD.csv`
  - `inventory_change_log_YYYYMMDD.csv`
  - `daily_logs_YYYYMMDD.csv`
- 修改风险：这是续跑、DB 落库和审计的共同契约。

##### `_log_event(event_type, message)`
- 类型：重要内部
- 作用：把关键事件写进 `daily_logs`。

##### `get_summary_statistics(date)`
- 类型：公开
- 输出：库存数、库存总量、open deployment 数、in transit 数、GR 数、shipment 数等。

### 28.2 `src/modules/deployment_planning/main.py` 深度说明

#### `_validate_deployment_shipment_constraint(...)`
- 类型：重要内部
- 运行顺序：M5 全部部署结果生成后、输出前。
- 输入：`deployment_plan_df`, `config['ShipmentLog']`, `sim_date`
- 输出：只写验证日志，不改结果。
- 业务意义：监控“部署量是否明显超过订单量”。
- 修改风险：这是监控，不是硬裁剪；不要误改成业务强约束。

#### `_initialize_soh_dict(...)`
- 类型：重要内部
- 运行顺序：M5 初始化库存时。
- 输入：`InventoryLog`, `SupplyDemandLog`, `SafetyStock`, `OrderLog`
- 输出：`(material, location) -> qty` 字典。
- 业务意义：给 M5 建立自己的 SOH 基准。
- 修改风险：会检查 `(material, location)` 重复，不能无脑允许重复行。

#### `_process_layer_demands(...)`
- 类型：重要内部
- 运行顺序：M5 每天每一层开始时。
- 输入：当前层 `all_pairs`，各种缓存和索引。
- 输出：`node_demands_map`
- 业务意义：为该层所有节点收集需求。
- 实现优先级：
  1. 向量化批量
  2. 多进程
  3. 线程池
  4. 串行回退
- 修改风险：`sorted(all_pairs)` 是为了稳定顺序，不要去掉。

#### `_allocate_pipeline_sources(...)`
- 类型：重要内部
- 运行顺序：库存优先级分配后。
- 输入：`future_intransit`, `open_deployment_inbound`, `future_production`
- 输出：回写每条需求的 `deploy_qty_with_plan_order` 及细分字段。
- 业务意义：先用“已经在路上/未来会来的货”去填缺口，减少新部署。
- 修改风险：只给 self 行分配 pipeline supply，逻辑上不能随意扩散到 cross-node 行。

#### `_process_gaps_and_create_plans(...)`
- 类型：重要内部
- 运行顺序：pipeline 覆盖之后。
- 输入：`demand_rows`, `adjusted_qtys`, upstream/network/lead time/cache 等。
- 输出：
  - `deployment_plan_rows`
  - `unfulfilled_rows`
  - `up_gap_next`
- 业务意义：把缺口变成调拨计划和上游净需求传递。
- 修改风险：
  - 与基线保持一致，不对 `df_gap_pos` 排序
  - `planned_delivery_date` 和 `leadtime` 的决定逻辑直接影响后续 M6 和上游 gap

#### `_update_soh_dict(...)`
- 类型：重要内部
- 运行顺序：每天 M5 逻辑最后。
- 输入：beginning inventory、production GR、in transit、delivery GR、shipment、deployment_plan_rows。
- 输出：
  - 更新 `soh_dict`
  - 追加 `stock_on_hand_log`
- 业务意义：把今天的收发货结果滚到明天库存。
- 修改风险：只对 `sending != receiving` 的 `deployed_qty_invCon` 做真实出库扣减。

#### `main(...)`
- 类型：公开
- 运行顺序：M5 模块总入口。
- 输入：文件模式或集成模式参数。
- 核心执行顺序：
  1. 加载配置
  2. 校验配置
  3. 生成 location layer map
  4. 构建缓存和索引
  5. 初始化库存 `soh_dict`
  6. 对每个仿真日：
     - 构建生产、在途、delivery GR、shipment、open deployment 字典
     - 计算 projected inventory 和 available inventory
     - 逐层收需求、应用 MOQ/RV、优先级分配、pipeline 分配、生成 gap 和计划
     - 执行 push/soft-push
     - 更新 `soh_dict`
  7. 应用 receiving space quota
  8. 约束检查
  9. 输出 Excel 或直接返回 DataFrame
- 输出：
  - `deployment_plan`
  - `unfulfilled_log`
  - `stock_on_hand_log`
  - `validation_log`
  - `statistics`

### 28.3 `src/modules/production_planning/capacity_allocator.py` 深度说明

#### `centralized_capacity_allocation_with_changeover(...)`
- 类型：公开
- 运行顺序：M4 无约束计划生成后。
- 输入：无约束计划、产能、产率、换产矩阵/定义、ML 配置、前一日产线状态、历史已占产能。
- 输出：`(plan_log, exceed_log)`
- 业务意义：把理想生产量转换成真实可排产量。

#### `CapacityAllocator.__init__(...)`
- 类型：重要内部
- 作用：建立 `rate_map`, `co_mat`, `co_def`, `mct_map`, `cap_map` 等核心索引。
- 修改风险：`cap_df` 是否带 `location` 会影响 capacity key 结构。

#### `_build_capacity_map(cap_df)`
- 类型：重要内部
- 输出：
  - 有 location：`(location, line, date) -> capacity`
  - 无 location：`(line, date) -> capacity`
- 修改风险：key 结构一改，后续所有分配都会错位。

#### `allocate(uncon)`
- 类型：公开
- 运行顺序：分配主循环。
- 作用：按 `line + simulation_date` 分组逐条线分配。
- 修改风险：排序字段会影响换产序列和产能消耗顺序。

#### `_allocate_line_group(...)`
- 类型：重要内部
- 作用：对单条线的多个批次排顺序并逐批分配。
- 关键逻辑：如果批次数大于 1，先跑 `optimal_changeover_sequence(...)`。

#### `_init_line_state(line)`
- 类型：重要内部
- 作用：从前一日产线状态恢复未完成换产。
- 修改风险：跨天换产连续性全靠这里承接。

#### `_allocate_batch(...)`
- 类型：重要内部
- 作用：给单个批次计算 planning window 和换产开销，再进入 horizon 分配。
- 输入关键字段：`lsk`, `ptf`, `mct`。
- 修改风险：`mlcfg.iloc[0]` 默认配置唯一，重复配置会有隐患。

#### `_calculate_changeover(...)`
- 类型：重要内部
- 作用：判断当前批次是否需要换产，以及换产剩余多少时间。

#### `_lookup_changeover(...)`
- 类型：重要内部
- 作用：查 `ChangeoverMatrix` 和 `ChangeoverDefinition`。
- 修改风险：缺配置时会退回默认时间，不是静默忽略；这会直接改变产能结果。

#### `_allocate_to_horizon(...)`
- 类型：重要内部
- 作用：在窗口内按天分配产能。
- 输出：若生产量没排完，会生成超额记录。

#### `_allocate_day(...)`
- 类型：重要内部
- 作用：单日分配核心。
- 顺序：
  1. 查当天产能
  2. 扣历史已占产能
  3. 消耗换产时间
  4. 用剩余小时按 rate 换算可生产量
  5. 生成单日计划记录
- 修改风险：`int(today_cap * rate)` 的向下取整行为不能轻改。

#### `_adjust_for_previous_allocation(...)`
- 类型：重要内部
- 作用：从名义产能中减去历史已分配小时数。
- key 结构：`location|line|YYYY-MM-DD`

#### `_consume_changeover(...)`
- 类型：重要内部
- 作用：先把当天一部分产能消耗在换产上。

#### `extract_allocated_capacity_from_plan(...)`
- 类型：公开
- 作用：把生产计划反推成“这天这条线用了多少小时”。
- 下游用途：跨日防重复占产能。

#### `_calculate_group_hours(...)`
- 类型：重要内部
- 作用：按分组统计生产小时 + 换产小时。

#### `validate_capacity_allocation(...)`
- 类型：公开
- 作用：检查当前计划是否与历史占用冲突。

#### `extract_line_states_from_plan(...)`
- 类型：公开
- 作用：把当天计划提炼成产线日末状态。

#### `_build_line_state(...)`
- 类型：重要内部
- 作用：构造最终可持久化的 line state。

#### `_analyze_end_of_day_changeover(...)`
- 类型：重要内部
- 作用：根据剩余产能推断是否有未完成换产。

#### `_check_line_changeover(...)`
- 类型：重要内部
- 作用：对某条线某一天检查剩余产能是否接近典型换产时间，从而推断“当天结束时正卡在换产中”。

#### `calculate_changeover_metrics(...)`
- 类型：公开
- 作用：汇总换产计数、时间、成本、MU 损失。

#### `_group_changeovers(...)` / `_prepare_changeover_def(...)` / `_create_changeover_record(...)`
- 类型：重要内部
- 作用：把 changeover log 整理成可输出汇总的记录。

#### `simulate_production(plan, pr_cfg, seed)`
- 类型：公开
- 运行顺序：M4 最后一步之一。
- 作用：用 Binomial 分布模拟真实产出 `produced_qty`。
- 修改风险：不能重新排序 `plan`，否则随机数消费顺序改变，结果就不再可复现。

### 28.4 `src/modules/logistics_execution/` 包深度说明（原 `module6.py`）

#### `run_daily_physical_flow(...)`
- 类型：公开
- 运行顺序：M6 单日入口。
- 输出：`delivery_plan`, `vehicle_log`, `truck_usage`, `unsatisfied_log`, `validation_log`, `bypass_log`, `statistics`
- 业务意义：供主流程直接按天调用。

#### `run_physical_flow_module(...)`
- 类型：公开
- 运行顺序：M6 总入口。
- 顺序：参数初始化 -> 数据准备 -> 仿真循环 -> 输出整理。

#### `_init_integrated_params(...)`
- 类型：重要内部
- 作用：集成模式只跑单天，直接加载 `load_integrated_config(...)`。

#### `_init_standalone_params(...)`
- 类型：重要内部
- 作用：独立模式根据 DeploymentPlan 决定仿真日期范围。

#### `_prepare_data(...)`
- 类型：重要内部
- 作用：
  - 校验 DeploymentPlan / TruckReleaseCon
  - 去重 DemandPriority / MaterialMD / TruckTypeSpecs
  - 构建 priority/material/spec 映射
  - 处理物料元数据
  - 准备 DeploymentPlan
  - 处理重复 UID
  - 建立容量映射
- 修改风险：这里是 M6 的数据总闸门。

#### `_filter_empty_demand_element(...)`
- 类型：重要内部
- 作用：过滤掉空 demand_element 记录。

#### `_build_priority_map(...)` / `_build_material_map(...)` / `_build_spec_map(...)`
- 类型：重要内部
- 作用：把配置表转换成 O(1) lookup dict。

#### `_process_material_metadata(...)`
- 类型：重要内部
- 作用：给部署行补 `demand_unit_to_weight` 和 `demand_unit_to_volume`。
- 修改风险：缺物料元数据时会默认用 1.0，这会影响装载率而不是简单报 warning。

#### `_log_missing_materials(...)`
- 类型：重要内部
- 作用：把缺失物料元数据写进验证日志。

#### `_prepare_deployment_plan(...)`
- 类型：重要内部
- 作用：标准化 `planned_deployment_date`，稳定排序，补 `ori_deployment_uid`，补 `priority`、`waiting_days`、`simulation_date`、`route_type_debug`。
- 修改风险：排序是重复 UID 去重和结果稳定的前提。

#### `_handle_uid_duplicates(...)`
- 类型：重要内部
- 作用：发现重复 `ori_deployment_uid` 时保留第一条。
- 修改风险：前面排序一旦变化，这里留下哪条也会变。

#### `_run_simulation_loop(...)`
- 类型：重要内部
- 作用：每日仿真总循环。
- 顺序：初始化 `agg_status` -> 日期循环 -> 读取库存 -> 处理当天需求。

#### `_init_aggregation_status(dp_dict)`
- 类型：重要内部
- 作用：把 DeploymentPlan 转成聚合状态，跟踪每个 UID 剩余待发数量和等待天数。

#### `_process_daily_demands(...)`
- 类型：重要内部
- 作用：收集当天待处理需求，筛出跨节点线路，按优先级排序后交给 route 级处理。

#### `_collect_pending_demands(...)`
- 类型：重要内部
- 作用：从 `agg_status` 提取还没发完的需求。
- 修改风险：刻意保持原始字典迭代顺序，不使用 `sorted()`。

#### `_process_routes(...)`
- 类型：重要内部
- 作用：按路线处理需求。
- 修改风险：路线处理顺序按“全局排序后的首次出现”决定，不是简单 groupby 排序。

#### `_process_single_route(...)`
- 类型：重要内部
- 作用：给单路线选择车型并尝试发车。

#### `_process_truck_type(...)`
- 类型：重要内部
- 作用：单车型装载和发车核心。
- 顺序：第一轮装载 -> 旁路和阈值判断 -> 第二轮补载 -> 生成发运记录 -> 更新剩余需求。
- 修改风险：这是车辆利用率、触发时机和物流结果最敏感的位置。

#### `_first_pass_loading(...)` / `_second_pass_loading(...)`
- 类型：重要内部
- 作用：分别完成保守装载和触发后的补载。

#### `_build_context(...)`
- 类型：重要内部
- 作用：给旁路规则表达式构造上下文。

#### `_generate_shipment_records(...)`
- 类型：重要内部
- 作用：真正生成 DeliveryPlan、VehicleLog、BypassRuleHitLog，并更新 `agg_status` 和库存。
- 修改风险：`vehicle_uid`、delay sampling、lead time 缺失异常处理都集中在这里。

#### `_handle_remaining_demands(...)`
- 类型：重要内部
- 作用：对超过 `max_wait_days` 仍未发出的需求生成 `unsatisfied_log`。

#### `_enforce_shipment_constraint(...)`
- 类型：重要内部但当前禁用
- 作用：理论上会把 delivery_qty 裁剪到不超过 shipment_qty。
- 当前状态：故意禁用，以保持与 Dev 版一致。

#### `_validate_shipment_delivery_constraint(...)`
- 类型：重要内部
- 作用：验证 delivery 总量是否超过 shipment 总量，只报警不裁剪。

#### `_generate_outputs(...)`
- 类型：重要内部
- 作用：构建各输出 DataFrame，做约束验证，并可写 Excel。

#### `_build_delivery_plan_df / _build_vehicle_df / _build_usage_df / _build_unsat_df / _build_validation_df / _build_bypass_df`
- 类型：重要内部
- 作用：构建标准输出表。
- 修改风险：空表是否保留列结构，要和基线保持一致。

#### `_write_excel_output(...)`
- 类型：重要内部
- 输出 Sheet：`DeliveryPlan`, `VehicleLog`, `TruckUsageLog`, `UnsatisfiedMDQLog`, `ValidationLog`, `BypassRuleHitLog`

### 28.5 `pgsql_db/module_data_writer.py` 深度说明

#### `ModuleDataWriter.__init__(db, config_name)`
- 类型：公开
- 作用：初始化数据库写入上下文。

#### `truncate_output_tables(run_id)`
- 类型：公开
- 运行顺序：数据库模式写入前。
- 作用：只删除当前 `run_id` 的历史数据，不影响其他运行。
- 修改风险：绝不能改成全表清空。

#### `write_summary_only(output_dir, run_id, if_exists)`
- 类型：公开
- 作用：只写 summary 和 orchestrator 数据，适合后处理优化场景。

#### `_write_summary_files_fast(...)`
- 类型：重要内部
- 作用：把固定命名的 summary 文件快速映射到数据库表。

#### `write_module_output(module_name, output_dir, run_id, sim_date, if_exists)`
- 类型：公开
- 作用：扫描某模块输出目录，把 xlsx/csv 写到数据库。

#### `_write_excel_file(...)`
- 类型：重要内部
- 输入：单个每日 Excel 文件。
- 输出：每个 sheet 对应写入 `{module_name}_output_{sheet}` 表。
- 自动追加：`file_date`, `sim_date`, `run_id`
- 修改风险：空表仍要建表，这是和基线保持一致的重要约定。

#### `write_all_modules(run_output_dir, run_id, if_exists)`
- 类型：公开
- 作用：全量写入 `module1/module3/module4/module5/module6/orchestrator/summary`。

#### `write_orchestrator_data(orchestrator_dir, run_id, sim_date, if_exists)`
- 类型：公开
- 作用：把 `orchestrator/*.csv` 合并并写入 `orchestrator_*` 表。
- 输入文件模式：
  - `unrestricted_inventory_*.csv`
  - `open_deployment_*.csv`
  - `open_deployment_pastdue_cleanup_*.csv`
  - `planning_intransit_*.csv`
  - `space_quota_*.csv`
  - `delivery_gr_*.csv`
  - `production_gr_*.csv`
  - `production_plan_backlog_*.csv`
  - `shipment_log_*.csv`
  - `delivery_shipment_log_*.csv`
  - `inventory_change_log_*.csv`
  - `daily_logs_*.csv`

#### `write_module_results_from_dict(all_results, run_id, sim_date, if_exists, truncate_first)`
- 类型：公开
- 运行顺序：DB 模式主链路核心写入函数。
- 输入：主流程内存结果字典。
- 输出：按模块把 DataFrame 直接写数据库。
- 关键特性：
  - 先按 `run_id` 清旧数据
  - 即使没有数据，也创建空表结构
  - 为每一天结果补 `sim_date`
- 修改风险：模块返回字典键名和这里的映射要严格一致。

#### `_clean_name(name)`
- 类型：重要内部
- 作用：把 sheet 名或文件名转成合法数据库表名片段。

#### `get_written_tables()` / `print_summary()`
- 类型：公开
- 作用：输出本次写库情况。

#### `generate_summary_reports_from_db(run_id, start_date, end_date, if_exists)`
- 类型：公开
- 作用：DB 模式下直接从模块输出表生成 7 张 summary 表。

#### `_generate_order_shipment_cut_summary(...)`
- 类型：重要内部
- 作用：从 `module1_output_orderlog/shipmentlog/cutlog` 聚合 summary。
- 修改风险：会先对订单去重，解决 AO 订单在多个 `simulation_date` 被重复记录的问题。

#### `_generate_changeover_summary(...)`
- 类型：重要内部
- 作用：从 `module4_output_changeoverlog` 生成换产 summary。
- 日期过滤：优先看 `changeover_end_date`，否则看 `date`。

#### `_generate_capacity_exceed_summary(...)`
- 类型：重要内部
- 作用：从 `module4_output_capacityexceed` 生成超产能 summary。

#### `_generate_production_plan_summary(...)`
- 类型：重要内部
- 作用：从 `module4_output_productionplan` 生成生产汇总。
- 日期过滤字段：`available_date`

#### `_generate_deployment_plan_summary(...)`
- 类型：重要内部
- 作用：从 `module5_output_deploymentplan` 生成部署汇总。
- 日期过滤字段候选：`deployment_date`, `arrival_date`, `ship_date`, `date`

#### `_generate_delivery_plan_summary(...)`
- 类型：重要内部
- 作用：从 `module6_output_deliveryplan` 生成交付汇总。
- 日期过滤字段候选：`planned_deploy_date`, `actual_ship_date`, `date`

#### `_generate_truck_usage_summary(...)`
- 类型：重要内部
- 作用：从 `module6_output_truckusagelog` 生成卡车使用汇总。

#### `write_run_data_to_db(...)`
- 类型：公开脚本入口
- 作用：一键把整个 run 输出导入数据库。
- 顺序：
  1. 建立数据库连接
  2. 测试连接
  3. `write_all_modules(...)`
  4. `print_summary()`
  5. `build_all_pending_indexes()`
- 修改风险：先写数据再统一建索引是性能优化关键，不要改成边写边建索引。

## 29. 修改高风险清单

1. `src/core/orchestrator/` 包里的标识符标准化和 UID 逻辑不能局部绕过（原 `orchestrator.py`）。
2. `src/core/orchestrator/processors.py` 的 `_process_delivery_arrivals()` 必须保持在日初执行（原 `orchestrator.py`）。
3. `deployment_planning/main.py` 的层级遍历、排序和 pipeline 覆盖顺序不能轻易调整。
4. `capacity_allocator.py` 中的 plan 顺序、换产时间消耗、`int(today_cap * rate)` 取整方式不能随意改。
5. `src/modules/logistics_execution/` 包（原 `module6.py`）中 route 顺序、truck type 顺序、第一轮/第二轮装载、UID 去重顺序都会影响最终结果。
6. `module_data_writer.py` 中 `run_id` 删除策略必须保持“按运行隔离”，不能误删其他场景。
7. `module_data_writer.py` 中空表建表能力必须保留，否则 DB 模式 summary 和对比工具会断。

## 附录 A. 建议配套阅读

- `docs/handover/refactored/02-api/function_dependency_matrix.md`
- `docs/handover/refactored/01-architecture/module_sequence_diagrams.md`
- `docs/handover/refactored/00-overview/handover.md`
- `archive/handover_docs/dev/chainsight_dev_source_summary.md`

## 附录 B. 更新规则

- 新增或删除主路径函数时，必须更新本文。
- 变更输入输出文件名、Sheet 名、表名时，必须更新本文。
- 修改复杂文件中的高风险函数后，建议同步补一条修订记录。
