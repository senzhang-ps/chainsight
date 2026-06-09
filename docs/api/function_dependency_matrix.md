# ChainSight 函数上下游依赖矩阵

## 封面信息

- 文档名称：ChainSight 函数上下游依赖矩阵
- 文档编号：CS-DM-20260306-001
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
| v1.0 | 2026-03-06 | 陈显跃 | 首次形成上下游依赖矩阵和排障速查 |
| v1.2 | 2026-04-10 | 陈显跃 | 第三阶段重构：单体 `run.py` / `main_integration.py` / `orchestrator.py` / `module1~6.py` 全部拆分为包目录，本矩阵中所有 `xxx.py::fn()` 请按下方"路径迁移指引"映射理解 |

## 路径迁移指引（第三阶段，2026-04-10）

| 旧路径（本文正文中仍出现的写法） | 新实际位置 |
|---|---|
| `src/core/run.py` | 包 `src/core/run/`，入口：`run_main.py` |
| `src/core/main_integration.py` | 包 `src/core/main_integration/`，入口：`simulation_file.py` / `simulation_db.py` / `production_runner.py` |
| `src/core/orchestrator.py` | 包 `src/core/orchestrator/`，入口：`orchestrator_main.py`；按模块处理在 `processors.py`；快照持久化在 `persistence.py` |
| `src/core/parallel_executor.py` | 包 `src/core/parallel_executor/`，入口：`parallel_executor_main.py` |
| `src/modules/module1.py` | `src/modules/demand_planning/`（兼容别名仍保留） |
| `src/modules/module3.py` | `src/modules/mrp_planning/` |
| `src/modules/module4.py` | `src/modules/production_planning/` |
| `src/modules/module5.py` | `src/modules/deployment_planning/` |
| `src/modules/module6.py` | `src/modules/logistics_execution/` |

## 文档定位

- 本文强调“谁读谁、谁写谁、谁依赖谁”。
- 它是交接排障和影响分析时最适合先翻的文档之一。

## 1. 使用说明

- 这份矩阵不是重复解释代码，而是回答一个更直接的问题：某个函数读了什么、改了什么、产物最后流向哪里。
- 交接时如果你要查“结果为什么不对”“哪一层把数据改坏了”“某张表为什么没有值”，优先看这份矩阵。
- 阅读方法建议：先看第 2 章总链路，再按 Core -> M1 -> M4 -> M5 -> M6 -> M3 -> DB 的顺序看。

## 2. 端到端总链路矩阵

| 步骤 | 核心函数 | 上游输入 | 下游输出 | 主要落点 |
|---|---|---|---|---|
| 1 | `src/core/run/run_main.py::main()` | 命令行参数、配置路径、数据库参数 | 运行模式选择、本次 run 目录 | `outputs/<config>/run_*` |
| 2 | `src/core/main_integration/simulation_file.py::run_integrated_simulation()` | Excel 配置 / `config_dict` | 启动 Orchestrator、按天驱动模块 | 各模块日输出 + Orchestrator 状态 |
| 3 | `src/modules/demand_planning/integration.py::run_daily_order_generation()` | `M1_*` 配置、Orchestrator 库存 | `orders_df/shipment_df/cut_df/supply_demand_df` | `module1_output_YYYYMMDD.xlsx` |
| 4 | `src/core/orchestrator/processors.py::process_module1_shipments()` | M1 `shipment_df` | 扣库存、记录 shipment log | `shipment_log_YYYYMMDD.csv` |
| 5 | `src/core/main_integration/production_runner.py::run_daily_production_planning_integrated()` | M4 配置、M3 净需求 | `production_df/exceed_log/issues_df/changeover_log` | `Module4Output_YYYYMMDD.xlsx` |
| 6 | `src/core/orchestrator/processors.py::process_module4_production()` | M4 `production_df` | backlog、production GR、增加库存 | `production_gr_YYYYMMDD.csv` / `production_plan_backlog_YYYYMMDD.csv` |
| 7 | `src/modules/deployment_planning/main.py::run_daily_deployment_planning()` | M1 输出、M4 输出、Orchestrator 状态、M5 配置 | `deployment_plan/unfulfilled_log/stock_on_hand_log/validation_log` | `Module5Output_YYYYMMDD.xlsx` |
| 8 | `src/core/orchestrator/processors.py::process_module5_deployment()` | M5 `deployment_plan` | 写 open deployment + UID | `open_deployment_YYYYMMDD.csv` |
| 9 | `src/modules/logistics_execution/main.py::run_daily_physical_flow()` | OpenDeployment、M6 配置、当前库存 | `delivery_plan/vehicle_log/truck_usage/unsatisfied_log` | `Module6Output_YYYYMMDD.xlsx` |
| 10 | `src/core/orchestrator/processors.py::process_module6_delivery()` | M6 `delivery_plan` | 扣 open deployment、扣发送库存、写在途或 delivery GR | `planning_intransit_YYYYMMDD.csv` / `delivery_gr_YYYYMMDD.csv` |
| 11 | `src/modules/mrp_planning/integration.py::run_integrated_mode()` | M1 输出、Orchestrator 状态、M3/M4/M5 配置 | `net_demand_df` | `Module3Output_YYYYMMDD.xlsx` |
| 12 | `src/core/orchestrator/persistence.py::save_daily_state()` | 当日全部状态 | 全量状态快照 | `orchestrator/*.csv` |
| 13 | `src/services/summary_report_generator.py::generate_all_reports()` | M1/M4/M5/M6 输出 + Orchestrator CSV | 汇总报表 | `summary/*.xlsx` / `.csv` |
| 14 | `pgsql_db/module_data_writer.py::write_module_results_from_dict()` | 内存模块结果 | DB 输出表 | `module*_output_*` |
| 15 | `pgsql_db/module_data_writer.py::write_orchestrator_data()` | `orchestrator/*.csv` | DB 状态表 | `orchestrator_*` |
| 16 | `pgsql_db/module_data_writer.py::generate_summary_reports_from_db()` | DB 模块输出表 | DB 汇总表 | `summary_output_*` |

## 3. 核心层依赖矩阵

### 3.1 `src/core/run/`（入口：`run_main.py`）

| 函数 | 上游依赖 | 下游依赖 | 文件/表影响 |
|---|---|---|---|
| `main()` | 命令行参数 | `_ensure_output_dir`, `run_integrated_simulation`, `_run_with_database` | 选择本地或 DB 模式 |
| `_ensure_output_dir()` | 配置路径、resume 参数 | run 目录选择 | 创建 `outputs/<config>/run_*` |
| `_run_with_database()` | DB 连接参数、`config_name` | `DatabaseInitializer.initialize`, `_load_config_from_database`, `run_integrated_simulation_from_dict`, `ModuleDataWriter.*` | DB 模式端到端流程 |
| `_load_config_from_database()` | `cfg_*` 配置表 | `config_dict` | 读 `cfg_global_network`, `cfg_m1_demandforecast`, `cfg_m4_linecapacity` 等 |

### 3.2 `src/core/main_integration/`（入口：`simulation_file.py` / `simulation_db.py` / `production_runner.py`）

| 函数 | 上游依赖 | 下游依赖 | 文件/表影响 |
|---|---|---|---|
| `run_integrated_simulation()` | Excel 配置、起止日期 | M1/M4/M5/M6/M3、Orchestrator、Summary | 产生全部本地输出 |
| `run_integrated_simulation_from_dict()` | DB 读出的 `config_dict` | 同上 + DB writer | 产生内存结果，供落库 |
| `load_configuration()` | 配置 Excel | `config_dict` | 读所有配置 Sheet |
| `detect_last_complete_date()` | `orchestrator/*.csv` | `check_resume_capability` | 扫描状态文件集 |
| `check_resume_capability()` | 输出目录、日期范围 | 主流程 resume 决策 | 不直接写文件 |
| `restore_orchestrator_state()` | `unrestricted_inventory_*`, `open_deployment_*`, `planning_intransit_*`, `production_gr_*`, `delivery_gr_*`, `shipment_log_*`, `delivery_shipment_log_*`, `daily_logs_*` | 恢复后的 `Orchestrator` | 读取状态 CSV |
| `run_daily_production_planning_integrated()` | M4 配置、M3 输出目录/内存结果 | M4 `production_df/exceed_log/issues_df/changeover_log` | `Module4Output_YYYYMMDD.xlsx` |
| `load_current_date_production_gr()` | `Module4Output_YYYYMMDD.xlsx` | M1/M5/Orchestrator 的当日可入库生产供给 | 读取 `ProductionPlan` |

### 3.3 `src/core/orchestrator/`（入口：`orchestrator_main.py`）

| 函数 | 上游依赖 | 下游依赖 | 文件/表影响 |
|---|---|---|---|
| `initialize_inventory()` | `M1_InitialInventory` | `unrestricted_inventory` | 初始化内存状态 |
| `set_space_capacity()` | `Global_SpaceCapacity` | `space_capacity` | 初始化内存状态 |
| `get_unrestricted_inventory_view()` | `unrestricted_inventory` | M1/M5/M6/落盘/DB writer | `unrestricted_inventory_YYYYMMDD.csv` |
| `get_open_deployment_view()` | `open_deployment` | M6、落盘、DB writer | `open_deployment_YYYYMMDD.csv` |
| `get_planning_intransit_view()` | `in_transit` | M5、落盘、DB writer | `planning_intransit_YYYYMMDD.csv` |
| `get_space_quota_view()` | `space_capacity` + `unrestricted_inventory` | M5 | `space_quota_YYYYMMDD.csv` |
| `process_module1_shipments()` | M1 `shipment_df` | 库存、shipment log | `shipment_log_YYYYMMDD.csv` |
| `process_module4_production()` | M4 `production_df` | backlog、production GR、库存 | `production_gr_YYYYMMDD.csv`, `production_plan_backlog_YYYYMMDD.csv` |
| `process_module5_deployment()` | M5 `DeploymentPlan` | `open_deployment` | `open_deployment_YYYYMMDD.csv` |
| `process_module6_delivery()` | M6 `DeliveryPlan` | `open_deployment`, `in_transit`, `delivery_gr`, `delivery_shipment_log`, 库存 | `planning_intransit_YYYYMMDD.csv`, `delivery_gr_YYYYMMDD.csv`, `delivery_shipment_log_YYYYMMDD.csv` |
| `_process_delivery_arrivals()` | `in_transit` | `delivery_gr`, 库存 | 更新状态并体现在 `delivery_gr_YYYYMMDD.csv` |
| `cleanup_past_due_open_deployments()` | `open_deployment` | 清理后的 `open_deployment`、审计日志 | `open_deployment_pastdue_cleanup_YYYYMMDD.csv` |
| `generate_inventory_change_log()` | 期初/期末库存、production GR、delivery GR、shipment、delivery shipment | 库存对账日志 | `inventory_change_log_YYYYMMDD.csv` |
| `save_daily_state()` | 当日全部状态 | CSV 快照 | `orchestrator/*.csv` |

## 4. Module1 依赖矩阵

| 函数 | 上游依赖 | 下游依赖 | 文件/表影响 |
|---|---|---|---|
| `run_daily_order_generation()` | `M1_DemandForecast`, `M1_ForecastError`, `M1_OrderCalendar`, `M1_AOConfig`, `M1_DPSConfig`, `M1_SupplyChoiceConfig`, Orchestrator 库存 | `generate_daily_orders`, `simulate_shipment_for_single_day`, `generate_supply_demand_log_for_integration`, `save_module1_output_with_supply_demand` | `module1_output_YYYYMMDD.xlsx` |
| `prepare_daily_forecasts()` | 周预测、DPS、SupplyChoice | 订单预测、SDL 预测 | 内存 DataFrame |
| `generate_daily_orders()` | 日预测、AO 配置、订单日历、ForecastError | `orders_df`, 消耗后的 forecast | 内存 DataFrame |
| `consume_orders_vectorized()` | 订单池、forecast | 被消耗后的 forecast | 不直接写文件 |
| `simulate_shipment_for_single_day()` | 订单池、当前库存 | `shipment_df`, `cut_df`, 更新库存视图 | 内存 DataFrame |
| `generate_supply_demand_log_for_integration()` | 日预测 | `SupplyDemandLog` | 内存 DataFrame |
| `save_module1_output_with_supply_demand()` | `orders_df`, `shipment_df`, `cut_df`, `supply_demand_df` | 标准 M1 输出文件 | `OrderLog`, `ShipmentLog`, `CutLog`, `SupplyDemandLog`, `Summary` |

## 5. Module3 依赖矩阵

| 函数 | 上游依赖 | 下游依赖 | 文件/表影响 |
|---|---|---|---|
| `run_integrated_mode()` | M1 输出、Orchestrator 状态、M3/M4/M5 配置 | `run_mrp_layered_simulation_daily`, `_save_daily_output` | `Module3Output_YYYYMMDD.xlsx` |
| `load_module1_daily_outputs()` | `module1_output_YYYYMMDD.xlsx` | M3 输入装配 | 读 `OrderLog`, `ShipmentLog`, `SupplyDemandLog` |
| `assign_location_layers()` | `Global_Network` | M3 分层上下文 | 内存层级映射 |
| `determine_lead_time()` | `Global_LeadTime`, M4 配置 | horizon 计算、gap 传递 | 内存数值 |
| `calculate_daily_net_demand()` | 库存、在途、GR、生产、开放调拨、订单、SDL、安全库存 | AO/FC/SS gap | 内存字典/记录 |
| `run_mrp_layered_simulation_daily()` | 分层结果、所有日输入数据 | `net_demand_df` | `NetDemand` sheet |

## 6. Module4 依赖矩阵

| 函数 | 上游依赖 | 下游依赖 | 文件/表影响 |
|---|---|---|---|
| `run_daily_production_planning()` | M4 配置、M3 输出目录 | `DailyProductionPlanner.run()` | `Module4Output_YYYYMMDD.xlsx` |
| `load_daily_net_demand()` | `Module3Output_(前一天).xlsx` | `build_unconstrained_plan_for_single_day` | 读 `NetDemand` |
| `build_unconstrained_plan_for_single_day()` | layer=0 净需求、ML 产线配置 | 无约束计划 | 内存 DataFrame |
| `optimal_changeover_sequence()` | 批次列表、ChangeoverMatrix | 批次顺序 | 内存顺序 |
| `centralized_capacity_allocation_with_changeover()` | 无约束计划、LineCapacity、rate_map、Changeover 配置、历史状态 | `plan_log`, `exceed_log` | 内存 DataFrame |
| `extract_line_states_from_plan()` | `plan_log` | 次日 `line_states_*.json` | JSON 状态文件 |
| `extract_allocated_capacity_from_plan()` | `plan_log` | 次日 `allocated_capacity_*.json` | JSON 状态文件 |
| `simulate_production()` | `plan_log`, `M4_ProductionReliability` | `produced_qty` | 更新 `ProductionPlan` |
| `write_output()` | `plan`, `exc`, `issues`, `changeover_log` | 标准 M4 输出 | `ProductionPlan`, `CapacityExceed`, `Validation`, `ChangeoverLog` |

## 7. Module5 依赖矩阵

| 函数 | 上游依赖 | 下游依赖 | 文件/表影响 |
|---|---|---|---|
| `run_daily_deployment_planning()` | M1 输出、M4 输出、Orchestrator 视图、M5 配置、Global 配置 | 全层部署规划、push/soft-push、SOH 更新、输出写盘 | `Module5Output_YYYYMMDD.xlsx` |
| `load_integrated_config()` | `config_dict`, `module1_output`, `module4_output`, Orchestrator | M5 运行配置包 | 内存字典 |
| `_initialize_soh_dict()` | `InventoryLog`, `SupplyDemandLog`, `SafetyStock`, `OrderLog` | `soh_dict` | 内存字典 |
| `_process_layer_demands()` | layer 节点、缓存、SDL/SS/Order/DeployConfig 索引 | `node_demands_map` | 内存字典 |
| `collect_node_demands()` | SDL、SS、Order、up_gap_buffer、Network、LeadTime | 节点需求行 | 内存 list |
| `apply_grouped_moq_rv()` | 节点需求行、DeployConfig、shipment 限制 | `adjusted_qtys` | 内存字典 |
| `apply_priority_allocation_vectorized()` | `adjusted_qtys`, 当前库存, DemandPriority | `deployed_qty_invCon` | 回写需求行 |
| `_allocate_pipeline_sources()` | 未来在途、开放调拨入库、未来生产 | `deploy_qty_with_plan_order` 系列字段 | 回写需求行 |
| `_process_gaps_and_create_plans()` | gap、upstream、LeadTime、PTF/LSK | `deployment_plan_rows`, `unfulfilled_rows`, `up_gap_next` | 内存 list |
| `push_softpush_allocation()` | 当前计划、PushPullModel、SafetyStock、LeadTime、projected_soh | push 计划行 | 追加到 `DeploymentPlan` |
| `_update_soh_dict()` | 部署计划、production GR、in transit、delivery GR、shipment | 次日库存 + `StockOnHandLog` | `StockOnHandLog` |
| `apply_receiving_space_quota()` | `DeploymentPlan`, `ReceivingSpace` | 裁剪后的计划 + 空间未满足日志 | `DeploymentPlan`, `UnfulfilledLog` |
| `log_outputs()` | 4 张输出表 | 标准 M5 输出文件 | `DeploymentPlan`, `UnfulfilledLog`, `StockOnHandLog`, `Validation` |

## 8. Module6 依赖矩阵

| 函数 | 上游依赖 | 下游依赖 | 文件/表影响 |
|---|---|---|---|
| `run_daily_physical_flow()` | `config_dict`, Orchestrator, 当前日 | `run_physical_flow_module()` | `Module6Output_YYYYMMDD.xlsx` |
| `run_physical_flow_module()` | 独立模式 Excel 或集成模式配置 | `_prepare_data`, `_run_simulation_loop`, `_generate_outputs` | M6 全套输出 |
| `load_integrated_config()` | OpenDeployment、LeadTime、Truck 配置、MDQ 规则、DemandPriority | M6 运行配置包 | 内存字典 |
| `_prepare_data()` | `DeploymentPlan`, `TruckReleaseCon`, `TruckTypeSpecs`, `MaterialMD`, `DemandPriority` | `dp`, `prio_map`, `spec_map`, `cap_map`, `validation_log` | 内存结构 |
| `_prepare_deployment_plan()` | 原始部署计划、priority map | 标准化后的 DeploymentPlan | 补 `ori_deployment_uid`, `priority`, `waiting_days` |
| `_handle_uid_duplicates()` | 标准化后的 DeploymentPlan | 去重后的 DeploymentPlan | 影响后续 route 处理 |
| `_run_simulation_loop()` | prepared_data、sim_dates、可选 Orchestrator 实物库存 | `delivery_plan`, `vehicle_log`, `unsat_log`, `bypass_log` | 内存 list |
| `_collect_pending_demands()` | `agg_status`, `dp_dict` | route 级待处理需求 | 内存 list |
| `_process_routes()` | 跨节点需求、Truck 配置 | 单路线处理 | 内存 list |
| `_process_truck_type()` | 车型配置、剩余需求、库存限制、bypass 规则 | 实际装载、发运记录 | `delivery_plan`, `vehicle_log` |
| `_first_pass_loading()` / `_second_pass_loading()` | `VehiclePacker`, 可用库存 | 更新 packer 装载状态 | 内存对象 |
| `_generate_shipment_records()` | 装载结果、LeadTime、DelayDistribution、agg_status | `DeliveryPlan`, `VehicleLog`, `BypassRuleHitLog`, 更新 `agg_status` 和库存 | 对应 M6 输出 sheet |
| `_handle_remaining_demands()` | 剩余需求、`max_wait_days` | `UnsatisfiedMDQLog` | `UnsatisfiedMDQLog` |
| `_validate_shipment_delivery_constraint()` | `delivery_plan_df`, Orchestrator shipment log | 验证日志 | `ValidationLog` |
| `_generate_outputs()` | delivery/vehicle/unsat/bypass/validation 内存结果 | 标准 M6 输出 | `DeliveryPlan`, `VehicleLog`, `TruckUsageLog`, `UnsatisfiedMDQLog`, `ValidationLog`, `BypassRuleHitLog` |

## 9. Summary 层依赖矩阵

| 函数 | 上游依赖 | 下游输出 | 文件/表影响 |
|---|---|---|---|
| `generate_all_reports()` | 各模块日输出 + Orchestrator CSV | 8 张汇总报表 | `summary/*.xlsx` / `.csv` |
| `_generate_order_shipment_cut_report()` | M1 `OrderLog`, `ShipmentLog`, `CutLog` | 订单/发货/缺货汇总 | `full_order_shipment_cut_report.xlsx` |
| `_generate_delivery_report()` | M6 `DeliveryPlan` | 交付汇总 | `full_delivery_plan_report.xlsx` |
| `_generate_truck_usage_report()` | M6 `TruckUsageLog` | 卡车使用汇总 | `full_truck_usage_report.xlsx` |
| `_generate_capacity_exceed_report()` | M4 `CapacityExceed` | 超产能汇总 | `full_exceed_capacity_report.xlsx` |
| `_generate_changeover_report()` | M4 `ChangeoverLog` | 换产汇总 | `full_changeover_report.xlsx` |
| `_generate_deployment_report()` | M5 `DeploymentPlan` | 部署汇总 | `full_deployment_plan_report.xlsx` / `.csv` |
| `_generate_production_report()` | M4 `ProductionPlan` | 生产汇总 | `full_production_plan_report.xlsx` |
| `_generate_historical_inventory_report()` | Orchestrator CSV + M1 输出 | 历史库存汇总 | `historical_inventory_record.csv` |

## 10. DB 写入层依赖矩阵

### 10.1 数据库初始化与配置读取

| 函数 | 上游依赖 | 下游依赖 | 表影响 |
|---|---|---|---|
| `DatabaseInitializer.initialize()` | DB 连接参数、`config_name`、本地 Excel（可选自动导入） | `ExcelImporter.import_excel_file`, DB 模式仿真入口 | 建库、补配置表 |
| `ExcelImporter.import_excel_file()` | 配置 Excel | `cfg_*` 表 | `cfg_global_network`, `cfg_m1_demandforecast`, `cfg_m4_linecapacity` 等 |
| `_load_config_from_database()` | `cfg_*` 表 | `config_dict` | 把表读回内存字典 |

### 10.2 模块输出直写数据库

| 函数 | 上游依赖 | 下游输出表 | 关键字段 |
|---|---|---|---|
| `write_module_results_from_dict()` | 主流程内存结果 `all_results` | `module*_output_*` | `sim_date`, `run_id`, `config_name` |
| `write_orchestrator_data()` | `orchestrator/*.csv` | `orchestrator_*` | `file_date`, `sim_date`, `run_id` |
| `truncate_output_tables()` | `run_id` | 按 run 删除旧数据 | 保证幂等重跑 |
| `create_table_from_df()` | DataFrame、表名 | 自动建表/补列/写库 | 可追加 `config_name`, `db_write_time` |

### 10.3 结果表映射矩阵

| 模块键 | DataFrame 键 | 输出表名 |
|---|---|---|
| module1 | `orders_df` | `module1_output_orderlog` |
| module1 | `shipment_df` | `module1_output_shipmentlog` |
| module1 | `cut_df` | `module1_output_cutlog` |
| module1 | `supply_demand_df` | `module1_output_supplydemandlog` |
| module1 | `summary_df` | `module1_output_summary` |
| module3 | `net_demand_df` | `module3_output_netdemand` |
| module4 | `production_df` | `module4_output_productionplan` |
| module4 | `exceed_log` | `module4_output_capacityexceed` |
| module4 | `issues_df` | `module4_output_validation` |
| module4 | `changeover_log` | `module4_output_changeoverlog` |
| module5 | `deployment_plan` | `module5_output_deploymentplan` |
| module5 | `unfulfilled_log` | `module5_output_unfulfilledlog` |
| module5 | `stock_on_hand_log` | `module5_output_stockonhandlog` |
| module5 | `validation_log` | `module5_output_validation` |
| module6 | `delivery_plan` | `module6_output_deliveryplan` |
| module6 | `vehicle_log` | `module6_output_vehiclelog` |
| module6 | `truck_usage` | `module6_output_truckusagelog` |
| module6 | `unsatisfied_log` | `module6_output_unsatisfiedmdqlog` |
| module6 | `validation_log` | `module6_output_validationlog` |
| module6 | `bypass_log` | `module6_output_bypassrulehitlog` |

### 10.4 DB Summary 生成矩阵

| 函数 | 上游表 | 下游表 | 备注 |
|---|---|---|---|
| `_generate_order_shipment_cut_summary()` | `module1_output_orderlog`, `module1_output_shipmentlog`, `module1_output_cutlog` | `summary_output_ordershipmentcutsummary` | 会先对订单去重 |
| `_generate_changeover_summary()` | `module4_output_changeoverlog` | `summary_output_fullchangeoverlog` | 过滤 `changeover_end_date` / `date` |
| `_generate_capacity_exceed_summary()` | `module4_output_capacityexceed` | `summary_output_fullcapacityexceed` | 按 `date` 过滤 |
| `_generate_production_plan_summary()` | `module4_output_productionplan` | `summary_output_fullproductionplan` | 按 `available_date` 过滤 |
| `_generate_deployment_plan_summary()` | `module5_output_deploymentplan` | `summary_output_fulldeploymentplan` | 过滤多种日期列 |
| `_generate_delivery_plan_summary()` | `module6_output_deliveryplan` | `summary_output_fulldeliveryplan` | 过滤 `planned_deploy_date` / `actual_ship_date` |
| `_generate_truck_usage_summary()` | `module6_output_truckusagelog` | `summary_output_fulltruckusage` | 按 `date` 过滤 |

## 11. 五个复杂文件的重点依赖矩阵

### 11.1 `orchestrator/`（包）

| 函数 | 直接读取 | 直接修改/写出 |
|---|---|---|
| `process_module1_shipments()` | M1 `shipment_df` | `unrestricted_inventory`, `shipment_log`, `shipment_log_by_date` |
| `process_module4_production()` | M4 `production_df` | `production_plan_backlog`, `production_gr`, `unrestricted_inventory` |
| `process_module5_deployment()` | M5 `deployment_df` | `open_deployment`, `uid_sequence` |
| `process_module6_delivery()` | M6 `delivery_df` | `open_deployment`, `unrestricted_inventory`, `delivery_shipment_log`, `delivery_gr`, `in_transit` |
| `_process_delivery_arrivals()` | `in_transit` | `unrestricted_inventory`, `delivery_gr`, `delivery_gr_by_date` |
| `save_daily_state()` | 全部状态对象 | `orchestrator/*.csv` |

### 11.2 `deployment_planning/main.py`

| 函数 | 直接读取 | 直接修改/写出 |
|---|---|---|
| `_initialize_soh_dict()` | `InventoryLog`, `SupplyDemandLog`, `SafetyStock`, `OrderLog` | `soh_dict` |
| `_process_layer_demands()` | layer 节点、缓存、索引 | `node_demands_map` |
| `_allocate_pipeline_sources()` | `future_intransit`, `open_deployment_inbound`, `future_production` | 回写 `demand_rows` pipeline 字段 |
| `_process_gaps_and_create_plans()` | `demand_rows`, `adjusted_qtys`, `Network`, `LeadTime`, `PTF/LSK` | `deployment_plan_rows`, `unfulfilled_rows`, `up_gap_next` |
| `_update_soh_dict()` | production GR、in transit、delivery GR、shipment、deployment_plan_rows | `soh_dict`, `stock_on_hand_log` |
| `run_daily_deployment_planning()` | 全部配置和状态输入 | `DeploymentPlan`, `UnfulfilledLog`, `StockOnHandLog`, `Validation` |

### 11.3 `capacity_allocator.py`

| 函数 | 直接读取 | 直接修改/写出 |
|---|---|---|
| `allocate()` | `uncon` | `plans_log`, `exceed_log` |
| `_allocate_line_group()` | 分线批次组 | 单线计划、超额记录 |
| `_allocate_batch()` | `mlcfg`, `state`, `batch` | 传给 horizon 分配 |
| `_allocate_to_horizon()` | planning window、changeover | 多天计划、超额记录 |
| `_allocate_day()` | `cap_map`, `previously_allocated`, `rate_map`, `mct_map` | 单日计划、回写 `cap_map` |
| `extract_allocated_capacity_from_plan()` | `plan_df`, `rate_map`, `changeover_def` | `allocated_capacity` 字典 |
| `extract_line_states_from_plan()` | `plan_df`, 可选 changeover 分析输入 | `line_states` 字典 |

### 11.4 `logistics_execution/`（包，原 `module6.py`）

| 函数 | 直接读取 | 直接修改/写出 |
|---|---|---|
| `_prepare_data()` | DeploymentPlan、TruckReleaseCon、TruckTypeSpecs、MaterialMD、DemandPriority | `dp`, `prio_map`, `spec_map`, `cap_map`, `validation_log` |
| `_collect_pending_demands()` | `agg_status`, `dp_dict` | `pending_rows` |
| `_process_truck_type()` | route demand、truck config、inventory | `delivery_plan`, `vehicle_log`, `bypass_log`, `agg_status`, `remaining_demands` |
| `_generate_shipment_records()` | `packer.load_records`, lead time, delay distribution | `delivery_plan`, `vehicle_log`, `bypass_log`, 库存、`agg_status` |
| `_handle_remaining_demands()` | route remaining、`agg_status` | `unsat_log`, `agg_status` |
| `_generate_outputs()` | 仿真结果内存 list | M6 输出 DataFrame / Excel |

### 11.5 `module_data_writer.py`

| 函数 | 直接读取 | 直接修改/写出 |
|---|---|---|
| `_write_excel_file()` | 单个每日 Excel 文件 | `module*_output_*` 表 |
| `write_orchestrator_data()` | `orchestrator/*.csv` | `orchestrator_*` 表 |
| `write_module_results_from_dict()` | `all_results` 内存结构 | `module*_output_*` 表 |
| `generate_summary_reports_from_db()` | DB 模块输出表 | `summary_output_*` 表 |

## 12. 排障速查

| 问题 | 先看函数 | 再看文件/表 |
|---|---|---|
| 续跑失败 | `detect_last_complete_date()`, `restore_orchestrator_state()` | `orchestrator/*.csv` |
| M5 计划量异常大 | `apply_grouped_moq_rv()`, `_allocate_pipeline_sources()`, `_process_gaps_and_create_plans()` | `Module5Output_*.xlsx`, `open_deployment_*.csv` |
| M6 发货量异常 | `_process_truck_type()`, `_generate_shipment_records()`, `_validate_shipment_delivery_constraint()` | `Module6Output_*.xlsx`, `shipment_log_*.csv` |
| M3 净需求异常 | `calculate_daily_net_demand()`, `run_mrp_layered_simulation_daily()` | `Module3Output_*.xlsx`, `module1_output_*.xlsx`, `orchestrator/*.csv` |
| DB 表没数据 | `write_module_results_from_dict()`, `write_orchestrator_data()` | `module*_output_*`, `orchestrator_*`, `run_id` |
| DB Summary 不对 | `_generate_*_summary()` | `summary_output_*`, 源输出表 |

## 13. 结论

- 这套矩阵最核心的价值，是把“函数级逻辑”变成“上下游依赖关系图谱”。
- 真正高风险的不是某个函数长不长，而是它读写了哪些状态、哪些文件、哪些表。
- 对 ChainSight 来说，最不能随便改的有三类：
  - 状态主键：`material/location/sending/receiving/ori_deployment_uid/vehicle_uid`
  - 运行顺序：`M1 -> M4 -> M5 -> M6 -> M3`
  - 数据契约：每日 Excel、Orchestrator CSV、DB 输出表名与空表结构

## 附录 A. 使用建议

- 评估改动影响面时先看本文。
- 查结果空表、错表、断链时先看本文。
- 查函数细节时再配合函数级手册一起看。
