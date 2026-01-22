# ChainSight 重构代码对照文档

## 📋 概述

本文档详细说明了 ChainSight 供应链仿真系统重构前后的代码对应关系。重构的主要目标是将单体模块拆分为更小、更可维护的组件，同时**保持数据处理逻辑完全一致**。

### 目录结构对比

| 重构前 (code_vo/) | 重构后 (src/) | 说明 |
|------------------|--------------|------|
| `code_vo/` (扁平结构) | `src/` (模块化结构) | 主代码目录 |
| - | `src/core/` | 核心入口和编排器 |
| - | `src/modules/` | 业务模块 |
| - | `src/services/` | 服务层 |
| - | `src/utils/` | 工具函数 |

---

## 🎯 核心组件对照

### 1. 入口与集成层

| 重构前文件 | 重构后文件 | 功能说明 |
|-----------|-----------|---------|
| `code_vo/run.py` | `src/core/run.py` | 命令行入口脚本 |
| `code_vo/main_integration.py` | `src/core/main_integration.py` | 集成仿真主程序 |
| `code_vo/orchestrator.py` | `src/core/orchestrator.py` | 状态编排器 (库存、部署、在途) |

### 2. 工具类

| 重构前文件 | 重构后文件 | 功能说明 |
|-----------|-----------|---------|
| `code_vo/logger_config.py` | `src/utils/logger_config.py` | 日志配置 |
| `code_vo/time_manager.py` | `src/utils/time_manager.py` | 时间管理器 |
| `code_vo/config_validator.py` | `src/utils/config_validator.py` | 配置验证器 |
| `code_vo/validation_manager.py` | `src/utils/validation_manager.py` | 验证管理器 |
| `code_vo/inventory_balance_checker.py` | `src/utils/inventory_balance_checker.py` | 库存平衡检查 |

### 3. 服务层

| 重构前文件 | 重构后文件 | 功能说明 |
|-----------|-----------|---------|
| `code_vo/summary_report_generator.py` | `src/services/summary_report_generator.py` | 汇总报告生成 |
| `code_vo/performance_profiler.py` | `src/services/performance_profiler.py` | 性能分析器 |

---

## 📦 Module1 - 订单生成模块 (Demand Planning)

### 模块概述

Module1 负责需求计划与订单生成，包括：
- 周度预测的DPS拆分与日度展开
- 基于AO配置的订单生成
- 预测消耗逻辑
- 发货模拟与供需日志生成

### 文件对照

| 重构前 | 重构后 | 说明 |
|-------|-------|------|
| `code_vo/module1.py` | `src/modules/module1.py` | Facade入口 |
| - | `src/modules/demand_planning/__init__.py` | 子包入口 |
| - | `src/modules/demand_planning/config.py` | 配置加载 |
| - | `src/modules/demand_planning/constants.py` | 常量定义 |
| - | `src/modules/demand_planning/normalization.py` | 标识符标准化 |
| - | `src/modules/demand_planning/dps.py` | DPS处理 |
| - | `src/modules/demand_planning/forecast.py` | 预测拆分 |
| - | `src/modules/demand_planning/order.py` | 订单生成 |
| - | `src/modules/demand_planning/consume.py` | 预测消耗 |
| - | `src/modules/demand_planning/shipment.py` | 发货处理 |
| - | `src/modules/demand_planning/io_utils.py` | IO工具 |
| - | `src/modules/demand_planning/integration.py` | 集成接口 |

### 函数详细说明

#### 配置与标准化函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `load_config(filename, sheet_mapping)` | `demand_planning/config.py` | 从Excel文件加载各配置页为DataFrame字典，对已存在的sheet进行解析并调用`_normalize_identifiers`保证键规范，对不存在的sheet使用默认空表或None填充 |
| `_normalize_location(location_str)` | `demand_planning/normalization.py` | 规范化地点字符串：将数值或字符串形式的地点编号统一为4位、左侧补零的字符串（如"7"→"0007"），对None/NaN返回空字符串 |
| `_normalize_material(material_str)` | `demand_planning/normalization.py` | 规范化物料字符串：将输入统一转为字符串，对None/NaN返回空字符串，确保合并与分组时的键一致 |
| `_normalize_identifiers(df)` | `demand_planning/normalization.py` | 统一规范化标识符列（material/location/sending/receiving/sourcing/dps_location）：全部转换为字符串类型，缺失值填充为空字符串，location使用向量化zfill(4)保证4位编号 |

#### DPS与供应选择函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `apply_dps(df, dps_cfg)` | `demand_planning/dps.py` | 按DPS配置进行地点拆分：先在MaterialLocationWeek粒度聚合，再按百分比分割为"保留量"和"拆分量"，拆分量的地点改为dps_location，输出重新在MaterialLocationWeek粒度汇总 |
| `apply_supply_choice(df, supply_cfg)` | `demand_planning/dps.py` | 应用供应选择对周度预测进行数量调整：在MaterialLocationWeek粒度合并adjust_quantity并进行向量化加总，缺失调整量按0处理 |

#### 预测拆分函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `expand_forecast_to_days_integer_split(demand_weekly, start_date, num_weeks, simulation_end_date)` | `demand_planning/forecast.py` | 将周度预测均匀拆分为7天的日度预测（整数分配）：每周数量按base_qty=quantity//7分配，余数remainder的前remainder天各加1，仅生成至simulation_end_date |

#### 订单生成函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `generate_daily_orders(sim_date, original_forecast, current_forecast, ao_config, order_calendar, forecast_error)` | `demand_planning/order.py` | 根据仿真日期、预测数据和AO配置生成当日订单，支持AO（提前订单）和Normal（常规订单）两种类型，应用预测误差进行数量调整 |
| `generate_quantity_with_percent_error(mean_qty, material, location, order_type, forecast_error)` | `demand_planning/order.py` | 根据预测误差配置为订单数量添加随机误差，使用截断正态分布确保数量非负 |

#### 预测消耗函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `consume_forecast_ao_logic(forecast_df, material, location, order_date, consume_qty)` | `demand_planning/consume.py` | AO订单的预测消耗逻辑：按AO配置的提前天数窗口，从订单日期向后查找预测进行消耗，支持跨多日消耗直到数量耗尽 |
| `consume_forecast_normal_logic(forecast_df, material, location, order_date, consume_qty)` | `demand_planning/consume.py` | Normal订单的预测消耗逻辑：仅消耗订单日期当天的预测数量 |
| `_consume_ao_for_ml_worker(args)` | `demand_planning/consume.py` | AO消耗的并行工作函数：按material-location分组进行并行处理以提升性能 |
| `_consume_normal_for_ml_worker(args)` | `demand_planning/consume.py` | Normal消耗的并行工作函数：按material-location分组进行并行处理 |

#### 发货与集成函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `simulate_shipment_for_single_day(simulation_date, order_log, inventory_dict, config_dict)` | `demand_planning/shipment.py` | 模拟单日发货：根据当日到期订单和可用库存计算实际发货量和缺货量，优先满足高优先级订单 |
| `run_daily_order_generation(config_dict, simulation_date, m1_output_dir, orchestrator, ...)` | `demand_planning/integration.py` | 运行单日订单生成的主入口函数：协调配置加载、订单生成、预测消耗、发货模拟和输出保存 |
| `generate_supply_demand_log_for_integration(config, current_forecast, orders_df, shipment_df, simulation_date)` | `demand_planning/integration.py` | 为集成模式生成供需日志：汇总当前预测、订单和发货数据形成统一的供需视图 |
| `save_module1_output_with_supply_demand(m1_output_dir, simulation_date, orders_df, shipment_df, cut_df, forecast_df, supply_demand_log)` | `demand_planning/io_utils.py` | 保存Module1输出到Excel文件：包含OrderLog、ShipmentLog、CutLog、ForecastLog和SupplyDemandLog等sheet |
| `_load_previous_orders(m1_output_dir, current_date, max_advance_days)` | `demand_planning/io_utils.py` | 加载历史订单文件：读取之前仿真日期生成的未来订单，用于构建完整的订单池 |
| `_build_available_inventory_from_orchestrator(orchestrator, simulation_date)` | `demand_planning/integration.py` | 从Orchestrator构建可用库存字典：获取当前库存状态用于发货模拟 |

---

## 📦 Module3 - MRP/净需求计算模块 (MRP Planning)

### 模块概述

Module3 负责物料需求计划（MRP）计算，包括：
- 网络层级分配
- 提前期计算
- 净需求计算（考虑库存、在途、安全库存）
- 分层MRP仿真

### 文件对照

| 重构前 | 重构后 | 说明 |
|-------|-------|------|
| `code_vo/module3.py` | `src/modules/module3.py` | Facade入口 |
| - | `src/modules/mrp_planning/__init__.py` | 子包入口 |
| - | `src/modules/mrp_planning/config_loader.py` | 配置加载 |
| - | `src/modules/mrp_planning/constants.py` | 常量定义 |
| - | `src/modules/mrp_planning/layer_assignment.py` | 层级分配 |
| - | `src/modules/mrp_planning/lead_time.py` | 提前期计算 |
| - | `src/modules/mrp_planning/net_demand.py` | 净需求计算 |
| - | `src/modules/mrp_planning/mrp_simulation.py` | MRP仿真 |
| - | `src/modules/mrp_planning/node_processor.py` | 节点处理 |
| - | `src/modules/mrp_planning/integration.py` | 集成接口 |
| - | `src/modules/mrp_planning/utils.py` | 工具函数 |

### 函数详细说明

#### 配置加载函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `load_config(config_path)` | `mrp_planning/config_loader.py` | 从Excel文件加载MRP配置数据：包括M3_SafetyStock（安全库存）、Global_Network（网络配置）、Global_LeadTime（提前期配置） |
| `load_module1_daily_outputs(module1_output_dir, simulation_date)` | `mrp_planning/config_loader.py` | 读取Module1当天的输出：SupplyDemandLog（已消耗预测）、ShipmentLog（当日发货）、OrderLog（订单池），用于MRP计算的需求输入 |

#### 层级分配函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `assign_location_layers(network_df)` | `mrp_planning/layer_assignment.py` | 为网络中的每个location分配层级：根据sourcing关系构建有向图，使用BFS从根节点（无sourcing）向下分配层级，用于MRP的分层计算顺序 |

#### 提前期计算函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `determine_lead_time(material, sending, receiving, config_dict, ...)` | `mrp_planning/lead_time.py` | 确定两个节点间的提前期：Plant节点使用max(MCT, PDT+GR)+PTF+LSK-1计算口径，DC节点使用PDT+GR，返回(horizon_start, horizon_end)窗口 |
| `_build_ptf_lsk_cache_m3(m4_mlcfg_df)` | `mrp_planning/lead_time.py` | 构建PTF/LSK缓存字典：从M4_MaterialLocationLineCfg提取(material, location) -> (ptf, lsk)映射，避免重复查询 |
| `_get_ptf_lsk(material, site, m4_mlcfg_df, cache)` | `mrp_planning/lead_time.py` | 获取指定物料和站点的PTF（计划时间围栏）和LSK（回顾周期），优先从缓存获取 |
| `_compute_root_horizon(material, simulation_date, ...)` | `mrp_planning/lead_time.py` | 计算根节点（Plant）的计划窗口：考虑PTF、LSK和MCT等参数 |

#### 净需求计算函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `calculate_daily_net_demand(material, location, simulation_date, ...)` | `mrp_planning/net_demand.py` | 计算单个物料-地点的每日净需求：净需求 = 毛需求 + 安全库存目标 - 现有库存 - 在途 - 计划收货，结果不小于0 |
| `apply_moq_rv(qty, moq, rv, is_cross_node)` | `mrp_planning/utils.py` | 应用MOQ（最小订货量）和RV（重订量）约束：若qty<moq则补到moq，否则向上取整到rv的倍数，自循环调运不应用约束 |
| `_lookup_moq_rv_three_keys(deploy_config_df, material, sending, receiving)` | `mrp_planning/utils.py` | 在deploy_config中按(material, sending, receiving)三级键查找MOQ/RV配置，回退到(material, sending)，未命中返回(1,1) |

#### MRP仿真函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `run_mrp_layered_simulation_daily(simulation_date, config_dict, ...)` | `mrp_planning/mrp_simulation.py` | 运行分层MRP仿真：按层级从上到下（从终端客户向上游工厂）计算各节点净需求，将本层缺口传递为上层的派生需求 |
| `infer_sending_location_type(material, sending, network_df, sim_date)` | `mrp_planning/utils.py` | 推断发送节点的类型（Plant/DC/Customer）：根据network配置中的location_type字段判断 |

#### 集成函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `run_integrated_mode(config_dict, simulation_date, output_dir, ...)` | `mrp_planning/integration.py` | 运行集成模式的MRP计算：协调配置加载、层级分配、净需求计算和输出保存 |
| `_apportion_largest_remainder(values, target)` | `mrp_planning/utils.py` | 最大余数法保和分配：将target按values的比例分配为整数，保证合计等于target |

---

## 📦 Module4 - 生产计划模块 (Production Planning)

### 模块概述

Module4 负责工业级APS生产计划的日度执行逻辑，包括：
- 配置加载与校验
- 净需求读取与计划构建
- 产能分配与换产优化
- 跨天产线状态连续性
- 生产计划输出

### 文件对照

| 重构前 | 重构后 | 说明 |
|-------|-------|------|
| `code_vo/module4.py` | `src/modules/module4.py` | Facade入口 |
| - | `src/modules/production_planning/__init__.py` | 子包入口 |
| - | `src/modules/production_planning/config_loader.py` | 配置加载 |
| - | `src/modules/production_planning/constants.py` | 常量定义 |
| - | `src/modules/production_planning/types.py` | 类型定义 |
| - | `src/modules/production_planning/state_manager.py` | 状态管理 |
| - | `src/modules/production_planning/demand_loader.py` | 需求加载 |
| - | `src/modules/production_planning/capacity_allocator.py` | 产能分配 |
| - | `src/modules/production_planning/plan_builder.py` | 计划构建 |
| - | `src/modules/production_planning/output_writer.py` | 输出写入 |
| - | `src/modules/production_planning/main.py` | 主入口 |
| - | `src/modules/production_planning/utils.py` | 工具函数 |

### 函数详细说明

#### 配置与状态管理函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `load_config(filepath)` | `production_planning/config_loader.py` | 加载生产计划配置：MaterialLocationLineCfg（物料产线配置）、LineCapacity（产线产能）、ChangeoverMatrix（换产矩阵）、ChangeoverDefinition（换产定义） |
| `validate_config(cfg)` | `production_planning/config_loader.py` | 校验配置完整性：检查必要字段存在、数据类型正确、关联关系一致 |
| `get_or_init_simulation_start(output_dir, provided_start)` | `production_planning/utils.py` | 读取或初始化仿真开始日期：首次运行写入simulation_start.txt，后续运行读取以保证审查日计算一致 |

#### 状态持久化函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `save_line_state(output_dir, simulation_date, line_states)` | `production_planning/state_manager.py` | 保存产线状态（最后物料与剩余换产时间）：写入line_states_YYYYMMDD.json，用于跨天连续性 |
| `load_line_state(output_dir, simulation_date)` | `production_planning/state_manager.py` | 加载前一天的产线状态：读取JSON文件恢复last_material和remaining_changeover等状态 |
| `save_allocated_capacity(output_dir, simulation_date, allocated_capacity)` | `production_planning/state_manager.py` | 保存已分配产能（小时）：按location\|line\|production_date键存储，避免多个仿真日对同一生产日重复分配 |
| `load_allocated_capacity(output_dir, simulation_date)` | `production_planning/state_manager.py` | 加载当前仿真日的已分配产能：用于同一天内避免重复分配 |
| `load_all_previous_capacity(output_dir, simulation_date)` | `production_planning/state_manager.py` | 汇总所有历史仿真日的已分配产能：遍历历史JSON文件合并为统一字典供校验使用 |

#### 需求加载函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `load_daily_net_demand(module3_output_dir, simulation_date)` | `production_planning/demand_loader.py` | 从Module3输出读取当日净需求数据：筛选层级与日期，构建生产计划的需求输入 |

#### 计划构建函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `build_unconstrained_plan_for_single_day(net_demand_df, mlcfg, simulation_date, simulation_start, issues)` | `production_planning/plan_builder.py` | 构建单日无约束生产计划：根据净需求和产线配置生成理论生产计划，不考虑产能限制 |
| `optimal_changeover_sequence(batches, co_mat, co_def, line)` | `production_planning/plan_builder.py` | 计算最优换产序列：使用贪心算法或动态规划优化批次排序，最小化总换产时间 |
| `extract_line_states_from_plan(plan_df, cap_df, ...)` | `production_planning/plan_builder.py` | 从生产计划提取产线状态：分析每条产线在日末的物料和换产状态 |
| `analyze_end_of_day_changeover_state(plan_df, cap_df, ...)` | `production_planning/plan_builder.py` | 分析日末换产状态：计算未完成换产在次日需要延续的时间 |

#### 产能分配函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `compute_planning_window(simulation_date, ptf, lsk)` | `production_planning/capacity_allocator.py` | 计算计划窗口：根据PTF（计划时间围栏）和LSK（回顾周期）确定可排产的日期范围 |
| `extract_allocated_capacity_from_plan(plan_df, rate_map, changeover_def)` | `production_planning/capacity_allocator.py` | 从生产计划提取已分配产能：汇总每个location/line/date的生产与换产耗时 |
| `validate_capacity_allocation(plan_log, previously_allocated_capacity, ...)` | `production_planning/capacity_allocator.py` | 校验产能分配：检查是否超出产线可用产能，记录超额情况 |
| `calculate_changeover_metrics(production_plan, changeover_def)` | `production_planning/plan_builder.py` | 计算换产指标：统计换产次数、总换产时间、产能利用率等KPI |

#### 工具函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `is_review_day(simulation_date, simulation_start, lsk, day)` | `production_planning/utils.py` | 判断是否为审查日：根据LSK周期判断当天是否需要审查特定生产日的计划 |
| `_normalize_location(location_str)` | `production_planning/utils.py` | 规范化地点字符串：统一为4位补零格式 |
| `_cast_identifiers_to_str(df, cols)` | `production_planning/utils.py` | 将标识符列转换为字符串：确保合并时类型一致 |

---

## 📦 Module5 - 部署计划模块 (Deployment Planning)

### 模块概述

Module5 负责多层级部署规划，在给定网络、需求、库存与产运数据下，按优先级与约束生成跨节点调拨计划：
- 需求收集与优先级排序
- 库存可用性计算
- 分配算法（Pull/Push/Soft-Push）
- MOQ/RV约束应用

### 文件对照

| 重构前 | 重构后 | 说明 |
|-------|-------|------|
| `code_vo/module5.py` | `src/modules/module5.py` | Facade入口 |
| - | `src/modules/deployment_planning/__init__.py` | 子包入口 |
| - | `src/modules/deployment_planning/data_loader.py` | 数据加载 |
| - | `src/modules/deployment_planning/constants.py` | 常量定义 |
| - | `src/modules/deployment_planning/cache_utils.py` | 缓存工具 |
| - | `src/modules/deployment_planning/normalizer.py` | 标准化器 |
| - | `src/modules/deployment_planning/inventory.py` | 库存计算 |
| - | `src/modules/deployment_planning/demand_collector.py` | 需求收集 |
| - | `src/modules/deployment_planning/allocation.py` | 分配算法 |
| - | `src/modules/deployment_planning/push_allocation.py` | Push分配 |
| - | `src/modules/deployment_planning/validation.py` | 验证逻辑 |
| - | `src/modules/deployment_planning/main.py` | 主入口 |

### 函数详细说明

#### 数据加载函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `load_module1_daily_shipment(module1_output_dir, current_date)` | `deployment_planning/data_loader.py` | 加载Module1当日发货数据（ShipmentLog）：用于当日可用库存的扣减（对客发货） |
| `load_module1_daily_orders(module1_output_dir, current_date)` | `deployment_planning/data_loader.py` | 加载Module1当日订单池（OrderLog）：AO/normal订单参与窗口需求分配与缺口传递 |
| `load_orchestrator_delivery_gr(orchestrator, current_date)` | `deployment_planning/data_loader.py` | 从Orchestrator加载当日收货（GR）视图：用于当日可用库存的增加（收货入库） |
| `load_orchestrator_open_deployment(orchestrator, current_date)` | `deployment_planning/data_loader.py` | 从Orchestrator加载未完成部署视图：已计划但未执行的调拨，用于计算预期到货 |

#### 缓存与提前期函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `_build_ptf_lsk_cache(m4_mlcfg_df)` | `deployment_planning/cache_utils.py` | 构建PTF/LSK缓存：从M4配置提取(material, location) -> (ptf, lsk)映射 |
| `_build_lead_time_cache(lead_time_df)` | `deployment_planning/cache_utils.py` | 构建提前期缓存：从Global_LeadTime提取(sending, receiving) -> (pdt, gr, mct)映射 |
| `_get_ptf_lsk(material, site, m4_mlcfg_df, cache)` | `deployment_planning/cache_utils.py` | 获取指定物料和站点的PTF/LSK，优先从缓存获取 |
| `determine_lead_time(material, sending, receiving, ...)` | `deployment_planning/cache_utils.py` | 确定两节点间的提前期，计算部署的计划窗口(horizon_start, horizon_end) |

#### 库存计算函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `build_open_deployment_inbound(open_deployment_df)` | `deployment_planning/inventory.py` | 构建未完成部署的入库汇总：按(receiving, material)聚合expected_qty，用于计算预期库存 |
| `calculate_projected_inventory(material, location, current_inventory, ...)` | `deployment_planning/inventory.py` | 计算预测库存：当前库存 + 在途 + 计划收货 - 计划出库，考虑时间窗口 |
| `calculate_available_inventory(material, location, projected_inventory, ...)` | `deployment_planning/inventory.py` | 计算可用库存：预测库存 - 安全库存，结果不小于0 |

#### 需求收集函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `assign_location_layers(network_df)` | `deployment_planning/demand_collector.py` | 为网络节点分配层级：使用BFS从根节点向下分配，用于分层收集需求 |
| `get_upstream(location, material, network_df, sim_date, ...)` | `deployment_planning/demand_collector.py` | 获取上游供应节点：根据network配置查找material在location的sourcing |
| `get_sending_location_type(material, sending, network_df, sim_date)` | `deployment_planning/demand_collector.py` | 获取发送节点类型（Plant/DC）：用于确定提前期计算方式 |

#### 分配算法函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `apply_moq_rv(qty, moq, rv, is_cross_node)` | `deployment_planning/allocation.py` | 应用MOQ/RV约束：跨节点调运qty<moq则补到moq，否则向上取整到rv倍数；自循环不应用约束 |
| `_lookup_moq_rv(deploy_cfg, material, sending, receiving)` | `deployment_planning/allocation.py` | 在deploy_config中查找MOQ/RV配置：优先(material,sending,receiving)，回退(material,sending) |
| `apply_grouped_moq_rv(demand_rows, location)` | `deployment_planning/allocation.py` | 按路径分组应用MOQ/RV：将同一(material,sending,receiving)路径的需求合并后应用约束 |
| `apply_priority_allocation_vectorized(demand_rows, adjusted_qtys, current_stock, demand_priority_map)` | `deployment_planning/allocation.py` | 向量化优先级分配：按优先级顺序分配可用库存，高优先级订单优先满足，返回实际分配量和未满足量 |

#### 标准化函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `_normalize_location(location_str)` | `deployment_planning/normalizer.py` | 规范化地点编码：补齐为4位数字字符串 |
| `_normalize_material(material_str)` | `deployment_planning/normalizer.py` | 规范化物料编码为字符串 |
| `_normalize_identifiers(df)` | `deployment_planning/normalizer.py` | 标识字段统一为字符串并格式化地点字段 |

---

## 📦 Module6 - 物流执行模块 (Logistics Execution)

### 模块概述

Module6 负责物流物理流执行，将计划的部署转化为实际的发运和收货：
- 装车规则评估
- 车辆装载优化
- 交付延迟采样
- 库存更新与GR生成

### 文件对照

| 重构前 | 重构后 | 说明 |
|-------|-------|------|
| `code_vo/module6.py` | `src/modules/module6.py` | Facade入口 |
| - | `src/modules/logistics_execution/__init__.py` | 子包入口 |
| - | `src/modules/logistics_execution/config_loader.py` | 配置加载 |
| - | `src/modules/logistics_execution/expression_evaluator.py` | 表达式求值 |
| - | `src/modules/logistics_execution/capacity_manager.py` | 容量管理 |
| - | `src/modules/logistics_execution/inventory_manager.py` | 库存管理 |
| - | `src/modules/logistics_execution/vehicle_packer.py` | 装车算法 |
| - | `src/modules/logistics_execution/delivery_processor.py` | 交付处理 |
| - | `src/modules/logistics_execution/validators.py` | 验证器 |

### 函数详细说明

#### 配置加载函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `load_standalone_config(input_excel)` | `logistics_execution/config_loader.py` | 加载独立模式配置：从Excel读取TruckReleaseCon、TruckCapacityPlan、TruckTypeSpecs、MaterialMD等 |
| `load_integrated_config(config_dict, orchestrator, current_date)` | `logistics_execution/config_loader.py` | 加载集成模式配置：从config_dict和Orchestrator获取OpenDeployment和各项M6配置 |

#### 表达式求值类

| 类/函数名 | 重构后位置 | 功能说明 |
|---------|-----------|---------|
| `class SafeExpressionEvaluator` | `logistics_execution/expression_evaluator.py` | 安全表达式求值器：解析和执行TruckReleaseCon中的发车条件表达式，支持AND/OR/NOT逻辑和比较运算符，限制可用变量以防止代码注入 |
| `SafeExpressionEvaluator.eval(expr, context)` | - | 执行表达式求值：将表达式字符串解析为AST并在给定context下求值返回布尔结果 |

#### 验证函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `_check_and_deduplicate(df, key_column, sheet_name, validation_log)` | `logistics_execution/validators.py` | 检查并去重DataFrame：基于key_column检测重复记录，保留第一条并记录验证日志 |
| `_generate_validation_report(validation_log, output_file)` | `logistics_execution/validators.py` | 生成验证报告：将validation_log中的问题按severity分组输出到文本文件 |
| `should_bypass_mdq(material, sending, receiving, bypass_rules)` | `logistics_execution/validators.py` | 检查是否跳过MDQ约束：根据MDQBypassRules配置判断特定路线是否豁免最小交付量限制 |

#### 容量管理函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `_normalize_capacity_plan(truck_cap_df, current_date)` | `logistics_execution/capacity_manager.py` | 标准化运力计划：将TruckCapacityPlan按日期和路线展开，计算每条路线的可用车辆数和容量 |

#### 交付处理函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `sample_delivery_delay(sending, receiving, dist_df)` | `logistics_execution/delivery_processor.py` | 采样交付延迟天数：根据DeliveryDelayDistribution配置的概率分布随机采样，用于模拟实际交付的不确定性 |
| `run_daily_physical_flow(deployment_plan, config, current_date, ...)` | `logistics_execution/delivery_processor.py` | 运行单日物理流：处理当日到期的部署计划，执行发车规则评估、装车优化、延迟采样，生成发运和预期到货记录 |

#### 库存管理函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `calculate_physical_inventory(orchestrator, current_date)` | `logistics_execution/inventory_manager.py` | 计算物理库存：从Orchestrator获取当前unrestricted库存和在途库存的汇总视图 |

#### 主入口函数

| 函数名 | 重构后位置 | 功能说明 |
|-------|-----------|---------|
| `run_physical_flow_module(config_dict, orchestrator, current_date, ...)` | `logistics_execution/__init__.py` | 运行物流执行模块：协调配置加载、物理流处理、Orchestrator状态更新和输出保存 |

---

## 📦 Orchestrator - 状态编排器

### 模块概述

Orchestrator是供应链仿真的中央状态管理器，负责：
- 库存状态维护（unrestricted inventory）
- 部署跟踪（open deployment）
- 在途管理（in transit）
- 生产/交付入库记录（production GR / delivery GR）
- 跨模块状态协调

### 文件对照

| 重构前 | 重构后 | 说明 |
|-------|-------|------|
| `code_vo/orchestrator.py` | `src/core/orchestrator.py` | 编排器核心 |

### 类与方法详细说明

#### DeploymentUID 类

| 方法名 | 功能说明 |
|-------|---------|
| `__init__(material, sending, receiving, planned_deploy_date, demand_element, sequence)` | 初始化部署唯一标识符，用于全程跟踪部署记录 |
| `to_string()` | 将UID转换为字符串表示：`material\|sending\|receiving\|date\|demand_element\|sequence` |
| `from_string(uid_str)` | 从字符串解析UID对象 |

#### Orchestrator 类 - 初始化与配置

| 方法名 | 功能说明 |
|-------|---------|
| `__init__(start_date, output_dir)` | 初始化编排器：设置仿真开始日期、输出目录，初始化所有状态字典和列表 |
| `initialize_inventory(initial_inventory_df)` | 从M1_InitialInventory配置初始化物理库存 |
| `set_space_capacity(space_capacity_df)` | 设置空间容量配置，用于收货空间限制 |

#### Orchestrator 类 - 库存管理

| 方法名 | 功能说明 |
|-------|---------|
| `get_unrestricted_inventory_view(date)` | 获取指定日期的unrestricted库存视图DataFrame |
| `get_current_unrestricted_inventory()` | 获取当前库存的字典形式：{(material, location): quantity} |
| `update_inventory(material, location, delta)` | 更新库存：增加或减少指定物料-地点的数量 |
| `save_beginning_inventory(date)` | 保存指定日期的期初库存快照 |
| `save_ending_inventory(date)` | 保存指定日期的期末库存快照 |

#### Orchestrator 类 - 部署管理

| 方法名 | 功能说明 |
|-------|---------|
| `add_open_deployment(deployment_record)` | 添加未完成部署记录，生成唯一UID |
| `get_open_deployment(date)` | 获取指定日期的未完成部署视图 |
| `get_open_deployment_view(date)` | 同上，返回DataFrame格式 |
| `process_module5_deployment(deployment_df, simulation_date)` | 处理Module5生成的部署计划：**注意**此函数包含关键的排序逻辑以确保UID生成稳定性 |
| `close_deployment(uid)` | 关闭已完成的部署记录 |

#### Orchestrator 类 - 在途管理

| 方法名 | 功能说明 |
|-------|---------|
| `add_in_transit(transit_record)` | 添加在途记录：物料已发出但未到达 |
| `get_in_transit_view(date)` | 获取指定日期的在途视图DataFrame |
| `get_planning_intransit_view(date)` | 获取用于计划的在途视图（包含预期到达信息） |
| `process_arrivals(date)` | 处理到达：将到期的在途记录转为库存 |

#### Orchestrator 类 - GR管理

| 方法名 | 功能说明 |
|-------|---------|
| `add_production_gr(gr_record)` | 添加生产入库记录：工厂生产完成入库 |
| `get_production_gr_view(date)` | 获取指定日期的生产GR视图 |
| `add_delivery_gr(gr_record)` | 添加交付入库记录：从上游到货 |
| `get_delivery_gr_view(date)` | 获取指定日期的交付GR视图 |

#### Orchestrator 类 - Module6集成

| 方法名 | 功能说明 |
|-------|---------|
| `process_module6_shipment(shipment_df, simulation_date)` | 处理Module6发运：从发送方扣减库存，创建在途记录，关闭对应的open deployment |
| `add_delivery_shipment_log(shipment_record)` | 添加发运出库日志 |
| `get_delivery_shipment_log(date)` | 获取指定日期的发运出库日志 |

#### Orchestrator 类 - 持久化

| 方法名 | 功能说明 |
|-------|---------|
| `save_state(date)` | 保存当前状态到JSON文件：包含库存、部署、在途等所有状态 |
| `load_state(date)` | 从JSON文件加载状态：用于断点续跑 |
| `save_daily_outputs(date)` | 保存每日输出到CSV文件：inventory、open_deployment、in_transit、production_gr、delivery_gr等 |

---

## 🔑 关键修复说明

### 修复1：Orchestrator 排序逻辑

**问题**: 重构初期 `src/core/orchestrator.py` 缺少 `process_module5_deployment()` 中的排序逻辑，导致 `ori_deployment_uid` 生成不一致。

**修复位置**: `src/core/orchestrator.py` 第688行左右

**添加代码**:
```python
# 排序以确保UID生成的稳定性 (与code_vo保持一致)
sort_cols = [
    col for col in ['material', 'sending', 'receiving', 'planned_deployment_date', 'demand_element', 'deployed_qty']
    if col in deployment_df.columns
]
if sort_cols:
    deployment_df = deployment_df.sort_values(by=sort_cols, kind='mergesort')
```

**原因**: UID的生成依赖于行的处理顺序，不排序会导致相同输入产生不同的UID序列。

---

## 📊 验证结果

已通过完整对比测试验证：

| 对比项 | 文件数 | 匹配率 |
|-------|-------|--------|
| orchestrator/ | 60 | 100% |
| module1/ | 5 | 100% |
| module3/ | 5 | 100% |
| module4/ | 7 | 100% |
| module5/ | 5 | 100% |
| module6/ | 10 | 100% |
| summary/ | 8 | 100% |
| **总计** | **100** | **100%** |

关键数值对比：
- `deployed_qty`: 完全一致
- `ending_inventory`: 完全一致  
- `in_transit`: 完全一致
- `production_gr`: 完全一致
- `delivery_gr`: 完全一致
- `order`: 完全一致

---

## 🏗️ 重构设计原则

1. **Facade模式**: 每个模块保留顶层facade (`module1.py`, `module3.py` 等)，内部拆分为子包
2. **单一职责**: 每个子模块文件负责单一功能域
3. **向后兼容**: 保持所有公开API和函数签名不变
4. **数据一致**: 所有数据处理逻辑完全保持一致
5. **配置不变**: 所有配置参数和阈值保持原样

---

## 📁 导入路径对照

### 使用示例

```python
# 重构前
from code_vo.module1 import run_daily_order_generation

# 重构后 (方式1 - 通过facade)
from src.modules.module1 import run_daily_order_generation

# 重构后 (方式2 - 直接导入)
from src.modules.demand_planning.integration import run_daily_order_generation
```

### 模块导入映射

| 功能 | 重构前 | 重构后 |
|-----|-------|-------|
| 订单生成 | `code_vo.module1` | `src.modules.module1` 或 `src.modules.demand_planning` |
| MRP计算 | `code_vo.module3` | `src.modules.module3` 或 `src.modules.mrp_planning` |
| 生产计划 | `code_vo.module4` | `src.modules.module4` 或 `src.modules.production_planning` |
| 部署计划 | `code_vo.module5` | `src.modules.module5` 或 `src.modules.deployment_planning` |
| 物流执行 | `code_vo.module6` | `src.modules.module6` 或 `src.modules.logistics_execution` |
| 编排器 | `code_vo.orchestrator` | `src.core.orchestrator` |
| 主集成 | `code_vo.main_integration` | `src.core.main_integration` |

---

## 📝 版本信息

- 文档创建日期: 2026-01-19
- 验证测试日期: 2026-01-19
- 测试配置: BC_S5.xlsx
- 测试日期范围: 2025-10-06 至 2025-10-10 (5天)

---

## 🔍 如何验证一致性

运行以下命令进行对比测试：

```powershell
# 1. 运行重构前代码
cd c:\Users\25936\Desktop\Code\chainsight\code_vo
python run.py --config path/to/config.xlsx --start-date 2025-10-06 --end-date 2025-10-10 --force-restart --non-interactive

# 2. 运行重构后代码
cd c:\Users\25936\Desktop\Code\chainsight
python -m src.core.main_integration --config path/to/config.xlsx --output outputs/test --start-date 2025-10-06 --end-date 2025-10-10 --force-restart

# 3. 运行对比脚本
python compare_all_outputs.py
```

对比脚本会自动比较所有输出文件并报告差异。
