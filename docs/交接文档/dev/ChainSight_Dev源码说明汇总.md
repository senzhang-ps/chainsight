# ChainSight_Dev 源码说明汇总

## 文档信息

| 项目 | 内容 |
|---|---|
| 文档名称 | ChainSight_Dev 源码说明汇总 |
| 文档定位 | Dev 版本源码统一说明文档 |
| 适用范围 | `ChainSight_Dev/` 主要 Python 脚本 |
| 目标读者 | 开发、测试、维护、交接人员 |
| 编写目的 | 将原有“一个脚本一个文档”的内容统一收敛为单一交接文档 |

## 1. 阅读说明

本文档用于统一说明 `ChainSight_Dev/` 目录下主要脚本的职责、输入输出、上下游依赖以及函数功能。建议阅读顺序为：入口脚本、总控脚本、状态中心、业务模块、支撑脚本、工具与测试脚本。对交接人员而言，只需保留本文件和必要的总览类文档即可满足日常阅读与排障需要。

## 2. 主运行链路说明

### 2.1 `run.py`

`run.py` 是 Dev 版本的命令行统一入口，负责解析启动参数、识别历史运行目录、决定输出目录，并最终调用主集成流程。它本身不承载业务算法，但它决定了整个仿真是新建运行、续跑已有运行，还是按指定目录恢复执行。

#### 输入输出与上下游依赖

| 项目 | 说明 |
|---|---|
| 输入文件 | 配置 Excel、历史 `outputs/<config>/run_*` 目录 |
| 输出文件 | 不直接产出业务结果，负责确定本次运行目录 |
| 上游依赖 | 用户 CLI 参数 |
| 下游依赖 | `main_integration.py::main()` / `run_integrated_simulation()` |
| 是否主链路必经 | 是 |

#### 核心函数

| 函数 | 功能 | 作用说明 |
|---|---|---|
| `_parse_args()` | 解析命令行参数 | 读取配置文件、日期范围、输出目录和续跑参数 |
| `_list_existing_runs()` | 扫描已有运行目录 | 列出某个配置下已经存在的 `run_*` 目录 |
| `_prompt_user_run_selection()` | 交互选择运行目录 | 当存在多个可续跑目录时提示用户选择 |
| `_ensure_output_dir()` | 创建或复用输出目录 | 决定本次运行写入新目录还是历史目录 |
| `get_or_init_simulation_start()` | 固化仿真起始日 | 保持续跑场景下日期口径一致 |
| `main()` | CLI 主入口 | 串联参数解析、输出目录决策和主流程调用 |

### 2.2 `main_integration.py`

`main_integration.py` 是 Dev 版本最核心的总控脚本，负责加载配置、恢复状态、设置随机种子、按日调度 M1/M4/M5/M6/M3，并在结束后执行库存平衡校验和总结报告生成。对接手人员而言，这是理解 Dev 版整体运行顺序的第一入口。

#### 输入输出与上下游依赖

| 项目 | 说明 |
|---|---|
| 输入文件 | 配置 Excel、历史 `orchestrator/*.csv`、历史 M4 输出 |
| 输出文件 | 各模块每日输出、`orchestrator/*.csv`、summary 报表 |
| 上游依赖 | `run.py` |
| 下游依赖 | `orchestrator.py`、M1/M4/M5/M6/M3、库存校验与汇总服务 |
| 是否主链路必经 | 是 |

#### 顶层函数

| 函数 | 功能 | 作用说明 |
|---|---|---|
| `detect_last_complete_date()` | 检测最后完整仿真日 | 扫描 Orchestrator 状态文件集合，判断是否可以续跑 |
| `restore_orchestrator_state()` | 恢复状态中心 | 从 CSV 恢复库存、开放调拨、在途、GR 和日志 |
| `check_resume_capability()` | 续跑能力检查 | 给出是否能续跑、从哪一天续跑的结构化结论 |
| `_normalize_location()` | 统一地点格式 | 规范地点编码，避免跨模块连接失败 |
| `_normalize_material()` | 统一物料格式 | 规范物料编码，避免 `.0` 和字符串混用 |
| `_normalize_sending()` | 统一发送地格式 | 保证 M5/M6/Orchestrator 对发送地识别一致 |
| `_normalize_receiving()` | 统一接收地格式 | 保证接收侧主键一致 |
| `_normalize_identifiers()` | DataFrame 标识标准化 | 批量标准化关键主键列 |
| `run_module4_integrated()` | 集成模式调用 M4 | 把主流程中的配置与净需求交给 M4 执行 |
| `load_current_date_production_gr()` | 读取当日生产入库 | 从 M4 输出中找出今天真正可入库的生产记录 |
| `load_module4_production_output()` | 读取 M4 输出文件 | 回读生产计划，用于兼容和调试 |
| `load_global_seed()` | 读取全局随机种子 | 统一仿真随机性来源 |
| `set_module_seeds()` | 设置模块随机种子 | 保证跨模块随机过程可复现 |
| `load_configuration()` | 加载全量配置 | 读取 Excel 所有关键配置表并做标准化处理 |
| `run_integrated_simulation()` | 执行完整仿真 | Dev 版本本地仿真总入口 |
| `main()` | 脚本入口 | 供命令行直接执行 |

#### 主运行顺序

1. 读取配置并做预处理。
2. 判断是否存在历史运行并尝试恢复状态。
3. 初始化随机种子和 Orchestrator。
4. 按日顺序执行 M1 -> M4 -> M5 -> M6 -> M3。
5. 每日结束后保存状态快照。
6. 运行完成后执行库存平衡检查和 summary 报表生成。

### 2.3 `orchestrator.py`

`orchestrator.py` 是 Dev 版本的状态中枢脚本，负责维护库存、开放调拨、在途、生产收货、交付收货、发货日志和库存变化日志。所有业务模块的关键执行结果最终都要回写到这里，因此它是理解模块间状态流转的核心文件。

#### 输入输出与上下游依赖

| 项目 | 说明 |
|---|---|
| 输入文件 | 初始库存配置、各模块回写结果、续跑恢复 CSV |
| 输出文件 | `unrestricted_inventory_*`、`open_deployment_*`、`planning_intransit_*`、`production_gr_*`、`delivery_gr_*`、`shipment_log_*` 等状态 CSV |
| 上游依赖 | `main_integration.py`、M1/M4/M5/M6 输出 |
| 下游依赖 | M1/M5/M6/M3、库存校验、summary、续跑恢复 |
| 是否主链路必经 | 是 |

#### 标识标准化函数

| 函数 | 功能 | 作用说明 |
|---|---|---|
| `_normalize_material()` | 统一物料编码 | 保证物料主键口径一致 |
| `_normalize_location()` | 统一地点编码 | 保证地点键一致 |
| `_normalize_sending()` | 统一发送地编码 | 用于调拨和物流主键规范化 |
| `_normalize_receiving()` | 统一接收地编码 | 用于接收节点主键规范化 |
| `_normalize_identifiers()` | 批量标准化 DataFrame 列 | 统一处理 `material/location/sending/receiving` |

#### `DeploymentUID`

| 方法 | 功能 | 作用说明 |
|---|---|---|
| `to_string()` | 生成 UID 字符串 | 将部署信息转成稳定唯一键 |
| `from_string()` | 解析 UID 字符串 | 把字符串 UID 恢复为结构化字段 |

#### `Orchestrator` 核心方法

| 方法 | 功能 | 作用说明 |
|---|---|---|
| `__init__()` | 初始化状态中心 | 建立库存、开放调拨、在途、GR、日志等内存对象 |
| `initialize_inventory()` | 初始化期初库存 | 从初始库存表建立库存账本 |
| `set_space_capacity()` | 设置空间配额 | 为 M5 提供接收侧空间上限依据 |
| `get_unrestricted_inventory_view()` | 输出库存视图 | 向模块或落盘过程提供库存 DataFrame |
| `get_current_unrestricted_inventory()` | 输出库存字典 | 供 M1/M5/M6 快速查库存 |
| `get_planning_intransit_view()` | 输出在途视图 | 供 M5、落盘和调试使用 |
| `get_open_deployment()` | 获取开放调拨 | 返回当前仍未完全执行的调拨记录 |
| `process_delivery_plan()` | 处理交付计划 | 将交付计划转换为状态更新 |
| `get_open_deployment_view()` | 输出开放调拨视图 | 供 M6 读取待发运调拨 |
| `get_space_quota_view()` | 输出空间配额视图 | 供 M5 做接收空间裁剪 |
| `get_production_plan_backlog_view()` | 输出生产 backlog | 提供未来生产供给视图 |
| `get_all_production_view()` | 输出生产视图 | 汇总生产相关记录 |
| `get_production_gr_view()` | 输出生产收货视图 | 提供生产入库记录 |
| `get_delivery_gr_view()` | 输出交付收货视图 | 提供物流到货入库记录 |
| `get_shipment_log_view()` | 输出客户发货视图 | 供 M6 和 summary 使用 |
| `get_delivery_shipment_log_view()` | 输出调拨发运视图 | 供后续核查和落盘 |
| `process_module1_shipments()` | 应用 M1 发货结果 | 扣减库存并写客户发货日志 |
| `process_module4_production()` | 应用 M4 生产结果 | 维护生产 backlog、production GR 和库存 |
| `process_module5_deployment()` | 应用 M5 调拨结果 | 将部署计划写入开放调拨池 |
| `process_module6_delivery()` | 应用 M6 物流结果 | 扣开放调拨、扣库存、生成在途或当天到货 |
| `run_daily_processing()` | 每日总处理入口 | 串联当天的状态变化处理 |
| `_process_delivery_arrivals()` | 到货入库内部函数 | 把今天到达的在途记录转为入库 |
| `_safe_convert_to_int()` | 稳健数值转换 | 避免脏数据导致状态更新失败 |
| `set_past_due_cleanup_grace_days()` | 设置清理宽限天数 | 控制超期开放调拨清理策略 |
| `cleanup_past_due_open_deployments()` | 清理超期调拨 | 删除过久未执行的开放调拨并输出审计 |
| `save_daily_state()` | 保存每日状态 | 写出全部 Orchestrator CSV 快照 |
| `_log_event()` | 写事件日志 | 记录关键状态事件 |
| `get_summary_statistics()` | 输出日统计 | 提供每日统计汇总 |
| `save_beginning_inventory()` | 保存期初库存 | 日初快照 |
| `save_ending_inventory()` | 保存期末库存 | 日末快照 |
| `get_beginning_inventory_view()` | 输出期初库存视图 | 供审计和比对使用 |
| `generate_inventory_change_log()` | 生成库存变化日志 | 用于库存守恒和审计对账 |
| `output_daily_inventory_summary()` | 输出库存摘要 | 生成简要库存总结 |
| `create_orchestrator()` | 工厂函数 | 按起始日创建 Orchestrator 实例 |

## 3. 业务模块说明

### 3.1 `module1.py`

`module1.py` 负责需求规划、订单生成、库存满足和供需日志输出，是 Dev 版本中最早参与每日业务计算的模块。它的输出直接影响 M4、M5、M6 和 M3 的后续输入。

#### 输入输出与上下游依赖

| 项目 | 说明 |
|---|---|
| 输入文件 | M1 配置 Sheet、历史 `module1_output_*.xlsx`、Orchestrator 库存视图 |
| 输出文件 | `module1_output_YYYYMMDD.xlsx` |
| 上游依赖 | `main_integration.py`、配置 Excel、Orchestrator |
| 下游依赖 | `orchestrator.py`、M3、M5、summary |
| 是否主链路必经 | 是 |

#### 函数说明

| 函数 | 功能 | 作用说明 |
|---|---|---|
| `_append_error_log()` | 记录错误日志 | 将模块运行中的错误附加到错误日志中 |
| `_normalize_location()` | 统一地点格式 | 避免地点编码不一致 |
| `_normalize_material()` | 统一物料格式 | 避免物料编码不一致 |
| `_normalize_identifiers()` | 批量标准化关键列 | 统一 DataFrame 主键列 |
| `load_config()` | 加载 M1 配置 | 读取 forecast、AO、订单日历、误差等配置 |
| `apply_dps()` | 执行 DPS 需求拆分 | 将需求按策略拆到不同地点 |
| `apply_supply_choice()` | 执行供给选择 | 调整不同 supply source 的需求归属 |
| `expand_forecast_to_days_integer_split()` | 周预测转日预测 | 将周级 forecast 分摊到日粒度 |
| `_consume_ao_for_ml_worker()` | AO 订单并行消耗 worker | 并行处理 AO 对 forecast 的消耗 |
| `_consume_normal_for_ml_worker()` | Normal 订单并行消耗 worker | 并行处理普通订单对 forecast 的消耗 |
| `generate_daily_orders()` | 生成当日订单 | 生成 AO 与普通订单，并准备后续发货输入 |
| `generate_quantity_with_percent_error()` | 生成带误差的订单量 | 模拟 forecast 偏差 |
| `consume_forecast_ao_logic()` | AO 消耗逻辑 | 按 AO 规则消耗 forecast |
| `consume_forecast_normal_logic()` | 普通订单消耗逻辑 | 按 normal 规则消耗 forecast |
| `simulate_shipment_for_single_day()` | 发货与 cut 模拟 | 按库存决定今天能发多少、缺多少 |
| `_load_previous_orders()` | 读取历史订单池 | 把前几天生成但未完成的订单带入今天 |
| `run_daily_order_generation()` | M1 每日主入口 | 串联配置、订单生成、消耗、发货和日志输出 |
| `generate_supply_demand_log_for_integration()` | 生成供需日志 | 为 M3 和 M5 提供下游输入 |
| `_apply_orders_consumption_to_forecast()` | 回写 forecast 消耗结果 | 把订单消耗后的 forecast 状态保存下来 |
| `save_module1_output_with_supply_demand()` | 保存 M1 输出 | 写出 OrderLog、ShipmentLog、CutLog、SupplyDemandLog、Summary |
| `_build_available_inventory_from_orchestrator()` | 从状态中心取库存 | 把 Orchestrator 库存转成 M1 可用格式 |
| `generate_shipment_with_inventory_check()` | 带库存校验的发货入口 | 在集成模式下结合库存视图生成发货 |

### 3.2 `module1_optimized.py`

`module1_optimized.py` 是 Dev 版本中针对 M1 的实验性优化脚本，不属于当前标准主运行路径，但保留了若干向量化或缓存化实现，可作为历史优化尝试的参考。

#### 输入输出与上下游依赖

| 项目 | 说明 |
|---|---|
| 输入文件 | M1 预测类输入 DataFrame |
| 输出文件 | 无固定标准输出文件 |
| 上游依赖 | Dev M1 优化试验 |
| 下游依赖 | 主要供对照和实验使用 |
| 是否主链路必经 | 否 |

#### 函数说明

| 函数 | 功能 | 作用说明 |
|---|---|---|
| `expand_forecast_to_days_integer_split()` | 周预测拆日预测 | 优化版日粒度 forecast 展开 |
| `generate_daily_orders()` | 生成当日订单 | 优化版订单生成逻辑 |
| `_normalize_location_cached()` | 带缓存的地点标准化 | 减少重复标准化开销 |
| `_normalize_identifiers()` | 标识符标准化 | 对输入 DataFrame 做批量统一处理 |

### 3.3 `module3.py`

`module3.py` 负责净需求与分层 MRP 计算，是 Dev 版本中用于回推网络缺口的核心脚本。它读取 M1 输出和 Orchestrator 状态，在层级网络中逐步传播需求缺口，为次日生产与调拨提供依据。

#### 输入输出与上下游依赖

| 项目 | 说明 |
|---|---|
| 输入文件 | M1 每日输出、M3 配置、Orchestrator 库存/在途/GR 视图 |
| 输出文件 | `Module3Output_YYYYMMDD.xlsx` |
| 上游依赖 | `main_integration.py`、M1、Orchestrator |
| 下游依赖 | 次日 M4、结果比对与汇总文档 |
| 是否主链路必经 | 是 |

#### 函数说明

| 函数 | 功能 | 作用说明 |
|---|---|---|
| `apply_moq_rv()` | 应用 MOQ/RV | 将需求量按最小起运量和取整规则调整 |
| `_lookup_moq_rv_three_keys()` | 查找 MOQ/RV 配置 | 按物料、发送地、接收地匹配部署规则 |
| `_apportion_largest_remainder()` | 最大余数分配 | 在整数约束下保持总量一致 |
| `_normalize_location()` | 统一地点编码 | 保证网络节点主键一致 |
| `_normalize_material()` | 统一物料编码 | 保证物料主键一致 |
| `_normalize_identifiers()` | 批量标准化关键列 | 统一输入 DataFrame 标识列 |
| `load_config()` | 加载 M3 配置 | 读取网络、LeadTime、安全库存和相关配置 |
| `load_module1_daily_outputs()` | 读取 M1 每日输出 | 取得订单、发货和供需日志 |
| `assign_location_layers()` | 计算网络层级 | 决定节点在 MRP 中的处理顺序 |
| `infer_sending_location_type()` | 推断发送地类型 | 区分 Plant、DC 等节点语义 |
| `_build_ptf_lsk_cache_m3()` | 构建 PTF/LSK 缓存 | 为 Plant 口径 horizon 计算提速 |
| `_get_ptf_lsk()` | 获取节点 PTF/LSK | 读取指定物料地点的计划约束 |
| `_compute_root_horizon()` | 计算根节点 horizon | 确定根层节点向前看的天数范围 |
| `determine_lead_time()` | 统一提前期口径 | 用于 gap 回推和供应窗口判断 |
| `calculate_daily_net_demand()` | 计算净需求 | 计算 AO、FC、SS 缺口 |
| `run_mrp_layered_simulation_daily()` | 每日 MRP 主逻辑 | 逐层处理网络并传播 gap |
| `load_excel_with_sheets()` | 兼容读取多 sheet Excel | 便于独立运行与调试 |
| `run_integrated_mode()` | 集成模式入口 | 供主流程调用执行每日 M3 |

### 3.4 `module4.py`

`module4.py` 负责生产计划、产能分配、换产处理和生产可靠率模拟，是 Dev 版本中承接净需求并把需求转成可执行生产的核心模块。

#### 输入输出与上下游依赖

| 项目 | 说明 |
|---|---|
| 输入文件 | 前一日 `Module3Output_*.xlsx`、M4 配置、历史 line state / capacity 状态文件 |
| 输出文件 | `Module4Output_YYYYMMDD.xlsx`、`line_state_*.json`、`allocated_capacity_*.json` |
| 上游依赖 | `main_integration.py`、M3、配置 Excel |
| 下游依赖 | `orchestrator.py`、M5、summary |
| 是否主链路必经 | 是 |

#### 函数说明

| 函数 | 功能 | 作用说明 |
|---|---|---|
| `get_or_init_simulation_start()` | 固化仿真起始日 | 保证审查日和跨天窗口口径一致 |
| `save_line_state()` | 保存产线状态 | 持久化日末产线状态 |
| `save_allocated_capacity()` | 保存已占产能 | 持久化各日产线已分配小时数 |
| `load_allocated_capacity()` | 加载指定日已占产能 | 支撑续排与容量累积 |
| `load_all_previous_capacity()` | 汇总历史产能占用 | 在新的仿真日扣减已使用产能 |
| `extract_allocated_capacity_from_plan()` | 从计划反推已占产能 | 把生产计划换算回小时占用 |
| `validate_capacity_allocation()` | 校验产能分配结果 | 检查是否超分配或冲突 |
| `load_line_state()` | 加载产线状态 | 为跨天换产延续提供输入 |
| `analyze_end_of_day_changeover_state()` | 分析日末换产状态 | 判断是否存在未完成换产 |
| `extract_line_states_from_plan()` | 从计划提取产线状态 | 为次日运行生成状态文件 |
| `_normalize_location()` | 统一地点编码 | 保证 merge 主键一致 |
| `_cast_identifiers_to_str()` | 标识列转字符串 | 避免类型不一致 |
| `_validate_merge_keys()` | 校验 merge 键 | 提前发现配置与结果键不匹配 |
| `load_daily_net_demand()` | 读取净需求 | 从 M3 输出中加载 layer=0 缺口 |
| `compute_planning_window()` | 计算计划窗口 | 决定本次排产可用日期区间 |
| `calculate_changeover_metrics()` | 汇总换产指标 | 生成换产时间、成本和损失日志 |
| `load_config()` | 加载 M4 配置 | 读取产线、产能、换产和可靠率配置 |
| `validate_config()` | 校验 M4 配置 | 检查核心配置完整性 |
| `is_review_day()` | 判断审查日 | 决定是否进入计划评审窗口 |
| `build_unconstrained_plan_for_single_day()` | 构建无约束生产计划 | 按净需求生成理想产量 |
| `optimal_changeover_sequence()` | 优化换产顺序 | 减少换产损耗 |
| `centralized_capacity_allocation_with_changeover()` | 集中产能分配 | 在容量和换产约束下生成可执行计划 |
| `simulate_production()` | 模拟生产可靠率 | 得到真实产出 `produced_qty` |
| `dedup_issues()` | 去重问题记录 | 整理校验输出 |
| `write_output()` | 写出 M4 输出 | 生成 ProductionPlan、CapacityExceed、Validation、ChangeoverLog |
| `run_daily_production_planning()` | M4 每日主入口 | 串联净需求读取、排产、模拟和输出 |
| `generate_consolidated_output()` | 汇总产出 | 对多个日结果做整理 |
| `main()` | 独立运行入口 | 支持脚本方式直接调用 |

### 3.5 `module5.py`

`module5.py` 负责部署规划，即在多层级供应网络中决定今天应当从哪里向哪里调拨多少货。它同时处理需求收集、库存投影、MOQ/RV、优先级分配、pipeline 覆盖、push/soft-push 和空间约束，是 Dev 版本中逻辑复杂度最高的模块之一。

#### 输入输出与上下游依赖

| 项目 | 说明 |
|---|---|
| 输入文件 | M1 输出、M4 输出、Orchestrator 状态视图、M5 配置 |
| 输出文件 | `Module5Output_YYYYMMDD.xlsx` |
| 上游依赖 | `main_integration.py`、M1、M4、Orchestrator |
| 下游依赖 | `orchestrator.py`、M6、M3、summary |
| 是否主链路必经 | 是 |

#### 函数说明

| 函数 | 功能 | 作用说明 |
|---|---|---|
| `_normalize_location()` | 统一地点格式 | 保持节点主键一致 |
| `_normalize_material()` | 统一物料格式 | 保持物料键一致 |
| `_normalize_identifiers()` | 批量标准化关键列 | 统一 DataFrame 关联主键 |
| `load_module1_daily_shipment()` | 读取 M1 发货结果 | 为部署规划提供客户发货参照 |
| `load_module1_daily_orders()` | 读取 M1 订单结果 | 为需求收集提供订单口径 |
| `load_orchestrator_delivery_gr()` | 读取交付收货记录 | 将已到货调拨纳入库存供给 |
| `load_orchestrator_open_deployment()` | 读取开放调拨记录 | 获取未执行完成的调拨池 |
| `build_open_deployment_inbound()` | 构造开放调拨入站供给 | 将开放调拨转换为接收端未来供给 |
| `calculate_projected_inventory()` | 计算预测库存 | 估计未来窗口内库存水平 |
| `calculate_available_inventory()` | 计算真实可用库存 | 判断今天真正可用于调拨的库存 |
| `get_upstream()` | 查找上游节点 | 为缺口回推与部署来源提供网络依据 |
| `apply_moq_rv()` | 应用 MOQ/RV | 对单条需求按最小起运量和取整规则调整 |
| `_lookup_moq_rv()` | 查询 MOQ/RV 配置 | 按物料、发送地、接收地匹配调拨规则 |
| `apply_grouped_moq_rv()` | 分组应用 MOQ/RV | 对同一路径需求统一做调整 |
| `apply_priority_allocation_vectorized()` | 按优先级分配库存 | 在库存不足时优先满足高优需求 |
| `_build_ptf_lsk_cache()` | 构建 PTF/LSK 缓存 | 提速审查窗口计算 |
| `_get_ptf_lsk()` | 获取 PTF/LSK | 为节点 horizon 计算提供参数 |
| `_build_lead_time_cache()` | 构建提前期缓存 | 提速路线 lead time 查询 |
| `determine_lead_time()` | 计算提前期 | 决定部署到货窗口 |
| `get_sending_location_type()` | 识别发送地类型 | 区分 Plant、DC 等发送端 |
| `assign_location_layers()` | 计算节点层级 | 为按层处理部署需求提供顺序 |
| `_build_active_network_cache()` | 构建活动网络缓存 | 快速判断当前日期有效网络关系 |
| `get_active_network()` | 获取有效网络 | 返回当前仿真日有效的网络路径 |
| `is_review_day()` | 判断审查日 | 决定某些 horizon 逻辑是否生效 |
| `compute_horizon()` | 计算 horizon | 决定向前看的时间范围 |
| `load_integrated_config()` | 集成模式加载配置 | 汇总 M1、M4、Orchestrator 与配置输入 |
| `load_config()` | 独立模式加载配置 | 直接从 Excel 读取部署规划输入 |
| `validate_config_before_run()` | 运行前校验 | 检查配置完整性并补充校验日志 |
| `collect_node_demands()` | 收集节点需求 | 汇总 SDL、订单、安全库存和上游 gap |
| `push_softpush_allocation()` | 执行 push / soft-push | 在基础部署之外补充主动补货 |
| `apply_receiving_space_quota()` | 应用接收空间约束 | 裁剪超过空间上限的部署量 |
| `log_outputs()` | 保存 M5 输出 | 写出 DeploymentPlan、UnfulfilledLog、StockOnHandLog、Validation |
| `main()` | M5 主入口 | 串联配置加载、需求收集、库存分配、部署生成和输出 |

### 3.6 `module6.py`

`module6.py` 负责物流执行，将 M5 生成的部署计划转换为真实发运、在途和到货结果。它同时处理装车、运输延迟、MDQ 规则、旁路规则和输出组织，是 Dev 版本中另一个高复杂度模块。

#### 输入输出与上下游依赖

| 项目 | 说明 |
|---|---|
| 输入文件 | OpenDeployment 视图、M6 配置、运力计划、MDQ 规则、Orchestrator 库存 |
| 输出文件 | `Module6Output_YYYYMMDD.xlsx` |
| 上游依赖 | `main_integration.py`、M5、Orchestrator |
| 下游依赖 | `orchestrator.py`、M3、summary |
| 是否主链路必经 | 是 |

#### 类与函数说明

| 函数/类 | 功能 | 作用说明 |
|---|---|---|
| `SafeExpressionEvaluator` | 安全求值器 | 用于执行旁路规则表达式 |
| `_check_and_deduplicate()` | 配置去重校验 | 检查关键主键列重复并输出验证日志 |
| `load_standalone_config()` | 独立模式加载配置 | 直接从 Excel 装载物流执行输入 |
| `load_integrated_config()` | 集成模式加载配置 | 汇总 OpenDeployment、运力、规则和元数据 |
| `_generate_validation_report()` | 生成验证报告 | 输出物流执行过程中的校验信息 |
| `should_bypass_mdq()` | 判断是否绕过 MDQ | 决定路线是否可跳过最小发运量约束 |
| `_normalize_capacity_plan()` | 归一化运力计划 | 将运力计划整理成按日可用格式 |
| `sample_delivery_delay()` | 采样运输延迟 | 按配置分布决定额外延迟天数 |
| `run_daily_physical_flow()` | 每日物流入口 | 供主流程按天调用 |
| `calculate_physical_inventory()` | 计算实物库存 | 从 Orchestrator 提取今天物流可见库存 |
| `run_physical_flow_module()` | M6 总入口 | 串联配置装配、装车、发运、日志和输出 |
| `main` | 兼容别名 | 与其他模块保持统一入口命名 |

## 4. 支撑脚本说明

### 4.1 `logger_config.py`

`logger_config.py` 提供 Dev 版本日志体系，包括双通道日志、`print` 重定向和简单文件日志器。

#### 输入输出与上下游依赖

| 项目 | 说明 |
|---|---|
| 输入文件 | 无 |
| 输出文件 | 日志文件、控制台日志 |
| 上游依赖 | 主流程与各模块初始化日志时调用 |
| 下游依赖 | 全部需要日志的脚本 |
| 是否主链路必经 | 是 |

#### 类与函数说明

| 函数/类 | 功能 | 作用说明 |
|---|---|---|
| `DualLogger` | 双通道日志器 | 创建文件和控制台双输出 logger |
| `PrintRedirector` | `print` 重定向器 | 将标准输出重定向到 logger |
| `setup_logging()` | 初始化日志体系 | 为主流程建立标准日志环境 |
| `create_simple_file_logger()` | 创建轻量日志器 | 用于工具脚本和简单测试 |

### 4.2 `time_manager.py`

`time_manager.py` 是 Dev 版本的统一时间管理脚本，用于控制当前仿真日、前后日期推导、文件日期格式转换和日期序列校验。

#### 输入输出与上下游依赖

| 项目 | 说明 |
|---|---|
| 输入文件 | 无 |
| 输出文件 | 无直接文件输出 |
| 上游依赖 | 主流程初始化仿真日期时调用 |
| 下游依赖 | M1/M4/M5/M6/M3 与输出命名逻辑 |
| 是否主链路必经 | 是 |

#### 核心内容

主要类为 `SimulationTimeManager`，负责当前日期获取、前后日期推导、日期序列生成、文件日期格式转换和日期顺序校验。顶层函数 `initialize_time_manager()`、`get_time_manager()` 和 `reset_time_manager()` 用于统一创建和管理全局时间上下文。

### 4.3 `validation_manager.py`

`validation_manager.py` 用于统一收集配置校验、运行校验和业务规则校验过程中产生的错误、警告和提示信息，是 Dev 版本验证链路的收口脚本。

#### 输入输出与上下游依赖

| 项目 | 说明 |
|---|---|
| 输入文件 | 各类待校验 DataFrame 与运行状态 |
| 输出文件 | 验证报告文本文件 |
| 上游依赖 | 配置校验、库存校验、模块运行校验 |
| 下游依赖 | 主流程错误决策、验证报告阅读 |
| 是否主链路必经 | 是 |

#### 核心内容

`ValidationManager` 负责统一收集错误、警告和信息，并支持日期转换、必需列校验、数值合法性校验和日期区间校验。辅助函数 `execute_with_validation()`、`is_critical_error()` 和 `get_default_result()` 用于简化验证调用逻辑。

### 4.4 `config_validator.py`

`config_validator.py` 负责在仿真开始前对全局配置和各业务模块配置进行预校验，是避免运行过程中因基础配置缺失或口径错误导致失败的重要前置脚本。

#### 输入输出与上下游依赖

| 项目 | 说明 |
|---|---|
| 输入文件 | 配置 Excel 和其对应 DataFrame |
| 输出文件 | 配置验证报告 |
| 上游依赖 | `main_integration.py` |
| 下游依赖 | `ValidationManager`、主流程启动决策 |
| 是否主链路必经 | 是 |

#### 核心内容

`ConfigValidator` 统一调度全局配置、M1/M3/M4/M5/M6 配置和跨模块一致性检查。顶层入口 `run_pre_simulation_validation()` 供主流程在正式启动前调用。

### 4.5 `inventory_balance_checker.py`

`inventory_balance_checker.py` 用于验证库存守恒关系，即期初库存加各类入库减各类出库是否等于期末库存，是 Dev 版本运行完成后的重要一致性校验工具。

#### 输入输出与上下游依赖

| 项目 | 说明 |
|---|---|
| 输入文件 | Orchestrator 状态 CSV、M6 输出、历史库存记录 |
| 输出文件 | 库存平衡检查结果与差异输出 |
| 上游依赖 | `main_integration.py` 运行完成后调用 |
| 下游依赖 | 交付验证、测试报告、问题排查 |
| 是否主链路必经 | 是 |

#### 核心内容

该脚本包含库存标准化函数及 `InventoryBalanceChecker` 类，支持单日检查、区间检查、负库存检查、明细差异输出与汇总报告生成。

### 4.6 `summary_report_generator.py`

`summary_report_generator.py` 用于在仿真完成后，对每日模块输出进行跨日期汇总，生成面向交付和分析使用的 summary 报表。

#### 输入输出与上下游依赖

| 项目 | 说明 |
|---|---|
| 输入文件 | M1/M4/M5/M6 每日输出、Orchestrator 状态文件 |
| 输出文件 | summary 目录下的各类汇总报表 |
| 上游依赖 | `main_integration.py` 收尾阶段 |
| 下游依赖 | 测试报告、交付分析、结果对比脚本 |
| 是否主链路必经 | 是 |

#### 核心内容

`SummaryReportGenerator` 提供订单发货、交付、卡车使用、超产能、换产、部署、生产和历史库存等多种汇总生成逻辑。

### 4.7 `performance_profiler.py`

`performance_profiler.py` 用于对模块或函数进行性能剖析，帮助识别热点函数和耗时瓶颈。

#### 输入输出与上下游依赖

| 项目 | 说明 |
|---|---|
| 输入文件 | 无固定输入文件 |
| 输出文件 | profile 文本报告 |
| 上游依赖 | 性能分析场景手动调用 |
| 下游依赖 | 优化诊断与性能比对 |
| 是否主链路必经 | 否 |

#### 核心内容

该脚本提供 `PerformanceProfiler` 上下文管理器和 `profile_function()` 装饰器两种方式，用于分析模块或函数的耗时。

## 5. 工具与测试脚本说明

### 5.1 `order_log_generator.py`

独立订单日志生成工具，用于在不执行完整仿真的情况下按日期区间单独生成订单日志结果。

### 5.2 `e2e_integration_test.py`

端到端集成测试脚本，用于构造测试配置、执行完整仿真并校验业务结果与状态一致性。

### 5.3 `test_logger.py`

日志模块测试脚本，用于验证 `logger_config.py` 中日志组件是否工作正常。

### 5.4 `wip_cov_generator.py`

独立 WIP CoV 计算脚本，用于从 forecast、BOM 和 CoV 配置中计算半成品需求波动系数。

### 5.5 `create_production_config.py`

用于构造生产配置样例的辅助脚本，主要服务于测试和示例场景。

### 5.6 `diagnose.py`

性能诊断脚本，用于比较不同数据处理方式对 M3 相关操作性能的影响。

### 5.7 `compare_runs.py`

运行结果对比脚本，用于比较两次历史运行在库存、订单发货、部署、生产和交付汇总上的差异。该脚本没有稳定的公共函数接口，更适合作为临时回归分析工具。

## 6. 维护建议

1. 若后续 Dev 版本脚本结构继续调整，应优先维护本汇总文档，再决定是否拆回单脚本文档。
2. 若某个脚本的公开函数、关键内部函数、输入输出文件或上下游依赖发生变化，应同步更新对应章节。
3. 若后续还需与重构版逐一对照，建议以本汇总文档为 Dev 基线，再在 `Refactored` 目录中建立同结构总文档。
