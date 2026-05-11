# Project Plan Implementation Details

更新时间：2026-05-08

本文是 `project_tracking_plan.xlsx` 中项目任务的实施细则，覆盖 P0 到 FINAL 的代码修改范围、实施步骤和验收口径。`INDEX.md` 不维护本文索引。

本文当前覆盖：

- P0：基线、风险、计划
- MLE-1：M1 订单生成逻辑改进
- MLE-2：M4 多产线排程
- MLE-3：filtered DataFrame 优化
- VAL-1：输入数据校验
- MLE-4：避免逐日 append
- VAL-2：中间数据校验
- TEST-1：E2E 网络构建
- DOC-1 / DOC-2 / DOC-3 / DOC-4：项目定义与分析文档
- FINAL：最终验收

## 1. 目标

M1 订单生成逻辑已在 `src/modules/demand_planning_refactor/` 中按“周级订单优先”口径落地：先用周度 forecast 和 forecast error(CoV) 生成周级订单总量，再用 AO 配置拆出 AO / normal 结构，最后按 `M1_OrderCalendar` 的有效下单日拆到每日订单。`OrderLog`、`ShipmentLog`、`CutLog`、`SupplyDemandLog` 和 `Summary` 的生产输出 schema 保持兼容。

当前文档口径：

- `src/modules/demand_planning/` 是旧版日度优先基线。
- `src/modules/demand_planning_refactor/` 是重构后代码事实。
- 本节 MLE-1 记录重构后实现边界和仍需回归的验收点，不再作为待实现代码清单。

## 2. 当前实现判断

重构版主入口仍是：

```text
src/modules/demand_planning_refactor/integration.py::run_daily_order_generation()
```

当前主链路：

1. `_validate_config()` 读取 `M1_DemandForecast`、`M1_ForecastError`、`M1_OrderCalendar`、`M1_AOConfig`，转换 `OrderCalendar.date`，将负需求裁零，并标准化标识符。
2. `_prepare_forecasts()` 生成三类基线：`weekly_for_orders`、`daily_for_consumption`、`daily_for_supply`。
3. `generate_daily_orders()` 在输入是周级 forecast 时进入 `_generate_daily_orders_from_weekly()`。
4. `_generate_daily_orders_from_weekly()` 调用 `generate_weekly_orders()` 生成周级 AO / normal 订单，再调用 `split_weekly_orders_to_daily()` 按有效下单日返回当前 `simulation_date` 对应订单。
5. `_merge_with_history()` 合并历史 `OrderLog` 和当日新增订单，保留 `date >= simulation_date` 的未到期订单。
6. `_generate_shipments()` 基于累计订单池和 Orchestrator 可用库存生成 `ShipmentLog` / `CutLog`。
7. `_apply_orders_consumption()` 用 `today_orders_df` 消耗 `daily_for_supply`，`generate_supply_demand_log_for_integration()` 输出未来窗口 `SupplyDemandLog`。
8. `_save_output()` 写出标准 `module1_output_YYYYMMDD.xlsx`。

如果输入 forecast 已经是日级结构，`generate_daily_orders()` 仍保留旧的兼容路径：先判断当天是否在 `M1_OrderCalendar` 中，再按日度 forecast 窗口生成 AO / normal。

## 3. 已落地的代码位置

| 文件 | 函数 | 当前职责 |
|---|---|---|
| `src/modules/demand_planning_refactor/integration.py` | `run_daily_order_generation()` | 串联配置校验、forecast 准备、订单生成、历史合并、发货、供需日志、输出 |
| `src/modules/demand_planning_refactor/integration.py` | `_prepare_forecasts()` | 返回 `weekly_for_orders`、`daily_for_consumption`、`daily_for_supply` 三类基线 |
| `src/modules/demand_planning_refactor/order.py` | `generate_daily_orders()` | 保持外部入口，按输入结构在周级路径和日级兼容路径之间分发 |
| `src/modules/demand_planning_refactor/order.py` | `generate_weekly_orders()` | 以周度 forecast、AO 配置和 forecast error(CoV) 生成守恒的周级 AO / normal 订单数量 |
| `src/modules/demand_planning_refactor/order.py` | `split_weekly_orders_to_daily()` | 按本周 `M1_OrderCalendar` 有效下单日拆分周订单，并只返回当前 `simulation_date` 的订单 |
| `src/modules/demand_planning_refactor/order.py` | `_split_integer_quantity_by_days()` | 整数拆分：`base = qty // days`，`remainder = qty % days`，余数给前 N 个有效下单日 |
| `src/modules/demand_planning_refactor/consume.py` | `consume_orders()` | 调度订单对 forecast 的消耗，默认走优化路径 |
| `src/modules/demand_planning_refactor/integration.py` | `_apply_orders_consumption()` | 生成供需日志前，对 `daily_for_supply` 应用当日订单消耗 |

## 4. 周级订单内部函数

| 函数 | 职责 |
|---|---|
| `_build_weekly_ml_demand(...)` | 按 `material/location/week` 聚合周需求，并稳定排序 |
| `_split_weekly_demand_by_ao_means(...)` | 按 `AOConfig.ao_percent` 和 `advance_days` 拆出 AO / normal 周级均值 |
| `_generate_weekly_total_orders(...)` | 依据 total/weekly CoV 生成 material/location/week 周总订单量 |
| `_generate_weekly_component_orders(...)` | 依据 AO / normal CoV 生成临时周级结构 |
| `_reconcile_weekly_component_orders_to_total(...)` | 将临时 AO / normal 结构按权重归一回周总量 |
| `_stable_normal_quantities(...)` / `_stable_seed(...)` | 使用 `M1_RandomSeed` 和业务 key 生成稳定随机数量 |
| `_build_week_start_map(...)` | 优先从 `reference_daily_forecast` 的 `week/date` 推导周起点，缺失时用日历日期兜底 |

## 5. 不修改的输出契约

除非有明确业务要求，不要改变下列输出列：

### OrderLog

```text
date, material, location, demand_type, quantity, simulation_date, advance_days
```

### ShipmentLog

```text
date, material, location, quantity, demand_type, order_id
```

### SupplyDemandLog

```text
date, material, location, quantity, demand_element
```

`source_week`、`weekly_quantity`、`split_rule`、`order_date` 等调试字段当前不默认写入生产 `OrderLog`。下单日通过 `simulation_date` 表达，需求日期通过 `date` 表达。

## 6. 仍需回归的边界

| 文件 | 回归原因 |
|---|---|
| `src/modules/demand_planning_refactor/integration.py::_apply_orders_consumption()` | 当日订单只消耗 `daily_for_supply`，需要确认 AO / normal 消耗顺序和窗口偏移符合预期 |
| `src/modules/demand_planning_refactor/io_utils.py::save_module1_output_with_supply_demand()` | 输出 sheet 名和列名必须保持兼容 |
| `src/modules/deployment_planning/data_loader.py` | M5 读取 M1 `OrderLog` |
| `src/modules/deployment_planning/demand_collector.py` | M5 消费 M1 订单需求 |
| `src/modules/mrp_planning/config_loader.py` | M3 读取 M1 输出 |
| `src/services/summary_report_generator.py` | 汇总报表读取 M1 `OrderLog` / `ShipmentLog` / `CutLog` |

## 7. 当前回归步骤

1. 代码级单元验证
   - `generate_weekly_orders()`：周总量、AO / normal 结构、CoV 兜底、稳定随机。
   - `split_weekly_orders_to_daily()`：只落在有效下单日、整数拆分守恒、余数稳定。
   - `generate_daily_orders()`：周级路径和日级兼容路径都返回 `(orders_df, consumed_forecast)`。

2. 集成验证
   - `run_daily_order_generation()` 返回 `orders_df`、`shipment_df`、`cut_df`、`supply_demand_df`、`summary_df`、`output_file`、`all_orders_for_next_day`。
   - 文件模式下 `OrderLog` 写累计订单池。
   - DB 模式传入 `previous_orders_df` 时跳过历史文件读取。

3. 输出兼容验证
   - `OrderLog.date` 仍是需求日期，`simulation_date` 是下单日。
   - AO: `date = simulation_date + advance_days`。
   - normal: `date = simulation_date`，`advance_days = 0`。
   - 下游不需要理解周级中间表。

4. 性能和结果验证
   - 对比旧版和新版关键 KPI，而不是逐行强制一致。
   - 记录 M1 单日平均耗时和功能测试耗时。
   - 若主仿真入口尚未切换到 `demand_planning_refactor`，需单独记录切换点和回退方案。

## 8. 验收清单

| 项目 | 标准 |
|---|---|
| 数量守恒 | 每个 material / location / week 的周订单量等于拆分后日订单量总和 |
| AO 逻辑 | AO 百分比、`advance_days`、误差配置生效 |
| normal 逻辑 | normal = total demand - AO demand，且非负 |
| 随机稳定性 | 相同 seed、相同输入、相同排序下输出一致 |
| 输出兼容 | M1 Excel sheet 和 DB 写入列不变 |
| 下游兼容 | M3 / M5 / summary report 可读取并正常运行 |
| 性能 | M1 优化后不慢于基线，或有明确性能说明和回退开关 |

## 9. 风险与决策点

| 风险 | 需要确认的决策 |
|---|---|
| 主仿真入口切换 | 当前重构模块与旧模块并行存在，主入口若切换到 refactor 需要单独回归 |
| 周起点口径漂移 | `split_weekly_orders_to_daily()` 优先用 `reference_daily_forecast.week/date` 推导周起点，缺失时用日历兜底 |
| 输出差异不可避免 | 新旧算法口径不同，不能只用逐行 diff 判断对错，需要同时看业务 KPI |
| schema 扩展影响大 | 调试字段默认不要写入生产 `OrderLog` |

## 10. 当前改动边界

当前保持：

- `run_daily_order_generation()` 函数签名不变。
- `generate_daily_orders()` 对外返回结构不变。
- M1 输出 sheet 名称和列名不变。
- M3 / M5 / 报表读取逻辑不改。

重构实现集中在 `src/modules/demand_planning_refactor/` 内，旧 `src/modules/demand_planning/` 作为对照基线保留。

## 11. MLE-2：M4 多产线排程改进计划

### 11.1 目标

将 M4 生产排程逻辑从“一个 SKU / 地点默认落到单条产线”调整为“一个 SKU / 地点可映射多条候选产线，并在产能、生产速率、优先级、换产和可靠性约束下分配生产量”。

核心验收口径：

- 一个 material / location 可以配置多条候选产线。
- 候选产线展开后不得重复放大净需求。
- 每条产线每天分配的生产小时不得超过可用产能。
- `prd_rate`、`priority`、`LineCapacity`、`ChangeoverMatrix`、`ChangeoverDefinition`、`ProductionReliability` 均正确生效。
- 输出 `ProductionPlan`、`ExceedLog`、`ChangeoverLog`、产线状态和已分配产能结构保持兼容。
- 文件模式和 DB 内存模式结果语义一致。

### 11.2 当前实现判断

当前 M4 主链路：

1. `src/modules/production_planning/integration.py::run_daily_production_planning_integrated()` 是模块公开入口。
2. 实际集成逻辑委托给 `src/core/main_integration/production_runner.py::run_daily_production_planning_integrated()`。
3. `src/modules/production_planning/plan_builder.py::build_unconstrained_plan_for_single_day()` 构建无约束计划。
4. `src/modules/production_planning/capacity_allocator.py::centralized_capacity_allocation_with_changeover()` 分配产能和换产。
5. `simulate_production()` 应用生产可靠性。
6. `output_writer.py` 写出 M4 输出。

需要重点注意两处风险：

- `plan_builder.py` 当前按 `MaterialLocationLineCfg` 每一行循环；如果同一个 material / location 配多条产线，可能重复处理同一份净需求。
- `capacity_allocator.py::_allocate_batch()` 当前通过 `self.mlcfg[(material/location)].iloc[0]` 取配置；多产线场景下会错误地使用第一条配置，不能可靠区分 line-specific 的 `lsk`、`ptf`、`MCT`、`prd_rate` 等字段。

### 11.3 必须修改的代码位置

| 文件 | 函数 | 修改目的 |
|---|---|---|
| `src/core/main_integration/production_runner.py` | `run_daily_production_planning_integrated()` | 保持集成入口签名不变，但确认 M4 配置标准化、内存模式净需求过滤、多产线结果返回结构正确 |
| `src/modules/production_planning/plan_builder.py` | `build_unconstrained_plan_for_single_day()` | 先按 material / location / requirement_date 聚合净需求，再展开 SKU-line 候选，避免多产线配置导致需求重复放大 |
| `src/modules/production_planning/plan_builder.py` | `_build_plan_for_material()` | 改为处理 material / location 粒度需求，而不是直接按配置行循环 |
| `src/modules/production_planning/plan_builder.py` | `_merge_with_config()` | 明确生成 candidate line table，保留 `delegate_line`、`prd_rate`、`priority`、`lsk`、`ptf`、`MCT` |
| `src/modules/production_planning/plan_builder.py` | `_create_plan_record()` | 输出每条候选产线的无约束计划，字段中必须包含 `line` 或标准化后的 `delegate_line` |
| `src/modules/production_planning/capacity_allocator.py` | `CapacityAllocator.__init__()` | 建立 line-aware 的配置索引，例如 `(material, location, line)`，替代仅 `(material, location)` 的 `mct_map` |
| `src/modules/production_planning/capacity_allocator.py` | `CapacityAllocator.allocate()` | 按 line / simulation_date 分组前确认候选计划已去重且排序稳定 |
| `src/modules/production_planning/capacity_allocator.py` | `_allocate_batch()` | 不再用 `material/location` 取 `.iloc[0]`；改为按 `material/location/line` 取当前批次对应配置 |
| `src/modules/production_planning/capacity_allocator.py` | `_allocate_day()` | 确认 `rate_map[(material, line)]`、location-aware capacity key、历史已分配产能 key 一致 |
| `src/modules/production_planning/capacity_allocator.py` | `validate_capacity_allocation()` | 校验多产线场景下每日每线总生产小时、换产小时和历史占用小时不超产能 |
| `src/modules/production_planning/capacity_allocator.py` | `simulate_production()` | 确认生产可靠性按 material / line 或当前配置口径应用 |
| `src/modules/production_planning/config_loader.py` | `validate_config()` | 增加多产线配置校验：重复候选、缺失产能、缺失速率、缺失换产定义、priority 非法 |
| `src/modules/production_planning/output_writer.py` | `write_output()` | 输出列保持兼容，必要时只增加可选 debug sheet，不破坏生产输出 |

### 11.4 建议新增的内部函数

| 新函数 | 建议位置 | 职责 |
|---|---|---|
| `build_sku_line_candidates(...)` | `plan_builder.py` | 将 material / location 净需求展开为 SKU-line 候选表 |
| `deduplicate_net_demand_for_candidates(...)` | `plan_builder.py` | 防止多条产线配置重复放大同一净需求 |
| `rank_candidate_lines(...)` | `plan_builder.py` | 按 priority、prd_rate、capacity 可用性等规则排序候选产线 |
| `build_line_config_index(...)` | `capacity_allocator.py` | 建立 `(material, location, line)` 配置索引 |
| `get_batch_line_config(...)` | `capacity_allocator.py` | 为单个批次取 line-specific 配置，替代 `.iloc[0]` |
| `validate_multi_line_candidates(...)` | `config_loader.py` | 在配置加载阶段暴露多产线配置问题 |

### 11.5 实施步骤

1. 建立 M4 基线
   - 跑当前样例配置，保存 `ProductionPlan`、`ExceedLog`、`ChangeoverLog`、line state、allocated capacity。
   - 记录 M4 平均单日耗时。

2. 明确配置语义
   - 确认 `M4_MaterialLocationLineCfg.delegate_line` 是否就是候选产线。
   - 确认多条产线的 `priority` 规则：优先级越小越优先，还是越大越优先。
   - 确认产能不足时是否允许跨线拆量，还是按优先级填满一条再到下一条。

3. 改造无约束计划
   - 先对净需求去重和聚合。
   - 再生成 SKU-line candidate table。
   - 不允许候选产线数量改变原始净需求总量，除非明确做拆分分配。

4. 改造产能分配
   - 按 line-aware config 获取 `lsk`、`ptf`、`MCT`、`prd_rate`。
   - 确保换产序列在每条产线内部独立计算。
   - 确保 location-aware capacity 和历史 allocated capacity 使用同一 key 口径。

5. 改造校验
   - 增加每日 line capacity 校验。
   - 增加 multi-line demand conservation 校验。
   - 增加缺失 changeover / rate / capacity 的配置校验。

6. 回归
   - 单产线配置下结果应与旧版保持一致或差异可解释。
   - 多产线配置下检查产量分配、换产、超额日志和生产可靠性。
   - 文件模式和 DB 模式都要跑。

### 11.6 验收清单

| 项目 | 标准 |
|---|---|
| 候选产线 | 一个 material / location 可展开为多条候选 line |
| 需求守恒 | 多产线展开后，总 unconstrained demand 不被重复放大 |
| 产能约束 | 每条 line 每天生产小时 + 换产小时 + 历史占用小时不超过 capacity |
| 换产逻辑 | 每条 line 内部换产顺序和跨天未完成换产状态正确 |
| 可靠性 | `ProductionReliability` 对最终计划生效 |
| 输出兼容 | `ProductionPlan` 等核心输出列不破坏下游读取 |
| 回归 | 单产线场景不回退，多产线场景通过边界 case |

## 12. MLE-3：filtered DataFrame 优化计划

### 12.1 目标

系统性清理 filtered DataFrame 再访问、链式索引和视图/副本不确定行为，降低 pandas `SettingWithCopy` 风险，并减少重复过滤带来的性能开销。

核心验收口径：

- 不再出现高风险链式索引写法，例如 `df[mask]['col'] = value`。
- 对需要修改的筛选结果统一使用 `.loc[...]` 或显式 `.copy()`。
- 热点路径避免在循环里重复做大 DataFrame mask。
- 输出结果与改造前保持一致，除非原逻辑存在明确 bug。

### 12.2 必须审计的代码范围

| 范围 | 重点文件 | 原因 |
|---|---|---|
| M1 | `src/modules/demand_planning_refactor/integration.py`、`order.py`、`consume.py`、`consume_optimized.py` | 订单生成、周级拆日、消耗和 SupplyDemandLog 使用大量过滤和局部更新 |
| M3 | `src/modules/mrp_planning/mrp_simulation.py`、`node_processor.py`、`config_loader.py` | 网络节点筛选、层级处理和 `.iloc[0]` 读取较多 |
| M4 | `src/modules/production_planning/plan_builder.py`、`capacity_allocator.py`、`demand_loader.py` | M4 多产线改造会增加候选表过滤和配置查找 |
| M5 | `src/modules/deployment_planning/main.py`、`demand_collector.py`、`demand_collector_vectorized.py`、`allocation.py`、`data_loader.py` | 部署计划和需求收集存在多处 filtered df、循环 append 和局部写回 |
| Core | `src/core/main_integration/simulation_file.py`、`simulation_db.py`、`production_runner.py` | 集成层对各模块输出做按日期过滤和复制 |

### 12.3 必须修改的代码位置

| 文件 | 位置 | 修改目的 |
|---|---|---|
| `src/modules/production_planning/plan_builder.py` | `_get_material_demands()` | 当前 `net_demand_df[mask].copy()` 可保留，但应避免在多产线循环中重复过滤；改为预索引或 groupby |
| `src/modules/production_planning/plan_builder.py` | `_merge_with_config()` | `cfg_slice = mlcfg[...]` 应明确 `.copy()`，并在多产线候选展开时避免重复 merge |
| `src/modules/production_planning/plan_builder.py` | `_filter_by_date()` | 对传入子表修改 `requirement_date` 前应保证是独立副本，或用 `.loc[:, 'requirement_date']` |
| `src/modules/production_planning/capacity_allocator.py` | `_allocate_batch()` | 避免 `self.mlcfg[(mask)].iloc[0]` 反复扫描；改用预建 config index |
| `src/modules/production_planning/capacity_allocator.py` | `_lookup_changeover()` | 对 MultiIndex `.loc` 返回 Series / scalar 的分支保留，但需加唯一性校验 |
| `src/core/main_integration/production_runner.py` | `run_daily_production_planning_integrated()` | 日期过滤后用于后续修改的 DataFrame 必须 `.copy()`；核心路径避免重复 normalize |
| `src/modules/deployment_planning/main.py` | 需求收集和分配主循环 | 将循环内 DataFrame filter 改为预索引、字典或批量 merge |
| `src/modules/deployment_planning/demand_collector.py` | `_collect_order_demands()` 等 | 已支持 `order_index`，需扩大到其他需求源，减少重复筛选 |
| `src/modules/mrp_planning/mrp_simulation.py` | 网络候选查找 | 将 `active_network[(mask)].iloc[0]` 改为预构建索引查询 |

### 12.4 统一编码规则

| 场景 | 推荐写法 | 禁止或慎用写法 |
|---|---|---|
| 筛选后要修改 | `subset = df.loc[mask, cols].copy()` | `subset = df[mask]` 后直接赋值 |
| 原表局部写回 | `df.loc[idx, 'col'] = value` | `df[mask]['col'] = value` |
| 循环内查找 | 预建 dict / MultiIndex / groupby map | 每次循环重新 `df[(a) & (b)]` |
| 取唯一配置 | 先校验唯一性，再 `.iloc[0]` | 不校验直接 `.iloc[0]` |
| 合并配置 | `merge(validate='m:1')` 或先去重 | 无校验 merge 导致行数膨胀 |

### 12.5 实施步骤

1. 审计
   - 使用 `rg "\\]\\[|\\.loc\\[|\\.iloc\\[|copy\\(|append\\(|concat\\(" src/modules src/core -g "*.py"` 建立清单。
   - 对每个命中点标记为：安全、需 `.copy()`、需 `.loc`、需预索引、需去重校验。

2. 先改热点模块
   - 优先处理 M4，因为 MLE-2 多产线会放大 filtered DataFrame 风险。
   - 其次处理 M1/M5，因为数据量和调用频率高。

3. 建立索引
   - 对配置表建立 `(material, location)`、`(material, location, line)`、`(line, date)` 等索引。
   - 对每日动态表按日期、物料、地点建立 group map。

4. 替换写法
   - 链式赋值全部改成 `.loc`。
   - 需要修改筛选结果时全部显式 `.copy()`。
   - 循环内重复 filter 改成索引查询。

5. 回归
   - 对比关键输出表行数、主键、数量字段和日期字段。
   - 对比 M1/M4/M5 平均耗时。
   - 保留差异说明。

### 12.6 验收清单

| 项目 | 标准 |
|---|---|
| 链式赋值 | 无 `df[mask]['col'] = ...` 类写法 |
| 视图/副本 | 需要修改的筛选表均显式 `.copy()` 或 `.loc` 写回 |
| 行数稳定 | merge 前后行数变化符合预期，有 `validate` 或唯一性校验 |
| 热点性能 | 循环内重复过滤被预索引替代 |
| 输出一致 | M1/M3/M4/M5 关键输出与改造前一致或差异可解释 |
| 可维护性 | 新增索引 helper 有清晰命名，避免分散手写 mask |

## 13. P0：基线、风险、计划实施细则

### 13.1 目标

在正式改造前冻结可复现的代码、配置、输出和性能基线，形成后续所有代码变更的对照标准。

### 13.2 代码和工具修改位置

| 文件或目录 | 修改目的 |
|---|---|
| `run.py` / `run.ps1` | 确认本地运行入口可复现，必要时补充固定参数示例 |
| `src/core/run/` | 检查 CLI 分发、输出目录、local / db runner 的运行参数是否能稳定复现 |
| `src/core/main_integration/simulation_file.py` | 文件模式基线运行入口，记录 validation、summary、balance report 产物 |
| `src/core/main_integration/simulation_db.py` | DB 模式基线运行入口，记录 checkpoint、summary、module output 表口径 |
| `src/services/performance_profiler.py` | 如现有能力不足，补充 M1/M4/M5/M6/M3 单模块耗时采集 |
| `src/utils/config_validator.py` | 预校验作为基线的一部分，记录当前配置脏数据和风险 |
| `docs/project_tracking_plan.xlsx` | 维护计划、工时、状态和甘特图，不作为代码行为依据 |
| `docs/project_execution_plan.md` | 高层计划说明，不作为运行入口 |

### 13.3 工作细分

| 子任务 | 输出 |
|---|---|
| 冻结代码基线 | git commit hash、运行命令、配置文件名 |
| 跑文件模式基线 | 模块输出、summary、inventory balance report |
| 跑 DB 模式基线 | module output 表、summary output 表、checkpoint 状态 |
| 建立性能基线 | M1/M4/M5/M6/M3 每日平均耗时和总耗时 |
| 风险审计 | 长函数、重复 DataFrame filter、循环 append、schema 风险清单 |

### 13.4 验收标准

- 同一配置、同一日期范围可重复运行。
- 文件模式和 DB 模式核心 KPI 口径可解释。
- 每个后续任务都有可对照的输出和性能基线。

## 14. VAL-1：输入数据校验实施细则

### 14.1 目标

把用户输入配置的 schema、类型、必填字段、范围和跨表一致性校验前置到仿真运行前，减少运行中失败和隐式脏数据传播。

### 14.2 必须修改的代码位置

| 文件 | 修改目的 |
|---|---|
| `src/utils/config_validator.py` | 主校验入口，增加 M1/M3/M4/M5/M6 schema、类型、范围和跨模块一致性规则 |
| `src/utils/validation_manager.py` | 统一错误、警告、信息收集和报告格式 |
| `src/core/main_integration/simulation_file.py` | 文件模式运行前调用预校验，失败时阻断或显式跳过 |
| `src/core/main_integration/simulation_db.py` | DB 模式运行前调用等价预校验，避免文件模式和 DB 模式校验差异 |
| `src/core/main_integration/config_loader.py` | 配置加载和字段标准化阶段补充类型转换、字段别名和缺失列检查 |
| `src/modules/production_planning/config_loader.py` | M4 多产线配置专项校验 |
| `src/modules/deployment_planning/validation.py` | M5 demand priority、OrderLog、SupplyDemandLog 等输入一致性校验 |
| `src/modules/logistics_execution/validators.py` | M6 truck、priority、threshold、deployment plan 校验 |

### 14.3 工作细分

| 子任务 | 输出 |
|---|---|
| schema 清单 | 每张配置表必填列、可选列、类型要求 |
| 类型校验 | 日期、数字、枚举、material/location 标识符 |
| 范围校验 | 非负数量、比例范围、产能、提前期、服务参数 |
| 跨表校验 | 网络、物料、地点、产线、lead time、capacity、changeover 一致性 |
| 坏数据样例 | 每类错误至少一个最小复现样例 |
| 报告输出 | 清晰定位到 sheet、字段、行、错误类型 |

### 14.4 验收标准

- 缺字段、错类型、非法范围、网络不一致可被识别。
- 校验报告能定位到具体 sheet / column / row。
- 文件模式和 DB 模式校验规则一致。

## 15. MLE-4：避免逐日 append 实施细则

### 15.1 目标

清理热点路径里的逐日 `append`、循环 `concat` 和逐行 DataFrame 构造，改为 list 累积后一次构造、批量 concat、预分配数组或索引化更新。

### 15.2 必须审计和修改的代码位置

| 文件 | 修改目的 |
|---|---|
| `src/modules/demand_planning_refactor/integration.py` | `_merge_with_history()`、`_apply_orders_consumption()` 的 concat / 过滤路径性能检查 |
| `src/modules/demand_planning_refactor/order.py` | 周级 AO / normal 订单生成、日级拆分保持批量 DataFrame 构造 |
| `src/modules/demand_planning_refactor/io_utils.py` | 历史订单读取 rows list + concat 保持批量方式，避免日循环内重复读写 |
| `src/modules/production_planning/plan_builder.py` | `plans.append(plan)` 后一次 concat；多产线改造时避免候选表循环 concat |
| `src/modules/production_planning/capacity_allocator.py` | `plans`、`exceeds` 使用 list of dict 保持，避免循环 DataFrame append |
| `src/modules/production_planning/output_writer.py` | 合并每日输出时保持 list + concat，不在循环中扩张 DataFrame |
| `src/modules/deployment_planning/main.py` | `demand_rows`、`deployment_plan_rows`、`unfulfilled_rows` 等热点列表保持批量构造 |
| `src/modules/deployment_planning/demand_collector.py` | 需求收集结果统一 list 累积，减少循环 filter 和 append |
| `src/modules/logistics_execution/output_writer.py` | shipment / validation 输出批量构造 |
| `src/core/run/local_writer.py` | 多日模块输出合并保持 list + concat，并避免重复写同一文件 |

### 15.3 工作细分

| 子任务 | 输出 |
|---|---|
| append 审计 | 所有 `append(`、`pd.concat(`、循环 DataFrame 构造命中清单 |
| 热点分类 | 必改、可接受、非热点三类 |
| 批量构造替换 | list of dict / list of DataFrame 后一次构造 |
| 预索引替换 | 高频查找改为 dict / MultiIndex / group map |
| 性能验证 | 改造前后平均耗时和内存峰值 |

### 15.4 验收标准

- 热点路径不再循环扩张 DataFrame。
- 输出行数、主键、数量字段与改造前一致或差异可解释。
- M1/M4/M5 至少有性能 benchmark。

## 16. VAL-2：中间数据校验实施细则

### 16.1 目标

在模块间传递的关键中间表上增加校验点，尽早发现负数、重复 key、缺失列、异常日期、库存不平衡和跨模块口径漂移。

### 16.2 必须修改或新增的代码位置

| 文件 | 修改目的 |
|---|---|
| `src/utils/validation_manager.py` | 复用统一报告机制，输出中间校验报告 |
| `src/utils/inventory_balance_checker.py` | 扩展库存平衡校验，纳入 M1/M4/M5/M6/M3 关键流转 |
| `src/core/main_integration/simulation_file.py` | 每日模块运行后插入中间校验点 |
| `src/core/main_integration/simulation_db.py` | DB 模式批次写入前插入中间校验点，失败时不推进 checkpoint |
| `src/core/main_integration/db_helpers.py` | 批量写 DB 前校验输出表 schema、run_id、simulation_date |
| `src/core/orchestrator/views.py` | 对 orchestrator 视图输出增加最小字段和日期口径校验 |
| `src/modules/deployment_planning/validation.py` | 扩展 M5 输入输出一致性校验 |
| `src/modules/logistics_execution/output_writer.py` | 保留 shipment / delivery 约束校验 |

### 16.3 建议新增文件

| 文件 | 职责 |
|---|---|
| `src/utils/intermediate_validator.py` | 集中定义中间表 checkpoint 校验规则 |
| `docs/testing/intermediate_validation_checklist.md` | 记录每个 checkpoint 的输入、输出和验收口径 |

### 16.4 工作细分

| 子任务 | 输出 |
|---|---|
| checkpoint 清单 | M1/M3/M4/M5/M6/Orchestrator 关键表 |
| schema 校验 | 每张中间表必填列、日期列、主键口径 |
| 数值校验 | 负数、NaN、无限值、异常大值 |
| 日期校验 | simulation_date、planned_date、available_date、requirement_date 合法性 |
| 重复校验 | 关键业务键重复检查 |
| 报告输出 | Excel/CSV/文本格式 validation report |

### 16.5 验收标准

- 每日关键中间表可校验。
- 错误能定位到表、字段、日期和业务键。
- DB 模式不得在模块失败或中间校验失败时推进 checkpoint。

## 17. TEST-1：E2E 网络构建实施细则

### 17.1 目标

构建最小但完整的端到端测试网络，覆盖 M1 -> M4 -> M5 -> M6 -> M3 的主链路，并能稳定跑通 1 天、7 天和 30 天仿真。

### 17.2 代码和测试修改位置

| 文件或目录 | 修改目的 |
|---|---|
| `run.py` | 保持 E2E 命令入口稳定 |
| `src/core/run/run_main.py` | 确认参数解析、文件模式 / DB 模式调度正确 |
| `src/core/main_integration/simulation_file.py` | 文件模式 E2E 主链路 |
| `src/core/main_integration/simulation_db.py` | DB 模式 E2E、checkpoint、summary 主链路 |
| `src/core/orchestrator/` | 验证库存、GR、shipment、open deployment 状态流转 |
| `src/services/summary_report_generator.py` | E2E 结束后生成汇总报告 |
| `src/utils/inventory_balance_checker.py` | E2E 库存平衡验收 |
| `config/` | 增加或整理最小 E2E 配置样例 |
| `tests/e2e/` 或 `scripts/e2e/` | 建议新增自动化 E2E runner 和 compare 脚本 |

### 17.3 工作细分

| 子任务 | 输出 |
|---|---|
| 最小网络设计 | plant / DC / customer、BOM、lead time、capacity |
| 基础配置生成 | 可运行 Excel 或 DB config |
| 1 天 smoke | 验证模块顺序和输出文件 |
| 7 天 smoke | 验证跨天状态、库存和历史订单 |
| 30 天 E2E | 验证长期运行、checkpoint、summary |
| 自动化脚本 | 一键 run + compare + report |
| E2E 报告 | 关键 KPI、失败点、已知限制 |

### 17.4 验收标准

- 30 天 E2E 稳定跑通。
- 库存平衡通过。
- M1/M3/M4/M5/M6 关键输出均存在且 schema 正确。
- 文件模式和 DB 模式差异有解释。

## 18. DOC-2 / DOC-3 / DOC-4：分析文档实施细则

### 18.1 DOC-2 历史回测文档

#### 代码和数据位置

| 文件或目录 | 修改目的 |
|---|---|
| `src/services/summary_report_generator.py` | 复用订单、发货、缺货、生产、调拨、物流汇总 |
| `pgsql_db/module_data_writer.py` | DB 模式 summary 表生成 |
| `archive/ChainSight_Dev/compare_runs.py` | 可参考旧对比逻辑，必要时迁移到 `scripts/` |
| `docs/testing/` | 输出回测报告和证据链 |

#### 工作细分

| 子任务 | 输出 |
|---|---|
| 回测资料整理 | baby case / 历史输出 / 实际 KPI |
| KPI 定义 | simulated vs actual 指标字典 |
| 差异分析 | 误差、偏差、异常月份 |
| 图表生成 | KPI 对比图、偏差图 |
| 证据链 | 数据源、脚本、输出文件链接 |

### 18.2 DOC-3 trade-off 分析

#### 代码和数据位置

| 文件或目录 | 修改目的 |
|---|---|
| `src/services/summary_report_generator.py` | 复用 summary 输出，补充 cost / cash / service 汇总所需字段 |
| `src/core/run/` | 支持多 scenario 运行参数或脚本调度 |
| `scripts/` | 建议新增 scenario runner 和结果汇总脚本 |
| `docs/testing/` 或 `docs/analysis/` | 输出 trade-off 报告 |

#### 工作细分

| 子任务 | 输出 |
|---|---|
| 指标框架 | cost / cash / service KPI |
| 场景设计 | baseline / high service / low cost 等 |
| 批量运行 | 多 scenario 输出 |
| 汇总结果 | 结果表、Pareto 或关键拐点 |
| 图表 | trade-off 曲线和矩阵 |

### 18.3 DOC-4 PG 数据分布分析

#### 代码和数据位置

| 文件或目录 | 修改目的 |
|---|---|
| `src/core/main_integration/config_loader.py` | 确认 PG 数据源加载和字段标准化 |
| `src/utils/config_validator.py` | 检查 PG 数据字段、日期范围、缺失和异常 |
| `scripts/` | 建议新增 distribution profiling 脚本 |
| `docs/analysis/` | 建议输出 PG 数据分布报告 |

#### 工作细分

| 子任务 | 输出 |
|---|---|
| 数据盘点 | 数据源、字段、时间范围 |
| demand 分布 | 均值、方差、长尾、季节性 |
| inventory 分布 | 库存、缺货、异常库存 |
| leadtime 分布 | 均值、波动、异常值 |
| production 分布 | 产能、产量、可靠性 |
| 建模建议 | 分布对仿真参数的影响 |

## 19. DOC-1：scope / problem / success criteria 实施细则

### 19.1 目标

形成项目范围、问题定义、成功标准和非目标说明，作为最终验收的文字标准。

### 19.2 代码修改范围

DOC-1 本身原则上不需要修改业务代码。若发现验收标准无法被现有输出证明，才需要补充观测或报告能力：

| 文件 | 可能修改目的 |
|---|---|
| `src/services/summary_report_generator.py` | 增加可支撑成功标准的汇总指标 |
| `src/utils/inventory_balance_checker.py` | 增加库存平衡和异常说明 |
| `src/utils/validation_manager.py` | 输出最终验收所需 validation report |
| `docs/project_execution_plan.md` | 如项目范围变化，更新高层计划 |
| `docs/project_tracking_plan.xlsx` | 更新状态、工时和完成率 |

### 19.3 工作细分

| 子任务 | 输出 |
|---|---|
| Scope 定义 | in-scope / out-of-scope |
| Problem 定义 | 业务问题和技术问题 |
| Success criteria | 性能、准确性、可维护性、验证标准 |
| Non-goals | 明确不做的事项 |
| 评审修订 | review notes 和 final scope doc |

## 20. FINAL：最终验收实施细则

### 20.1 目标

确认代码、验证、测试、文档、风险关闭全部完成，形成可交付版本。

### 20.2 必须检查的代码和产物位置

| 范围 | 检查内容 |
|---|---|
| `src/modules/demand_planning_refactor/` | M1 订单生成、输出兼容、性能；旧 `src/modules/demand_planning/` 作为基线对照 |
| `src/modules/production_planning/` | M4 多产线、产能、换产、可靠性 |
| `src/modules/deployment_planning/` | M5 对 M1/M4 输出读取和分配逻辑 |
| `src/modules/logistics_execution/` | M6 交付、车辆、validation |
| `src/modules/mrp_planning/` | M3 净需求和下游回流 |
| `src/core/main_integration/` | 文件模式 / DB 模式主链路、checkpoint、summary |
| `src/core/orchestrator/` | 库存、GR、shipment、open deployment 状态 |
| `src/services/summary_report_generator.py` | 汇总报告 |
| `pgsql_db/` | DB schema、写入、summary 表、checkpoint |
| `docs/` | 实施计划、测试报告、分析报告、验收记录 |

### 20.3 工作细分

| 子任务 | 输出 |
|---|---|
| 最终集成验收 | final checklist |
| 风险关闭 | open risks / known issues closure |
| 交付包整理 | 代码、配置、报告、测试结果、运行说明 |
| sign-off | 验收记录和最终状态 |

### 20.4 验收标准

- 所有 `Progress Tracker` 任务状态与证据一致。
- E2E、summary、inventory balance、validation report 均可追溯。
- 已知风险有关闭记录或明确接受说明。
- 交付包可以让接手者复现关键结果。
