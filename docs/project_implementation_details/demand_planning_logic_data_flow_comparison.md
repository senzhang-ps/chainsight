# demand_planning 与 demand_planning_refactor 实现逻辑及数据流对比

更新时间：2026-05-08

本文对比以下两个模块的实现逻辑顺序和数据流顺序：

- `src/modules/demand_planning`
- `src/modules/demand_planning_refactor`

## 1. 总体结论

`demand_planning` 原版是“日度优先”实现：先将预测数据准备成日度数据，再直接按当前仿真日生成 AO 和 Normal 订单。

`demand_planning_refactor` 重构版是“周级订单优先”实现：先识别周级预测基线，在周粒度生成订单总量，再拆分到具体下单日，最后进入后续消耗、发货和输出流程。

后段流程，包括历史订单合并、发货/缺货计算、供需日志生成和 Excel 输出，整体顺序基本保持一致。主要变化集中在 `integration.py`、`order.py` 和默认优化消耗路径。

## 2. 实现逻辑顺序对比

| 对比项 | `demand_planning` 原版 | `demand_planning_refactor` 重构版 |
|---|---|---|
| 模块入口 | `integration.py::run_daily_order_generation()`，由 `module1.run_daily_order_generation` 暴露 | 同样以 `integration.py::run_daily_order_generation()` 为主入口，但 `__init__.py` 额外导出 weekly 相关 API |
| 配置校验 | 先校验 `M1_DemandForecast`、`M1_ForecastError`、`M1_OrderCalendar`、`M1_AOConfig` | 校验项基本一致 |
| 预测准备 | `_prepare_forecasts()` 中执行 DPS、SupplyChoice、周转日，产出 `daily_for_orders`、`daily_for_supply` | `_prepare_forecasts()` 返回 `weekly_for_orders`、`daily_for_consumption`、`daily_for_supply` |
| 订单生成 | 直接按日度预测窗口生成当日 AO 订单和 Normal 订单 | 若输入是周级预测，先 `generate_weekly_orders()`，再 `split_weekly_orders_to_daily()`，最后进入消耗 |
| AO/Normal 顺序 | 先生成 AO，再生成 Normal，然后聚合订单 | 周级路径先按 AOConfig 拆 AO / normal 均值，再分别抽样总量和 AO / normal 结构，最后归一回周总量并拆到日 |
| 预测消耗 | `order.generate_daily_orders()` 内部调用 `consume_orders()` 消耗订单预测 | 同样调用 `consume_orders()`，默认走 optimized/vectorized 消耗路径 |
| 历史订单合并 | 订单生成后 `_merge_with_history()` 合并历史 `OrderLog` | 顺序一致：今日订单生成后再合并历史订单 |
| 发货/缺货计算 | `_generate_shipments()` 基于 orchestrator 库存生成 `shipment_df`、`cut_df` | 发货逻辑基本一致，仍基于库存计算 ship/cut |
| 供需日志 | `_apply_orders_consumption()` 消耗 `daily_for_supply` 后生成 `supply_demand_df` | 顺序一致，但更强调供需基线与订单基线分离 |
| 输出 | `io_utils.save_module1_output_with_supply_demand()` 写 `module1_output_YYYYMMDD.xlsx` | 输出边界基本不变，仍写相同结构的 Excel |

## 3. 原版 `demand_planning` 逻辑顺序

```text
run_daily_order_generation()
-> _validate_config()
-> _prepare_forecasts()
-> generate_daily_orders()
-> _merge_with_history()
-> _generate_shipments()
-> _apply_orders_consumption()
-> generate_supply_demand_log_for_integration()
-> _save_output()
```

关键说明：

1. `_prepare_forecasts()` 先将周度 forecast 展开为日度 forecast。
2. `generate_daily_orders()` 以当前 `simulation_date` 为核心判断是否生成订单。
3. 订单生成时直接计算 AO 和 Normal 订单。
4. 订单生成后再合并历史订单。
5. 后续发货、缺货、供需日志和 Excel 输出按固定顺序执行。

## 4. 重构版 `demand_planning_refactor` 逻辑顺序

```text
run_daily_order_generation()
-> _validate_config()
-> _prepare_forecasts()
-> generate_daily_orders()
-> generate_weekly_orders()
-> split_weekly_orders_to_daily()
-> consume_orders()
-> _merge_with_history()
-> generate_shipment_with_inventory_check()
-> _apply_orders_consumption()
-> generate_supply_demand_log_for_integration()
-> save_module1_output_with_supply_demand()
```

关键说明：

1. 重构版保留原主入口，外部调用边界基本不变。
2. 当预测数据仍是周级结构时，订单生成优先走周级订单路径。
3. 周级路径先计算 AO / normal 周级均值，再分别生成 total 周总量和 AO / normal 临时结构，并归一回 total 后按有效订单日拆分。
4. 订单拆分后仍进入统一的预测消耗、历史合并、发货和输出流程。
5. 重构版通过 weekly API 将周级订单逻辑显式暴露，便于单独测试和复用。

## 5. 数据流顺序对比

| 数据阶段 | `demand_planning` 原版 | `demand_planning_refactor` 重构版 |
|---|---|---|
| 输入配置 | `config_dict` 中读取 forecast、error、calendar、AO 配置 | 输入配置结构基本一致 |
| 预测数据准备 | `M1_DemandForecast` → DPS → SupplyChoice → 日度 forecast | `M1_DemandForecast` → DPS/SupplyChoice → 周级订单基线 + 日级消耗基线 + 供需基线 |
| 订单输入数据 | `daily_for_orders` | 周级路径使用 `weekly_for_orders`，兼容返回使用 `daily_for_consumption` |
| 订单生成结果 | `today_orders_df` | `today_orders_df`，但来源可能是周级订单拆日结果 |
| 订单历史 | `today_orders_df` + previous `OrderLog` → `all_orders_df` | 顺序一致 |
| 发货输入 | `all_orders_df` + orchestrator inventory | 顺序一致 |
| 发货结果 | `shipment_df`、`cut_df` | 顺序一致 |
| 供需日志输入 | 消耗后的 `daily_for_supply` | `_apply_orders_consumption()` 用当日订单消耗独立供需基线 `daily_for_supply` |
| 最终输出 | `OrderLog`、`ShipmentLog`、`CutLog`、`SupplyDemandLog`、`Summary` | 输出 sheet 结构保持一致 |

## 6. 原版数据流

```text
config_dict
-> demand_forecast / forecast_error / order_calendar / ao_config
-> daily_for_orders / daily_for_supply
-> today_orders_df
-> all_orders_df
-> shipment_df / cut_df
-> consumed_supply
-> supply_demand_df
-> module1_output_YYYYMMDD.xlsx
```

原版特点：

- 数据流以日度 forecast 为核心。
- `daily_for_orders` 和 `daily_for_supply` 是主要中间数据。
- 订单生成直接发生在当前仿真日。
- 周级预测在订单生成前已经被展开为日级数据。

## 7. 重构版数据流

```text
config_dict
-> demand_forecast / forecast_error / order_calendar / ao_config
-> weekly_for_orders / daily_for_consumption / daily_for_supply
-> weekly_orders_df
-> today_orders_df
-> all_orders_df
-> shipment_df / cut_df
-> supply_demand_df
-> module1_output_YYYYMMDD.xlsx
```

重构版特点：

- 数据流显式区分订单生成、订单消耗和供需展示三类基线。
- 周级订单先在周粒度生成，再拆到可下单日期。
- `daily_for_consumption` 是订单层返回 `consumed_forecast` 的兼容基线。
- `daily_for_supply` 用于供需日志，`_apply_orders_consumption()` 会用当日订单对它单独消耗。

## 8. 关键差异总结

1. 原版是“日度优先”：周预测先展开为日预测，然后直接按日度窗口生成订单。
2. 重构版是“周级订单优先”：先在周粒度生成守恒的 AO / normal 订单数量，再拆分到具体订单日。
3. 原版数据流较集中，订单生成、订单消耗、供需展示的输入边界相对弱。
4. 重构版将订单生成、订单消耗、供需展示的数据基线拆分得更清楚，职责边界更明确。
5. 发货、缺货、供需日志、Excel 输出这些后段流程基本保持一致。
6. 主要代码变化集中在 `integration.py`、`order.py` 和 `__init__.py`；`consume.py` / `consume_optimized.py` 负责默认优化消耗路径。
7. `forecast.py`、`shipment.py`、`io_utils.py` 等支撑模块的整体边界基本保持稳定。

## 9. 涉及文件

| 文件 | 说明 |
|---|---|
| `src/modules/demand_planning/integration.py` | 原版端到端编排主入口 |
| `src/modules/demand_planning_refactor/integration.py` | 重构版端到端编排主入口 |
| `src/modules/demand_planning/order.py` | 原版日级订单生成逻辑 |
| `src/modules/demand_planning_refactor/order.py` | 重构版周级订单生成、周到日拆分、日级消费逻辑 |
| `src/modules/demand_planning/forecast.py` | 原版周转日预测展开逻辑 |
| `src/modules/demand_planning_refactor/forecast.py` | 重构版预测展开支撑逻辑 |
| `src/modules/demand_planning/shipment.py` | 原版发货/缺货计算逻辑 |
| `src/modules/demand_planning_refactor/shipment.py` | 重构版发货/缺货计算逻辑 |
| `src/modules/demand_planning/io_utils.py` | 原版历史订单读取与 Excel 输出 |
| `src/modules/demand_planning_refactor/io_utils.py` | 重构版历史订单读取与 Excel 输出 |

## 10. 有效下单日判断对比

### 10.1 原版 `demand_planning`

原版只在日级订单生成入口判断当前仿真日是否为有效下单日：

```text
src/modules/demand_planning/order.py::generate_daily_orders()
-> is_order_day = not order_calendar[order_calendar['date'] == sim_date].empty
-> if not is_order_day: return empty orders_df
```

判断口径：

1. `M1_OrderCalendar.date` 必须精确等于当前 `sim_date`。
2. 如果当天不在 `M1_OrderCalendar` 中，直接返回空订单，不再计算 AO / normal。
3. 如果当天在 `M1_OrderCalendar` 中，才继续基于日级 forecast 窗口计算 `avg_daily_demand`。
4. `OrderCalendar` 不参与未来 7 天 forecast 窗口筛选，也不参与 AO 需求日期校验。
5. AO 的 `OrderLog.date = sim_date + advance_days`，这个需求日期可以不是有效下单日。

因此，原版的有效下单日判断是“**是否允许今天生成订单**”。它不负责把一周订单分配到哪些日期，也不保证 AO 需求日期仍落在 `M1_OrderCalendar` 内。

### 10.2 重构版 `demand_planning_refactor`

重构版在周级路径下把有效下单日判断拆成两个层次：

```text
src/modules/demand_planning_refactor/order.py::generate_daily_orders()
-> _uses_weekly_order_generation()
-> _generate_daily_orders_from_weekly()
-> generate_weekly_orders()
-> split_weekly_orders_to_daily()
```

其中：

```text
generate_weekly_orders()
```

只负责：

```text
weekly forecast -> AO / normal mean split
                -> total/weekly CoV sampled total qty
                -> AO / normal CoV provisional qty
                -> reconciled weekly AO / normal qty
```

它不接收 `order_calendar`，也不接收 `simulation_date`，所以不做有效下单日判断。

真正判断有效下单日的位置是：

```text
src/modules/demand_planning_refactor/order.py::split_weekly_orders_to_daily()
```

判断逻辑：

```text
simulation_date = normalize(simulation_date)
calendar_dates = normalize(M1_OrderCalendar.date)
if simulation_date not in calendar_dates:
    return empty orders_df

valid_order_dates = [
    d for d in calendar_dates
    if week_start <= d < week_end
]

if simulation_date not in valid_order_dates:
    continue
```

判断口径：

1. 先规范化 `simulation_date` 和 `M1_OrderCalendar.date`，避免时间分量影响匹配。
2. 当天必须存在于 `M1_OrderCalendar.date`，否则直接返回空订单。
3. 对每个周级订单，再筛选该周范围内的有效下单日 `valid_order_dates`。
4. 周级订单量只分配到 `valid_order_dates` 中的日期。
5. 每次日仿真只返回 `valid_order_dates` 中等于当前 `simulation_date` 的那一份。
6. AO 的 `advance_days` 在拆到有效下单日之后才应用，得到最终 `OrderLog.date`。

因此，重构版的有效下单日判断是“**先用本周有效下单日决定周订单如何拆日，再让每日仿真只取当前下单日对应份额**”。它不影响周级总量生成，但会影响周量分配和当日是否输出订单。

### 10.3 关键差异

| 对比项 | 原版 `demand_planning` | 重构版 `demand_planning_refactor` |
|---|---|---|
| 判断函数 | `generate_daily_orders()` | `split_weekly_orders_to_daily()` |
| 判断阶段 | 日级订单生成入口 | 周订单拆日阶段 |
| 是否影响周级总量 | 不存在周级总量 | 不影响，`generate_weekly_orders()` 不检查日历 |
| 是否使用本周全部有效下单日 | 否，只判断当天 | 是，先找出本周 `valid_order_dates` |
| 非有效下单日结果 | 返回空订单 | 返回空订单 |
| 有效下单日的业务含义 | 今天是否允许生成订单 | 周订单可以落在哪些下单日，以及今天取哪一份 |
| AO `advance_days` 与日历关系 | 先判断 `sim_date`，再算 AO 需求日期 | 先拆到有效下单日，再算 AO 需求日期 |
| `OrderLog.date` 是否必须是有效下单日 | 否，它是需求日期 | 否，它仍是需求日期；有效下单日由 `simulation_date` 表达 |

结论：当前新旧版本都通过 `M1_OrderCalendar.date` 判断有效下单日，但判断职责不同。原版是日级入口开关；重构版是周订单拆日约束加每日仿真过滤。重构版不应该、也没有在 `generate_weekly_orders()` 中检查下单日。
