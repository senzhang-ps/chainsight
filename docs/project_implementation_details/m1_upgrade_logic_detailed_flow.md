# M1 Refactor Logic Detailed Flow

更新时间：2026-05-08

本文按 `src/modules/demand_planning_refactor/` 的当前代码记录 M1 重构后流程。旧 `src/modules/demand_planning/` 只作为日度优先基线对照。

## 1. 主入口和总体顺序

主入口：

```text
src/modules/demand_planning_refactor/integration.py::run_daily_order_generation()
```

函数签名保持集成模式兼容：

```python
run_daily_order_generation(
    config_dict,
    simulation_date,
    output_dir,
    orchestrator=None,
    skip_file_output=False,
    previous_orders_df=None,
)
```

执行顺序：

```text
run_daily_order_generation()
-> _validate_config()
-> _prepare_forecasts()
-> generate_daily_orders()
   -> _uses_weekly_order_generation()
   -> _generate_daily_orders_from_weekly()
      -> generate_weekly_orders()
      -> split_weekly_orders_to_daily()
      -> consume_orders()
-> _merge_with_history()
-> _generate_shipments()
-> _apply_orders_consumption()
-> generate_supply_demand_log_for_integration()
-> _save_output()
-> _build_summary_df()
```

## 2. 输入校验

`_validate_config()` 读取并校验以下 M1 必填表：

| 配置键 | 用途 |
|---|---|
| `M1_DemandForecast` | 周级或日级 demand forecast |
| `M1_ForecastError` | total/weekly、AO、normal 的 CoV 配置 |
| `M1_OrderCalendar` | 有效下单日 |
| `M1_AOConfig` | AO 比例和 `advance_days` |

处理规则：

1. 缺少任一必填表时抛出 `ValueError`。
2. `M1_OrderCalendar.date` 转为 datetime。
3. `M1_DemandForecast.quantity` 负数裁剪为 0。
4. 对 forecast、forecast error、AO config 做 material / location 等标识符标准化。

## 3. 三类 Forecast 基线

`_prepare_forecasts()` 是重构后 M1 的关键边界。若输入 forecast 含 `week`，它会输出三类视图：

| 视图 | 来源 | 用途 |
|---|---|---|
| `weekly_for_orders` | `M1_DemandForecast` 经 `apply_dps()` 后仍保留周级 | 周级订单总量和 AO / normal 结构生成 |
| `daily_for_consumption` | `weekly_for_orders` 经 `expand_forecast_to_days_integer_split()` 后生成日级 | `generate_daily_orders()` 返回兼容用的 consumed forecast |
| `daily_for_supply` | DPS 后再经 `apply_supply_choice()`，然后展开为日级 | `_apply_orders_consumption()` 和 `SupplyDemandLog` 使用 |

若输入 forecast 不含 `week`，三类视图都退化为输入 forecast 的 copy，并走日级兼容订单生成路径。

## 4. 周级订单生成

`generate_daily_orders()` 会先用 `_uses_weekly_order_generation()` 判断 `original_forecast` 是否是周级订单基线：需要包含 `week` 且不包含 `date`。满足条件时进入 `_generate_daily_orders_from_weekly()`。

周级订单函数：

```text
generate_weekly_orders(weekly_forecast, ao_config, forecast_error)
```

内部顺序：

1. `_build_weekly_ml_demand()` 按 `material/location/week` 聚合周需求并稳定排序。
2. `_split_weekly_demand_by_ao_means()` 按 `AOConfig.ao_percent` 和 `advance_days` 拆出 AO / normal 周级均值。
3. `_generate_weekly_total_orders()` 使用 total/weekly CoV 生成整周订单总量。
4. `_generate_weekly_component_orders()` 使用 AO / normal CoV 生成临时周级结构。
5. `_reconcile_weekly_component_orders_to_total()` 将临时结构按权重归一回整周订单总量。

输出中间表：

```text
material, location, week, demand_type, advance_days, quantity
```

该表只在内部使用，不默认写入生产 `OrderLog`。

## 5. Forecast Error 口径

重构版将 forecast error 分成两层使用：

| 使用点 | 取值规则 |
|---|---|
| 周总量 CoV | 优先使用 `order_type in ('total', 'weekly')`，否则使用 normal，仍缺失时使用该 material/location 的最大值 |
| AO / normal 结构 CoV | 只使用 `order_type in ('ao', 'normal')` 的配置 |

随机数使用 `M1_RandomSeed` 加业务 key 生成稳定 seed：

```text
weekly_total | material | location | week
weekly_component | demand_type | material | location | week | advance_days
```

同一输入、同一 seed、同一业务 key 下，周总量和 AO / normal 结构可复现。

## 6. 周订单拆到有效下单日

拆分函数：

```text
split_weekly_orders_to_daily(
    weekly_orders,
    simulation_date,
    order_calendar,
    reference_daily_forecast,
)
```

核心规则：

1. 规范化 `simulation_date` 和 `M1_OrderCalendar.date`。
2. 如果 `simulation_date` 不在 `M1_OrderCalendar.date` 中，返回空订单。
3. 对每条周级订单，找出 `[week_start, week_start + 7)` 内的有效下单日。
4. 使用 `_split_integer_quantity_by_days(quantity, len(valid_order_dates))` 平均拆分，余数给前 N 个有效下单日。
5. 每日仿真只返回当前 `simulation_date` 对应的那一份。

周起点由 `_build_week_start_map()` 决定：优先从 `reference_daily_forecast` 中每个 `week` 的最小 `date` 推导；如果参考日级 forecast 缺失，则用日历最小日期和周号兜底。

## 7. 日期语义

当前代码的日期语义如下：

| 字段 | 语义 | 生成规则 |
|---|---|---|
| `simulation_date` | 下单日 / 当前仿真日 | 必须是 `M1_OrderCalendar` 有效下单日 |
| `date` | 需求日期 / 到期日期，最终写入 `OrderLog.date` | `simulation_date + advance_days` |
| `advance_days` | AO 提前期；normal 为 0 | 来自 AO config 或 normal 默认值 |

因此：

- normal：`date = simulation_date`，`advance_days = 0`。
- AO：`date = simulation_date + advance_days`。
- `OrderCalendar` 只决定下单日，不要求 AO 需求日期也落在下单日历中。

## 8. 历史订单、发货和供需日志

`_merge_with_history()`：

- DB 模式传入 `previous_orders_df` 时直接使用内存历史订单。
- 文件模式调用 `load_previous_orders(output_dir, simulation_date, max_advance_days)`。
- 只保留 `date >= simulation_date` 的未到期订单。
- 按可用字段去重并标准化，输出累计订单池 `all_orders_df`。

`_generate_shipments()`：

- 调用 `generate_shipment_with_inventory_check()`。
- 只筛选 `OrderLog.date == simulation_date` 的到期订单。
- 可用库存来自 Orchestrator 的 beginning inventory、production GR、delivery GR。

`_apply_orders_consumption()`：

- 使用 `today_orders_df` 消耗 `daily_for_supply`。
- AO 先消耗，normal 后消耗。
- 消耗偏移顺序为 `0, -1, -2, 1, 2, 3`。
- `generate_supply_demand_log_for_integration()` 只输出 `simulation_date < date <= future_cutoff` 的 forecast demand。

## 9. 输出契约

生产输出仍写入：

| Sheet / 返回值 | 列 |
|---|---|
| `OrderLog` / `orders_df` | `date`, `material`, `location`, `demand_type`, `quantity`, `simulation_date`, `advance_days` |
| `ShipmentLog` / `shipment_df` | `date`, `material`, `location`, `quantity`, `demand_type`, `order_id` |
| `CutLog` / `cut_df` | `date`, `material`, `location`, `quantity` |
| `SupplyDemandLog` / `supply_demand_df` | `date`, `material`, `location`, `quantity`, `demand_element` |
| `Summary` / `summary_df` | `Total_Orders`, `Total_Shipments`, `Total_Cuts`, `Total_SupplyDemand`, `Date` |

`run_daily_order_generation()` 返回的 `orders_df` 是累计订单池，与本地 Excel `OrderLog` 保持一致；`all_orders_for_next_day` 保留同一累计订单池供下一日仿真使用。

## 10. 验收点

| 验收点 | 标准 |
|---|---|
| 周总量守恒 | 每个 material/location/week 的周级 total 等于 AO / normal 归一后数量之和 |
| 拆日守恒 | 同一周级订单在本周有效下单日上的拆分量之和等于周级数量 |
| 下单日约束 | 非 `M1_OrderCalendar` 日期返回空新增订单 |
| 日期语义 | `simulation_date` 是下单日，`date` 是需求日期 |
| 输出兼容 | 生产 sheet 名称和列名不变 |
| 随机稳定 | 同一 seed 和业务 key 下周总量、组件结构可复现 |
| 下游透明 | M3 / M5 / Summary / DB writer 不需要理解周级中间表 |
