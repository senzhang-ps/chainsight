# Demand Planning Mode Comparison

更新时间：2026-05-08

本文对比原 `src/modules/demand_planning` 模式与重构后 `src/modules/demand_planning_refactor` 模式，重点说明周度 forecast 到日级订单的处理方式、日期语义、数据流和输出兼容边界。

## 1. 对比结论

| 项目 | 原 `demand_planning` | 新 `demand_planning_refactor` |
|---|---|---|
| 订单生成口径 | 先将周度 forecast 拆成日度 forecast，再按每日 `simulation_date` 生成订单 | 先基于周度 forecast 生成周级订单量，再按有效下单日拆成日级订单 |
| `M1_OrderCalendar` 作用 | 只判断当天是否生成订单 | 决定周订单拆分后可以落在哪些下单日，并让每日仿真只取当前下单日份额 |
| AO `advance_days` 作用点 | 在日级订单生成时直接作用于 `simulation_date` | 周级订单先拆成日级下单量，再根据下单日 `simulation_date` 加 `advance_days` 得到需求日期 |
| `simulation_date` 语义 | 订单生成日 | 订单生成日 / 下单日 |
| `OrderLog.date` 语义 | 需求日期；AO 为 `simulation_date + advance_days`，normal 为 `simulation_date` | 需求日期；AO 为拆分后的下单日 + `advance_days`，normal 为拆分后的下单日 |
| 是否要求拆分后落在订单日 | 不要求。AO 需求日期可能是非订单日 | 要求。拆分出来的 `simulation_date` 必须是 `M1_OrderCalendar` 有效日期 |
| 输出字段 | 保持旧字段 | 保持旧字段不变 |

## 2. 原 demand_planning 模式

### 2.1 主入口

原模式入口：

```text
src/modules/demand_planning/integration.py::run_daily_order_generation()
```

主流程：

```text
_validate_config()
-> _prepare_forecasts()
-> generate_daily_orders()
-> _merge_with_history()
-> _generate_shipments()
-> _apply_orders_consumption()
-> generate_supply_demand_log_for_integration()
-> _save_output()
-> _build_summary_df()
```

### 2.2 周度 forecast 处理

原模式在 `_prepare_forecasts()` 中先把周度 forecast 转为日度 forecast：

```text
src/modules/demand_planning/forecast.py::expand_forecast_to_days_integer_split()
```

拆分规则：

```text
week_start = orchestrator.start_date + (week - 1) * 7 days
base_qty = quantity // 7
remainder = quantity % 7
前 remainder 天每天多分配 1
```

因此，原模式的订单生成输入已经是日度 forecast，而不是周度 forecast。

### 2.3 订单生成逻辑

原模式调用：

```text
src/modules/demand_planning/order.py::generate_daily_orders()
```

处理逻辑：

```text
1. 判断 simulation_date 是否存在于 M1_OrderCalendar.date
2. 如果不是订单日，返回空 orders_df
3. 如果是订单日，从日度 forecast 中取 simulation_date 起未来 7 天窗口
4. 按 material / location 计算 avg_daily_demand
5. 生成 AO 订单
6. 生成 normal 订单
7. 聚合订单
8. 内部调用 consume_orders() 消耗 forecast，但集成入口忽略 returned consumed_forecast
```

### 2.4 AO 和 normal 日期语义

AO：

```text
ao_daily_avg = avg_daily_demand * ao_percent
quantity = normal(ao_daily_avg, ao_daily_avg * error_std_percent)
OrderLog.date = simulation_date + advance_days
simulation_date = 当前仿真日
advance_days = AOConfig.advance_days
```

normal：

```text
normal_daily_avg = avg_daily_demand * (1 - total_ao_percent)
quantity = normal(normal_daily_avg, normal_daily_avg * error_std_percent)
OrderLog.date = simulation_date
simulation_date = 当前仿真日
advance_days = 0
```

关键点：

- `M1_OrderCalendar` 只控制是否在某一天生成订单。
- AO 订单加 `advance_days` 后，`OrderLog.date` 可能不是有效订单日。
- 原模式没有“周订单总量拆到日后守恒”的概念，因为它没有先生成周级订单量。

## 3. 新 demand_planning_refactor 模式

### 3.1 新模块位置

新模块完整放在：

```text
src/modules/demand_planning_refactor/
```

原模块保留不变：

```text
src/modules/demand_planning/
```

### 3.2 主入口兼容

新模块保留外部入口：

```text
src/modules/demand_planning_refactor/integration.py::run_daily_order_generation()
```

函数签名保持不变：

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

### 3.3 forecast 准备层

新 `_prepare_forecasts()` 返回三类视图：

| 视图 | 用途 |
|---|---|
| `weekly_for_orders` | 订单生成使用的周级基线 |
| `daily_for_consumption` | 周级订单路径中 `consume_orders()` 的兼容消耗基线 |
| `daily_for_supply` | `_apply_orders_consumption()` 和 `SupplyDemandLog` 使用 |

对应主流程：

```text
_prepare_forecasts()
-> weekly_for_orders
-> daily_for_consumption
-> daily_for_supply
```

### 3.4 周级订单生成

新模式在 `order.py` 中新增周级订单逻辑：

```text
generate_weekly_orders()
_build_weekly_ml_demand()
_generate_weekly_total_orders()
_split_weekly_demand_by_ao_means()
_generate_weekly_component_orders()
_reconcile_weekly_component_orders_to_total()
```

处理逻辑：

```text
1. 按 material / location / week 聚合周需求
2. 先使用 AOConfig 将 weekly demand 拆成 AO / normal 周级均值
3. 基于 weekly demand forecast 和 total/weekly forecast error(CoV) 按正态分布生成整周订单总量
4. 基于 AO / normal 周级均值和各自 order_type 的 CoV 生成临时 AO / normal 周级数量
5. 将临时 AO / normal 数量按比例归一到整周订单总量，归一后的周级数量作为后续拆日的守恒基线
```

为避免同一周在多个下单日重复随机导致周总量和 AO / normal 结构漂移，新模式对总量和结构分别做稳定随机：

```text
total:     base_seed + weekly_total + material + location + week
component: base_seed + weekly_component + demand_type + material + location + week + advance_days
```

这样同一周、同一物料地点在不同仿真日看到的是同一个周级总订单量和同一套 AO / normal 内部结构。`total/weekly` CoV 控制总盘子，`AO` / `normal` CoV 控制盘子内部结构，最后通过整数分配把 AO / normal 数量归一回总量。

### 3.5 周订单拆到有效下单日

新模式拆分函数：

```text
split_weekly_orders_to_daily()
_split_integer_quantity_by_days()
```

拆分规则：

```text
valid_order_dates = M1_OrderCalendar 中落在当前 week 范围内的日期
base_qty = weekly_order_qty // len(valid_order_dates)
remainder = weekly_order_qty % len(valid_order_dates)
前 remainder 个有效下单日每天多分配 1
```

关键语义：

- `generate_weekly_orders()` 不检查 `M1_OrderCalendar`，只生成总量守恒后的周级 AO / normal。
- `split_weekly_orders_to_daily()` 会先取本周有效下单日，再只返回当前 `simulation_date` 的份额。
- 拆分后的 `simulation_date` 必须是 `M1_OrderCalendar` 有效下单日。
- 如果某天不是有效下单日，则不生成订单。
- 周订单总量等于该周所有有效下单日拆分量之和。

### 3.6 新模式日期语义

用户确认的新规则：

```text
weekly qty
-> daily qty
-> 基于 M1_OrderCalendar 确定下单日 simulation_date
-> 最后基于 advance_days 调整需求日期 OrderLog.date
```

AO：

```text
simulation_date = 有效下单日
OrderLog.date = simulation_date + advance_days
advance_days = AOConfig.advance_days
```

normal：

```text
simulation_date = 有效下单日
OrderLog.date = simulation_date
advance_days = 0
```

## 4. 输出兼容

新模式不新增生产输出字段，不修改下游可见 schema。

### OrderLog

```text
date, material, location, demand_type, quantity, simulation_date, advance_days
```

### ShipmentLog

```text
date, material, location, quantity, demand_type, order_id
```

### CutLog

```text
date, material, location, quantity
```

### SupplyDemandLog

```text
date, material, location, quantity, demand_element
```

### Summary

```text
Total_Orders, Total_Shipments, Total_Cuts, Total_SupplyDemand, Date
```

## 5. 新旧模式差异示例

假设：

```text
week 1 quantity = 70
AO percent = 0.2
advance_days = 2
M1_OrderCalendar = 2026-05-04, 2026-05-06
forecast error = 0
```

### 原模式

```text
周 forecast 70 先拆为 7 天，每天 10
2026-05-04 是订单日：
  avg_daily_demand = 10
  AO quantity = 10 * 0.2 = 2
  normal quantity = 10 * 0.8 = 8
  AO OrderLog.date = 2026-05-06
  normal OrderLog.date = 2026-05-04
```

### 新模式

```text
先按周生成：
  weekly total quantity = normal(70, 70 * forecast_error)
  forecast error = 0 时，weekly total quantity = 70
  weekly AO quantity = 70 * 0.2 = 14
  weekly normal quantity = 70 - 14 = 56

week 1 有两个有效下单日：
  2026-05-04
  2026-05-06

拆到每个下单日：
  AO: 14 / 2 = 7, 7
  normal: 56 / 2 = 28, 28

2026-05-04 当天生成：
  AO quantity = 7
  AO OrderLog.date = 2026-05-06
  normal quantity = 28
  normal OrderLog.date = 2026-05-04
```

## 6. 下游影响

由于输出字段保持不变，下游读取层无需因字段变化而修改：

| 下游 | 依赖 M1 输出 |
|---|---|
| M5 Deployment Planning | `OrderLog`、`SupplyDemandLog`、`ShipmentLog` |
| M3 MRP Planning | M1 需求输出 |
| Summary Report | `OrderLog`、`ShipmentLog`、`CutLog`、`SupplyDemandLog` |
| DB Writer | `module1_output_*` 相关表 |

但业务数值会变化，这是预期差异：

- 原模式是“日均需求驱动订单”。
- 新模式是“周订单量守恒后拆到订单日”。
- 因此新旧结果不应按逐行完全一致验收，应按 schema、数量守恒、日期语义、关键 KPI 对比验收。

## 7. 功能测试覆盖

新增测试目录：

```text
functional_tests/
```

当前覆盖：

| 测试文件 | 覆盖内容 |
|---|---|
| `functional_tests/module_tests/test_m1_refactor_weekly_orders.py` | 周订单拆分、订单日约束、`advance_days` 最后作用、周量守恒、total/weekly CoV 总量控制、AO / normal CoV 结构控制与归一、total CoV 兜底 |
| `functional_tests/feature_tests/test_m1_refactor_vs_legacy.py` | 新旧模式输出字段兼容、关键 KPI 对比 |
| `functional_tests/integration_tests/test_m1_refactor_contract.py` | M1 输出下游契约字段和非负数量校验 |

运行方式：

```powershell
D:\project\chainsight\.venv\Scripts\python.exe functional_tests\run_functional_tests.py
```

当前结果：

```text
11 functional tests passed
```

## 8. 当前保留边界

- 原 `src/modules/demand_planning/` 不修改，作为基线和回退路径。
- 新 `src/modules/demand_planning_refactor/` 作为并行模块存在。
- 当前未修改 `src/core/main_integration` 的模块导入路径，因此主仿真仍默认使用原 M1，除非后续显式切换。
- `docs/INDEX.md` 不加入本文索引。

## 9. 有效下单日判断细化

### 9.1 原模式如何判断有效下单日

原 `src/modules/demand_planning` 的有效下单日判断发生在日级订单生成入口：

```text
src/modules/demand_planning/order.py::generate_daily_orders()
```

核心逻辑：

```text
is_order_day = not order_calendar[order_calendar['date'] == sim_date].empty
if not is_order_day:
    return empty orders_df
```

含义：

1. 只判断当前 `simulation_date` 是否存在于 `M1_OrderCalendar.date`。
2. 如果不存在，当天不生成 AO / normal 订单。
3. 如果存在，继续使用日级 forecast 计算未来 7 天平均日需求。
4. `M1_OrderCalendar` 不参与 forecast 窗口选择，也不参与 AO `OrderLog.date` 的合法性判断。
5. AO 的 `OrderLog.date = simulation_date + advance_days`，所以 AO 需求日期可能不是有效下单日。

原模式下，`M1_OrderCalendar` 的角色是“**当天是否允许下单的开关**”。

### 9.2 新模式如何判断有效下单日

新 `src/modules/demand_planning_refactor` 的周级订单路径中，周级订单生成和有效下单日判断是分开的。

周级订单生成：

```text
src/modules/demand_planning_refactor/order.py::generate_weekly_orders()
```

只做：

```text
weekly forecast -> AO / normal mean split
                -> total/weekly CoV sampled total qty
                -> AO / normal CoV provisional qty
                -> reconciled weekly AO / normal qty
```

这个函数不接收 `M1_OrderCalendar`，也不接收 `simulation_date`，因此不判断有效下单日。

有效下单日判断发生在：

```text
src/modules/demand_planning_refactor/order.py::split_weekly_orders_to_daily()
```

核心逻辑：

```text
calendar_dates = normalize(M1_OrderCalendar.date)
if simulation_date not in calendar_dates:
    return empty orders_df

valid_order_dates = calendar_dates within [week_start, week_end)
if simulation_date not in valid_order_dates:
    skip this weekly order

split_qty = weekly_order_qty allocated by index in valid_order_dates
OrderLog.date = simulation_date + advance_days
```

含义：

1. 先根据 `M1_OrderCalendar` 找出本周有效下单日。
2. 周级 AO / normal 数量只分配到这些有效下单日。
3. 每个仿真日只取 `simulation_date` 对应的那一份。
4. 如果当天不是有效下单日，返回空订单。
5. `advance_days` 最后才作用到需求日期 `OrderLog.date`。

新模式下，`M1_OrderCalendar` 的角色是“**周订单拆日约束 + 每日仿真取数过滤**”。

### 9.3 新旧判断差异

| 对比项 | 原 `demand_planning` | 新 `demand_planning_refactor` |
|---|---|---|
| 判断位置 | `generate_daily_orders()` 开始处 | `split_weekly_orders_to_daily()` |
| 判断对象 | 当前 `sim_date` 是否在 `M1_OrderCalendar.date` | 当前 `simulation_date` 是否在日历内，并且是否属于该周 `valid_order_dates` |
| 周级生成是否检查日历 | 无周级生成 | 不检查 |
| 是否先找本周有效下单日 | 否 | 是 |
| 非下单日输出 | 空订单 | 空订单 |
| 下单日字段语义 | `simulation_date` 是订单生成日 | `simulation_date` 是拆分后的有效下单日 |
| 需求日期字段语义 | `OrderLog.date` 是需求日期 | `OrderLog.date` 仍是需求日期 |
| AO 提前期应用时点 | 当天生成 AO 时直接加 | 周量拆到有效下单日后再加 |

结论：新旧版本都使用 `M1_OrderCalendar.date` 判断有效下单日，但原版只把它作为日级生成开关；新版把它作为周订单拆日和每日取数的约束。周级订单总量生成本身不应该检查下单日，当前重构版也是按这个边界实现的。
