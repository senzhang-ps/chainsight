# demand_planning 与 demand_planning_refactor 实现对比

测试与排查参考：本文对比 `src/modules/demand_planning` 原版与 `src/modules/demand_planning_refactor` 重构版在实现逻辑顺序和数据流顺序上的差异。

## 实现逻辑顺序对比

| 对比项 | demand_planning 原版 | demand_planning_refactor 重构版 |
|---|---|---|
| 模块入口 | `integration.py::run_daily_order_generation`，由 `module1.run_daily_order_generation` 暴露 | 同样以 `integration.py::run_daily_order_generation` 为主入口，`__init__.py` 额外导出 weekly 相关 API |
| 配置校验 | 先校验 `M1_DemandForecast`、`M1_ForecastError`、`M1_OrderCalendar`、`M1_AOConfig` | 校验项基本一致 |
| 预测准备 | `_prepare_forecasts()` 中执行 DPS、SupplyChoice、周转日，产出 `daily_for_orders`、`daily_for_supply` | 更明确拆成订单基线、消耗基线、供需基线，并支持周级订单路径 |
| 订单生成 | 直接按日度预测窗口生成当日 AO 订单和 Normal 订单 | 若输入是周级预测，先 `generate_weekly_orders()`，再 `split_weekly_orders_to_daily()`，最后进入消耗 |
| AO/Normal 顺序 | 先生成 AO，再生成 Normal，然后合并订单 | 仍保留 AO/Normal 逻辑，但周级路径先算周总量，再拆分到日 |
| 预测消耗 | `order.generate_daily_orders()` 内部调用 `consume_orders()` 消耗订单预测 | 同样调用 `consume_orders()`，默认走 optimized/vectorized 消耗路径 |
| 历史订单合并 | 订单生成后 `_merge_with_history()` 合并历史 `OrderLog` | 顺序一致：今日订单生成后再合并历史订单 |
| 发货/缺货 | `_generate_shipments()` 基于 orchestrator 库存生成 `shipment_df`、`cut_df` | 发货逻辑基本一致，仍基于库存计算 ship/cut |
| 供需日志 | `_apply_orders_consumption()` 消耗 `daily_for_supply` 后生成 `supply_demand_df` | 顺序一致，但 refactor 更强调供需基线与订单基线分离 |
| 输出 | `io_utils.save_module1_output_with_supply_demand()` 写 `module1_output_YYYYMMDD.xlsx` | 输出边界基本不变，仍写相同结构的 Excel |

## 数据流顺序对比

| 阶段 | demand_planning 原版数据流 | demand_planning_refactor 重构版数据流 |
|---|---|---|
| 输入配置 | `config_dict` → `demand_forecast`、`forecast_error`、`order_calendar`、`ao_config` | 同左 |
| 预测处理 | `demand_forecast` → DPS/SupplyChoice → `daily_for_orders`、`daily_for_supply` | `demand_forecast` → `demand_dps` 周级订单基线 → `daily_for_consumption` 日级消耗基线 → `daily_for_supply` 供需基线 |
| 订单计算 | `daily_for_orders` → 7 天均值 → AO/Normal → `today_orders_df` | `weekly_for_orders` → 周级总量 → AO/Normal 周级分量 → 日拆分 → `today_orders_df` |
| 订单消耗 | `today_orders_df` 消耗 `current_forecast` | `today_orders_df` 消耗 `daily_for_consumption` |
| 历史累计 | `today_orders_df` + 历史 `OrderLog` → `all_orders_df` | 同左 |
| 发货/缺货 | `all_orders_df` + orchestrator 库存 → `shipment_df`、`cut_df` | 同左 |
| 供需输出 | `daily_for_supply` 被订单消耗 → `supply_demand_df` | 同左，但与订单生成基线职责更清楚分离 |
| 文件输出 | `OrderLog`、`ShipmentLog`、`CutLog`、`SupplyDemandLog`、`Summary` | 同左 |

## 核心结论

1. 原版是“日度优先”：周预测先展开为日预测，然后直接按日度窗口生成订单。
2. 重构版是“周级订单优先”：先在周粒度生成订单总量，再拆分到具体订单日。
3. 原版数据流较集中，`current_forecast`、`daily_for_orders`、`daily_for_supply` 的职责边界相对弱。
4. 重构版把订单生成、订单消耗、供需展示的数据基线拆得更清楚，数据流职责更明确。
5. 发货、缺货、供需日志、Excel 输出这些后段流程基本保持一致，主要变化集中在 `integration.py` 和 `order.py`。
