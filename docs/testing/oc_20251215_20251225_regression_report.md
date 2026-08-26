# OC 11 天连续运行回归测试报告

**测试日期：** 2026-08-21；补充交叉验证：2026-08-26  
**测试范围：** 2025-12-15 至 2025-12-25（11 个仿真日）  
**场景：** `OC_Paste_S1_20251224_repare`  
**结论：** 业务结果一致；重构 Polars 实现端到端耗时为 legacy 的 **24.7%**（**4.05×** 加速）。补充的“固定 legacy M1 输入”交叉验证确认：pandas 与 Polars 的 M4→M5→M6→M3 在 11 天内均与 legacy 完全一致。Full Deployment Plan 汇总表仍保留 219 条 `NULL` / `0` 表示差异待确认。

---

## 1. 测试目标与范围

本次回归包含两类验证：

1. **性能回归**：比较 legacy 数据库实现与 refactor Polars 实现的 11 天连续运行耗时。
2. **数据库结果回归**：比较 `input`（legacy）与 `refactor`（重构）schema 的模块输出、运行状态和汇总报表。

两次运行使用相同的场景、日期范围和连续运行模式。性能报告中的 refactor 运行 ID 与数据库审计中的 refactor 运行 ID 一致；性能报告的元数据 `schema` 标注为 `test`，而数据库审计实际读取 `refactor` schema，详见“限制与注意事项”。

## 2. 测试基线

| 项目 | Legacy | Refactor |
|---|---|---|
| 实现 | `legacy` | `refactor`（Polars） |
| 运行模式 | `continuous` | `continuous` |
| 日期范围 | 2025-12-15 ~ 2025-12-25 | 2025-12-15 ~ 2025-12-25 |
| 运行 ID | `db_OC_Paste_S1_20251224_repare_20260821_104153` | `OC_Paste_S1_20251224_repare_20260821_110420` |
| 落库 schema（审计） | `input` | `refactor` |
| 性能报告 | [run_db_dq_20251215_20251225.json](../../outputs/performance_reports/run_db_dq_20251215_20251225.json) | [test_run_refactor_polars_dq_20251215_20251225.json](../../outputs/performance_reports/test_run_refactor_polars_dq_20251215_20251225.json) |

## 3. 性能结果

### 3.1 端到端耗时

| 指标 | Legacy | Refactor Polars | 改善 |
|---|---:|---:|---:|
| 总耗时 | 1,204.18 s（20.07 min） | 297.48 s（4.96 min） | **4.05× 加速；减少 75.3%** |
| 平均每仿真日 | 109.47 s | 27.04 s | 4.05× |
| 模块执行合计 | 1,075.09 s | 70.92 s | **15.16×** |
| 持久化与检查点 | 88.99 s | 60.43 s | 1.47× |

> Refactor 还包含 32.42 s 的 `finalise` 阶段；该阶段已计入端到端总耗时。

### 3.2 各模块累计耗时

| 模块 | Legacy | Refactor Polars | 加速倍数 | 耗时减少 |
|---|---:|---:|---:|---:|
| M1 | 26.28 s | 24.67 s | 1.07× | 6.1% |
| M3 | 212.85 s | 8.64 s | **24.62×** | 95.9% |
| M4 | 15.78 s | 0.93 s | **16.97×** | 94.1% |
| M5 | 808.36 s | 27.41 s | **29.49×** | 96.6% |
| M6 | 11.83 s | 9.26 s | 1.28× | 21.7% |

### 3.3 日度性能趋势

| 指标 | Legacy | Refactor Polars |
|---|---:|---:|
| 最快单日 | 89.43 s（2025-12-15） | 9.92 s（2025-12-15） |
| 最慢单日 | 122.84 s（2025-12-17） | 15.70 s（2025-12-25） |
| 运行趋势 | 各日约 89 ~ 123 s | 随累计状态增大由约 10 s 增至约 16 s |

性能提升主要来自 M3、M4 与 M5。当前端到端剩余主要耗时已转移至持久化、检查点及运行收尾阶段。

## 4. 数据一致性结果

审计脚本对 31 对表进行了比较，排除了明确的行序技术列，并忽略只有零数值指标的占位行。结果产物位于 [summary.md](../../outputs/legacy_refactor_db_compare/run_20260821_111935/summary.md)、[comparison_report.json](../../outputs/legacy_refactor_db_compare/run_20260821_111935/comparison_report.json) 与 [difference_details.csv](../../outputs/legacy_refactor_db_compare/run_20260821_111935/difference_details.csv)。

| 检查项 | 结果 |
|---|---|
| 已配对表 | 31 |
| 业务一致表 | **30** |
| 有差异表 | 1 |
| 表覆盖缺口 | 2 个 refactor 新增状态表 |
| 范围外表 | 12（空日志、运行审计日志、重复输出或新增内部状态） |

### 4.1 逐表对比明细

下表的“左/右原始行数”是落库的完整行数；“左独有”“右独有”“共同存在”是移除运行元数据与明确行序列、并按审计策略排除全零数值指标占位行后的比较结果。对于非唯一关联键，比较器会在同一键组内按稳定排序补充重复序号进行一对一配对，因此共同存在数不一定等于原始行数。`<完整业务行多重集>` 表示以两侧共有的全部业务字段规范化后进行无序多重集比较。

#### 模块输出

| 左表（input） | 右表（refactor） | 左原始行数 | 右原始行数 | 左独有 | 右独有 | 共同存在 | 关联键 |
|---|---|---:|---:|---:|---:|---:|---|
| `module1_output_cutlog` | `module1_output_cutlog` | 934 | 934 | 0 | 0 | 934 | `sim_date, date, material, location` |
| `module1_output_orderlog` | `module1_output_orderlog` | 49,608 | 49,608 | 0 | 0 | 49,608 | `sim_date, simulation_date, date, material, location` |
| `module1_output_shipmentlog` | `module1_output_shipmentlog` | 17,871 | 17,871 | 0 | 0 | 13,006 | `sim_date, date, material, location` |
| `module1_output_summary` | `module1_output_summary` | 11 | 11 | 0 | 0 | 11 | `sim_date, date` |
| `module1_output_supplydemandlog` | `module1_output_supplydemandlog` | 2,352,240 | 2,352,240 | 0 | 0 | 1,590,140 | `sim_date, date, material, location, demand_element` |
| `module3_output_netdemand` | `module3_output_netdemand` | 18,505 | 18,505 | 0 | 0 | 18,505 | `sim_date, simulation_date, requirement_date, material, location, demand_element` |
| `module4_output_capacityexceed` | `module4_output_capacityexceed` | 1,265 | 1,265 | 0 | 0 | 1,265 | `sim_date, simulation_date, material, location, line` |
| `module4_output_changeoverlog` | `module4_output_changeoverlog` | 77 | 77 | 0 | 0 | 77 | `sim_date, date, location, line, changeover_type` |
| `module4_output_productionplan` | `module4_output_productionplan` | 114 | 114 | 0 | 0 | 114 | `sim_date, simulation_date, production_plan_date, available_date, material, location, line, changeover_id` |
| `module5_output_deploymentplan` | `module5_output_deploymentplan` | 452,002 | 452,002 | 0 | 0 | 154,035 | `sim_date, date, material, sending, receiving, demand_element` |
| `module5_output_stockonhandlog` | `module5_output_stockonhandlog` | 55,969 | 55,969 | 0 | 0 | 13,312 | `sim_date, date, material, location` |
| `module5_output_unfulfilledlog` | `module5_output_unfulfilledlog` | 95,302 | 95,302 | 0 | 0 | 95,302 | `sim_date, date, sending, receiving, demand_element` |
| `module5_output_validation` | `module5_output_validation` | 90,200 | 90,200 | 0 | 0 | 90,200 | `sim_date` |
| `module6_output_deliveryplan` | `module6_output_deliveryplan` | 3,694 | 3,694 | 0 | 0 | 3,694 | `sim_date, ori_deployment_uid, vehicle_uid` |
| `module6_output_truckusagelog` | `module6_output_truckusagelog` | 65 | 65 | 0 | 0 | 65 | `sim_date, date, sending, receiving` |
| `module6_output_vehiclelog` | `module6_output_vehiclelog` | 70 | 70 | 0 | 0 | 70 | `sim_date, date, sending, receiving` |

#### 运行状态：`orchestrator` → `viewcontext`

| 左表（input） | 右表（refactor） | 左原始行数 | 右原始行数 | 左独有 | 右独有 | 共同存在 | 关联键 |
|---|---|---:|---:|---:|---:|---:|---|
| `orchestrator_delivery_gr` | `viewcontext_delivery_gr` | 788 | 788 | 0 | 0 | 788 | `sim_date, date, material, receiving` |
| `orchestrator_inventory_change_log` | `viewcontext_inventory_change_log` | 22,355 | 22,355 | 0 | 0 | 22,355 | `sim_date, date, material, location` |
| `orchestrator_open_deployment` | `viewcontext_open_deployment` | 3,901 | 3,901 | 0 | 0 | 3,901 | `sim_date, ori_deployment_uid` |
| `orchestrator_planning_intransit` | `viewcontext_planning_intransit` | 10,143 | 10,143 | 0 | 0 | 10,143 | `sim_date, transit_uid` |
| `orchestrator_production_gr` | `viewcontext_production_gr` | 55 | 55 | 0 | 0 | 55 | `sim_date, date, material, location` |
| `orchestrator_production_plan_backlog` | `viewcontext_production_plan_backlog` | 643 | 643 | 0 | 0 | 643 | `sim_date, available_date, material, location` |
| `orchestrator_space_quota` | `viewcontext_space_quota` | 0 | 0 | 0 | 0 | 0 | `sim_date` |
| `orchestrator_unrestricted_inventory` | `viewcontext_unrestricted_inventory` | 26,202 | 26,202 | 0 | 0 | 22,027 | `sim_date, date, material, location` |

#### 汇总报表

| 左表（input） | 右表（refactor） | 左原始行数 | 右原始行数 | 左独有 | 右独有 | 共同存在 | 关联键 |
|---|---|---:|---:|---:|---:|---:|---|
| `summary_output_fullcapacityexceed` | `summary_full_exceed_capacity_report` | 1,265 | 1,265 | 0 | 0 | 1,265 | `sim_date, simulation_date, material, location, line` |
| `summary_output_fullchangeoverlog` | `summary_full_changeover_report` | 77 | 77 | 0 | 0 | 77 | `sim_date, date, location, line, changeover_type` |
| `summary_output_fulldeliveryplan` | `summary_full_delivery_plan_report` | 3,694 | 3,694 | 0 | 0 | 3,694 | `ori_deployment_uid, vehicle_uid` |
| `summary_output_fulldeploymentplan` | `summary_full_deployment_plan_report` | 452,002 | 452,002 | 219 | 219 | 283,299 | `<完整业务行多重集>` |
| `summary_output_fullproductionplan` | `summary_full_production_plan_report` | 55 | 55 | 0 | 0 | 55 | `sim_date, simulation_date, production_plan_date, available_date, material, location, line, changeover_id` |
| `summary_output_fulltruckusage` | `summary_full_truck_usage_report` | 65 | 65 | 0 | 0 | 65 | `sim_date, date, sending, receiving` |
| `summary_output_ordershipmentcutsummary` | `summary_full_order_shipment_cut_report` | 21,379 | 21,379 | 0 | 0 | 14,252 | `date, material, location` |

### 4.2 已通过的关键输出

以下模块和主要状态/汇总输出的原始行数与业务值均一致：

- **M1**：`cutlog`、`orderlog`、`shipmentlog`、`summary`、`supplydemandlog`
- **M3**：`netdemand`
- **M4**：`capacityexceed`、`changeoverlog`、`productionplan`
- **M5**：`deploymentplan`、`stockonhandlog`、`unfulfilledlog`、`validation`
- **M6**：`deliveryplan`、`truckusagelog`、`vehiclelog`
- **状态表**：库存变动、开放部署、计划在途、生产收货、生产积压、空间配额、非限制库存等 8 对状态表
- **汇总表**：产能超限、换型、交付计划、生产计划、卡车使用、订单发货削减等 6 对汇总表

### 4.3 唯一差异：Full Deployment Plan 汇总

| 表对 | Legacy 原始行数 | Refactor 原始行数 | 有效业务行 | 精确匹配 | Legacy 独有 | Refactor 独有 |
|---|---:|---:|---:|---:|---:|---:|
| `summary_output_fulldeploymentplan` → `summary_full_deployment_plan_report` | 452,002 | 452,002 | 283,518 / 283,518 | 283,299 | 219 | 219 |

差异样例集中于 2025-12-19 的 `push replenishment` 记录：

- Legacy 在 `deploy_from_future_production`、`deploy_from_in_transit`、`deploy_from_open_deployment_inbound`、`deploy_qty_with_plan_order` 中写入 `NULL`。
- Refactor 在相同记录的上述字段写入 `0`。
- 样例中的日期、物料、来源/接收地点、需求量、计划量、已部署量、交付日期与路线属性相同。

因此，该差异当前判定为**字段表示一致性问题（`NULL` 与 `0`）**，不是已确认的计划数量或路线业务差异。若报表消费者将空值与零值视为同义，可在此表对的专项审计中归一化；若空值具有“未计算/不适用”语义，则应保留为待修复项。

此外，两侧存在非对等的 schema 字段：

| Legacy 独有 | Refactor 独有 |
|---|---|
| `deployed_qty_invcon_push` | `simulation_date` |

### 4.4 覆盖缺口

下列 refactor 状态表无 legacy 对等表，未计入业务一致性通过/失败：

- `viewcontext_delivery_shipment_log`
- `viewcontext_shipment_log`

### 4.5 固定 Legacy M1 的下游交叉验证（2026-08-26）

为隔离 M1 随机数消费顺序、全零记录及订单数量差异对下游的放大影响，新增了受控交叉验证：**不执行 refactor M1**，而是把 legacy 历史 M1 的日度结果注入 refactor 状态层；随后真实执行 refactor 的 M4→M5→M6→M3，并与同一 legacy run 的下游落库结果逐日比较。

固定输入来自 `test_bc` 数据库的 `input` schema：

| 项目 | 值 |
|---|---|
| Legacy M1 / 下游 Oracle run ID | `db_OC_Paste_S1_20251224_repare_20260826_130441` |
| 固定 M1 合同 | 当日 `OrderLog`、累计有效订单、`ShipmentLog`、`CutLog`、`SupplyDemandLog`、`Summary` |
| 累计订单规则 | `sim_date <= D` 且订单 `date >= D` |
| 写库 | 不写数据库；仅内存回放与本地审计报告 |
| 比较范围 | M4、M5、M6、M3 全部已注册业务输出，以及库存、在途、开放调拨、空间配额等日末状态 |
| 不纳入业务验收 | `module5.validation_log`；该表为实现诊断合同，虽行数相同但无稳定跨实现业务键 |

#### 4.5.1 结果

| Refactor 后端 | 比较项 | 业务差异 | 状态差异 | 结果 |
|---|---:|---:|---:|---|
| pandas | 165 张日度模块输出 | 0 | 0 / 44 | 通过 |
| Polars | 165 张日度模块输出 | 0 | 0 / 44 | 通过 |

Polars 最终审计报告：[fixed_legacy_m1_downstream_parity.json](../../outputs/fixed_legacy_m1_downstream_parity/run_20260826_162529/fixed_legacy_m1_downstream_parity.json)。

该验证说明：在完全相同的 M1 外部合同和日度状态写回下，重构下游没有独立业务差异；此前完整集成中的后期差异可以归因到已修复的状态与数值语义，而非 M4–M3 的功能性重构偏离。


## 5. 比较器修正

首次执行中，Full Deployment Plan 的 `0` 与 `0.0` 因字符串形式不同被视为不同整行。已在 [tests/compare_utils.py](../../tests/compare_utils.py) 中把部署类字段纳入数值规范化，使该表从 283,518 条全部未匹配收敛到 219 条明确的 `NULL` / `0` 差异。

## 6. 结论与后续动作

### 回归结论

- **性能：通过。** Refactor Polars 连续运行总耗时降低 75.3%，端到端加速 4.05×；M3/M4/M5 均达到显著加速。
- **功能：通过。** 原始数据库审计的 31 对业务表中 30 对完全一致；固定 legacy M1 的下游交叉验证进一步确认 pandas、Polars 均在 11 天内实现 M4→M5→M6→M3 的业务与状态一致。唯一仍待确认项是 Full Deployment Plan 汇总的 219 条 `NULL` / `0` 表示差异。

### 建议后续动作

1. 明确 Full Deployment Plan 四个 `deploy_from_*` 来源分量的空值业务语义。
2. 若 `NULL` 与 `0` 语义等价，在输出写入或报表专项比较中统一为一种表示后重新审计。
3. 修正/核验 refactor 性能报告中的 `schema: test` 元数据，使其与实际审计的 `refactor` schema 保持一致。
4. 将本次 11 天场景作为性能和落库一致性的固定回归基线，并在修改 M3、M4、M5、持久化逻辑后重复执行。

## 7. 复现命令

以下命令基于本次实际测试调用整理；执行前请确保目标 schema 中仅保留待比较的一个 run ID。

```powershell
# Legacy 数据库运行（input schema）
conda run --no-capture-output -n work python test/test_run.py --config input/OC/OC_Paste_S1_20251224_extension/OC_Paste_S1_20251224_repare.xlsx --start-date 2025-12-15 --end-date 2025-12-25 --engine pandas --use-db --performance-report "$PWD/outputs/performance_reports/run_db_dq_20251215_20251225.json" --run-mode continuous

# Refactor Polars 运行（将 --test-schema 替换为实际目标 schema）
conda run --no-capture-output -n work python test/test_run.py --config input/OC/OC_Paste_S1_20251224_extension/OC_Paste_S1_20251224_repare.xlsx --start-date 2025-12-15 --end-date 2025-12-25 --engine polars --test --test-schema refactor --performance-report "$PWD/outputs/performance_reports/test_run_refactor_polars_dq_20251215_20251225.json" --run-mode continuous

# input 与 refactor 落库审计
conda run --no-capture-output -n work python tests/_compare_legacy_refactor_schema_runs.py --legacy-schema input --refactor-schema refactor

# 固定 legacy M1 的下游交叉验证（pandas；Polars 时将 ENGINE 改为 polars）
$env:RUN_FIXED_LEGACY_M1_DOWNSTREAM='1'
$env:FIXED_LEGACY_M1_DATABASE='test_bc'
$env:FIXED_LEGACY_M1_SCHEMA='input'
$env:FIXED_LEGACY_M1_RUN_ID='db_OC_Paste_S1_20251224_repare_20260826_130441'
$env:FIXED_LEGACY_M1_REFACTOR_ENGINE='pandas'
conda run --no-capture-output -n work pytest tests/test_fixed_legacy_m1_downstream_parity.py -s -q
```
