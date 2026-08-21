# ChainSight 模块级时序图（Refactor Preview）

## 文档信息

| 项 | 内容 |
|---|---|
| 维护者 | 林宏南 |
| 文档版本 | Preview v2.0 |
| 最后更新 | 2026-08-21 |
| 文档状态 | Preview，随重构链路持续更新 |
| 适用范围 | `test/test_run.py`、`test/test_integration.py`、`src/modules/` |

> **版本说明**：本文描述当前以 `StateContext` 为状态单一事实源的重构集成链路。legacy 的 `main_integration`、旧 `Orchestrator` 状态处理和 CSV 快照流程仍保留用于兼容与回归，但不是本文的当前运行事实。
>
> **当前入口**：完整链路由 `test/test_run.py` 启动，并调用 `test/test_integration.py` 的 `run_integrated_simulation()`。该入口支持测试 schema、内存运行、DQ 跳过、模块耗时日志和性能报告等调试参数。

---

## 1. 阅读说明

本文说明重构链路每天按什么顺序执行、模块如何通过结果合同写回状态，以及哪些数据可以跨日使用。模块内部算法实现位于各业务包的 `integration_refactor.py` 和 `backends.py`；本文不替代专项算法设计文档。

权威模块顺序定义在 `src/core/orchestrator/models.py`：

```text
day_start → M1 → M4 → M5 → M6 → M3 → day_end
```

---

## 2. 启动与初始化时序

```mermaid
sequenceDiagram
    participant CLI as test_run.py
    participant INT as test_integration.py
    participant O as Orch
    participant CM as ConfigManager
    participant R as ConfigReader
    participant DQ as DataQualityChecker
    participant S as StateContext
    participant MOD as M1/M4/M5/M6/M3

    CLI->>INT: run_integrated_simulation(...)
    INT->>O: 创建 Orch
    O->>CM: bootstrap()
    CM->>CM: 读取 defaults.yaml
    opt 启用持久化
        CM->>CM: 连接 PostgreSQL 并 migrate
    end
    CM->>R: load_all()
    R-->>CM: model 投影后的 all_config
    opt 未使用 --skip-dq 且未命中 DQ 缓存
        CM->>DQ: validate(all_config)
        DQ-->>CM: 模型约束驱动的 DQ 结果
    end
    O-->>INT: 配置、run_id 与协作者就绪
    INT->>S: initialize(all_config)
    S->>S: 初始库存、空间容量或续跑状态恢复
    INT->>MOD: prepare()（每个模块仅一次）
```

### 2.1 初始化约束

- `ConfigReader` 按 `src/models/cfg.py` 的模型注册表投影配置字段；同名 CSV 可覆盖 Excel sheet。
- `ConfigInputDataQualityChecker` 从模型字段和表元数据中读取非空、枚举、范围、日期及主键约束，负责发现并记录问题；是否阻断由 `data_quality` 配置决定。
- `prepare()` 只用于静态配置、索引和预计算；模块的 `run()` 不应隐式重复执行 `prepare()`。

---

## 3. 每日总时序

```mermaid
sequenceDiagram
    participant I as 集成调度器
    participant S as StateContext
    participant M1 as ModuleOne
    participant M4 as ModuleFour
    participant M5 as ModuleFive
    participant M6 as ModuleSix
    participant M3 as ModuleThree
    participant O as Orch / PersistenceManager

    loop 每个 simulation_date
        I->>S: day_start(date)
        Note over S: 期初快照、过期调拨清理、到货/生产入库、刷新 View

        I->>M1: run() / output()
        M1-->>I: M1 结果合同
        I->>S: validate_module_result + apply_module_result(M1)

        I->>M4: 注入前一日 M3 和 M4 跨日状态
        I->>M4: run() / output()
        M4-->>I: M4 结果合同
        I->>S: validate_module_result + apply_module_result(M4)

        I->>M5: run() / output()
        M5-->>I: M5 结果合同
        I->>S: validate_module_result + apply_module_result(M5)

        I->>M6: run() / output()
        M6-->>I: M6 结果合同
        I->>S: validate_module_result + apply_module_result(M6)

        I->>M3: run() / output()
        M3-->>I: M3 结果合同
        I->>S: validate_module_result + apply_module_result(M3)
        I->>S: day_end(date)

        opt enable_persistence=True
            I->>O: 单批事务写模块输出、状态和 checkpoint
        end
    end
```

每个模块完成后，调度器先校验输出合同，再调用 `StateContext.apply_module_result()` 统一写回。模块本身不应直接修改库存、在途或开放调拨容器。

---

## 4. 日初状态处理

```mermaid
sequenceDiagram
    participant I as 集成调度器
    participant S as StateContext
    participant OD as open_deployment
    participant T as in_transit
    participant B as production_backlog
    participant V as 动态 Views

    I->>S: day_start(date)
    S->>S: 保存期初 unrestricted_inventory
    S->>OD: 清理超过宽限期的开放调拨
    S->>T: 接收到货，生成 delivery_gr 并增加收货库存
    S->>B: 将当日可用生产过账，生成 production_gr
    S->>V: 从最新状态重新计算模块输入 View
```

日初完成后，M1 至 M3 读取到的是已处理到货、生产入库和过期调拨清理后的当日状态。

---

## 5. M1：需求与客户发货

```mermaid
sequenceDiagram
    participant I as 集成调度器
    participant M1 as ModuleOne
    participant B as demand_planning backends
    participant S as StateContext

    I->>M1: run()
    M1->>B: 预测、订单、消耗、DPS/供给选择计算
    M1->>S: 读取当前库存及动态 View
    M1-->>I: orders_df / shipment_df / cut_df / supply_demand_df / summary_df
    I->>S: apply_module_result(module1)
    S->>S: 发货扣减库存，保存当日订单和供需事实
```

M1 的订单、发货、削减和供需日志均是结果合同的一部分。`shipment_df` 的库存影响仅在状态层受控处理。

---

## 6. M4：生产与严格一日滞后

```mermaid
sequenceDiagram
    participant I as 集成调度器
    participant S as StateContext
    participant M4 as ModuleFour
    participant B as production_planning backends

    I->>S: get_previous_m3_result(date)
    I->>S: 读取上一日产线状态和已分配产能
    I->>M4: 注入上述跨日输入并 run()
    M4->>B: 无约束计划、产能分配、换产和生产计算
    M4-->>I: production_df / exceed_log / issues_df / changeover_log / unconstrained_plan
    I->>S: apply_module_result(module4)
    S->>S: 更新 production_plan_backlog、产线状态和已分配产能
```

M4 只能消费前一个自然日 M3 的净需求。首日没有前一日 M3 结果时，状态层提供空合同；M4 产生的未来生产会保存在 backlog，待可用日到达后由后续 `day_start()` 入库。

---

## 7. M5：部署与当日 Planning Facts

```mermaid
sequenceDiagram
    participant I as 集成调度器
    participant M5 as ModuleFive
    participant B as deployment_planning backends
    participant S as StateContext
    participant M6 as ModuleSix
    participant M3 as ModuleThree

    I->>M5: run()
    M5->>S: 读取库存、需求、在途、生产和空间 View
    M5->>B: 网络分层、需求收集、分配、MOQ/RV 与校验
    M5-->>I: deployment_plan / unfulfilled_log / stock_on_hand_log / validation_log
    I->>S: apply_module_result(module5)
    S->>S: 稳定排序并创建 open_deployment / DeploymentUID
    S->>S: 发布当日 Planning Facts
    M6->>S: 读取 open_deployment
    M3->>S: 读取同日 Planning Facts
```

Planning Facts 只为同日 M5→M6→M3 数据交接而存在；新一天的 `day_start()` 会清空它们，且它们不会替代可恢复的跨日状态。

---

## 8. M6：物流执行、在途与到货

```mermaid
sequenceDiagram
    participant I as 集成调度器
    participant M6 as ModuleSix
    participant B as logistics_execution backends
    participant S as StateContext

    I->>M6: run()
    M6->>S: 读取开放调拨、库存和物流约束 View
    M6->>B: 路线、车辆、装载、MDQ 与等待规则计算
    M6-->>I: delivery_plan / vehicle_log / truck_usage / unsatisfied_log / validation_log / bypass_log
    I->>S: apply_module_result(module6)
    S->>S: 仅处理当日实际发运
    S->>S: 扣减发货库存与开放调拨，记录 delivery_shipment_log
    alt 当天到货
        S->>S: 增加收货库存并记录 delivery_gr
    else 未来到货
        S->>S: 创建 in_transit
    end
```

M6 的 `ori_deployment_uid` 与车辆关联依赖 M5 写回时的稳定排序；该排序和业务 UID 是回归兼容性边界。

---

## 9. M3：净需求与次日反馈

```mermaid
sequenceDiagram
    participant I as 集成调度器
    participant M3 as ModuleThree
    participant B as mrp_planning backends
    participant S as StateContext
    participant M4 as 次日 ModuleFour

    I->>M3: run()
    M3->>S: 读取 M5 当日 Planning Facts 和 M6 写回后的供应状态
    M3->>B: 网络层级、前置期与净需求计算
    M3-->>I: net_demand_df
    I->>S: apply_module_result(module3)
    S->>S: 按产出日期保存 m3_net_demand_by_date
    M4->>S: 次日读取 get_previous_m3_result()
```

这保证了同日 M3 能看到物流执行后的事实，同时 M4 仍保持严格一日滞后。

---

## 10. 日末、持久化与续跑

```mermaid
sequenceDiagram
    participant I as 集成调度器
    participant S as StateContext
    participant P as PersistenceManager
    participant DB as PostgreSQL

    I->>S: day_end(date)
    S->>S: 保存期末库存和汇总所需状态快照
    opt 启用持久化
        I->>P: batch_transaction()
        I->>P: 保存各模块 output()
        I->>P: 保存 StateContext Views、审计和 M4 跨日状态
        I->>P: 保存 checkpoint（最后完整日）
        P->>DB: 单次提交；异常时回滚
    end
```

checkpoint 只表示已经完整提交的最后一个仿真日。续跑复用原 `run_id`，从该日期的下一天开始，并从已持久化 View 恢复 `StateContext`。

---

## 11. 调试与排障顺序

建议用 `test/test_run.py` 的 `--no-persist`、`--test`、`--test-schema`、`--skip-dq`、`--verbose` 和 `--performance-report` 组合定位问题，并按以下路径追踪：

```text
配置读取 / 模型驱动 DQ
  → 日初 View
  → 模块结果合同
  → StateContext 写回
  → 日末 View 和跨日状态
  → PostgreSQL 持久化与 checkpoint
```

## 12. 相关文档

- [重构架构总览](architecture.md)
- [ChainSight 1.0 运行与维护手册](chainsight-1.0.md)
- [Core 详细说明](core.md)