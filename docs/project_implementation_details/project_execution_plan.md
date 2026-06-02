# 项目任务细分实施计划书与进度跟踪方案

> 来源：用户提供的任务排期截图  
> 生成日期：2026-05-06  
> M1 重构口径更新：2026-05-08，按 `src/modules/demand_planning_refactor/` 当前代码同步  
> 计划周期：2026-05-06 至 2026-10-15  
> 状态口径：pending / in-progress / done / blocked  

---

## 1. 项目目标

本计划将截图中的高层任务拆解为可执行、可跟踪、可验收的日级项目计划，覆盖以下工作流：

1. **Model Logic & Engineering**
   - Module 1：订单生成逻辑升级，先周度生成，再拆分到日。
   - Module 4：生产排程逻辑升级，支持一个 SKU 对应多条产线。
   - 避免通过 filtered DataFrame 访问信息。
   - 避免逐日 append DataFrame 新行。
2. **Model Validation**
   - 用户输入数据校验。
   - 中间结果数据校验。
3. **Model Testing**
   - E2E 测试网络构建。
4. **Documentations**
   - 项目范围、问题定义、成功标准。
   - 历史回测文档。
   - 成本 / 现金 / 服务水平 trade-off 分析。
   - PG 数据分布分析。
5. **Coding Design Problems**
   - 拆分 1000+ 行函数与嵌套函数。
   - 硬编码参数问题：已完成。
   - 死代码 / 注释代码块清理：已完成。

---

## 2. 总体里程碑

| 阶段 | 时间 | 主题 | 主要产出 |
|---|---:|---|---|
| Phase 0 | 5/6 - 5/9 | 启动、基线、代码审计 | 项目计划、性能基线、风险清单 |
| Phase 1 | 5/11 - 5/29 | M1 订单生成逻辑升级 | `demand_planning_refactor` 周→日订单生成、主入口切换评估、向量化回归报告 |
| Phase 2 | 6/1 - 6/26 | M4 多产线生产排程 | 多产线排程逻辑、测试案例、性能报告 |
| Phase 3 | 6/29 - 7/24 | DataFrame 性能优化 + Validation | filtered df 修复、append-free、输入/中间校验 |
| Phase 4 | 7/27 - 8/14 | E2E 测试网络 | E2E 配置、30 天仿真、自动化脚本 |
| Phase 5 | 8/17 - 9/25 | 分析类文档 | 回测、trade-off、PG 分布报告 |
| Phase 6 | 9/28 - 10/15 | 项目定义、收尾、验收 | scope/problem/success criteria、最终交付包 |

---

## 3. WBS 工作分解

| ID | 工作流 | 任务 | 子任务 | 交付物 | 初始状态 |
|---|---|---|---|---|---|
| P0 | 启动 | 项目启动与基线 | 计划确认、性能基线、风险审计 | 项目计划、baseline、风险清单 | pending |
| MLE-1 | Model Logic | M1 订单生成逻辑 | `demand_planning_refactor` 已落地周级订单、日度拆分、AO/normal、向量化；待主入口切换和回归 | M1 重构代码、流程图、回归报告 | in-progress |
| MLE-2 | Model Logic | M4 多产线排程 | SKU-line 候选、产能分配、换产、可靠性 | M4 多产线逻辑 | pending |
| MLE-3 | Model Logic | filtered DataFrame 优化 | chained indexing 审计、`.loc/.copy()` 规范 | DataFrame 访问规范与修复 | pending |
| MLE-4 | Model Logic | 避免逐日 append | list 累积、一次 concat、预分配 | append-free 实现 | pending |
| VAL-1 | Validation | 用户输入校验 | schema、必填、类型、范围、业务一致性 | input validator | pending |
| VAL-2 | Validation | 中间数据校验 | M1/M3/M4/M5 checkpoint、负数/重复/日期异常 | intermediate validator | pending |
| TEST-1 | Testing | E2E 测试网络 | mock network、smoke、30 天仿真、自动化 | E2E 测试包 | pending |
| DOC-1 | Documentation | 项目范围与成功标准 | scope、problem、success criteria、non-goals | 项目定义文档 | pending |
| DOC-2 | Documentation | 历史回测文档 | KPI 定义、simulated vs actual、偏差解释 | 回测报告 | pending |
| DOC-3 | Documentation | Trade-off 分析 | cost/cash/service KPI、场景、曲线 | trade-off 报告 | pending |
| DOC-4 | Documentation | PG 数据分布分析 | demand/inventory/leadtime/production 分布 | PG 数据报告 | pending |
| CDP-1 | Coding Design | 1000+ 行函数拆分 | 长函数识别、拆分、接口整理、测试保护 | 重构 PR | pending |
| CDP-2 | Coding Design | 硬编码参数清理 | 已完成 | - | done |
| CDP-3 | Coding Design | 死代码与注释块清理 | 已完成 | - | done |

---

## 4. 日级实施计划

### 4.1 5 月：启动与 M1 订单生成逻辑升级

| 日期 | 任务 ID | 当日任务 | 具体工作 | 交付物 | 状态 |
|---|---|---|---|---|---|
| 5/6 | P0 | 项目启动 | 确认截图任务范围、冻结代码基线 | 项目计划 v1 | pending |
| 5/7 | P0 | 性能基线 | 跑现有 M1/M4/M5/M6 端到端流程，记录耗时 | baseline 性能日志 | pending |
| 5/8 | P0 | 风险审计 | 标记 M1/M4 高风险函数、长函数、逐日 append 点 | 风险清单 | pending |
| 5/11 | MLE-1 | M1 现状梳理 | 以 `src/modules/demand_planning_refactor` 为准复核 forecast → weekly orders → daily orders → shipment | M1 流程图 | in-progress |
| 5/12 | MLE-1 | 周度订单设计复核 | 复核 week first, then split to valid order dates 的数据结构 | M1 设计说明 | in-progress |
| 5/13 | MLE-1 | 日度拆分规则复核 | 复核整数拆分、余数分配、OrderCalendar 有效下单日规则 | 拆分规则表 | in-progress |
| 5/14 | MLE-1 | AO/Normal 适配复核 | 复核 AO `advance_days` 最后作用、normal 当天下当天要 | AO/Normal 规则说明 | in-progress |
| 5/15 | MLE-1 | 单元测试补强 | 补齐 M1 周→日拆分和输出契约测试 | M1 unit test case | in-progress |
| 5/18 | MLE-1 | 周度订单生成回归 | 回归 `generate_weekly_orders()` 总量、结构、CoV 和随机稳定性 | M1 回归记录 | pending |
| 5/19 | MLE-1 | 日度拆分回归 | 回归 `split_weekly_orders_to_daily()` 数量守恒和下单日过滤 | 日订单输出 | pending |
| 5/20 | MLE-1 | 向量化处理回归 | 回归 `consume_orders()`、`_apply_orders_consumption()` 优化路径 | 向量化验证 | pending |
| 5/21 | MLE-1 | 随机误差一致性 | 验证 `M1_RandomSeed` 与稳定业务 key 不漂移 | 随机一致性检查 | pending |
| 5/22 | MLE-1 | M1 回归测试 | 对比旧版和重构版 `OrderLog` / `SupplyDemandLog` 关键 KPI | M1 diff 报告 | pending |
| 5/25 | MLE-1 | 性能优化 | 缓存周→日 forecast，避免每天重复展开 | M1 cache 实现 | pending |
| 5/26 | MLE-1 | 下游兼容 | 验证 M3/M5 读取 M1 输出不受影响 | 兼容性报告 | pending |
| 5/27 | CDP-1 | 函数拆分 | 拆分 M1 中长函数，保留行为一致 | 小函数化 PR | pending |
| 5/28 | MLE-1 | 性能 benchmark | 记录优化前后 M1 每日平均耗时 | M1 性能报告 | pending |
| 5/29 | MLE-1 | Phase 1 验收 | 代码审查、结果确认、风险关闭 | M1 验收记录 | pending |

### 4.2 6 月：M4 多产线生产排程

| 日期 | 任务 ID | 当日任务 | 具体工作 | 交付物 | 状态 |
|---|---|---|---|---|---|
| 6/1 | MLE-2 | M4 现状梳理 | 梳理 SKU→line、capacity、changeover 数据流 | M4 流程图 | pending |
| 6/2 | MLE-2 | 多产线需求定义 | 明确一个 SKU 可映射多条产线的业务规则 | 规则说明 | pending |
| 6/3 | MLE-2 | 数据模型设计 | 设计 SKU-line candidate table、priority、rate | 数据结构草案 | pending |
| 6/4 | MLE-2 | 约束识别 | 识别 capacity、changeover、reliability、available_date 约束 | 约束清单 | pending |
| 6/5 | MLE-2 | 测试场景设计 | 设计单 SKU 多线、产能不足、换产冲突用例 | M4 test cases | pending |
| 6/8 | MLE-2 | 多线候选生成 | 实现 SKU 可选产线展开 | candidate lines | pending |
| 6/9 | MLE-2 | 产能分配逻辑 | 按产能、rate、priority 分配生产量 | capacity allocation | pending |
| 6/10 | MLE-2 | 换产逻辑适配 | 多线情况下处理 changeover matrix | changeover 适配 | pending |
| 6/11 | MLE-2 | 可靠性适配 | 保持 production reliability 逻辑一致 | reliability 验证 | pending |
| 6/12 | MLE-2 | 单元测试 | 覆盖多产线分配核心函数 | M4 unit tests | pending |
| 6/15 | MLE-2 | 集成 M4 | 接入 production_runner.py 主流程 | M4 集成代码 | pending |
| 6/16 | MLE-2 | 边界场景 | 测试无产线、单产线、多产线、产能为 0 | 边界测试结果 | pending |
| 6/17 | CDP-1 | M4 长函数拆分 | 拆分 M4 生产排程中的 1000+ 行/嵌套函数 | M4 refactor | pending |
| 6/18 | MLE-2 | 性能测试 | 对比 M4 优化前后耗时 | M4 benchmark | pending |
| 6/19 | MLE-2 | 回归修复 | 修复输出差异、排序差异、类型差异 | 修复记录 | pending |
| 6/22 | MLE-2 | 下游兼容 | 验证 M4 输出传给 orchestrator / M5 不变 | 兼容性报告 | pending |
| 6/23 | MLE-2 | 验证报告 | 整理多产线用例与结果 | M4 验证报告 | pending |
| 6/24 | MLE-2 | 代码审查 | Review 多产线实现、命名、类型、性能 | review notes | pending |
| 6/25 | MLE-2 | 缺陷修复 | 修复 review 问题 | 修复 PR | pending |
| 6/26 | MLE-2 | Phase 2 验收 | M4 多产线逻辑冻结 | M4 验收记录 | pending |

### 4.3 7 月：DataFrame 优化 + Validation

| 日期 | 任务 ID | 当日任务 | 具体工作 | 交付物 | 状态 |
|---|---|---|---|---|---|
| 6/29 | MLE-3 | filtered df 审计 | 搜索 chained indexing、filtered df 再访问模式 | 问题清单 | pending |
| 6/30 | MLE-3 | 访问规范制定 | 明确 `.loc`、`.copy()`、索引缓存规范 | DataFrame 规范 | pending |
| 7/1 | MLE-3 | 修复 M1/M3 | 修复 M1/M3 中 filtered dataframe 风险点 | 修复提交 | pending |
| 7/2 | MLE-3 | 修复 M4/M5 | 修复 M4/M5 中 filtered dataframe 风险点 | 修复提交 | pending |
| 7/3 | MLE-3 | 回归验证 | 确认输出无变化 | diff 报告 | pending |
| 7/6 | VAL-1 | 输入校验梳理 | 整理所有输入表 schema、必填字段 | schema 清单 | pending |
| 7/7 | VAL-1 | KNIME 逻辑迁移 | 将已有 KNIME 校验逻辑映射到 Python | 迁移映射表 | pending |
| 7/8 | VAL-1 | 实现基础校验 | 空值、类型、日期、数值范围检查 | input validator | pending |
| 7/9 | VAL-1 | 实现业务校验 | material/location/network/leadtime 一致性检查 | business rules | pending |
| 7/10 | VAL-1 | 输入校验测试 | 构造坏数据，验证报错清晰 | validation test | pending |
| 7/13 | MLE-4 | append 审计 | 搜索逐日 append / 循环 concat | append 清单 | pending |
| 7/14 | MLE-4 | 设计替代方案 | list 累积、一次 concat、NumPy 预分配 | 优化方案 | pending |
| 7/15 | MLE-4 | 改造 M1 append | 替换 M1 中逐日 append 模式 | M1 append-free | pending |
| 7/16 | MLE-4 | 改造 M4/M5 append | 替换 M4/M5 中高频 append | M4/M5 append-free | pending |
| 7/17 | MLE-4 | 性能验证 | 对比 append 改造前后耗时和内存 | 性能报告 | pending |
| 7/20 | VAL-2 | 中间表定义 | 定义 M1/M3/M4/M5 中间输出检查点 | checkpoint 清单 | pending |
| 7/21 | VAL-2 | 中间校验实现 | 检查负数、重复 key、缺失列、异常日期 | intermediate validator | pending |
| 7/22 | VAL-2 | 集成校验 | 将校验接入每日流程或独立脚本 | validation runner | pending |
| 7/23 | VAL-2 | 校验报告 | 输出 Excel/CSV validation report | validation report | pending |
| 7/24 | MLE/VAL | 7 月验收 | 汇总 DataFrame 优化与 validation 结果 | 7 月验收记录 | pending |

### 4.4 8 月：E2E 测试网络 + 回测文档

| 日期 | 任务 ID | 当日任务 | 具体工作 | 交付物 | 状态 |
|---|---|---|---|---|---|
| 7/27 | TEST-1 | E2E 需求定义 | 明确最小网络规模、覆盖模块、成功标准 | E2E 设计 | pending |
| 7/28 | TEST-1 | 网络数据设计 | 设计 plant/DC/customer、BOM、leadtime、capacity | mock config | pending |
| 7/29 | TEST-1 | 构建基础配置 | 生成最小可运行 Excel/DB 配置 | test input | pending |
| 7/30 | TEST-1 | 单日 smoke | 跑 1 天全链路，修复阻断问题 | smoke result | pending |
| 7/31 | TEST-1 | 多日 smoke | 跑 7 天全链路 | 7-day result | pending |
| 8/3 | TEST-1 | E2E 网络扩展 | 增加多 SKU、多地点、多产线场景 | extended network | pending |
| 8/4 | TEST-1 | M1/M3 验证 | 检查订单、供需日志、净需求流转 | M1/M3 report | pending |
| 8/5 | TEST-1 | M4/M5 验证 | 检查生产计划、调拨计划 | M4/M5 report | pending |
| 8/6 | TEST-1 | M6/库存验证 | 检查物流、GR、库存平衡 | M6/inventory report | pending |
| 8/7 | TEST-1 | E2E 修复 | 修复端到端差异 | 修复记录 | pending |
| 8/10 | TEST-1 | 30 天测试 | 跑 30 天 E2E | 30-day output | pending |
| 8/11 | TEST-1 | 结果分析 | 检查库存、service、cuts、deployments | E2E analysis | pending |
| 8/12 | TEST-1 | 自动化脚本 | 固化 E2E run + compare 脚本 | e2e script | pending |
| 8/13 | TEST-1 | 报告整理 | 输出 E2E 测试报告 | E2E report | pending |
| 8/14 | TEST-1 | E2E 验收 | E2E 网络冻结 | E2E 验收记录 | pending |
| 8/17 | DOC-2 | 回测资料整理 | 整理 baby case 已完成回测结果 | 回测资料包 | pending |
| 8/18 | DOC-2 | KPI 定义 | 明确 simulated vs actual KPI 列表 | KPI 字典 | pending |
| 8/19 | DOC-2 | 差异分析 | 分析误差、偏差、异常月份 | 差异表 | pending |
| 8/20 | DOC-2 | 图表生成 | 输出 KPI 对比图 | charts | pending |
| 8/21 | DOC-2 | 回测初稿 | 写回测报告初稿 | backtesting draft | pending |
| 8/24 | DOC-2 | 业务解释 | 补充偏差原因、限制条件 | explanation | pending |
| 8/25 | DOC-2 | 证据链 | 链接数据源、脚本、输出文件 | evidence pack | pending |
| 8/26 | DOC-2 | Review | 内部审查回测报告 | review notes | pending |
| 8/27 | DOC-2 | 修订 | 修订图表和结论 | revised report | pending |
| 8/28 | DOC-2 | 回测验收 | 回测文档定稿 | final backtesting report | pending |

### 4.5 9 月：Trade-off 分析 + PG 数据分布

| 日期 | 任务 ID | 当日任务 | 具体工作 | 交付物 | 状态 |
|---|---|---|---|---|---|
| 8/31 | DOC-3 | 指标框架 | 定义 cost/cash/service 三类 KPI | KPI framework | pending |
| 9/1 | DOC-3 | 成本指标 | 设计生产、库存、物流、缺货成本口径 | cost metric | pending |
| 9/2 | DOC-3 | cash 指标 | 设计库存占用、现金周转相关指标 | cash metric | pending |
| 9/3 | DOC-3 | service 指标 | 设计 fill rate、cut rate、OTIF 类指标 | service metric | pending |
| 9/4 | DOC-3 | 场景设计 | 设计 baseline/high service/low cost 等场景 | scenario set | pending |
| 9/7 | DOC-3 | 跑场景 | 执行多个 scenario | scenario outputs | pending |
| 9/8 | DOC-3 | 汇总结果 | 汇总成本、现金、服务指标 | result table | pending |
| 9/9 | DOC-3 | trade-off 分析 | 识别 Pareto 或关键拐点 | trade-off analysis | pending |
| 9/10 | DOC-3 | 图表 | 生成 trade-off 曲线和矩阵 | charts | pending |
| 9/11 | DOC-3 | 报告定稿 | 输出 trade-off 报告 | final report | pending |
| 9/14 | DOC-4 | PG 数据盘点 | 确认 PG 数据源、字段、时间范围 | data inventory | pending |
| 9/15 | DOC-4 | demand 分布 | 分析需求均值、方差、长尾、季节性 | demand profile | pending |
| 9/16 | DOC-4 | inventory 分布 | 分析库存分布、缺货、异常库存 | inventory profile | pending |
| 9/17 | DOC-4 | leadtime 分布 | 分析 lead time 均值、波动、异常值 | leadtime profile | pending |
| 9/18 | DOC-4 | production 分布 | 分析产能、产量、可靠性分布 | production profile | pending |
| 9/21 | DOC-4 | 异常检测 | 标记离群点、缺失、异常时间段 | anomaly list | pending |
| 9/22 | DOC-4 | 分布图表 | 输出 histogram/boxplot/time series | charts | pending |
| 9/23 | DOC-4 | 建模建议 | 说明分布对仿真参数的影响 | modeling notes | pending |
| 9/24 | DOC-4 | 报告初稿 | 写 PG 数据分布报告 | draft report | pending |
| 9/25 | DOC-4 | 报告定稿 | 完成 PG 数据分析文档 | final PG report | pending |

### 4.6 10 月上：项目范围、成功标准、最终验收

| 日期 | 任务 ID | 当日任务 | 具体工作 | 交付物 | 状态 |
|---|---|---|---|---|---|
| 9/28 | DOC-1 | Scope 定义 | 明确本项目包含/不包含哪些模块和能力 | scope draft | pending |
| 9/29 | DOC-1 | Problem 定义 | 明确要解决的业务/技术问题 | problem statement | pending |
| 9/30 | DOC-1 | 成功标准 | 定义性能、准确性、可维护性、验证标准 | success criteria | pending |
| 10/1 | DOC-1 | 文档缓冲 | 国庆/异步整理材料 | doc buffer | pending |
| 10/2 | DOC-1 | 文档缓冲 | 补充图表、引用、术语表 | doc buffer | pending |
| 10/5 | DOC-1 | 文档缓冲 | 整合所有报告链接 | doc pack | pending |
| 10/6 | DOC-1 | 文档缓冲 | 检查范围与交付物一致性 | checklist | pending |
| 10/7 | DOC-1 | 文档缓冲 | 准备评审版本 | review version | pending |
| 10/8 | DOC-1 | 正式评审 | 项目范围、问题、成功标准评审 | review notes | pending |
| 10/9 | DOC-1 | 修订定稿 | 根据评审意见修订 | final scope doc | pending |
| 10/12 | ALL | 最终集成验收 | 检查代码、验证、测试、文档全部完成 | final checklist | pending |
| 10/13 | ALL | 风险关闭 | 关闭 open risks / known issues | risk closure | pending |
| 10/14 | ALL | 交付包整理 | 打包代码、报告、测试结果、说明 | delivery package | pending |
| 10/15 | ALL | 项目 sign-off | 最终汇报与签收 | sign-off record | pending |

---

## 5. 项目跟踪汇总表

| ID | 工作流 | 任务 | 计划开始 | 计划结束 | 工期 | 当前状态 | 依赖 | 验收标准 |
|---|---|---|---:|---:|---:|---|---|---|
| P0 | 启动 | 基线、风险、计划 | 5/6 | 5/9 | 3d | pending | 无 | baseline + risk list 完成 |
| MLE-1 | 工程 | M1 订单生成逻辑 | 5/11 | 5/29 | 15d | in-progress | P0 | `demand_planning_refactor` 已实现，待主入口切换、OrderLog/SupplyDemandLog 回归和性能确认 |
| MLE-2 | 工程 | M4 多产线排程 | 6/1 | 6/26 | 20d | pending | P0 | 多产线 case 通过，输出兼容 |
| MLE-3 | 工程 | filtered dataframe 优化 | 6/29 | 7/3 | 5d | pending | MLE-1/MLE-2 | 无 chained indexing 风险 |
| VAL-1 | 校验 | 输入数据校验 | 7/6 | 7/10 | 5d | pending | P0 | bad data 可被准确拦截 |
| MLE-4 | 工程 | 避免逐日 append | 7/13 | 7/17 | 5d | pending | MLE-3 | 高频 append 清零或可解释 |
| VAL-2 | 校验 | 中间数据校验 | 7/20 | 7/24 | 5d | pending | VAL-1 | 每日中间表校验报告 |
| TEST-1 | 测试 | E2E 网络构建 | 7/27 | 8/14 | 15d | pending | MLE/VAL | 30 天 E2E 成功跑通 |
| DOC-2 | 文档 | 历史回测文档 | 8/17 | 8/28 | 10d | pending | TEST-1 | simulated vs actual KPI 报告 |
| DOC-3 | 文档 | trade-off 分析 | 8/31 | 9/11 | 10d | pending | TEST-1 | cost/cash/service 报告 |
| DOC-4 | 文档 | PG 数据分布 | 9/14 | 9/25 | 10d | pending | P0 | PG 数据分布报告 |
| DOC-1 | 文档 | scope/problem/success criteria | 9/28 | 10/9 | 10d | pending | 所有阶段 | 最终范围与成功标准 |
| FINAL | 收尾 | 最终验收 | 10/12 | 10/15 | 4d | pending | 全部任务 | sign-off 完成 |

---

## 6. 验收标准

| 任务 | 必须满足的验收标准 |
|---|---|
| M1 周→日订单生成 | 总订单量守恒；AO/normal 逻辑一致；随机数可复现；M3/M5 不受影响 |
| M4 多产线排程 | 一个 SKU 可多线生产；产能不超；换产逻辑正确；边界 case 通过 |
| filtered DataFrame 优化 | 无 chained assignment warning；无视图/副本不确定行为 |
| 避免逐日 append | 热点路径不再循环 append；性能 benchmark 有改善 |
| 输入校验 | 缺字段、错类型、非法范围、网络不一致均可识别 |
| 中间校验 | 每日关键中间表可校验；错误定位到表/字段/日期 |
| E2E 测试网络 | 30 天仿真稳定跑通；库存平衡；关键 KPI 可解释 |
| 回测文档 | simulated vs actual KPI 有图表、有偏差解释 |
| trade-off 分析 | cost/cash/service 指标定义清晰，场景结果可复现 |
| PG 分布分析 | 主要字段分布、异常、建模影响均记录 |
| scope 文档 | 范围、问题、成功标准、非目标明确 |
| 最终验收 | 代码、验证、测试、文档、风险关闭全部完成 |

---

## 7. 高性能保障机制

| 机制 | 说明 |
|---|---|
| 基线先行 | 每项优化前先记录原始耗时、行数、输出 hash |
| 向量化优先 | DataFrame 构造使用批量生成，不逐日 append |
| 缓存重复计算 | 周→日预测、静态配置、索引映射只算一次 |
| Numba/NumPy 加速 | 对存在顺序依赖的循环，用 JIT/数组方式加速 |
| 回归对比 | 每项改动后对比 OrderLog、SupplyDemandLog、ShipmentLog |
| 性能门禁 | 优化后不得慢于基线；核心模块需记录平均耗时 |
| fallback 开关 | 高风险优化必须保留旧逻辑开关回退 |
