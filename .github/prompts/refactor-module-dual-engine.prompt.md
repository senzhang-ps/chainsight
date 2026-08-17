---
name: "重构模块为 pandas/Polars 双引擎"
description: "用于重构业务模块：先建立 pandas 正确性基线，再实现独立 Polars 引擎并完成真实数据性能验证"
argument-hint: "模块名称、目标目录及重构范围，例如：物流执行模块，src/modules/logistics_execution"
agent: "agent"
---

重构 $ARGUMENTS 所指定的模块，使其遵循本仓库已验证的“双引擎 + 显式门面编排”模式。以 M1、M4、M5 的重构实现为参考，但先理解目标模块现状和既有测试，不要机械复制代码。

## 目标架构

1. `integration_refactor.py`（或该模块的等价入口）只承担生命周期、公开门面和编排职责。
   - 公开 `prepare()` 与 `run()` 必须独立；`run()` 不得隐式调用 `prepare()`。
   - `prepare()` 只编排静态数据加载、标准化、验证、网络/索引等静态状态构建。
   - `run()` 只编排单日动态输入、计算步骤、输出收尾；不得包含 DataFrame 业务变换。
   - 每一个具有业务含义的阶段应有显式、可测试的门面方法；不要在 backend 中隐藏一个覆盖整个 `prepare()` 或 `run()` 的大包装方法。
   - facade 不得按引擎分支，也不得承担 pandas/Polars DataFrame 计算。

2. `backends.py` 定义统一的 backend 接口，并提供独立的 `_PandasBackend` 与 `_PolarsBackend`。
   - 两个 backend 都实现完整业务逻辑；Polars backend 不得继承、调用或通过转换委托 pandas backend 完成计算。
   - 在模块输入、输出以及与 `StateContext` 交界处保持既有 pandas 公共契约；Polars 仅在内部计算阶段使用。
   - 保持既有的业务输出 schema、列含义、日期语义、排序/稳定性和错误行为，除非本次需求明确改变它们。

3. 动态状态必须通过 `StateContext` 读取和写回。禁止为追求性能跳过状态更新或改变多日状态推进语义。

## 执行顺序：正确性优先

严格按下列顺序推进，未完成上一阶段不得提前宣称 Polars 提速成果：

### 阶段 A：理解与拆分

- 阅读目标模块、调用方、状态上下文、现有测试，以及 M1/M4/M5 的相关实现。
- 列出静态准备阶段、单日计算阶段、输入/输出表、状态读写点、顺序敏感规则和性能热点。
- 先制定最小重构计划；保持修改范围聚焦，避免重排无关代码或改变公共 API。

### 阶段 B：先完成 pandas 重构基线

- 先将全部业务计算迁移至 `_PandasBackend`，让 facade 只做显式编排。
- 为 pandas 增加或更新单元、契约、集成和真实数据测试。
- 验证 refactor pandas 与原实现的业务输出、状态推进和异常行为；有差异时先定位并修复 pandas，不能用 Polars 绕过。
- 若真实多日回放昂贵，缓存通过验证的 pandas oracle、必要输入快照和日初 `StateContext` 视图。缓存必须带版本、配置/run 标识和失效方式。

### 阶段 C：实现独立 Polars 引擎

- 在 pandas 基线已通过后，逐个门面步骤实现 `_PolarsBackend`。
- 优先使用原生 Polars 表达式、join、group_by、窗口与惰性执行；仅对天然顺序敏感的小范围逻辑使用 Python 循环。
- 谨慎处理 pandas/Polars 的日期、空值、字符串/分类、整数/浮点、重复键、join 保留行和排序语义差异。
- 不得为了让测试通过而减少业务数据、静默丢行、降低精度或使用 pandas 作为 Polars 的计算后备。

### 阶段 D：逐步严格 parity

- 将 pandas 作为 correctness oracle。每个仿真日比较业务输出的：
  1. 业务键集合；
  2. 匹配行字段值；
  3. 浮点精度差异；
  4. schema 与行数；
  5. 状态推进结果。
- 首先比较首个不一致日；使用缓存的 pandas 日初 Context，分别执行 pandas 和 Polars 的相同显式门面步骤。
- 对 `load_daily_inputs`、active network、route parameters、supply ledger、分层计划、push、空间约束及最终输出建立步骤级检查点，找出首个不一致的中间表后再修复。
- 先通过“隔离日初状态”逐日 parity，再通过“连续多日状态传播” strict parity。
- 测试应对零差异进行断言，而不是只写 JSON 报告后仍让差异静默通过。

### 阶段 E：性能验证与报告

- 仅在全量真实数据 strict parity 通过后，才运行并报告性能测试。
- 统一计时口径：明确是否包括 `prepare()`、`run()`、数据库读取、状态写回、报告写入；同一对比中必须一致。
- 使用真实的固定配置和多日历史输入，至少记录 pandas 与 Polars 的总耗时、日均耗时和加速比；若有 legacy，也保留 legacy/pandas/Polars 三方历史表。
- 性能报告以性能为主体，简洁说明“已通过严格一致性”这一可比前提即可；不要用未经严格 parity 的旧数据得出等价性能结论。

## 交付与验证要求

- 在每次修改后检查相关编辑器错误，并运行最小相关测试；完成后运行目标模块的回归、严格 parity 和性能测试。
- 测试或真实数据执行较慢时，先复用缓存 pandas oracle，避免反复执行完整 pandas 五日回放。
- 将测试产物写入模块对应的 `outputs/` 目录，报告应区分历史性能与当前严格等价性能。
- 完成时简要汇报：修改的架构边界、pandas 验证结果、Polars strict parity 结果、性能数据、缓存位置与重建条件。
- 若缺少真实输入、测试基础设施或无法验证的外部依赖，明确说明阻塞点；不要伪造性能或一致性结论。
