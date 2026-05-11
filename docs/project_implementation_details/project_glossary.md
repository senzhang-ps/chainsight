# ChainSight 项目专有名词解释

更新时间：2026-05-08

本文作为 `docs/project_implementation_details/` 下的项目级术语表，统一解释 ChainSight 仿真、配置、模块、输出和业务缩写中的专有名词。若具体模块文档中出现同名术语，以本文解释作为默认口径；模块文档可在局部场景中补充更细规则。

## 1. 全局仿真术语

| 术语 | 英文 / 缩写 | 解释 |
|---|---|---|
| ChainSight | ChainSight | 当前供应链计划与执行仿真项目。核心目标是在多模块链路中模拟订单、生产、调拨、物流和补货。 |
| 仿真日 | `simulation_date` | 当前正在运行的业务日期。多数模块以天为单位执行一次；在 M1 `OrderLog` 中同时表示订单下单日。 |
| 仿真开始日 | `start_date` | 整个仿真周期的第一天，常用于把 `week` 映射为实际日期。 |
| 配置字典 | `config_dict` | 从 Excel 或数据库加载后的配置集合。key 通常是 sheet 名，例如 `M1_DemandForecast`。 |
| 文件模式 | File Mode | 从本地 Excel 配置读取输入，并把模块输出写成本地文件。 |
| DB 模式 | Database Mode | 从数据库读取配置或状态，并把模块输出写入数据库表。 |
| Orchestrator | Orchestrator | 全局状态编排器。负责提供库存视图、生产入库、物流到货、open deployment 等跨模块状态。 |
| Checkpoint | Checkpoint | DB 模式下的中间状态保存点，用于断点恢复或批量写入控制。 |
| Summary | Summary | 每日仿真后的汇总输出，用于报表或数据库对账。 |
| Schema | Schema | 表结构定义，包括列名、类型、必填列和业务约束。 |
| Contract | Contract | 模块之间约定的输入/输出格式。改代码时应优先保持下游 contract 不变。 |

## 2. 模块名称

| 术语 | 英文 / 缩写 | 解释 |
|---|---|---|
| M1 | Demand Planning | 需求规划模块。负责需求 forecast、订单生成、客户发货、缺货和供需日志。 |
| M3 | MRP Planning | MRP 补货 / 净需求模块。根据库存、需求、供应和安全库存计算补货或净需求。 |
| M4 | Production Planning | 生产排程模块。根据产线、产能、换产和需求安排生产计划。 |
| M5 | Deployment Planning | 调拨规划模块。决定网络节点之间的库存调拨计划。 |
| M6 | Logistics Execution | 物流执行模块。将调拨计划转为车辆、装车、发运、到货和物流约束结果。 |
| Global | Global Config | 跨模块共享配置，例如网络、空间容量、提前期和需求优先级。 |

## 3. M1 需求规划术语

| 术语 | 英文 / 缩写 | 解释 |
|---|---|---|
| 需求预测 | Demand Forecast | 未来需求输入。项目中常见配置表为 `M1_DemandForecast`。 |
| 周度预测 | Weekly Forecast | 以 `week` 为粒度的需求预测，通常包含 `material`、`location`、`week`、`quantity`。 |
| 日度预测 | Daily Forecast | 以具体 `date` 为粒度的需求预测，通常由周度预测拆分得到。 |
| Forecast Error | Forecast Error | 预测误差配置，对应 `M1_ForecastError`。用于在生成订单时模拟需求波动。 |
| CoV | Coefficient of Variation | 变异系数，等于标准差除以均值，即 `std / mean`。在代码中通常体现为 `error_std_percent`，用于计算正态分布抽样标准差：`std_qty = mean_qty * error_std_percent`。 |
| `error_std_percent` | Error Standard Deviation Percent | 预测误差百分比。虽然字段名包含 `percent`，实际计算中按比例值使用，例如 `0.2` 表示标准差为均值的 20%。 |
| AO | Advance Order | 提前订单。重构版 M1 先在周级拆出 AO 周量，再按有效下单日拆到每日，最后用 `advance_days` 得到需求日期。 |
| Normal Order | Normal | 普通订单。通常表示扣除 AO 后的剩余订单，`advance_days = 0`。 |
| AOConfig | `M1_AOConfig` | AO 配置表。常见字段包括 `material`、`location`、`advance_days`、`ao_percent`。 |
| `ao_percent` | AO Percent | AO 占比。用于把需求或订单基线拆成提前订单份额。 |
| `advance_days` | Advance Days | 提前天数。AO 订单中，`OrderLog.date = simulation_date + advance_days`；normal 订单为 0。 |
| OrderCalendar | `M1_OrderCalendar` | 下单日历。重构版 M1 用它决定本周周订单可拆到哪些有效下单日，并过滤每日是否输出新增订单。 |
| DPS | Demand Planning Split | 需求地点拆分。根据 `M1_DPSConfig` 将部分需求从原 location 拆到 `dps_location`。 |
| Supply Choice | `M1_SupplyChoiceConfig` | 供给选择调整配置。用于调整供需日志或供应视图中的需求数量。 |
| weekly_for_orders | Weekly Order Baseline | DPS 后仍保留 `week` 的 forecast，用于 `generate_weekly_orders()`。 |
| daily_for_consumption | Daily Consumption Baseline | `weekly_for_orders` 拆成日级后的 forecast，用于订单生成返回值兼容和消耗逻辑。 |
| daily_for_supply | Daily Supply Baseline | DPS + SupplyChoice 后拆成日级的 forecast，用于 `SupplyDemandLog`。 |
| 周级订单量 | Weekly Order Quantity | 在一周粒度生成的订单数量，后续再拆到日级有效下单日。 |
| 守恒基线 | Conservation Baseline | 周级 AO / normal 数量作为拆日基线，要求拆到各日后的数量合计等于周级数量。 |
| 订单消耗 | Forecast Consumption | 订单生成后从 forecast 中扣减对应数量，用于后续供需日志或剩余需求展示。 |

## 4. M1 输出术语

| 术语 | 英文 / 缩写 | 解释 |
|---|---|---|
| OrderLog | Order Log | M1 订单输出。常见字段为 `date`、`material`、`location`、`demand_type`、`quantity`、`simulation_date`、`advance_days`。 |
| `OrderLog.date` | Requirement Date | 订单需求日期或到期日期。AO 通常为下单日加 `advance_days`；normal 通常等于下单日。 |
| `OrderLog.simulation_date` | Order Date | 订单生成日 / 下单日。重构版 M1 中该日期必须来自 `M1_OrderCalendar` 的有效下单日。 |
| ShipmentLog | Shipment Log | 客户发货输出。记录当日实际发出的客户订单数量。 |
| CutLog | Cut Log | 缺货 / 未满足客户订单输出。通常为订单需求量减去可发货量。 |
| SupplyDemandLog | Supply Demand Log | 供需日志。记录未来窗口内 forecast 被订单消耗后的剩余需求或供需视图。 |
| `demand_type` | Demand Type | 需求类型。M1 订单中常见值为 `AO`、`normal`；发货中可能为 `customer`。 |
| `quantity` | Quantity | 数量字段。不同表中含义取决于上下文，例如需求量、订单量、发货量、缺货量或库存量。 |

## 5. 库存、供应与执行术语

| 术语 | 英文 / 缩写 | 解释 |
|---|---|---|
| Material | Material | 物料编码。项目中一般需要按字符串处理，避免前导零或数字格式问题。 |
| Location | Location | 地点 / 节点编码。用于表示工厂、仓库、客户节点或网络节点。 |
| ML 粒度 | Material-Location Granularity | `material + location` 的组合粒度，是多数需求、库存、订单、发货计算的基础维度。 |
| SKU | Stock Keeping Unit | 库存计量单位。项目中常与 material 概念接近，但 SKU 更偏业务库存单位。 |
| Inventory | Inventory | 库存。可分为日初库存、可用库存、在途库存、期末库存等。 |
| Beginning Inventory | Beginning Inventory | 日初库存。仿真日开始时某个 material/location 的库存数量。 |
| Available Inventory | Available Inventory | 可用库存。通常由日初库存、当日生产入库、当日物流到货等组成。 |
| SOH | Stock On Hand | 在手库存。M5 中常用于表示某节点当前可用或账面库存。 |
| GR | Goods Receipt | 收货 / 入库。生产完成或物流到货后进入库存的动作。 |
| GI | Goods Issue | 发货 / 出库。从库存中发出货物的动作。若文档或代码中出现 GI，通常与库存扣减相关。 |
| Open Deployment | Open Deployment | 已计划但尚未完成的调拨或在途计划。 |
| In-Transit | In Transit | 在途库存或在途运输，已发出但尚未到达接收节点。 |

## 6. 计划与网络术语

| 术语 | 英文 / 缩写 | 解释 |
|---|---|---|
| Network | Network | 供应链网络关系，通常定义 sending / receiving 节点之间是否可以流动。 |
| Sending | Sending Location | 发出节点。 |
| Receiving | Receiving Location | 接收节点。 |
| Lead Time | Lead Time | 提前期。表示从发出、生产、运输到可用的时间延迟。 |
| PDT | Production Duration Time | 生产持续时间或生产处理时间。具体含义以 `Global_LeadTime` 配置口径为准。 |
| OTD | Order To Delivery | 下单到交付的时间或相关提前期字段。具体以 `Global_LeadTime` 配置口径为准。 |
| MCT | Manufacturing Cycle Time | 制造周期时间。具体以 `Global_LeadTime` 配置口径为准。 |
| Safety Stock | Safety Stock | 安全库存。M3 中用于判断补货或净需求的库存下限。 |
| Net Demand | Net Demand | 净需求。通常由需求、库存、在途、供应和安全库存综合计算得到。 |
| MRP | Material Requirements Planning | 物料需求计划。用于计算补货、生产或调拨需求。 |

## 7. M4 生产排程术语

| 术语 | 英文 / 缩写 | 解释 |
|---|---|---|
| Production Plan | Production Plan | 生产计划输出。记录某日、某产线、某物料的计划生产数量。 |
| Line | Production Line | 产线。生产排程中的基本能力单位。 |
| Line Capacity | Line Capacity | 产线产能。表示某产线在某时间窗口内可生产的最大数量或时长。 |
| Changeover | Changeover | 换产。从一个物料切换到另一个物料时产生的时间或能力损失。 |
| Changeover Matrix | Changeover Matrix | 换产矩阵。定义 from material 到 to material 的换产关系。 |
| Changeover Definition | Changeover Definition | 换产定义。定义某类换产在某产线上的具体时间或规则。 |
| Production Reliability | Production Reliability | 生产可靠性。用于模拟产线实际产出与计划产出的偏差或约束。 |
| Capacity Exceed | Capacity Exceed | 产能超限记录。表示生产计划超出可用产能。 |

## 8. M5 调拨规划术语

| 术语 | 英文 / 缩写 | 解释 |
|---|---|---|
| Deployment | Deployment | 调拨。将库存从一个节点计划移动到另一个节点。 |
| Deployment Plan | Deployment Plan | 调拨计划输出。作为 M6 物流执行的主要输入。 |
| Push / Pull | Push / Pull | 推式 / 拉式调拨策略。Push 偏按供应主动分配，Pull 偏按需求拉动补货。 |
| Unfulfilled Log | Unfulfilled Log | 未满足调拨或需求记录。表示调拨规划阶段无法满足的数量。 |
| Stock On Hand Log | SOH Log | 在手库存日志。记录 M5 规划过程中的库存状态。 |

## 9. M6 物流执行术语

| 术语 | 英文 / 缩写 | 解释 |
|---|---|---|
| Logistics Execution | Logistics Execution | 物流执行。将调拨计划转化为实际车辆、装车、发运和到货计划。 |
| Delivery Plan | Delivery Plan | 到货 / 配送计划。记录发运后预计到达接收节点的计划。 |
| Truck | Truck | 车辆或运输资源。 |
| Truck Usage | Truck Usage | 车辆使用记录。用于统计每日车辆占用、线路和运输量。 |
| Truck Type Specs | Truck Type Specs | 车型规格。定义车辆容量、限制或类型属性。 |
| Truck Capacity Plan | Truck Capacity Plan | 车辆容量计划。定义某日期、线路或节点可用运输能力。 |
| Delivery Delay Distribution | Delivery Delay Distribution | 到货延迟分布。用于模拟物流延迟。 |
| MDQ | Minimum Delivery Quantity | 最小交付量或最低发运量。若调拨量低于 MDQ，可能触发不满足、合并或 bypass 规则。 |
| MDQ Bypass Rule | MDQ Bypass Rule | 绕过最小交付量限制的规则。用于特定业务例外。 |
| Unsatisfied MDQ | Unsatisfied MDQ | 未满足最小交付量约束的记录。 |

## 10. 数据处理与测试术语

| 术语 | 英文 / 缩写 | 解释 |
|---|---|---|
| DataFrame | DataFrame | pandas 表格数据结构，是项目中主要的内存数据载体。 |
| Vectorized | Vectorized | 向量化处理。用批量数组或 DataFrame 操作替代逐行循环，以提升性能。 |
| Filtered DataFrame | Filtered DataFrame | 经过条件筛选后的 DataFrame。项目计划中要求减少重复 filtered DataFrame 访问以优化性能。 |
| Append-Free | Append-Free | 避免逐日或逐行 `append`，改用列表累积后一次 `concat` 或预分配。 |
| Validation | Validation | 校验。包括输入配置校验、中间结果校验和输出 contract 校验。 |
| E2E | End-to-End | 端到端测试。覆盖从配置加载到多模块执行再到输出校验的完整链路。 |
| Smoke Test | Smoke Test | 冒烟测试。用于快速确认主流程能否跑通。 |
| Regression Test | Regression Test | 回归测试。用于确认修改没有破坏既有行为。 |

## 11. 常见配置表速查

| 配置表 | 所属模块 | 作用 |
|---|---|---|
| `Global_Network` | Global | 网络路径和节点关系配置。 |
| `Global_SpaceCapacity` | Global | 空间或仓储容量配置。 |
| `Global_LeadTime` | Global | 节点间或业务过程的提前期配置。 |
| `Global_DemandPriority` | Global | 需求优先级配置。 |
| `M1_DemandForecast` | M1 | 需求预测输入。 |
| `M1_ForecastError` | M1 | 预测误差 / CoV 配置。 |
| `M1_OrderCalendar` | M1 | 下单日历。 |
| `M1_AOConfig` | M1 | 提前订单配置。 |
| `M1_DPSConfig` | M1 | 需求地点拆分配置。 |
| `M1_SupplyChoiceConfig` | M1 | 供给选择调整配置。 |
| `M3_SafetyStock` | M3 | 安全库存配置。 |
| `M4_MaterialLocationLineCfg` | M4 | 物料、地点、产线映射配置。 |
| `M4_LineCapacity` | M4 | 产线产能配置。 |
| `M4_ChangeoverMatrix` | M4 | 换产关系配置。 |
| `M4_ChangeoverDefinition` | M4 | 换产时间或规则配置。 |
| `M4_ProductionReliability` | M4 | 生产可靠性配置。 |
| `M5_PushPullModel` | M5 | 推拉式调拨策略配置。 |
| `M5_DeployConfig` | M5 | 调拨规划配置。 |
| `M6_TruckReleaseCon` | M6 | 车辆释放或运输约束配置。 |
| `M6_MaterialMD` | M6 | 物流物料主数据。 |
| `M6_DeliveryDelayDistribution` | M6 | 到货延迟分布配置。 |
| `M6_MDQBypassRules` | M6 | 最小交付量绕过规则。 |
| `M6_TruckTypeSpecs` | M6 | 车型规格配置。 |
| `M6_TruckCapacityPlan` | M6 | 车辆容量计划配置。 |
