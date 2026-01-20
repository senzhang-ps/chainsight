## 一、模块定位与目标

- 基于每日快照数据与编排视图，在多层网络中自下而上收集需求、分配可用库存、生成跨节点调拨计划，并在最后应用接收端仓容限制。
- 与 Module 3 的窗口与标识口径保持一致：Plant 与 DC 采用相同的提前期公式与标识符标准化，支持层级推进与缺口上传。

## 二、数据输入与标准化

### 静态配置

- **Network**：网络关系与有效期 `eff_from/eff_to`（仅使用 `sim_date` 有效的配置参与计算）
- **LeadTime**：`PDT`、`GR`、`MCT`（按 `sending+receiving`）
- **M4_MaterialLocationLineCfg**：`PTF`、`LSK`（按 `material+location`，大小写兼容；未命中默认 `PTF=0`、`LSK=1`）
- **SafetyStock**：安全库存目标（按 `date`）
- **DeployConfig**：路径参数（按 `material+sending` 维护 `moq/rv`；分组应用时按 `material+sending+receiving` 命中）
- **DemandPriority**：`demand_element -> priority` 映射（缺失时自动补 AO=1、normal=2、其他=9）
- **ReceivingSpace**：接收端 `max_qty` 当日限额（按 `receiving, date`）

### 每日快照（当日视图）

- **SupplyDemandLog**：forecast 与其他本地需求（窗口参与统计）
- **OrderLog**：AO/normal，窗口 `[sim_date, horizon_end]`
- **TodayShipment**：客户发货（`date==sim_date`）

### Orchestrator 动态数据（当日视图）

- Beginning Inventory、In-Transit、Delivery GR、Open Deployment（含未发运调拨）、生产 GR/计划（历史生产优先，必要时回退 Module4）

### 标识符标准化

- `material`、`location`、`sending`、`receiving`、`sourcing` 统一为字符串；地点左补零至 4 位；缺失置空字符串。

## 三、网络层级识别（assign_location_layers）

- 从 `Network` 构图（`sourcing→location`），BFS 自顶向下分配层级。
- 自动识别根节点：无父节点的节点；未显式出现在 `location` 但出现在 `sourcing` 也视作根；孤立点归到末层 + 1。
- 输出 `location→layer` 映射，用于逐层从下游到上游推进。

## 四、发送端类型与提前期口径

### 发送端类型

1. 若在 `Network` 有活动行则使用其 `location_type`
2. 若为根层（`layer==0`），视为 Plant
3. 否则默认 DC

### 提前期计算

- **DC**：`lead time = PDT + GR`
- **Plant**：`lead time = max(MCT, PDT+GR) + PTF + LSK - 1`（`PTF/LSK` 按 `(material, sending)` 命中，大小写兼容；默认 `PTF=0/LSK=1`）

### 根节点计划窗口

- 无上游时使用 Plant 公式的窗口：`horizon = max(MCT, PDT+GR) + PTF + LSK - 1`（`sending==location` 的所有行取最大值，保证 `≥ 1`）。

## 五、库存口径（projected 与 dynamic）

- **预计库存 projected_soh**：用于判断窗口内是否存在可覆盖能力（含在途/未来）。
  - `beginning + in_transit + delivery_gr + today_production + future_production − today_shipment − open_deployment_outbound`
- **当日可用库存 dynamic_soh**：用于计算当日可分配库存（客户、自补货、下游补货）。
  - `beginning + delivery_gr + today_production_gr − open_deployment_outbound`
- **开放调拨 inbound**：`(material, receiving) → sum(quantity)`，仅用于管道覆盖，不计入当日现货。

## 六、需求收集与分配顺序

- 需求来源窗口统一为 `[sim_date, horizon_end]`：
  - SupplyDemandLog：forecast 与其他（非 AO/normal）
  - SafetyStock：仅取 `horizon_end` 当天目标量
  - OrderLog：AO/normal
  - 上游传入 GAP：`net demand for xxx`（同样按窗口过滤）
- 分配顺序：
  - 先用当日可用库存 `dynamic_soh` 按优先级分配（高优先级先满，最后一个优先级按比例切分）
  - 对自补货剩余缺口，用管道供给（in-transit/open-deployment inbound/future production）按优先级比例覆盖
  - 对仍有缺口的行，生成 GAP 上传至上游（`demand_element = net demand for {原类型}`）
- 到货日与 leadtime：跨节点行计算 `leadtime` 与 `planned_delivery_date = sim_date + leadtime`；自补货行 `leadtime=0`。

## 七、路径级 MOQ/RV 放大与回分

- 分组维度：仅 `(material, sending, receiving)`；自循环（`sending==receiving`）不应用 MOQ/RV。
- 组总量 `S` 的一次放大：`T = apply_moq_rv(S, moq, rv)`（跨节点：`S<moq→moq`，否则 `ceil(S/rv)*rv`；自循环保持原值）。
- 组内回分：使用“最大余数法”按各行原始需求占比回分到行级，合计 `= T`。
- 应用时机：在库存分配（优先级向量化）之前计算分组后的 `adjusted_qtys`；跨节点计划行的 `planned_qty` 使用调整量，自补保持原始需求量。

## 八、接收仓容配额（ReceivingSpace）

- 仅对跨节点到货（`sending != receiving`）在当日应用限额。
- 组内按 `demand_priority` 升序处理；配额不足时对最后一个优先级组按比例切分。
- 被限额的差额记录于 `UnfulfilledLog`，原因 `space constraint`。

## 九、Push / Soft-Push 逻辑

- 前提：仅在当日非 push 需求（自补货与跨节点常规分配）都已满足且发送端仍有剩余库存时执行。
- 发送端剩余可用库存：以当日可用库存 `dynamic_soh` 为基线，扣除当日已分配后得到 `available_soh`；Soft-Push 在此基础上优先保留发送端当日安全库存，再参与下推。
- 目标与挡位：以接收端“到货日”的安全库存为目标，选择最高可行挡位 `L`（默认 `[1.2, 1.5, 2.0, 2.5, 3.0]`，可由 `config['M5_PushLevels']` 覆盖）。
- 需求缺口计算：`need_r = max(0, L * SS_r - PI_r)`，其中 `PI_r` 的基线优先采用 `projected_soh`（预测库存），若未提供则回退为 `dynamic_soh`（当日可用库存）。
- 分配策略：对可行挡位，按 `need_r` 比例分配 `available_soh`（向下取整），生成 push/soft-push 计划行；随后仍受接收空间限额裁剪。
- 产出字段：`date/material/sending/receiving/demand_element(planned as push/soft push replenishment)/planned_qty/deployed_qty_invCon/planned_delivery_date/leadtime/is_cross_node`。

说明（与代码一致）：
- `projected_soh` 在 push/soft-push 中被用作接收端库存基线的优先选项；若未传入，则回退到 `dynamic_soh`。参见 [module5.py](module5.py#L1699-L1701)、[module5.py](module5.py#L2230) 与调用处 [module5.py](module5.py#L2616-L2620)。

## 十、逐层并行与性能优化

- 按层逐步从下游至上游处理；节点使用线程池并行，失败任务回退串行重试。
- 构建缓存加速：
  - `PTF/LSK` 缓存（15–20×）
  - `LeadTime` 基础参数缓存 `PDT/GR/MCT`（10–15×）
  - `Network` 活动行缓存（按有效期）

## 十一、输出

- 调拨计划行（跨节点与自补货）：
  - 字段：`date`、`material`、`sending`、`receiving`、`demand_element`、`planned_qty`、`deployed_qty_invCon`、`planned_delivery_date`、`leadtime`、`is_cross_node`
- 未满足日志：`UnfulfilledLog`（例如接收空间限制），包含 `reason=space constraint` 等标注。

## 十二、容错与默认值

- 缺失或未命中时的默认：
  - `lead time` 最低为 `1`
  - `PTF=0`、`LSK=1`
  - `MOQ/RV=1,1`
  - `DemandPriority` 自动补齐：AO=1、normal=2、其他=9
- 规则与边界：
  - 自循环不应用 `MOQ/RV`
  - `Network` 不允许同一 `(material, location)` 存在多个 `sourcing`
  - 标识符统一字符串化与地点补零，避免匹配失败

## 十三、与 Module 3 的关键对齐点

- Plant/DC 提前期公式一致：`max(MCT, PDT+GR) + PTF + LSK - 1` / `PDT + GR`
- 分层推进与 GAP 语义一致：自下而上传递缺口，父层参与窗口消化。
- 标识符与有效期过滤一致：`eff_from/eff_to` 按 `sim_date` 过滤，字符串化统一。

## 十四、适用场景

- 多层网络日度滚动调拨与补货计划（含接收空间限额）
- 与上游模块（M1/M4）及编排系统（Orchestrator）集成运行，支持 push/soft-push 策略