## 一、模块定位与目标

- 基于每日快照数据，按供应网络的层级自下而上计算各节点的净需求（AO/预测/安全库存缺口），并将缺口按路径向上游传递，应用 MOQ/RV 放大后再回分。
- 与 Module 5 的提前期与 MOQ/RV 口径对齐，支持 Plant 与 DC 的不同提前期计算。

## 二、数据输入与标准化

### 静态配置

- **Global_Network**：网络关系、location_type、有效期 `eff_from`/`eff_to`（仅使用模拟日期 `sim_date` 有效的配置参与计算）
- **Global_LeadTime**：`PDT`、`GR`、`MCT`（按 `sending+receiving` 匹配）
- **M4_MaterialLocationLineCfg**：`PTF`、`LSK`（按 `material+location`，大小写兼容，未命中默认 `PTF=0`、`LSK=1`）
- **M3_SafetyStock**：安全库存目标
- **M5_DeployConfig**：部署路径的 `MOQ`、`RV`（优先三键 `material+sending+receiving`，回退到二键 `material+sending`，未命中默认 `1,1`）

### 每日快照（当日视图）

- **SupplyDemandLog**：M1 输出的“订单消耗后，考虑了 supply choice 的 supply demand”
- **ShipmentLog**：当日客户出货（`date==sim_date`）
- **OrderLog**：当日版本的订单全量（历史生成但未来分货 + 当日分货单订单），用于本地 AO 需求窗口统计

### Orchestrator 动态数据（当日视图）

- Beginning Inventory、In-Transit、Delivery GR、All Production（含今天+未来）、Open Deployment、Delivery Shipment Log

### 标识符标准化

- `material`、`location`、`sending`、`receiving`、`sourcing` 统一为字符串；地点左补零至 4 位；缺失置空字符串

## 三、网络层级识别（assign_location_layers）

- 从 Global_Network 构图（`sourcing→location`），BFS 自顶向下分配层级
- 自动识别根节点：无父节点的节点；若未显式出现在 `location` 但出现在 `sourcing` 也视作根；孤立点归到末层 + 1
- 输出 `location→layer` 映射，用于自下而上的逐层计算

## 四、发送端类型与提前期口径

### 发送端类型推断

1. 若存在同物料在 `location==sending` 的显式配置且在有效期内，直接取其 `location_type`
2. 若 `sending` 为根节点（`layer==0`），视为 Plant
3. 若只出现在 `sourcing`、从不出现在 `location`，也视为 Plant
4. 其他默认为 DC

### 提前期计算

- **DC**：`lead time = PDT + GR`
- **Plant**：`lead time = max(MCT, PDT+GR) + PTF + LSK - 1`（`PTF/LSK` 从 M4 表按 `material+sending` 取，大小写兼容，未命中默认 `PTF=0`、`LSK=1`）

### 根节点计划窗口

- `horizon = max(MCT, PDT+GR) + PTF + LSK - 1`，`PDT/GR/MCT` 以 `sending==location` 的所有行最大值，保证 `≥ 1`

## 五、可用量口径（当日）

### 可用量来源

- 期初库存（当日 `date` 的 Beginning Inventory）
- 在途库存（`receiving=本节点`）
- 当日 GR（Delivery GR，`date==sim_date`）
- 当日生产 GR（All Production 中 `available_date==sim_date` 的 `produced_qty`）
- 未来确认生产（`available_date > sim_date` 的 `con_planned_qty`，若无则回退 `produced_qty/quantity`），作为未来 pipeline
- 开放调拨入库（Open Deployment，`receiving=本节点`，`date>sim_date` 的未来入库；数量字段优先 `deployed_qty`，无则回退 `quantity`）

### 扣减项

- 当日客户发货（ShipmentLog）
- 当日跨点发运（Delivery Shipment Log，作为发送端且当日）
- 当日开放调拨出库（Open Deployment，`sending=本节点`，排除自循环 `sending==receiving`）

## 六、需求侧与缺口顺序

- **AO 本地需求**：OrderLog 中 `demand_type=='AO'`，窗口 `[sim_date, sim_date+horizon]` 合计
- **预测本地需求**：SupplyDemandLog 在同一窗口内的 `quantity` 合计
- **安全库存本地需求**：在 `horizon_end` 当日的目标安全库存量
- **缺口消耗顺序**：AO → Forecast → Safety
  - 将“总可用量”依序消耗 AO/预测后，剩余用于安全库存；分别计算 `AO_gap`、`FC_gap`、`SS_gap`
- 叠加下游传入的 gap：AO/FC/SS 各自与本地叠加后参与消耗

## 七、缺口向上游的传递与 MOQ/RV 放大

- 对于每个子节点→父节点路径，先汇总该路径上的 AO/FC/SS 正向缺口 `S`
- 依据 M5_DeployConfig 命中 `(material, sending=父, receiving=子)` 的 `moq/rv`（回退到二键，再默认 `1,1`），对 `S` 做一次放大：`T = apply_moq_rv(S, moq, rv)`（跨节点口径；自循环不应用 MOQ/RV）
- 将 `T` 用“最大余数法”按比例回分到 AO/FC/SS 三类，保证合计 `= T`
- 在父节点端累加来自所有子节点的 AO/FC/SS gap，作为上层计算的下游缺口输入

## 八、逐层并行计算与性能

- 自最下层到最上层逐层计算；每层节点用线程池并行（默认最多 32 线程，失败任务会回退到串行重试）
- **PTF/LSK** 采用预构建字典缓存，避免反复 DataFrame 过滤（15–20 倍加速）
- **节点覆盖扩展**：为所有层级中的地点与网络中出现的物料补齐缺失的 `material-location` 组合进行净需求评估，保证根层也可生成 net demand

## 九、输出

- 对每个节点、每类缺口（若 `> 0`）生成一条净需求记录：
  - 字段：`material`、`location`、`requirement_date=sim_date+1`、`quantity`（负值表示缺口）、`demand_element`（`net demand for AO` / `net demand for forecast` / `net demand for safety`）、`layer`、`simulation_date`、`horizon_days`
- 层内聚合：相同 key（`material, location, requirement_date, demand_element, layer`）分组汇总 `quantity`，保留首个 `simulation_date/horizon_days`
- **集成模式 run_integrated_mode**
  - 按日期范围循环，逐日加载 Module1 与 Orchestrator 数据，执行计算
  - 输出每日日志文件：`Module3Output_YYYYMMDD.xlsx`，Sheet: `NetDemand`
  - 返回汇总信息（记录数、处理天数、输出文件名）

## 十、容错与默认值

### 缺失配置或匹配不到行时的默认

- `lead time` / 根节点 `horizon` 最低为 `1`
- `PTF=0`、`LSK=1`；`MOQ/RV=1,1`

### 其他容错措施

- 标识符均做字符串化与地点补零，避免连接/匹配失败
- 读取/计算异常会打印 Warning 并使用空表或默认值继续

## 十一、与 M5 的关键对齐点

- Plant 提前期与根节点 `horizon` 口径：`max(MCT, PDT+GR) + PTF + LSK - 1`
- 路径级 `MOQ/RV` 一次放大 + 最大余数法回分，确保对上游的需求为放大后的整数且可解释
- 自循环不应用 `MOQ/RV`

## 十二、适用场景

- 多层网络、Plant-DC 混合场景的日度 MRP 推演
- 与上游模块（M1/M4）和编排系统（Orchestrator）对接的日级滚动运行