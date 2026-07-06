# fem-cs-test 配置人工修订记录（2026-05-22）

## 1. 本次已完成的修改

### M1_DemandForecast
- 已将 `week` 从 `WEEK06~WEEK18` 重编码为数字 `1~13`。
- 这一步解决了 `week` 非数字的问题。

### M4_MaterialLocationLineCfg
- 已重整表行结构，使其按表头落到标准列。
- 已按确认口径写入：
  - `ptf = 7`
  - `day = 2`
  - `MCT = 0`
- 删除了不在当前 benchmark / forecast 范围内的物料：
  - `90464033`

### Global_Network
- 已按 `M1_DemandForecast` 的 `material + location` 组合裁剪超范围行。

### M1_InitialInventory
- 已将 `location = 386` 统一更正为 `0386`。
- 已补齐缺失的 6 个 `material-location` 组合，`quantity = 0`。

### M1_ForecastError
- 已补齐缺失的 13 个 `material-location` 组合。
- 新增行口径：
  - `order_type = normal`
  - `error_std_percent = 0.75`
- 已删除不需要的 extra 行，保留 `0386`，其余超出 `M1_DemandForecast` 的组合已删除。

### M3_SafetyStock
- 已按 `M4_MaterialLocationLineCfg.material` 裁剪，仅保留 M4 物料范围内的 safety stock。

### M5_DeployConfig
- 已补齐缺失的 14 个 `material + receiving` 组合。
- 新增行口径：
  - `moq = 1`
  - `rv = 1`
  - `lsk = 1`
  - `day = 1`
- 已按 `M4_MaterialLocationLineCfg.material` 裁剪多余物料。

### M6_MaterialMD
- 已按 `M4_MaterialLocationLineCfg.material` 裁剪多余物料。

### Global_DemandPriority
- 已补齐多层 `net demand for ...` 缺失模式，共新增 10 行。

### M1_OrderCalendar
- 已补齐完整仿真周期日期：`2026-02-02 ~ 2026-05-03`。
- `order_day_flag` 统一填 `1`。

### M4_ChangeoverMatrix
- 现有表只维护了一半对称转产矩阵。
- 已按现有非 self pair 自动补齐反向 pair，共新增 435 行。

### M5_PushPullModel
- 已删除所有 `model = pull` 的行，共删除 14 行。

---

## 2. 本次确认的业务决定

### 关于 0386
- `0386` 被视为 **sourcing-only location**。
- 因此：
  - 不要求它作为 `Global_Network` 的 network location 出现。
  - 与 `0386` 相关的部分校验项应视为规则问题，而不是配置问题。

### 关于 M3_SafetyStock 缺失组合
- 当前缺失的 31 个 `material-location` 组合，本轮先视为“业务上没问题”，暂不补。

### 关于 M4_ChangeoverDefinition
- `cost` 可留空。
- `mu_loss` 可留空。

### 关于 M1_ForecastError
- 缺失行统一按 `error_std_percent = 0.75` 补齐。

### 关于 M5_DeployConfig
- 缺失行统一按：
  - `moq = 1`
  - `rv = 1`
  - `lsk = 1`
  - `day = 1`
  补齐。

### 关于 M5_PushPullModel
- 非法值 `pull` 不改写为 `push`，而是直接删除对应行。

---

## 3. 仍需人工补充 / 人工确认的内容

### 3.1 M1_DemandForecast
- `week 14`
- `week 15`
- 当前仅完成 `1~13` 周重编码，后两周仍需人工补充。

### 3.2 M3_SafetyStock
#### 缺失但暂未补的 31 个 material-location
- `80799573-E568`
- `80799574-C563`
- `80799575-E556`
- `80799576-E556`
- `80799577-E556`
- `80817373-D594`
- `80817373-E230`
- `80817374-D594`
- `80817374-E230`
- `80817375-D594`
- `80817375-E230`
- `80817378-D594`
- `80817378-E230`
- `80858047-C867`
- `80858048-C867`
- `80858048-D608`
- `80858048-D874`
- `80858048-D876`
- `80878361-C563`
- `80878361-C867`
- `80878361-D608`
- `80878361-D874`
- `80878361-D876`
- `80878370-C867`
- `80878370-D608`
- `80878370-D874`
- `80878370-D876`
- `80892811-E230`
- `82278637-D594`
- `82278637-E230`
- `83900927-D594`

#### daily coverage
- `M3_SafetyStock` 仍需人工补齐全周期 daily rows。

### 3.3 M4_MaterialLocationLineCfg
- `rv` 当前仍为空，需要人工补。

### 3.4 M4_LineCapacity
- 全周期日期覆盖仍需人工补。

### 3.5 M6_MaterialMD
以下 7 个 material 仍需人工补主数据：
- `80817373`
- `80817374`
- `80817375`
- `80817378`
- `80892811`
- `82278637`
- `83900927`

---

## 4. 本轮明确跳过 / 暂不处理的项
- `Global_LeadTime` 的 `OTD/PDT` 异常暂不处理。
- `M6_TruckReleaseCon` lane 与 `Global_Network` 不一致暂不处理。
- 与 `0386` 作为 sourcing-only location 相关的若干 location 校验项暂按规则问题处理。
- `M1_InitialInventory` remaining extra 组合暂保留。

---

## 5. skill / 校验规则待更新
- config-validation skill 需要更新：
  - 对于 `0386` 这种 sourcing-only location，不应强制要求它作为 `Global_Network` 的 network location 出现。
