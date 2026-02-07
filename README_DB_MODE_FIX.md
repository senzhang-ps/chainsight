# DB 模式输出一致性修复记录

> **目标**: 重构版本 (`D:\PG\Code\chainsight`) 的 DB 模式输出必须与 Dev 基线 (`ChainSight_Dev`) 完全一致。  
> **日期**: 2026-02-07  
> **配置**: `OC_Paste_S1_20251224` / `BC_S5`

---

## 1. 对比环境

| 项目 | 说明 |
|------|------|
| **重构版本** | `D:\PG\Code\chainsight` — 入口 `src/core/run.py` |
| **Dev 基线** | `D:\PG\Code\chainsight\ChainSight_Dev` — 入口 `run.py` |
| **Python** | 3.12.10 (`.venv`) |
| **数据库** | PostgreSQL `test_db` @ `localhost:5432` (user=postgres) |
| **76天 DB 模式命令** | `run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2026-02-28 --use-db` |
| **76天 File 模式命令** | `run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2026-02-28` |
| **Dev 基线输出目录** | `config\OC_Paste_S1_20251224\run_20260127_142402` |

---

## 2. 对比规则与方法

### 2.1 对比脚本

| 脚本 | 用途 |
|------|------|
| `compare_76day_perf.py` | 76天全量对比: DB vs Dev + File vs Dev (684项检查) |
| `extract_timing.py` | 从仿真日志提取各模块计时数据 |
| `quick_check_delivery.py` | DeliveryPlan 浮点精度诊断工具 |
| `compare_oc_paste_db.py` | DB 模式输出 vs Dev Excel 基线（6 天对比） |
| `compare_bc_s5_db.py` | BC_S5 DB 模式输出 vs Dev Excel 基线（5 天对比） |
| `debug_filemode_compare.py` | File 模式输出 vs Dev Excel 基线（验证代码逻辑正确性） |
| `debug_config_dict_compare.py` | DB 加载的 config_dict vs File 加载的 config_dict（列/行差异定位） |
| `debug_dedup_check.py` | 检查 DB 中配置表的重复行情况 |
| `debug_config_name.py` | 检查 DB config_name 值 |
| `debug_m3_root_cause.py` | DB vs File 配置输入逐表对比（根因定位） |

### 2.2 对比逻辑

1. **行数对比**: 两侧 DataFrame 行数是否一致
2. **内容对比**: 按业务主键对齐后逐行逐列比较
   - 数值列: 允许 `1e-6` 误差 (`np.isclose`)
   - 字符串列: 精确匹配
   - 日期列: 统一转 `datetime` 后比较
3. **列名对比**: DB 模式列名经 `column_mapping` 恢复大小写后，须与 File 模式一致
4. **关键验证指标**: `[OK]` = 完全一致, `[DIFF]` = 内容差异, `[ROW_DIFF]` = 行数不同
5. **浮点精度处理**: `FLOAT_ROUND_DIGITS=6`，用于处理 PostgreSQL TEXT 列中浮点数序列化差异（如 `0.136460695` → `0.13646069500000002`），6位小数确保两者舍入一致
6. **批量加载优化**: 小表批量加载，大表 (>500K行) 按日期逐日查询

### 2.3 核心原则

> **File 模式 100% 一致 = 代码逻辑正确**  
> 所有 DB 模式差异均为「DB 数据加载路径」或「仿真执行路径」问题，而非算法/模块逻辑问题。

---

## 3. 已完成的修复

### 3.1 ✅ BC_S5 File 模式 — 100% 一致

BC_S5 配置在 File 模式下所有模块输出与 Dev 完全一致，无需修复。

---

### 3.2 ✅ OC_Paste File 模式 — 100% 一致

OC_Paste 配置在 File 模式下 M3/M4/M5/M6 全部 `[OK]`，证明重构模块逻辑完全正确。

---

### 3.3 ✅ 修复: 仿真模式由 memory 改为 file

**问题**: DB 模式走了 `memory` 模式仿真，而 Dev 走的是 `file` 模式。  
**根因**: `run.py` 中 DB 模式调用了 `run_optimized_simulation_from_dict` (memory)，而非 `run_integrated_simulation_from_dict` (file)。  
**修复**: 切换 DB 模式为 `file` 模式仿真入口。  
**文件**: `src/core/run.py`

---

### 3.4 ✅ 修复: DB 模式跳过 file_output

**问题**: DB 模式在 `skip_file_output=True` 时不输出中间 Excel 文件。  
**修复**: 保证 DB 模式仍然输出中间文件以供读取。  
**文件**: `src/core/run.py`, `pgsql_db/module_data_writer.py`

---

### 3.5 ✅ 修复: M4 `_normalize_identifiers` 过度标准化

**问题**: `run_module4_integrated` 返回的 `production_df` 被 `_normalize_identifiers` 处理，导致 location 字段从 `386` 变成 `0386`，与 Dev 不一致。  
**根因**: M4 输出本身应为 int 型 location（Excel 读取自然失去前导零），不应做 normalize。  
**修复**: M4 返回时不再调用 `_normalize_identifiers`。  
**文件**: `src/core/main_integration.py`

---

### 3.6 ✅ 修复: M6 零输出 — config_name 路径不匹配

**问题**: M6 在 DB 模式下输出 0 行数据。  
**根因**: 
- CLI 传入 `config_name = "config/OC_Paste_S1_20251224"` (带路径前缀)
- DB 中存储的是 `config_name = "OC_Paste_S1_20251224"` (无路径前缀)
- `_load_config_from_database` 做精确匹配 → 0 行

**修复**: `_load_config_from_database` 优先精确匹配，无数据时回退到 basename 匹配:
```python
filtered = df[df['config_name'] == config_name]
if filtered.empty:
    config_basename = config_name.split('/')[-1]
    if config_basename != config_name:
        filtered = df[df['config_name'] == config_basename]
```
同步修改 `db_initializer.py` 的 `check_config_data_exists` 方法。

**文件**: `src/core/run.py`, `pgsql_db/db_initializer.py`

---

### 3.7 ✅ 修复: M1 OrderLog 累积 vs 当日

**问题**: DB 模式写入的 OrderLog 只包含当日订单 (`today_orders_df`)，而 Dev 写入的是累积订单 (`all_orders_df`)。  
**根因**: `integration.py` 返回 `today_orders_df` 作为 `orders_df`，DB writer 据此写入。  
**修复**: 改为返回 `all_orders_df` (累积订单):
```python
# 修改前
'orders_df': today_orders_df
# 修改后
'orders_df': all_orders_df
```
**文件**: `src/modules/demand_planning/integration.py`, `pgsql_db/module_data_writer.py`

---

### 3.8 ✅ 修复: 配置表重复行导致 M4 崩溃

**问题**: `cannot convert the series to <class 'float'>` — M4 计算 `rate_map.get()` 时返回多行。  
**根因**: config_name 双匹配（精确匹配 + basename 回退同时命中），导致每张配置表行数翻倍。  
**修复**: 优先精确匹配，仅在精确匹配为空时尝试 basename:
```python
filtered = df[df['config_name'] == config_name]
if filtered.empty:
    # 仅此时才回退
    ...
```
**文件**: `src/core/run.py` (`_load_config_from_database`)

---

### 3.9 ✅ 修复: DB 配置表额外列 (unnamed_*, 非 ASCII 列)

**问题**: DB 中 `Global_LeadTime` 多出 `unnamed_6/7/8` 列，`Global_Network` 多出 `unnamed_6` 和 `加上c816到下游dc` 列。  
**根因**: 多个配置（BC_S5、OC_Paste）共享同一张 `cfg_*` 表，BC_S5 的 Excel 有额外列，追加写入时自动扩展了表结构，导致 OC_Paste 行中这些列为 NULL。  
**修复**: 在两处加载点清理多余列:

**`_load_config_from_database` (run.py)**:
```python
drop_cols = [c for c in filtered.columns 
             if c in ('config_name', 'config_type', 'db_write_time')
             or c.startswith('unnamed') or c.startswith('Unnamed')]
filtered = filtered.drop(columns=drop_cols, errors='ignore')
filtered = filtered.dropna(axis=1, how='all')  # 删除全 NULL 列
```

**`load_configuration_from_dict` (main_integration.py)**:
```python
drop_cols = [c for c in df_copy.columns 
             if c in ('config_name', 'config_type', 'db_write_time')
             or c.lower().startswith('unnamed')
             or (not c.isascii() and c not in column_mapping.values())]
df_copy = df_copy.drop(columns=drop_cols, errors='ignore')
df_copy = df_copy.dropna(axis=1, how='all')
```

**文件**: `src/core/run.py`, `src/core/main_integration.py`

---

### 3.10 ✅ 新增: config_type 列区分 OC/BC 配置

**需求**: 在数据库中新增 `config_type` 列 (`OC` / `BC` / `OTHER`)，方便快速筛选不同类型的配置。  
**实现**:
- `excel_importer.py`: 新增 `_derive_config_type()` 静态方法，根据 config_name 前缀推导类型
- `db_connection.py`: `create_table_from_df` 新增 `config_type` 参数
- `run.py` / `main_integration.py`: 加载时将 `config_type` 列从业务数据中移除

**推导规则**:
| config_name 前缀 | config_type |
|---|---|
| `OC*` | `OC` |
| `BC*` | `BC` |
| 其他 | `OTHER` |

**文件**: `pgsql_db/excel_importer.py`, `pgsql_db/db_connection.py`, `src/core/run.py`, `src/core/main_integration.py`

---

### 3.11 ✅ 修复: DB 模式仿真引擎路径不一致（M3/M4/M6 差异根因）

**现象**: M3 NetDemand 全 6 天均有 ~25-30% 行差异（行数一致但 quantity/horizon_days/material 不同），级联导致 M4 ProductionPlan、M6 DeliveryPlan/VehicleLog 也出现差异。

**根因定位过程**:

1. **排除配置输入差异**: 运行 `debug_m3_root_cause.py` 逐表对比 DB 模式与 File 模式的所有配置输入:

   | 配置表 | 行数 | 结果 |
   |--------|------|------|
   | M3_SafetyStock | 180,576 | ✅ MATCH |
   | Global_LeadTime | 46 | ✅ MATCH |
   | Global_Network | 4,397 | ✅ MATCH |
   | Global_DemandPriority | 22 | ✅ MATCH |
   | M1_InitialInventory | 2,368 | ✅ MATCH |
   | M1_DemandForecast | 97,416 | ✅ MATCH |
   | Global_Seed | seed=42 | ✅ MATCH |

   两个微小差异（不影响计算）:
   - `Global_SpaceCapacity`: File 模式 0 行 DataFrame 有列名 `['capacity','eff_from','eff_to','location']`；DB 模式 0 行 DataFrame 列名为 `[]`（`dropna(axis=1, how='all')` 对空 DataFrame 清除了列）
   - `Global_Network_old`: File 模式名为 `Global_Network_old`，DB 模式名为 `global_network_old`（sheet_mapping 未覆盖此变体）

2. **结论: 配置输入 100% 一致，差异必然在仿真执行路径。**

3. **发现执行路径分歧**: `src/core/run.py` 的 `_run_with_database()` (原 line 651-674):
   ```python
   # DB 模式 — 优先走 DuckDB 高性能引擎
   try:
       from pgsql_db.optimized_simulation import run_optimized_simulation_from_dict
       result = run_optimized_simulation_from_dict(...)  # ← DuckDB 引擎
   except ImportError:
       from .main_integration import run_integrated_simulation_from_dict
       result = run_integrated_simulation_from_dict(...)  # ← 标准引擎（与 File 模式相同）
   ```
   由于 DuckDB 已安装，DB 模式**始终走 `run_optimized_simulation_from_dict`**，该引擎对 M3/M4 的计算逻辑与标准引擎不一致。

   File 模式则通过 `run_integrated_simulation` → 内部调用各模块 → 输出与 Dev 100% 一致。

**修复方案**: 将 DB 模式的仿真引擎从 DuckDB 高性能引擎改为标准引擎，确保与 File 模式使用完全相同的代码路径:
```python
# 修改后 (src/core/run.py)
# [FIX §3.11] 始终使用标准仿真引擎（与文件模式相同的代码路径）
from .main_integration import run_integrated_simulation_from_dict
logger.info("[RUN] 使用标准仿真引擎运行（与文件模式一致）...")
result = run_integrated_simulation_from_dict(
    config_data=config_data,
    config_name=config_name,
    start_date=start_date,
    end_date=end_date,
    output_base_dir=str(temp_output),
    skip_validation=True
)
```

**说明**: DuckDB 高性能引擎 (`pgsql_db/optimized_simulation.py`) 保留在代码库中不删除，未来可作为性能优化方案，但需先修正其 M3/M4 计算逻辑后再启用。

**文件**: `src/core/run.py`

---

### 3.12 ✅ 修复: BC_S5 配置自动导入被旧格式表误判跳过

**现象**: BC_S5 DB 模式仿真运行时所有模块输出为空 (0 行)，报错 "缺少必需的配置数据"。  
**根因**: `check_config_data_exists()` (db_initializer.py:148-168) 的 `else` 分支将旧格式 `bc_s5_*` 表（31 张、51,615 行，无 `config_name` 列）计入有效配置数据:
```python
# BUG: 旧格式表无 config_name 列，但仍按前缀匹配计入
old_prefix = config_name.lower().replace("-", "_").replace(" ", "_") + "_"
if tbl.startswith(old_prefix) or tbl.startswith('cfg_'):
    result = self.db.execute_query(f'SELECT COUNT(*) FROM "{tbl}"')
    total_rows += result[0][0] if result else 0
```
返回 `(True, 51615)` → `initialize()` 判定配置已存在 → 跳过自动从 Excel 导入 → `_load_config_from_database` 在 `cfg_*` 表中找不到 `config_name='BC_S5'` 的数据 → 所有配置 DataFrame 为空。

**修复**: `else` 分支不再计入旧格式表:
```python
else:
    # 没有config_name列 -> 旧格式表，不计入统一配置检测
    # 旧格式表（如 bc_s5_*）无法被 _load_config_from_database 读取，
    # 计入会导致误判"配置已存在"而跳过自动导入
    pass
```

**文件**: `pgsql_db/db_initializer.py`

---

## 4. 当前状态

### 4.1 ✅ DB 模式 6 天仿真运行成功 (历史记录)

**最近一次成功运行**: `2026-02-06 17:52:10` ~ `18:10:23` (总耗时 18 分 13 秒)  
**输出目录**: `outputs/db_OC_Paste_S1_20251224_20260206_175210`  
**数据库**: PostgreSQL `test_db`，配置 29 表，输出 17 表 (1,747,862 行)，Orchestrator 12 表，Summary 7 表 (255,192 行)

### 4.2 已验证一致的模块 (§3.11 修复前历史记录)

| 模块 | 数据表 | 状态 | 备注 |
|------|--------|------|------|
| M1 | OrderLog | ✅ [OK] 全 6 天 | 4532→15669 行匹配 |
| M1 | ShipmentLog | ✅ [OK] 全 6 天 | |
| M1 | CutLog | ✅ [OK] 全 6 天 | |
| M5 | DeploymentPlan | ✅ [OK] 全 6 天 | 31196→44803 行匹配 |
| M6 | DeliveryPlan | ✅ [OK] Day 1-4 | Day 5-6 受上游差异级联影响 |
| M6 | VehicleLog | ✅ [OK] Day 1-4 | Day 5-6 受上游差异级联影响 |

### 4.3 ✅ §3.11 修复后验证 — OC_Paste 全模块 [OK] (6天)

§3.11 修改完成后重跑仿真 + `compare_oc_paste_db.py`，**全 9 种模块输出 × 6 天 = 54 项检查全部 `[OK]`**。

| 模块 | 数据表 | 修复前状态 | 修复后状态 |
|------|--------|------------|------------|
| M3 | NetDemand | ❌ [DIFF] 全 6 天 | ✅ [OK] 全 6 天 |
| M4 | ProductionPlan | ❌ [DIFF] Day 2-6 | ✅ [OK] 全 6 天 |
| M4 | CapacityExceed | ❌ [DIFF] Day 2-6 | ✅ [OK] 全 6 天 |
| M6 | DeliveryPlan | ❌ [DIFF] Day 5-6 | ✅ [OK] 全 6 天 |
| M6 | VehicleLog | ❌ [DIFF] Day 5-6 | ✅ [OK] 全 6 天 |

### 4.4 ✅ BC_S5 DB 模式验证 — 全模块 [OK] (5天)

**日期**: 2026-02-06  
**配置**: BC_S5 (2025-10-06 ~ 2025-10-10, 5 天)  
**对比脚本**: `compare_bc_s5_db.py`  
**文件基线**: `config/BC_S5/run_20260206_101033`  

**验证结果: 45/45 [OK]**

| 模块 | 数据表 | 天数 | 状态 |
|------|--------|------|------|
| M1 | OrderLog | 5 天 | ✅ [OK] (738→2562 行) |
| M1 | ShipmentLog | 5 天 | ✅ [OK] (282 行/天) |
| M1 | CutLog | 5 天 | ✅ [OK] (282 行/天) |
| M3 | NetDemand | 5 天 | ✅ [OK] (494→559 行) |
| M4 | ProductionPlan | 5 天 | ✅ [OK] (0/64/0/0/0 行) |
| M4 | CapacityExceed | 5 天 | ✅ [OK] (0/15/0/0/0 行) |
| M5 | DeploymentPlan | 5 天 | ✅ [OK] (10636→12962 行) |
| M6 | DeliveryPlan | 5 天 | ✅ [OK] (0/103/52/391/91 行) |
| M6 | VehicleLog | 5 天 | ✅ [OK] (0/1/1/7/2 行) |

### 4.5 ✅ 76天全量对比验证 — OC_Paste_S1_20251224

**日期**: 2026-02-07
**配置**: OC_Paste_S1_20251224 (2025-12-15 ~ 2026-02-28, 76天)
**对比脚本**: `compare_76day_perf.py`
**Dev基线**: `config/OC_Paste_S1_20251224/run_20260127_142402/`
**Src输出**: `outputs/OC_Paste_S1_20251224/run_20260206_234914/`
**DB输出**: `outputs/db_OC_Paste_S1_20251224_20260206_231947/` + PostgreSQL test_db

**验证结果: 1368/1368 全部通过 (684 DB + 684 File)**

DB vs Dev (9模块 × 76天 = 684项):

| 模块 | 数据表 | 天数 | 状态 |
|------|--------|------|------|
| M1 | OrderLog | 76天 | ✅ [OK] 76/76 |
| M1 | ShipmentLog | 76天 | ✅ [OK] 76/76 |
| M1 | CutLog | 76天 | ✅ [OK] 76/76 |
| M3 | NetDemand | 76天 | ✅ [OK] 76/76 |
| M4 | ProductionPlan | 76天 | ✅ [OK] 76/76 |
| M4 | CapacityExceed | 76天 | ✅ [OK] 76/76 |
| M5 | DeploymentPlan | 76天 | ✅ [OK] 76/76 |
| M6 | DeliveryPlan | 76天 | ✅ [OK] 76/76 |
| M6 | VehicleLog | 76天 | ✅ [OK] 76/76 |

File vs Dev (9模块 × 76天 = 684项):

| 模块 | 数据表 | 天数 | 状态 |
|------|--------|------|------|
| M1 | OrderLog | 76天 | ✅ [OK] 76/76 |
| M1 | ShipmentLog | 76天 | ✅ [OK] 76/76 |
| M1 | CutLog | 76天 | ✅ [OK] 76/76 |
| M3 | NetDemand | 76天 | ✅ [OK] 76/76 |
| M4 | ProductionPlan | 76天 | ✅ [OK] 76/76 |
| M4 | CapacityExceed | 76天 | ✅ [OK] 76/76 |
| M5 | DeploymentPlan | 76天 | ✅ [OK] 76/76 |
| M6 | DeliveryPlan | 76天 | ✅ [OK] 76/76 |
| M6 | VehicleLog | 76天 | ✅ [OK] 76/76 |

**性能数据**:
| 版本 | 总运行时间 | 平均每天 |
|------|------------|----------|
| Dev | 98,389秒 (27.33小时) | 1294.6秒 |
| Src | 15,414秒 (4.28小时) | 202.8秒 |
| DB | 8,228秒 (2.29小时) | 108.3秒 |

**加速倍数**: Src 6.38x / DB 11.96x (vs Dev)

**浮点精度处理**: `FLOAT_ROUND_DIGITS=6` — `truck_load_pct` 在 PostgreSQL TEXT列序列化时产生浮点表示差异，6位小数容差确保正确比对。

**性能报告**: 详见 `docs/算法优化测试报告.md` (v4.0)

---

## 5. 已完成: Module6 `optimal_type` 修复

### 5.1 ✅ Module6 `'optimal_type'` KeyError — 已修复

**现象**: DB 模式运行时 M6 每天都报 `❌ Module6 失败: 'optimal_type'`  
**根因定位**:
- `M6_TruckReleaseCon` 表中 `optimal_type` 列在 Excel 原始数据中**全为 NaN**（该配置未设定最优车型）
- `load_configuration_from_dict` (main_integration.py:1265) 执行 `dropna(axis=1, how='all')` 删除全 NULL 列
- DB 模式下 `optimal_type` 列因全 NaN 被移除
- `get_optimal_truck_sequence` (capacity_manager.py:294) 访问 `truck_cfgs['optimal_type']` → **KeyError**
- File 模式不受影响（Excel 读取时自然保留全 NaN 列，不走 `load_configuration_from_dict`）

**说明**: `optimal_type` 字段只在 DB 表结构中存在，Dev 版本不需要额外添加该字段。

**修复方案**: 在 `get_optimal_truck_sequence` 中增加防御性检查:
```python
# 修改前 (capacity_manager.py:293-300)
optimal_types = truck_cfgs[truck_cfgs['optimal_type'] == 'Y']['truck_type'].tolist()
all_types = truck_cfgs['truck_type'].tolist()
non_optimal = [t for t in all_types if t not in optimal_types]
return optimal_types + non_optimal

# 修改后
all_types = truck_cfgs['truck_type'].tolist()
# optimal_type 列可能不存在（如 DB 模式下全 NaN 被 dropna 移除）
if 'optimal_type' in truck_cfgs.columns:
    optimal_types = truck_cfgs[truck_cfgs['optimal_type'] == 'Y']['truck_type'].tolist()
    non_optimal = [t for t in all_types if t not in optimal_types]
    return optimal_types + non_optimal
return all_types
```

**效果**: M6 全 6 天正常运行，输出 delivery_plan 798 行、vehicle_log 16 行、truck_usage_log 14 行。

**文件**: `src/modules/logistics_execution/capacity_manager.py`

---

## 6. 未完成 / 待处理

### 6.1 ✅ 重跑 DB 模式仿真 + 对比验证 — 已完成
OC_Paste 6天: 54/54 [OK] (§4.3)
BC_S5 5天: 45/45 [OK] (§4.4)

### 6.2 ✅ BC_S5 配置 DB 模式测试 — 已完成
修复内容 (§3.12): check_config_data_exists() 中旧格式表误判 bug
验证结果: 45/45 [OK]

### 6.3 ✅ 76天全量对比测试 — 已完成
OC_Paste 76天: DB 684/684 + File 684/684 = 1368/1368 [OK] (§4.5)
性能报告: docs/算法优化测试报告.md (v4.0)

### 6.4 🟡 调试脚本清理

以下调试脚本在修复完成后应清理:

| 文件 | 用途 |
|------|------|
| `debug_config_compare.py` | 对比 DB vs File 配置表结构 |
| `debug_config_dict_compare.py` | 对比 config_dict 列差异 |
| `debug_config_name.py` | 检查 DB config_name 值 |
| `debug_dedup_check.py` | 检查配置表重复行 |
| `debug_filemode_compare.py` | File 模式输出对比 |
| `debug_db_tables.py` | 列出 DB 所有表 |
| `debug_m3_root_cause.py` | M3 根因分析（配置输入逐表对比） |
| `debug_m6.py` | M6 调试 |
| `debug_truckreleasecon.py` | TruckReleaseCon 调试 |
| `truncate_output.py` | 清空输出表 |

---

## 7. 涉及修改的文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `compare_76day_perf.py` | 新增 | 76天全量对比脚本 (DB vs Dev + File vs Dev, FLOAT_ROUND_DIGITS=6) |
| `extract_timing.py` | 新增 | 仿真日志计时数据提取 |
| `quick_check_delivery.py` | 新增 | DeliveryPlan 浮点精度诊断 |
| `docs/算法优化测试报告.md` | 更新 | 算法优化测试报告 v4.0 (76天 OC_Paste) |
| `src/core/run.py` | 多处修改 | `_load_config_from_database` config_name 匹配、extra columns 清理、config_type 清理；§3.11 仿真引擎改为标准模式 |
| `src/core/main_integration.py` | 多处修改 | `load_configuration_from_dict` extra columns 清理、M4 取消 normalize |
| `src/modules/demand_planning/integration.py` | 修改 | M1 返回 `all_orders_df` 而非 `today_orders_df` |
| `src/modules/logistics_execution/capacity_manager.py` | 修改 | `get_optimal_truck_sequence` 增加 `optimal_type` 列存在性检查 (§5.1) |
| `pgsql_db/excel_importer.py` | 新增 | `_derive_config_type()` 方法、导入时传递 config_type |
| `pgsql_db/db_connection.py` | 修改 | `create_table_from_df` 新增 `config_type` 参数 |
| `pgsql_db/db_initializer.py` | 修改 | `check_config_data_exists` basename 回退匹配；§3.12 旧格式表不再误判为有效配置 |
| `pgsql_db/module_data_writer.py` | 修改 | M1 orders_df 映射注释更新 |

---

## 8. 数据库表结构说明

### 8.1 共享配置表 (cfg_* 表)

所有配置（BC_S5、OC_Paste 等）写入**同一张表**，通过 `config_name` + `config_type` 列区分:

```
cfg_m6_truckreleasecon
├── sending          (text)
├── receiving        (text)
├── truck_type       (text)
├── optimal_type     (text)
├── wfr              (double precision)
├── vfr              (double precision)
├── mdq              (bigint)
├── config_name      (text)        ← 'OC_Paste_S1_20251224' / 'BC_S5'
├── config_type      (text)        ← 'OC' / 'BC' (新增)
└── db_write_time    (timestamp)
```

### 8.2 已知风险与缓解

> **共享表的列超集问题**: 不同配置的 Excel 可能有不同列，追加写入时 PostgreSQL 会自动  
> `ALTER TABLE ADD COLUMN`，导致其他配置行中该列为 NULL。  
> **缓解方案 1**: `_load_config_from_database` 和 `load_configuration_from_dict` 中  
> 加入了 `dropna(axis=1, how='all')` 清理全 NULL 列。  
> **缓解方案 2**: 模块代码对可能被 `dropna` 移除的列做防御性检查（如 `optimal_type`，见 §5.1）。

---

## 9. 快速操作指南

```powershell
# 1. 删除全部表，重新建表导入
.venv/Scripts/python.exe truncate_tables.py          # 清空输出表
# (需额外脚本 DROP 所有 cfg_* 表后重新导入)

# 2. 重跑 DB 模式 (6天)
.venv/Scripts/python.exe run.py --config config/OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-20 --use-db

# 3. 对比输出
.venv/Scripts/python.exe compare_oc_paste_db.py

# 4. File 模式验证 (作为基准)
.venv/Scripts/python.exe run.py --config config/OC_Paste_S1_20251224.xlsx --start-date 2025-12-15 --end-date 2025-12-24
.venv/Scripts/python.exe debug_filemode_compare.py

# 5. 76天全量 DB 模式运行
.venv/Scripts/python.exe run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2026-02-28 --use-db

# 6. 76天全量 File 模式运行
.venv/Scripts/python.exe run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2026-02-28

# 7. 76天全量对比 (DB vs Dev + File vs Dev)
.venv/Scripts/python.exe compare_76day_perf.py --dev-dir "config/OC_Paste_S1_20251224/run_20260127_142402" --ref-dir "outputs/OC_Paste_S1_20251224/run_20260206_234914"
```