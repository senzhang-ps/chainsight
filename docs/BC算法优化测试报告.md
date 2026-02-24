# ChainSight 供应链仿真系统 算法优化测试报告

## 1. 测试概述

### 1.1 测试目的
本报告对 ChainSight 供应链仿真系统进行了全面的三版本对比测试，验证代码重构后的功能正确性和性能提升效果。

### 1.2 测试版本
| 版本 | 说明 | 代码位置 |
|------|------|----------|
| **Dev** | 原始开发版本（基准版本） | `ChainSight_Dev/` |
| **Src** | 重构本地运行版本 | `src/` |
| **DB** | 重构数据库集成版本 | `src/` + `--use-db` |

### 1.3 测试环境
- **操作系统**: Windows 11
- **Python版本**: 3.12.9
- **数据库**: PostgreSQL 5432 + DuckDB (内存模式)
- **测试数据**: BC_S5.xlsx 配置文件
- **仿真周期**: 87天 (2025-10-05 至 2025-12-30)
- **随机种子**: 42（确保可重复性）
- **测试时间**: 2026-02-05

### 1.4 测试输出目录
| 版本 | 输出路径 |
|------|----------|
| Dev | `ChainSight_Dev/BC_S5/run_20260205_135019/` |
| Src | `outputs/BC_S5/run_20260204_232353/` |
| DB | `outputs/db_config/BC_S5_20260204_233026/` |

---

## 2. 性能对比分析

### 2.1 总体运行时间

| 版本 | 开始时间 | 结束时间 | 总运行时间 | 平均每天 |
|------|----------|----------|------------|----------|
| **Dev** | 13:50:19 | 16:31:20 | **9,661秒** (161.0分钟) | 111.05秒 |
| **Src** | 23:23:53 | 00:39:49 | **4,556秒** (75.9分钟) | 52.37秒 |
| **DB** | 23:30:26 | 00:09:49 | **2,363秒** (39.4分钟) | 27.16秒 |

### 2.2 性能提升比例

| 对比项 | 加速倍数 | 时间减少百分比 |
|--------|----------|----------------|
| **Src vs Dev** | **2.12x** | 52.8% |
| **DB vs Dev** | **4.09x** | 75.5% |
| **DB vs Src** | **1.93x** | 48.1% |

### 2.3 各模块平均耗时对比（秒/天）

| 模块 | Dev | Src | DB | Src加速 | DB加速 |
|------|-----|-----|-----|---------|--------|
| **M1 订单生成** | ~7.5 | ~1.2 | ~1.2 | 6.25x | 6.25x |
| **M3 流量计算** | ~22.9 | ~3.4 | ~2.7 | 6.74x | 8.48x |
| **M5 部署规划** | ~61.6 | ~26.1 | ~11.4 | 2.36x | 5.40x |

### 2.4 关键模块每日详细耗时（最后一天示例）

#### Day 87 (2025-12-30) 各模块耗时（秒）
| 模块 | Dev | Src | DB |
|------|-----|-----|-----|
| M1 订单生成 | 5.47 | 1.18 | ~1.2 |
| M3 净需求计算 | 22.94 | 3.45 | ~2.7 |
| M5 部署规划 | 61.61 | 26.06 | ~11.4 |
| M6 物流执行 | ~2.0 | ~1.2 | ~1.2 |

### 2.5 DB版本资源使用
| 项目 | 值 |
|------|------|
| DuckDB 内存限制 | 4GB |
| 线程数 | 14 |
| 仿真核心运行 | 39.4分钟 (含数据库写入) |
| 数据库写入开销 | 1088.97秒 |
| 纯仿真时间 | ~21.2分钟 |

---

## 3. 数据一致性验证

### 3.1 总体验证结果

| 验证项 | Dev | Src | DB | 一致性 |
|--------|-----|-----|-----|--------|
| **历史库存记录** | 49,688 | 49,688 | 49,688 | ✅ 100% |
| **订单发货削减汇总** | 24,252 | 24,252 | 24,252 | ✅ 100% |
| **部署计划总数** | 1,149,959 | 1,149,959 | 1,149,959 | ✅ 100% |
| **交付计划总数** | 43,055 | 43,055 | 43,055 | ✅ 100% |
| **生产计划总数** | 379 | 379 | 379 | ✅ 100% |
| **换型报告总数** | 146 | 146 | 146 | ✅ 100% |
| **超容量报告** | 15 | 15 | 15 | ✅ 100% |
| **卡车使用报告** | 192 | 192 | 192 | ✅ 100% |

### 3.2 每日库存一致性验证

| 验证项 | 结果 |
|--------|------|
| **仿真天数** | 87天 |
| **库存不一致天数** | 0天 |
| **一致性比例** | 100% |
| **最终库存数量** | 396,489 |

### 3.3 汇总文件详细对比

| 文件名 | Dev行数 | Src行数 | 一致性 |
|--------|---------|---------|--------|
| full_changeover_report.xlsx | 146 | 146 | ✅ MATCH |
| full_delivery_plan_report.xlsx | 43,055 | 43,055 | ✅ MATCH |
| full_deployment_plan_report.csv | 1,149,959 | 1,149,959 | ✅ MATCH |
| full_exceed_capacity_report.xlsx | 15 | 15 | ✅ MATCH |
| full_order_shipment_cut_report.xlsx | 24,252 | 24,252 | ✅ MATCH |
| full_production_plan_report.xlsx | 379 | 379 | ✅ MATCH |
| full_truck_usage_report.xlsx | 192 | 192 | ✅ MATCH |
| historical_inventory_record.csv | 49,688 | 49,688 | ✅ MATCH |

### 3.4 最终Orchestrator状态对比

| 指标 | Dev | Src | DB | 一致性 |
|------|-----|-----|-----|--------|
| **日期** | 2025-12-30 | 2025-12-30 | 2025-12-30 | ✅ |
| **库存项数** | 569 | 569 | 569 | ✅ |
| **总库存量** | 396,489 | 396,489 | 396,489 | ✅ |
| **开放部署数** | 944 | 944 | 944 | ✅ |
| **在途数量** | 3,649 | 3,649 | 3,649 | ✅ |
| **生产入库** | 6 | 6 | 6 | ✅ |
| **交付入库** | 490 | 490 | 490 | ✅ |
| **发货数** | 282 | 282 | 282 | ✅ |

---

## 4. Bug修复记录

### 4.1 历史修复的问题

在测试过程中，发现并修复了以下数据一致性问题：

#### Bug 1: Module1 数据类型不匹配导致合并失败 (关键修复)
**问题描述**: 订单生成模块在DataFrame合并时，`material`列数据类型不一致（string vs int64），导致合并静默失败

**根本原因**: 
- `ml_avg_demand` 的 `material` 列为 **string** (已标准化)
- `ao_config` 的 `material` 列为 **int64** (原始Excel数据)
- `forecast_error` 的 `material` 列为 **int64** (原始Excel数据)

**影响**: 前2天订单生成失败，库存偏差从2.63%增长到17.65%

**修复文件**: 
- `src/modules/demand_planning/order.py` - `_generate_ao_orders()` 和 `_generate_normal_orders()` 函数
- `src/modules/demand_planning/integration.py` - `_validate_config()` 函数

```python
# 修复: 在合并前调用 normalize_identifiers() 确保类型一致
ao_cfg = normalize_identifiers(ao_cfg)
fe_normalized = normalize_identifiers(forecast_error.copy())
```

#### Bug 2: Module1 OrderLog 累积订单字段不一致
**问题描述**: DB版本写入数据库时使用了 `orders_df` 而非累积订单 `all_orders_for_next_day`

**修复文件**: `pgsql_db/module_data_writer.py` (约第659行)
```python
# 修复前
'all_orders_for_next_day': ('module1_output_orderlog', 'orders_df'),

# 修复后  
'all_orders_for_next_day': ('module1_output_orderlog', 'all_orders_for_next_day'),
```

#### Bug 3: Module1 集成模块累积订单逻辑错误
**问题描述**: 主集成模块使用了错误的字段名

**修复文件**: `src/core/main_integration.py` (约第2258行)
```python
# 修复前
accumulated_orders = day_results.get('orders_df', pd.DataFrame())

# 修复后
accumulated_orders = day_results.get('all_orders_for_next_day', pd.DataFrame())
```

#### Bug 4: Module4 净需求处理逻辑不完整
**问题描述**: 净需求应只包含下游需求（Layer 0），且数量需取绝对值

**修复文件**: `src/core/main_integration.py` (约第797-812行)
```python
# 修复后添加
net_demand_df = net_demand_df[net_demand_df['location_layer'] == 0].copy()
if 'quantity' in net_demand_df.columns:
    net_demand_df['quantity'] = net_demand_df['quantity'].abs()
```

#### Bug 5: Module5 内存模式数据加载问题
**问题描述**: Module5 在内存模式下未能正确读取累积订单数据，导致 AO (Auto-Order) 需求丢失

**修复文件**: `src/modules/deployment_planning/data_loader.py` (约第342行)
```python
# 修复前
orders_df = module1_result.get('orders_df', pd.DataFrame())

# 修复后
orders_df = module1_result.get('all_orders_for_next_day', module1_result.get('orders_df', pd.DataFrame()))
```

**影响**: 此Bug导致 DB 版本的 Module5 部署计划数量与 Dev 基准不一致。修复后三版本数据完全一致。

#### Bug 6: 数据库写入前未清空历史数据
**问题描述**: 多次运行会导致数据累积

**修复文件**: `pgsql_db/module_data_writer.py`
- 添加 `truncate_output_tables()` 方法
- 修改 `write_module_results_from_dict()` 在写入前自动清空表

---

## 5. 技术优化详解

### 5.1 代码架构优化

| 优化项 | 优化前 (Dev) | 优化后 (Src/DB) |
|--------|-------------|-----------------|
| **代码结构** | 单文件混合 | 模块化分层 |
| **数据处理** | 逐行操作 | 向量化批处理 |
| **内存管理** | 多次复制 | 原地操作 |
| **I/O操作** | 每天写文件 | 批量/内存缓存 |
| **数据存储** | 纯Excel文件 | DuckDB内存 + PostgreSQL |

### 5.2 模块级优化

#### Module 1 订单生成 (6.25x 加速)
- 使用 NumPy 向量化操作替代 Python 循环
- AO/Normal 消耗计算完全向量化
- 批量生成随机数，避免逐行调用

#### Module 3 净需求计算 (8.48x 加速)
- DuckDB 批量计算替代逐节点 Python 计算
- 按层级批量处理（Layer 2 → 1 → 0）
- 单次 DuckDB 查询处理数百节点

#### Module 5 部署规划 (5.40x 加速)
- 预构建索引加速查找
- 需求收集算法优化
- 分配算法向量化处理

### 5.3 DB版本特有优化

| 技术 | 说明 |
|------|------|
| **DuckDB 内存模式** | 列式存储，OLAP 查询优化 |
| **批量写入** | 使用 `copy_from` 高效导入 |
| **内存数据传递** | 模块间直接传递 DataFrame |
| **自动表清理** | 写入前自动 TRUNCATE |

---

## 6. 测试结论

### 6.1 功能验证结论

✅ **重构版本(Src/DB)与原始版本(Dev)输出100%一致**

验证覆盖：
- 87天仿真周期全部通过
- Module1: 订单、发运、削减日志 (24,252条)
- Module3: 净需求计算结果
- Module4: 生产计划 (379条)
- Module5: 部署计划 (1,149,959条)
- Module6: 交付计划 (43,055条)
- 库存记录: 49,688条
- 所有业务指标数值完全匹配

### 6.2 性能提升结论

| 版本 | 相对Dev加速 | 推荐场景 |
|------|-------------|----------|
| **Src** | 2.12x | 常规仿真运行、调试开发 |
| **DB** | 4.09x | 大规模数据分析、生产环境 |

### 6.3 优化效果总结

| 指标 | 优化前(Dev) | Src版本 | DB版本 | 最佳提升 |
|------|-------------|---------|--------|----------|
| 单次仿真时间(87天) | 161.0分钟 | 75.9分钟 | 39.4分钟 | **75.5%** |
| M1耗时/天 | ~7.5秒 | ~1.2秒 | ~1.2秒 | **84.0%** |
| M3耗时/天 | ~22.9秒 | ~3.4秒 | ~2.7秒 | **88.2%** |
| M5耗时/天 | ~61.6秒 | ~26.1秒 | ~11.4秒 | **81.5%** |

### 6.4 建议

1. **生产环境**: 推荐使用 DB 版本，性能提升显著
2. **开发调试**: Src 版本足够使用，无需数据库依赖
3. **数据分析**: DB 版本支持 SQL 查询，便于数据分析

---

## 7. 附录

### 7.1 测试日志文件
| 版本 | 日志路径 |
|------|----------|
| Dev | `ChainSight_Dev/BC_S5/run_20260205_135019/simulation_log_20260205_135019.txt` |
| Src | `outputs/BC_S5/run_20260204_232353/simulation_log_20260204_232353.txt` |
| DB | `outputs/db_config/BC_S5_20260204_233026/simulation_log_20260204_233026.txt` |

### 7.2 测试命令
```bash
# Dev版本
cd ChainSight_Dev
python run.py --config BC_S5.xlsx --start-date 2025-10-05 --end-date 2025-12-30

# Src版本（本地运行）
python run.py --config BC_S5 --start-date 2025-10-05 --end-date 2025-12-30

# DB版本（数据库集成）
python run.py --config BC_S5 --start-date 2025-10-05 --end-date 2025-12-30 --use-db
```

### 7.3 数据库连接
```python
import psycopg
conn = psycopg.connect('host=localhost port=5432 dbname=test_db user=postgres password=123456')
```

### 7.4 数据对比脚本
```bash
# 对比 DB vs Dev 基准
python compare_db_vs_dev.py

# 清空数据库输出表
python truncate_tables.py
```

---

**报告生成时间**: 2026-02-05  
**测试执行人**: chenxianyue002@chinasofti.com  
**版本**: v3.0
