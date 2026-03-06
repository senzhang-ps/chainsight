# ChainSight 供应链仿真系统 BC算法优化测试报告 (完整版 Day 01-87)

## 1. 测试概述

### 1.1 测试目的
本报告对 ChainSight 供应链仿真系统全仿真周期 Day 01-87 (2025-10-05 至 2025-12-30) 的仿真结果进行三版本对比测试，验证代码重构后的功能正确性和性能提升效果。

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
- **测试时间**: 2026-02-12

### 1.4 测试输出目录
| 版本 | 输出路径 |
|------|----------|
| Dev | `outputs/dev_output/BC_S5/run_20260211_181635/` |
| Src | `outputs/BC_S5/run_20260301_111720/` |
| DB | `outputs/db_BC_S5_20260301_190902/` |

---

## 2. 性能对比分析

### 2.1 全周期运行时间 (共87天)
| 版本 | 总时间 | 平均每天 |
|------|--------|----------|
| **Dev** | 9540秒 (159.0分钟) | 109.7秒 |
| **Src** | 2897秒 (48.3分钟) | 33.3秒 |
| **DB** | 1563秒 (26.1分钟) | 18.0秒 |

### 2.2 性能提升比例
| 对比项 | 加速倍数 | 时间减少百分比 |
|--------|----------|----------------|
| **Src vs Dev** | **3.29x** | 69.6% |
| **DB vs Dev**  | **6.10x**  | 83.6% |
| **DB vs Src**  | **1.85x**  | 46.0% |

### 2.3 各模块平均耗时对比（秒/天，全87天均值）
| 模块 | Dev | Src | DB | Src加速 | DB加速 |
|------|-----|-----|-----|---------|--------|
| **M1 订单生成** | ~18.3 | ~5.7 | ~5.6 | 3.18x | 3.29x |
| **M3 净需求计算** | ~24.1 | ~5.8 | ~2.3 | 4.14x | 10.70x |
| **M5 部署规划** | ~60.6 | ~18.4 | ~7.1 | 3.29x | 8.54x |

---

## 3. 数据一致性验证

### 3.1 总体验证结果

| 模块 | 表名 | Dev总行数 | Src总行数 | DB总行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|-----------|-----------|----------|------------|-----------|-----------|
| Module1 | 订单日志 (OrderLog) | 424,848 | 424,848 | 424,848 | PASS | PASS | PASS |
| Module1 | 发货日志 (ShipmentLog) | 24,252 | 24,252 | 24,252 | PASS | PASS | PASS |
| Module1 | 削减日志 (CutLog) | 24,252 | 24,252 | 24,252 | PASS | PASS | PASS |
| Module1 | 供需日志 (SupplyDemandLog) | 2,084,295 | 2,084,295 | 2,084,295 | PASS | PASS | PASS |
| Module1 | 每日汇总 (Summary) | 87 | 87 | 87 | PASS | PASS | PASS |
| Module3 | 净需求 (NetDemand) | 30,702 | 30,702 | 30,702 | PASS | PASS | PASS |
| Module4 | 生产计划 (ProductionPlan) | 379 | 379 | 379 | PASS | PASS | PASS |
| Module4 | 超容量报告 (CapacityExceed) | 15 | 15 | 15 | PASS | PASS | PASS |
| Module4 | 换型日志 (ChangeoverLog) | 146 | 146 | 146 | PASS | PASS | PASS |
| Module5 | 部署计划 (DeploymentPlan) | 1,149,959 | 1,149,959 | 1,149,959 | PASS | PASS | PASS |
| Module5 | 未满足日志 (UnfulfilledLog) | 513,212 | 513,212 | 513,212 | PASS | PASS | PASS |
| Module5 | 库存日志 (StockOnHandLog) | 72,471 | 72,471 | 72,471 | PASS | PASS | PASS |
| Module5 | 验证 (Validation) | 40,368 | 40,368 | 40,368 | PASS | PASS | PASS |
| Module6 | 交付计划 (DeliveryPlan) | 43,055 | 43,055 | 43,055 | PASS | PASS | PASS |
| Module6 | 车辆日志 (VehicleLog) | 258 | 258 | 258 | PASS | PASS | PASS |
| Module6 | 卡车使用日志 (TruckUsageLog) | 192 | 192 | 192 | PASS | PASS | PASS |

> **Dev vs Src**: 16/16 表全部PASS  
> **Dev vs DB**:  16/16 表PASS  
> **Src vs DB**:  16/16 表PASS

---

### 3.2 Module1 - 订单生成模块

#### 订单日志 (OrderLog)

- **总行数 (全87天)**: Dev=424,848, Src=424,848, DB=424,848
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-10-05 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 02 | 2025-10-06 | 738 | 738 | 738 | PASS | PASS | PASS |
| Day 03 | 2025-10-07 | 1,194 | 1,194 | 1,194 | PASS | PASS | PASS |
| Day 04 | 2025-10-08 | 1,650 | 1,650 | 1,650 | PASS | PASS | PASS |
| Day 05 | 2025-10-09 | 2,106 | 2,106 | 2,106 | PASS | PASS | PASS |
| Day 06 | 2025-10-10 | 2,562 | 2,562 | 2,562 | PASS | PASS | PASS |
| Day 07 | 2025-10-11 | 3,018 | 3,018 | 3,018 | PASS | PASS | PASS |
| Day 08 | 2025-10-12 | 3,246 | 3,246 | 3,246 | PASS | PASS | PASS |
| Day 09 | 2025-10-13 | 3,474 | 3,474 | 3,474 | PASS | PASS | PASS |
| Day 10 | 2025-10-14 | 3,702 | 3,702 | 3,702 | PASS | PASS | PASS |
| Day 11 | 2025-10-15 | 3,930 | 3,930 | 3,930 | PASS | PASS | PASS |
| Day 12 | 2025-10-16 | 4,158 | 4,158 | 4,158 | PASS | PASS | PASS |
| Day 13 | 2025-10-17 | 4,386 | 4,386 | 4,386 | PASS | PASS | PASS |
| Day 14 | 2025-10-18 | 4,614 | 4,614 | 4,614 | PASS | PASS | PASS |
| Day 15 | 2025-10-19 | 4,842 | 4,842 | 4,842 | PASS | PASS | PASS |
| Day 16 | 2025-10-20 | 5,070 | 5,070 | 5,070 | PASS | PASS | PASS |
| Day 17 | 2025-10-21 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 18 | 2025-10-22 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 19 | 2025-10-23 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 20 | 2025-10-24 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 21 | 2025-10-25 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 22 | 2025-10-26 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 23 | 2025-10-27 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 24 | 2025-10-28 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 25 | 2025-10-29 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 26 | 2025-10-30 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 27 | 2025-10-31 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 28 | 2025-11-01 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 29 | 2025-11-02 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 30 | 2025-11-03 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 31 | 2025-11-04 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 32 | 2025-11-05 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 33 | 2025-11-06 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 34 | 2025-11-07 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 35 | 2025-11-08 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 36 | 2025-11-09 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 37 | 2025-11-10 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 38 | 2025-11-11 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 39 | 2025-11-12 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 40 | 2025-11-13 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 41 | 2025-11-14 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 42 | 2025-11-15 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 43 | 2025-11-16 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 44 | 2025-11-17 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 45 | 2025-11-18 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 46 | 2025-11-19 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 47 | 2025-11-20 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 48 | 2025-11-21 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 49 | 2025-11-22 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 50 | 2025-11-23 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 51 | 2025-11-24 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 52 | 2025-11-25 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 53 | 2025-11-26 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 54 | 2025-11-27 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 55 | 2025-11-28 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 56 | 2025-11-29 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 57 | 2025-11-30 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 58 | 2025-12-01 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 59 | 2025-12-02 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 60 | 2025-12-03 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 61 | 2025-12-04 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 62 | 2025-12-05 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 63 | 2025-12-06 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 64 | 2025-12-07 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 65 | 2025-12-08 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 66 | 2025-12-09 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 67 | 2025-12-10 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 68 | 2025-12-11 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 69 | 2025-12-12 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 70 | 2025-12-13 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 71 | 2025-12-14 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 72 | 2025-12-15 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 73 | 2025-12-16 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 74 | 2025-12-17 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 75 | 2025-12-18 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 76 | 2025-12-19 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 77 | 2025-12-20 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 78 | 2025-12-21 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 79 | 2025-12-22 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 80 | 2025-12-23 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 81 | 2025-12-24 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 82 | 2025-12-25 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 83 | 2025-12-26 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 84 | 2025-12-27 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 85 | 2025-12-28 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 86 | 2025-12-29 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| Day 87 | 2025-12-30 | 5,298 | 5,298 | 5,298 | PASS | PASS | PASS |
| **合计** | | **424,848** | **424,848** | **424,848** | **87/87天PASS** | **87/87天PASS** | **87/87天PASS** |

#### 发货日志 (ShipmentLog)

- **总行数 (全87天)**: Dev=24,252, Src=24,252, DB=24,252
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-10-05 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 02 | 2025-10-06 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 03 | 2025-10-07 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 04 | 2025-10-08 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 05 | 2025-10-09 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 06 | 2025-10-10 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 07 | 2025-10-11 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 08 | 2025-10-12 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 09 | 2025-10-13 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 10 | 2025-10-14 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 11 | 2025-10-15 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 12 | 2025-10-16 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 13 | 2025-10-17 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 14 | 2025-10-18 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 15 | 2025-10-19 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 16 | 2025-10-20 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 17 | 2025-10-21 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 18 | 2025-10-22 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 19 | 2025-10-23 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 20 | 2025-10-24 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 21 | 2025-10-25 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 22 | 2025-10-26 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 23 | 2025-10-27 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 24 | 2025-10-28 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 25 | 2025-10-29 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 26 | 2025-10-30 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 27 | 2025-10-31 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 28 | 2025-11-01 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 29 | 2025-11-02 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 30 | 2025-11-03 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 31 | 2025-11-04 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 32 | 2025-11-05 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 33 | 2025-11-06 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 34 | 2025-11-07 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 35 | 2025-11-08 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 36 | 2025-11-09 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 37 | 2025-11-10 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 38 | 2025-11-11 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 39 | 2025-11-12 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 40 | 2025-11-13 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 41 | 2025-11-14 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 42 | 2025-11-15 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 43 | 2025-11-16 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 44 | 2025-11-17 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 45 | 2025-11-18 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 46 | 2025-11-19 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 47 | 2025-11-20 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 48 | 2025-11-21 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 49 | 2025-11-22 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 50 | 2025-11-23 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 51 | 2025-11-24 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 52 | 2025-11-25 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 53 | 2025-11-26 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 54 | 2025-11-27 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 55 | 2025-11-28 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 56 | 2025-11-29 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 57 | 2025-11-30 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 58 | 2025-12-01 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 59 | 2025-12-02 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 60 | 2025-12-03 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 61 | 2025-12-04 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 62 | 2025-12-05 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 63 | 2025-12-06 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 64 | 2025-12-07 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 65 | 2025-12-08 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 66 | 2025-12-09 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 67 | 2025-12-10 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 68 | 2025-12-11 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 69 | 2025-12-12 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 70 | 2025-12-13 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 71 | 2025-12-14 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 72 | 2025-12-15 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 73 | 2025-12-16 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 74 | 2025-12-17 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 75 | 2025-12-18 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 76 | 2025-12-19 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 77 | 2025-12-20 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 78 | 2025-12-21 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 79 | 2025-12-22 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 80 | 2025-12-23 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 81 | 2025-12-24 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 82 | 2025-12-25 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 83 | 2025-12-26 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 84 | 2025-12-27 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 85 | 2025-12-28 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 86 | 2025-12-29 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 87 | 2025-12-30 | 282 | 282 | 282 | PASS | PASS | PASS |
| **合计** | | **24,252** | **24,252** | **24,252** | **87/87天PASS** | **87/87天PASS** | **87/87天PASS** |

#### 削减日志 (CutLog)

- **总行数 (全87天)**: Dev=24,252, Src=24,252, DB=24,252
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-10-05 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 02 | 2025-10-06 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 03 | 2025-10-07 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 04 | 2025-10-08 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 05 | 2025-10-09 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 06 | 2025-10-10 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 07 | 2025-10-11 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 08 | 2025-10-12 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 09 | 2025-10-13 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 10 | 2025-10-14 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 11 | 2025-10-15 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 12 | 2025-10-16 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 13 | 2025-10-17 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 14 | 2025-10-18 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 15 | 2025-10-19 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 16 | 2025-10-20 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 17 | 2025-10-21 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 18 | 2025-10-22 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 19 | 2025-10-23 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 20 | 2025-10-24 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 21 | 2025-10-25 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 22 | 2025-10-26 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 23 | 2025-10-27 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 24 | 2025-10-28 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 25 | 2025-10-29 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 26 | 2025-10-30 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 27 | 2025-10-31 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 28 | 2025-11-01 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 29 | 2025-11-02 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 30 | 2025-11-03 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 31 | 2025-11-04 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 32 | 2025-11-05 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 33 | 2025-11-06 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 34 | 2025-11-07 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 35 | 2025-11-08 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 36 | 2025-11-09 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 37 | 2025-11-10 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 38 | 2025-11-11 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 39 | 2025-11-12 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 40 | 2025-11-13 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 41 | 2025-11-14 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 42 | 2025-11-15 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 43 | 2025-11-16 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 44 | 2025-11-17 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 45 | 2025-11-18 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 46 | 2025-11-19 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 47 | 2025-11-20 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 48 | 2025-11-21 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 49 | 2025-11-22 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 50 | 2025-11-23 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 51 | 2025-11-24 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 52 | 2025-11-25 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 53 | 2025-11-26 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 54 | 2025-11-27 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 55 | 2025-11-28 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 56 | 2025-11-29 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 57 | 2025-11-30 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 58 | 2025-12-01 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 59 | 2025-12-02 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 60 | 2025-12-03 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 61 | 2025-12-04 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 62 | 2025-12-05 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 63 | 2025-12-06 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 64 | 2025-12-07 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 65 | 2025-12-08 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 66 | 2025-12-09 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 67 | 2025-12-10 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 68 | 2025-12-11 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 69 | 2025-12-12 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 70 | 2025-12-13 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 71 | 2025-12-14 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 72 | 2025-12-15 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 73 | 2025-12-16 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 74 | 2025-12-17 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 75 | 2025-12-18 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 76 | 2025-12-19 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 77 | 2025-12-20 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 78 | 2025-12-21 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 79 | 2025-12-22 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 80 | 2025-12-23 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 81 | 2025-12-24 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 82 | 2025-12-25 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 83 | 2025-12-26 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 84 | 2025-12-27 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 85 | 2025-12-28 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 86 | 2025-12-29 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 87 | 2025-12-30 | 282 | 282 | 282 | PASS | PASS | PASS |
| **合计** | | **24,252** | **24,252** | **24,252** | **87/87天PASS** | **87/87天PASS** | **87/87天PASS** |

#### 供需日志 (SupplyDemandLog)

- **总行数 (全87天)**: Dev=2,084,295, Src=2,084,295, DB=2,084,295
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-10-05 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 02 | 2025-10-06 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 03 | 2025-10-07 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 04 | 2025-10-08 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 05 | 2025-10-09 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 06 | 2025-10-10 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 07 | 2025-10-11 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 08 | 2025-10-12 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 09 | 2025-10-13 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 10 | 2025-10-14 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 11 | 2025-10-15 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 12 | 2025-10-16 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 13 | 2025-10-17 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 14 | 2025-10-18 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 15 | 2025-10-19 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 16 | 2025-10-20 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 17 | 2025-10-21 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 18 | 2025-10-22 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 19 | 2025-10-23 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 20 | 2025-10-24 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 21 | 2025-10-25 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 22 | 2025-10-26 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 23 | 2025-10-27 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 24 | 2025-10-28 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 25 | 2025-10-29 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 26 | 2025-10-30 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 27 | 2025-10-31 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 28 | 2025-11-01 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 29 | 2025-11-02 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 30 | 2025-11-03 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 31 | 2025-11-04 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 32 | 2025-11-05 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 33 | 2025-11-06 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 34 | 2025-11-07 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 35 | 2025-11-08 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 36 | 2025-11-09 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 37 | 2025-11-10 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 38 | 2025-11-11 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 39 | 2025-11-12 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 40 | 2025-11-13 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 41 | 2025-11-14 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 42 | 2025-11-15 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 43 | 2025-11-16 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 44 | 2025-11-17 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 45 | 2025-11-18 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 46 | 2025-11-19 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 47 | 2025-11-20 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 48 | 2025-11-21 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 49 | 2025-11-22 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 50 | 2025-11-23 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 51 | 2025-11-24 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 52 | 2025-11-25 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 53 | 2025-11-26 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 54 | 2025-11-27 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 55 | 2025-11-28 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 56 | 2025-11-29 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 57 | 2025-11-30 | 25,470 | 25,470 | 25,470 | PASS | PASS | PASS |
| Day 58 | 2025-12-01 | 25,187 | 25,187 | 25,187 | PASS | PASS | PASS |
| Day 59 | 2025-12-02 | 24,904 | 24,904 | 24,904 | PASS | PASS | PASS |
| Day 60 | 2025-12-03 | 24,621 | 24,621 | 24,621 | PASS | PASS | PASS |
| Day 61 | 2025-12-04 | 24,338 | 24,338 | 24,338 | PASS | PASS | PASS |
| Day 62 | 2025-12-05 | 24,055 | 24,055 | 24,055 | PASS | PASS | PASS |
| Day 63 | 2025-12-06 | 23,772 | 23,772 | 23,772 | PASS | PASS | PASS |
| Day 64 | 2025-12-07 | 23,489 | 23,489 | 23,489 | PASS | PASS | PASS |
| Day 65 | 2025-12-08 | 23,206 | 23,206 | 23,206 | PASS | PASS | PASS |
| Day 66 | 2025-12-09 | 22,923 | 22,923 | 22,923 | PASS | PASS | PASS |
| Day 67 | 2025-12-10 | 22,640 | 22,640 | 22,640 | PASS | PASS | PASS |
| Day 68 | 2025-12-11 | 22,357 | 22,357 | 22,357 | PASS | PASS | PASS |
| Day 69 | 2025-12-12 | 22,074 | 22,074 | 22,074 | PASS | PASS | PASS |
| Day 70 | 2025-12-13 | 21,791 | 21,791 | 21,791 | PASS | PASS | PASS |
| Day 71 | 2025-12-14 | 21,508 | 21,508 | 21,508 | PASS | PASS | PASS |
| Day 72 | 2025-12-15 | 21,225 | 21,225 | 21,225 | PASS | PASS | PASS |
| Day 73 | 2025-12-16 | 20,942 | 20,942 | 20,942 | PASS | PASS | PASS |
| Day 74 | 2025-12-17 | 20,659 | 20,659 | 20,659 | PASS | PASS | PASS |
| Day 75 | 2025-12-18 | 20,376 | 20,376 | 20,376 | PASS | PASS | PASS |
| Day 76 | 2025-12-19 | 20,093 | 20,093 | 20,093 | PASS | PASS | PASS |
| Day 77 | 2025-12-20 | 19,810 | 19,810 | 19,810 | PASS | PASS | PASS |
| Day 78 | 2025-12-21 | 19,527 | 19,527 | 19,527 | PASS | PASS | PASS |
| Day 79 | 2025-12-22 | 19,244 | 19,244 | 19,244 | PASS | PASS | PASS |
| Day 80 | 2025-12-23 | 18,961 | 18,961 | 18,961 | PASS | PASS | PASS |
| Day 81 | 2025-12-24 | 18,678 | 18,678 | 18,678 | PASS | PASS | PASS |
| Day 82 | 2025-12-25 | 18,395 | 18,395 | 18,395 | PASS | PASS | PASS |
| Day 83 | 2025-12-26 | 18,112 | 18,112 | 18,112 | PASS | PASS | PASS |
| Day 84 | 2025-12-27 | 17,829 | 17,829 | 17,829 | PASS | PASS | PASS |
| Day 85 | 2025-12-28 | 17,546 | 17,546 | 17,546 | PASS | PASS | PASS |
| Day 86 | 2025-12-29 | 17,263 | 17,263 | 17,263 | PASS | PASS | PASS |
| Day 87 | 2025-12-30 | 16,980 | 16,980 | 16,980 | PASS | PASS | PASS |
| **合计** | | **2,084,295** | **2,084,295** | **2,084,295** | **87/87天PASS** | **87/87天PASS** | **87/87天PASS** |

#### 每日汇总 (Summary)

- **总行数 (全87天)**: Dev=87, Src=87, DB=87
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-10-05 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 02 | 2025-10-06 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 03 | 2025-10-07 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 04 | 2025-10-08 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 05 | 2025-10-09 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 06 | 2025-10-10 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 07 | 2025-10-11 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 08 | 2025-10-12 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 09 | 2025-10-13 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 10 | 2025-10-14 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 11 | 2025-10-15 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 12 | 2025-10-16 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 13 | 2025-10-17 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 14 | 2025-10-18 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 15 | 2025-10-19 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 16 | 2025-10-20 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 17 | 2025-10-21 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 18 | 2025-10-22 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 19 | 2025-10-23 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 20 | 2025-10-24 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 21 | 2025-10-25 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 22 | 2025-10-26 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 23 | 2025-10-27 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 24 | 2025-10-28 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 25 | 2025-10-29 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 26 | 2025-10-30 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 27 | 2025-10-31 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 28 | 2025-11-01 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 29 | 2025-11-02 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 30 | 2025-11-03 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 31 | 2025-11-04 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 32 | 2025-11-05 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 33 | 2025-11-06 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 34 | 2025-11-07 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 35 | 2025-11-08 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 36 | 2025-11-09 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 37 | 2025-11-10 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 38 | 2025-11-11 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 39 | 2025-11-12 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 40 | 2025-11-13 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 41 | 2025-11-14 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 42 | 2025-11-15 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 43 | 2025-11-16 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 44 | 2025-11-17 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 45 | 2025-11-18 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 46 | 2025-11-19 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 47 | 2025-11-20 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 48 | 2025-11-21 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 49 | 2025-11-22 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 50 | 2025-11-23 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 51 | 2025-11-24 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 52 | 2025-11-25 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 53 | 2025-11-26 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 54 | 2025-11-27 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 55 | 2025-11-28 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 56 | 2025-11-29 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 57 | 2025-11-30 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 58 | 2025-12-01 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 59 | 2025-12-02 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 60 | 2025-12-03 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 61 | 2025-12-04 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 62 | 2025-12-05 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 63 | 2025-12-06 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 64 | 2025-12-07 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 65 | 2025-12-08 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 66 | 2025-12-09 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 67 | 2025-12-10 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 68 | 2025-12-11 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 69 | 2025-12-12 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 70 | 2025-12-13 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 71 | 2025-12-14 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 72 | 2025-12-15 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 73 | 2025-12-16 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 74 | 2025-12-17 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 75 | 2025-12-18 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 76 | 2025-12-19 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 77 | 2025-12-20 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 78 | 2025-12-21 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 79 | 2025-12-22 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 80 | 2025-12-23 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 81 | 2025-12-24 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 82 | 2025-12-25 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 83 | 2025-12-26 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 84 | 2025-12-27 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 85 | 2025-12-28 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 86 | 2025-12-29 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 87 | 2025-12-30 | 1 | 1 | 1 | PASS | PASS | PASS |
| **合计** | | **87** | **87** | **87** | **87/87天PASS** | **87/87天PASS** | **87/87天PASS** |

### 3.3 Module3 - 净需求计算模块

#### 净需求 (NetDemand)

- **总行数 (全87天)**: Dev=30,702, Src=30,702, DB=30,702
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-10-05 | 453 | 453 | 453 | PASS | PASS | PASS |
| Day 02 | 2025-10-06 | 497 | 497 | 497 | PASS | PASS | PASS |
| Day 03 | 2025-10-07 | 491 | 491 | 491 | PASS | PASS | PASS |
| Day 04 | 2025-10-08 | 498 | 498 | 498 | PASS | PASS | PASS |
| Day 05 | 2025-10-09 | 512 | 512 | 512 | PASS | PASS | PASS |
| Day 06 | 2025-10-10 | 529 | 529 | 529 | PASS | PASS | PASS |
| Day 07 | 2025-10-11 | 517 | 517 | 517 | PASS | PASS | PASS |
| Day 08 | 2025-10-12 | 507 | 507 | 507 | PASS | PASS | PASS |
| Day 09 | 2025-10-13 | 508 | 508 | 508 | PASS | PASS | PASS |
| Day 10 | 2025-10-14 | 510 | 510 | 510 | PASS | PASS | PASS |
| Day 11 | 2025-10-15 | 513 | 513 | 513 | PASS | PASS | PASS |
| Day 12 | 2025-10-16 | 469 | 469 | 469 | PASS | PASS | PASS |
| Day 13 | 2025-10-17 | 463 | 463 | 463 | PASS | PASS | PASS |
| Day 14 | 2025-10-18 | 419 | 419 | 419 | PASS | PASS | PASS |
| Day 15 | 2025-10-19 | 405 | 405 | 405 | PASS | PASS | PASS |
| Day 16 | 2025-10-20 | 373 | 373 | 373 | PASS | PASS | PASS |
| Day 17 | 2025-10-21 | 339 | 339 | 339 | PASS | PASS | PASS |
| Day 18 | 2025-10-22 | 292 | 292 | 292 | PASS | PASS | PASS |
| Day 19 | 2025-10-23 | 260 | 260 | 260 | PASS | PASS | PASS |
| Day 20 | 2025-10-24 | 223 | 223 | 223 | PASS | PASS | PASS |
| Day 21 | 2025-10-25 | 205 | 205 | 205 | PASS | PASS | PASS |
| Day 22 | 2025-10-26 | 197 | 197 | 197 | PASS | PASS | PASS |
| Day 23 | 2025-10-27 | 202 | 202 | 202 | PASS | PASS | PASS |
| Day 24 | 2025-10-28 | 222 | 222 | 222 | PASS | PASS | PASS |
| Day 25 | 2025-10-29 | 221 | 221 | 221 | PASS | PASS | PASS |
| Day 26 | 2025-10-30 | 240 | 240 | 240 | PASS | PASS | PASS |
| Day 27 | 2025-10-31 | 270 | 270 | 270 | PASS | PASS | PASS |
| Day 28 | 2025-11-01 | 266 | 266 | 266 | PASS | PASS | PASS |
| Day 29 | 2025-11-02 | 290 | 290 | 290 | PASS | PASS | PASS |
| Day 30 | 2025-11-03 | 307 | 307 | 307 | PASS | PASS | PASS |
| Day 31 | 2025-11-04 | 317 | 317 | 317 | PASS | PASS | PASS |
| Day 32 | 2025-11-05 | 277 | 277 | 277 | PASS | PASS | PASS |
| Day 33 | 2025-11-06 | 272 | 272 | 272 | PASS | PASS | PASS |
| Day 34 | 2025-11-07 | 237 | 237 | 237 | PASS | PASS | PASS |
| Day 35 | 2025-11-08 | 239 | 239 | 239 | PASS | PASS | PASS |
| Day 36 | 2025-11-09 | 249 | 249 | 249 | PASS | PASS | PASS |
| Day 37 | 2025-11-10 | 264 | 264 | 264 | PASS | PASS | PASS |
| Day 38 | 2025-11-11 | 268 | 268 | 268 | PASS | PASS | PASS |
| Day 39 | 2025-11-12 | 295 | 295 | 295 | PASS | PASS | PASS |
| Day 40 | 2025-11-13 | 295 | 295 | 295 | PASS | PASS | PASS |
| Day 41 | 2025-11-14 | 310 | 310 | 310 | PASS | PASS | PASS |
| Day 42 | 2025-11-15 | 282 | 282 | 282 | PASS | PASS | PASS |
| Day 43 | 2025-11-16 | 265 | 265 | 265 | PASS | PASS | PASS |
| Day 44 | 2025-11-17 | 289 | 289 | 289 | PASS | PASS | PASS |
| Day 45 | 2025-11-18 | 305 | 305 | 305 | PASS | PASS | PASS |
| Day 46 | 2025-11-19 | 327 | 327 | 327 | PASS | PASS | PASS |
| Day 47 | 2025-11-20 | 296 | 296 | 296 | PASS | PASS | PASS |
| Day 48 | 2025-11-21 | 277 | 277 | 277 | PASS | PASS | PASS |
| Day 49 | 2025-11-22 | 269 | 269 | 269 | PASS | PASS | PASS |
| Day 50 | 2025-11-23 | 291 | 291 | 291 | PASS | PASS | PASS |
| Day 51 | 2025-11-24 | 304 | 304 | 304 | PASS | PASS | PASS |
| Day 52 | 2025-11-25 | 319 | 319 | 319 | PASS | PASS | PASS |
| Day 53 | 2025-11-26 | 324 | 324 | 324 | PASS | PASS | PASS |
| Day 54 | 2025-11-27 | 346 | 346 | 346 | PASS | PASS | PASS |
| Day 55 | 2025-11-28 | 345 | 345 | 345 | PASS | PASS | PASS |
| Day 56 | 2025-11-29 | 371 | 371 | 371 | PASS | PASS | PASS |
| Day 57 | 2025-11-30 | 386 | 386 | 386 | PASS | PASS | PASS |
| Day 58 | 2025-12-01 | 401 | 401 | 401 | PASS | PASS | PASS |
| Day 59 | 2025-12-02 | 391 | 391 | 391 | PASS | PASS | PASS |
| Day 60 | 2025-12-03 | 399 | 399 | 399 | PASS | PASS | PASS |
| Day 61 | 2025-12-04 | 408 | 408 | 408 | PASS | PASS | PASS |
| Day 62 | 2025-12-05 | 353 | 353 | 353 | PASS | PASS | PASS |
| Day 63 | 2025-12-06 | 361 | 361 | 361 | PASS | PASS | PASS |
| Day 64 | 2025-12-07 | 374 | 374 | 374 | PASS | PASS | PASS |
| Day 65 | 2025-12-08 | 371 | 371 | 371 | PASS | PASS | PASS |
| Day 66 | 2025-12-09 | 373 | 373 | 373 | PASS | PASS | PASS |
| Day 67 | 2025-12-10 | 377 | 377 | 377 | PASS | PASS | PASS |
| Day 68 | 2025-12-11 | 363 | 363 | 363 | PASS | PASS | PASS |
| Day 69 | 2025-12-12 | 369 | 369 | 369 | PASS | PASS | PASS |
| Day 70 | 2025-12-13 | 351 | 351 | 351 | PASS | PASS | PASS |
| Day 71 | 2025-12-14 | 364 | 364 | 364 | PASS | PASS | PASS |
| Day 72 | 2025-12-15 | 369 | 369 | 369 | PASS | PASS | PASS |
| Day 73 | 2025-12-16 | 349 | 349 | 349 | PASS | PASS | PASS |
| Day 74 | 2025-12-17 | 363 | 363 | 363 | PASS | PASS | PASS |
| Day 75 | 2025-12-18 | 380 | 380 | 380 | PASS | PASS | PASS |
| Day 76 | 2025-12-19 | 392 | 392 | 392 | PASS | PASS | PASS |
| Day 77 | 2025-12-20 | 382 | 382 | 382 | PASS | PASS | PASS |
| Day 78 | 2025-12-21 | 388 | 388 | 388 | PASS | PASS | PASS |
| Day 79 | 2025-12-22 | 398 | 398 | 398 | PASS | PASS | PASS |
| Day 80 | 2025-12-23 | 405 | 405 | 405 | PASS | PASS | PASS |
| Day 81 | 2025-12-24 | 435 | 435 | 435 | PASS | PASS | PASS |
| Day 82 | 2025-12-25 | 407 | 407 | 407 | PASS | PASS | PASS |
| Day 83 | 2025-12-26 | 410 | 410 | 410 | PASS | PASS | PASS |
| Day 84 | 2025-12-27 | 395 | 395 | 395 | PASS | PASS | PASS |
| Day 85 | 2025-12-28 | 371 | 371 | 371 | PASS | PASS | PASS |
| Day 86 | 2025-12-29 | 392 | 392 | 392 | PASS | PASS | PASS |
| Day 87 | 2025-12-30 | 394 | 394 | 394 | PASS | PASS | PASS |
| **合计** | | **30,702** | **30,702** | **30,702** | **87/87天PASS** | **87/87天PASS** | **87/87天PASS** |

### 3.4 Module4 - 生产计划模块

#### 生产计划 (ProductionPlan)

- **总行数 (全87天)**: Dev=379, Src=379, DB=379
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-10-05 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 02 | 2025-10-06 | 65 | 65 | 65 | PASS | PASS | PASS |
| Day 03 | 2025-10-07 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 04 | 2025-10-08 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 05 | 2025-10-09 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 06 | 2025-10-10 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 07 | 2025-10-11 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 08 | 2025-10-12 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 09 | 2025-10-13 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 10 | 2025-10-14 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 11 | 2025-10-15 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 12 | 2025-10-16 | 17 | 17 | 17 | PASS | PASS | PASS |
| Day 13 | 2025-10-17 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 14 | 2025-10-18 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 15 | 2025-10-19 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 16 | 2025-10-20 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 17 | 2025-10-21 | 26 | 26 | 26 | PASS | PASS | PASS |
| Day 18 | 2025-10-22 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 19 | 2025-10-23 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 20 | 2025-10-24 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 21 | 2025-10-25 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 22 | 2025-10-26 | 23 | 23 | 23 | PASS | PASS | PASS |
| Day 23 | 2025-10-27 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 24 | 2025-10-28 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 25 | 2025-10-29 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 26 | 2025-10-30 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 27 | 2025-10-31 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 28 | 2025-11-01 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 29 | 2025-11-02 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 30 | 2025-11-03 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 31 | 2025-11-04 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 32 | 2025-11-05 | 43 | 43 | 43 | PASS | PASS | PASS |
| Day 33 | 2025-11-06 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 34 | 2025-11-07 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 35 | 2025-11-08 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 36 | 2025-11-09 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 37 | 2025-11-10 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 38 | 2025-11-11 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 39 | 2025-11-12 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 40 | 2025-11-13 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 41 | 2025-11-14 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 42 | 2025-11-15 | 31 | 31 | 31 | PASS | PASS | PASS |
| Day 43 | 2025-11-16 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 44 | 2025-11-17 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 45 | 2025-11-18 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 46 | 2025-11-19 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 47 | 2025-11-20 | 23 | 23 | 23 | PASS | PASS | PASS |
| Day 48 | 2025-11-21 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 49 | 2025-11-22 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 50 | 2025-11-23 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 51 | 2025-11-24 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 52 | 2025-11-25 | 14 | 14 | 14 | PASS | PASS | PASS |
| Day 53 | 2025-11-26 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 54 | 2025-11-27 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 55 | 2025-11-28 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 56 | 2025-11-29 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 57 | 2025-11-30 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 58 | 2025-12-01 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 59 | 2025-12-02 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 60 | 2025-12-03 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 61 | 2025-12-04 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 62 | 2025-12-05 | 63 | 63 | 63 | PASS | PASS | PASS |
| Day 63 | 2025-12-06 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 64 | 2025-12-07 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 65 | 2025-12-08 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 66 | 2025-12-09 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 67 | 2025-12-10 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 68 | 2025-12-11 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 69 | 2025-12-12 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 70 | 2025-12-13 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 71 | 2025-12-14 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 72 | 2025-12-15 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 73 | 2025-12-16 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 74 | 2025-12-17 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 75 | 2025-12-18 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 76 | 2025-12-19 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 77 | 2025-12-20 | 25 | 25 | 25 | PASS | PASS | PASS |
| Day 78 | 2025-12-21 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 79 | 2025-12-22 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 80 | 2025-12-23 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 81 | 2025-12-24 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 82 | 2025-12-25 | 34 | 34 | 34 | PASS | PASS | PASS |
| Day 83 | 2025-12-26 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 84 | 2025-12-27 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 85 | 2025-12-28 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 86 | 2025-12-29 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 87 | 2025-12-30 | 0 | 0 | 0 | PASS | PASS | PASS |
| **合计** | | **379** | **379** | **379** | **87/87天PASS** | **87/87天PASS** | **87/87天PASS** |

#### 超容量报告 (CapacityExceed)

- **总行数 (全87天)**: Dev=15, Src=15, DB=15
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-10-05 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 02 | 2025-10-06 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 03 | 2025-10-07 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 04 | 2025-10-08 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 05 | 2025-10-09 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 06 | 2025-10-10 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 07 | 2025-10-11 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 08 | 2025-10-12 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 09 | 2025-10-13 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 10 | 2025-10-14 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 11 | 2025-10-15 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 12 | 2025-10-16 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 13 | 2025-10-17 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 14 | 2025-10-18 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 15 | 2025-10-19 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 16 | 2025-10-20 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 17 | 2025-10-21 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 18 | 2025-10-22 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 19 | 2025-10-23 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 20 | 2025-10-24 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 21 | 2025-10-25 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 22 | 2025-10-26 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 23 | 2025-10-27 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 24 | 2025-10-28 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 25 | 2025-10-29 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 26 | 2025-10-30 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 27 | 2025-10-31 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 28 | 2025-11-01 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 29 | 2025-11-02 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 30 | 2025-11-03 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 31 | 2025-11-04 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 32 | 2025-11-05 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 33 | 2025-11-06 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 34 | 2025-11-07 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 35 | 2025-11-08 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 36 | 2025-11-09 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 37 | 2025-11-10 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 38 | 2025-11-11 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 39 | 2025-11-12 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 40 | 2025-11-13 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 41 | 2025-11-14 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 42 | 2025-11-15 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 43 | 2025-11-16 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 44 | 2025-11-17 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 45 | 2025-11-18 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 46 | 2025-11-19 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 47 | 2025-11-20 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 48 | 2025-11-21 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 49 | 2025-11-22 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 50 | 2025-11-23 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 51 | 2025-11-24 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 52 | 2025-11-25 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 53 | 2025-11-26 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 54 | 2025-11-27 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 55 | 2025-11-28 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 56 | 2025-11-29 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 57 | 2025-11-30 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 58 | 2025-12-01 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 59 | 2025-12-02 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 60 | 2025-12-03 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 61 | 2025-12-04 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 62 | 2025-12-05 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 63 | 2025-12-06 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 64 | 2025-12-07 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 65 | 2025-12-08 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 66 | 2025-12-09 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 67 | 2025-12-10 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 68 | 2025-12-11 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 69 | 2025-12-12 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 70 | 2025-12-13 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 71 | 2025-12-14 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 72 | 2025-12-15 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 73 | 2025-12-16 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 74 | 2025-12-17 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 75 | 2025-12-18 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 76 | 2025-12-19 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 77 | 2025-12-20 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 78 | 2025-12-21 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 79 | 2025-12-22 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 80 | 2025-12-23 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 81 | 2025-12-24 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 82 | 2025-12-25 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 83 | 2025-12-26 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 84 | 2025-12-27 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 85 | 2025-12-28 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 86 | 2025-12-29 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 87 | 2025-12-30 | 0 | 0 | 0 | PASS | PASS | PASS |
| **合计** | | **15** | **15** | **15** | **87/87天PASS** | **87/87天PASS** | **87/87天PASS** |

#### 换型日志 (ChangeoverLog)

- **总行数 (全87天)**: Dev=146, Src=146, DB=146
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-10-05 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 02 | 2025-10-06 | 25 | 25 | 25 | PASS | PASS | PASS |
| Day 03 | 2025-10-07 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 04 | 2025-10-08 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 05 | 2025-10-09 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 06 | 2025-10-10 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 07 | 2025-10-11 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 08 | 2025-10-12 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 09 | 2025-10-13 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 10 | 2025-10-14 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 11 | 2025-10-15 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 12 | 2025-10-16 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 13 | 2025-10-17 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 14 | 2025-10-18 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 15 | 2025-10-19 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 16 | 2025-10-20 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 17 | 2025-10-21 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 18 | 2025-10-22 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 19 | 2025-10-23 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 20 | 2025-10-24 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 21 | 2025-10-25 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 22 | 2025-10-26 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 23 | 2025-10-27 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 24 | 2025-10-28 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 25 | 2025-10-29 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 26 | 2025-10-30 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 27 | 2025-10-31 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 28 | 2025-11-01 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 29 | 2025-11-02 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 30 | 2025-11-03 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 31 | 2025-11-04 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 32 | 2025-11-05 | 17 | 17 | 17 | PASS | PASS | PASS |
| Day 33 | 2025-11-06 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 34 | 2025-11-07 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 35 | 2025-11-08 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 36 | 2025-11-09 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 37 | 2025-11-10 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 38 | 2025-11-11 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 39 | 2025-11-12 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 40 | 2025-11-13 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 41 | 2025-11-14 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 42 | 2025-11-15 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 43 | 2025-11-16 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 44 | 2025-11-17 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 45 | 2025-11-18 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 46 | 2025-11-19 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 47 | 2025-11-20 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 48 | 2025-11-21 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 49 | 2025-11-22 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 50 | 2025-11-23 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 51 | 2025-11-24 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 52 | 2025-11-25 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 53 | 2025-11-26 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 54 | 2025-11-27 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 55 | 2025-11-28 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 56 | 2025-11-29 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 57 | 2025-11-30 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 58 | 2025-12-01 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 59 | 2025-12-02 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 60 | 2025-12-03 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 61 | 2025-12-04 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 62 | 2025-12-05 | 28 | 28 | 28 | PASS | PASS | PASS |
| Day 63 | 2025-12-06 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 64 | 2025-12-07 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 65 | 2025-12-08 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 66 | 2025-12-09 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 67 | 2025-12-10 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 68 | 2025-12-11 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 69 | 2025-12-12 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 70 | 2025-12-13 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 71 | 2025-12-14 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 72 | 2025-12-15 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 73 | 2025-12-16 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 74 | 2025-12-17 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 75 | 2025-12-18 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 76 | 2025-12-19 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 77 | 2025-12-20 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 78 | 2025-12-21 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 79 | 2025-12-22 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 80 | 2025-12-23 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 81 | 2025-12-24 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 82 | 2025-12-25 | 14 | 14 | 14 | PASS | PASS | PASS |
| Day 83 | 2025-12-26 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 84 | 2025-12-27 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 85 | 2025-12-28 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 86 | 2025-12-29 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 87 | 2025-12-30 | 0 | 0 | 0 | PASS | PASS | PASS |
| **合计** | | **146** | **146** | **146** | **87/87天PASS** | **87/87天PASS** | **87/87天PASS** |

### 3.5 Module5 - 部署规划模块

#### 部署计划 (DeploymentPlan)

- **总行数 (全87天)**: Dev=1,149,959, Src=1,149,959, DB=1,149,959
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-10-05 | 9,742 | 9,742 | 9,742 | PASS | PASS | PASS |
| Day 02 | 2025-10-06 | 10,676 | 10,676 | 10,676 | PASS | PASS | PASS |
| Day 03 | 2025-10-07 | 11,212 | 11,212 | 11,212 | PASS | PASS | PASS |
| Day 04 | 2025-10-08 | 11,767 | 11,767 | 11,767 | PASS | PASS | PASS |
| Day 05 | 2025-10-09 | 12,285 | 12,285 | 12,285 | PASS | PASS | PASS |
| Day 06 | 2025-10-10 | 12,881 | 12,881 | 12,881 | PASS | PASS | PASS |
| Day 07 | 2025-10-11 | 13,396 | 13,396 | 13,396 | PASS | PASS | PASS |
| Day 08 | 2025-10-12 | 13,639 | 13,639 | 13,639 | PASS | PASS | PASS |
| Day 09 | 2025-10-13 | 13,886 | 13,886 | 13,886 | PASS | PASS | PASS |
| Day 10 | 2025-10-14 | 13,795 | 13,795 | 13,795 | PASS | PASS | PASS |
| Day 11 | 2025-10-15 | 13,834 | 13,834 | 13,834 | PASS | PASS | PASS |
| Day 12 | 2025-10-16 | 14,042 | 14,042 | 14,042 | PASS | PASS | PASS |
| Day 13 | 2025-10-17 | 14,117 | 14,117 | 14,117 | PASS | PASS | PASS |
| Day 14 | 2025-10-18 | 14,004 | 14,004 | 14,004 | PASS | PASS | PASS |
| Day 15 | 2025-10-19 | 13,821 | 13,821 | 13,821 | PASS | PASS | PASS |
| Day 16 | 2025-10-20 | 13,628 | 13,628 | 13,628 | PASS | PASS | PASS |
| Day 17 | 2025-10-21 | 13,683 | 13,683 | 13,683 | PASS | PASS | PASS |
| Day 18 | 2025-10-22 | 13,574 | 13,574 | 13,574 | PASS | PASS | PASS |
| Day 19 | 2025-10-23 | 13,261 | 13,261 | 13,261 | PASS | PASS | PASS |
| Day 20 | 2025-10-24 | 12,888 | 12,888 | 12,888 | PASS | PASS | PASS |
| Day 21 | 2025-10-25 | 12,407 | 12,407 | 12,407 | PASS | PASS | PASS |
| Day 22 | 2025-10-26 | 12,283 | 12,283 | 12,283 | PASS | PASS | PASS |
| Day 23 | 2025-10-27 | 12,130 | 12,130 | 12,130 | PASS | PASS | PASS |
| Day 24 | 2025-10-28 | 11,841 | 11,841 | 11,841 | PASS | PASS | PASS |
| Day 25 | 2025-10-29 | 11,790 | 11,790 | 11,790 | PASS | PASS | PASS |
| Day 26 | 2025-10-30 | 11,815 | 11,815 | 11,815 | PASS | PASS | PASS |
| Day 27 | 2025-10-31 | 11,866 | 11,866 | 11,866 | PASS | PASS | PASS |
| Day 28 | 2025-11-01 | 11,871 | 11,871 | 11,871 | PASS | PASS | PASS |
| Day 29 | 2025-11-02 | 12,047 | 12,047 | 12,047 | PASS | PASS | PASS |
| Day 30 | 2025-11-03 | 12,126 | 12,126 | 12,126 | PASS | PASS | PASS |
| Day 31 | 2025-11-04 | 12,097 | 12,097 | 12,097 | PASS | PASS | PASS |
| Day 32 | 2025-11-05 | 12,161 | 12,161 | 12,161 | PASS | PASS | PASS |
| Day 33 | 2025-11-06 | 12,420 | 12,420 | 12,420 | PASS | PASS | PASS |
| Day 34 | 2025-11-07 | 12,496 | 12,496 | 12,496 | PASS | PASS | PASS |
| Day 35 | 2025-11-08 | 12,508 | 12,508 | 12,508 | PASS | PASS | PASS |
| Day 36 | 2025-11-09 | 12,388 | 12,388 | 12,388 | PASS | PASS | PASS |
| Day 37 | 2025-11-10 | 12,504 | 12,504 | 12,504 | PASS | PASS | PASS |
| Day 38 | 2025-11-11 | 12,148 | 12,148 | 12,148 | PASS | PASS | PASS |
| Day 39 | 2025-11-12 | 12,134 | 12,134 | 12,134 | PASS | PASS | PASS |
| Day 40 | 2025-11-13 | 12,205 | 12,205 | 12,205 | PASS | PASS | PASS |
| Day 41 | 2025-11-14 | 12,237 | 12,237 | 12,237 | PASS | PASS | PASS |
| Day 42 | 2025-11-15 | 12,368 | 12,368 | 12,368 | PASS | PASS | PASS |
| Day 43 | 2025-11-16 | 12,451 | 12,451 | 12,451 | PASS | PASS | PASS |
| Day 44 | 2025-11-17 | 12,490 | 12,490 | 12,490 | PASS | PASS | PASS |
| Day 45 | 2025-11-18 | 12,780 | 12,780 | 12,780 | PASS | PASS | PASS |
| Day 46 | 2025-11-19 | 12,879 | 12,879 | 12,879 | PASS | PASS | PASS |
| Day 47 | 2025-11-20 | 12,866 | 12,866 | 12,866 | PASS | PASS | PASS |
| Day 48 | 2025-11-21 | 12,887 | 12,887 | 12,887 | PASS | PASS | PASS |
| Day 49 | 2025-11-22 | 12,913 | 12,913 | 12,913 | PASS | PASS | PASS |
| Day 50 | 2025-11-23 | 12,863 | 12,863 | 12,863 | PASS | PASS | PASS |
| Day 51 | 2025-11-24 | 12,895 | 12,895 | 12,895 | PASS | PASS | PASS |
| Day 52 | 2025-11-25 | 12,881 | 12,881 | 12,881 | PASS | PASS | PASS |
| Day 53 | 2025-11-26 | 13,023 | 13,023 | 13,023 | PASS | PASS | PASS |
| Day 54 | 2025-11-27 | 13,202 | 13,202 | 13,202 | PASS | PASS | PASS |
| Day 55 | 2025-11-28 | 13,522 | 13,522 | 13,522 | PASS | PASS | PASS |
| Day 56 | 2025-11-29 | 13,679 | 13,679 | 13,679 | PASS | PASS | PASS |
| Day 57 | 2025-11-30 | 13,619 | 13,619 | 13,619 | PASS | PASS | PASS |
| Day 58 | 2025-12-01 | 13,602 | 13,602 | 13,602 | PASS | PASS | PASS |
| Day 59 | 2025-12-02 | 13,717 | 13,717 | 13,717 | PASS | PASS | PASS |
| Day 60 | 2025-12-03 | 13,887 | 13,887 | 13,887 | PASS | PASS | PASS |
| Day 61 | 2025-12-04 | 13,961 | 13,961 | 13,961 | PASS | PASS | PASS |
| Day 62 | 2025-12-05 | 13,889 | 13,889 | 13,889 | PASS | PASS | PASS |
| Day 63 | 2025-12-06 | 14,046 | 14,046 | 14,046 | PASS | PASS | PASS |
| Day 64 | 2025-12-07 | 14,333 | 14,333 | 14,333 | PASS | PASS | PASS |
| Day 65 | 2025-12-08 | 14,561 | 14,561 | 14,561 | PASS | PASS | PASS |
| Day 66 | 2025-12-09 | 14,628 | 14,628 | 14,628 | PASS | PASS | PASS |
| Day 67 | 2025-12-10 | 14,607 | 14,607 | 14,607 | PASS | PASS | PASS |
| Day 68 | 2025-12-11 | 14,543 | 14,543 | 14,543 | PASS | PASS | PASS |
| Day 69 | 2025-12-12 | 14,156 | 14,156 | 14,156 | PASS | PASS | PASS |
| Day 70 | 2025-12-13 | 14,311 | 14,311 | 14,311 | PASS | PASS | PASS |
| Day 71 | 2025-12-14 | 14,123 | 14,123 | 14,123 | PASS | PASS | PASS |
| Day 72 | 2025-12-15 | 14,098 | 14,098 | 14,098 | PASS | PASS | PASS |
| Day 73 | 2025-12-16 | 14,016 | 14,016 | 14,016 | PASS | PASS | PASS |
| Day 74 | 2025-12-17 | 13,789 | 13,789 | 13,789 | PASS | PASS | PASS |
| Day 75 | 2025-12-18 | 13,936 | 13,936 | 13,936 | PASS | PASS | PASS |
| Day 76 | 2025-12-19 | 13,854 | 13,854 | 13,854 | PASS | PASS | PASS |
| Day 77 | 2025-12-20 | 13,886 | 13,886 | 13,886 | PASS | PASS | PASS |
| Day 78 | 2025-12-21 | 14,093 | 14,093 | 14,093 | PASS | PASS | PASS |
| Day 79 | 2025-12-22 | 14,178 | 14,178 | 14,178 | PASS | PASS | PASS |
| Day 80 | 2025-12-23 | 14,352 | 14,352 | 14,352 | PASS | PASS | PASS |
| Day 81 | 2025-12-24 | 14,494 | 14,494 | 14,494 | PASS | PASS | PASS |
| Day 82 | 2025-12-25 | 14,494 | 14,494 | 14,494 | PASS | PASS | PASS |
| Day 83 | 2025-12-26 | 14,614 | 14,614 | 14,614 | PASS | PASS | PASS |
| Day 84 | 2025-12-27 | 14,632 | 14,632 | 14,632 | PASS | PASS | PASS |
| Day 85 | 2025-12-28 | 14,691 | 14,691 | 14,691 | PASS | PASS | PASS |
| Day 86 | 2025-12-29 | 14,791 | 14,791 | 14,791 | PASS | PASS | PASS |
| Day 87 | 2025-12-30 | 14,754 | 14,754 | 14,754 | PASS | PASS | PASS |
| **合计** | | **1,149,959** | **1,149,959** | **1,149,959** | **87/87天PASS** | **87/87天PASS** | **87/87天PASS** |

#### 未满足日志 (UnfulfilledLog)

- **总行数 (全87天)**: Dev=513,212, Src=513,212, DB=513,212
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-10-05 | 8,657 | 8,657 | 8,657 | PASS | PASS | PASS |
| Day 02 | 2025-10-06 | 8,081 | 8,081 | 8,081 | PASS | PASS | PASS |
| Day 03 | 2025-10-07 | 9,337 | 9,337 | 9,337 | PASS | PASS | PASS |
| Day 04 | 2025-10-08 | 9,381 | 9,381 | 9,381 | PASS | PASS | PASS |
| Day 05 | 2025-10-09 | 9,701 | 9,701 | 9,701 | PASS | PASS | PASS |
| Day 06 | 2025-10-10 | 9,988 | 9,988 | 9,988 | PASS | PASS | PASS |
| Day 07 | 2025-10-11 | 10,066 | 10,066 | 10,066 | PASS | PASS | PASS |
| Day 08 | 2025-10-12 | 9,860 | 9,860 | 9,860 | PASS | PASS | PASS |
| Day 09 | 2025-10-13 | 9,817 | 9,817 | 9,817 | PASS | PASS | PASS |
| Day 10 | 2025-10-14 | 9,349 | 9,349 | 9,349 | PASS | PASS | PASS |
| Day 11 | 2025-10-15 | 8,983 | 8,983 | 8,983 | PASS | PASS | PASS |
| Day 12 | 2025-10-16 | 8,663 | 8,663 | 8,663 | PASS | PASS | PASS |
| Day 13 | 2025-10-17 | 8,387 | 8,387 | 8,387 | PASS | PASS | PASS |
| Day 14 | 2025-10-18 | 7,576 | 7,576 | 7,576 | PASS | PASS | PASS |
| Day 15 | 2025-10-19 | 6,685 | 6,685 | 6,685 | PASS | PASS | PASS |
| Day 16 | 2025-10-20 | 6,288 | 6,288 | 6,288 | PASS | PASS | PASS |
| Day 17 | 2025-10-21 | 5,918 | 5,918 | 5,918 | PASS | PASS | PASS |
| Day 18 | 2025-10-22 | 5,316 | 5,316 | 5,316 | PASS | PASS | PASS |
| Day 19 | 2025-10-23 | 4,856 | 4,856 | 4,856 | PASS | PASS | PASS |
| Day 20 | 2025-10-24 | 3,947 | 3,947 | 3,947 | PASS | PASS | PASS |
| Day 21 | 2025-10-25 | 3,211 | 3,211 | 3,211 | PASS | PASS | PASS |
| Day 22 | 2025-10-26 | 3,056 | 3,056 | 3,056 | PASS | PASS | PASS |
| Day 23 | 2025-10-27 | 2,551 | 2,551 | 2,551 | PASS | PASS | PASS |
| Day 24 | 2025-10-28 | 2,479 | 2,479 | 2,479 | PASS | PASS | PASS |
| Day 25 | 2025-10-29 | 2,385 | 2,385 | 2,385 | PASS | PASS | PASS |
| Day 26 | 2025-10-30 | 2,422 | 2,422 | 2,422 | PASS | PASS | PASS |
| Day 27 | 2025-10-31 | 2,838 | 2,838 | 2,838 | PASS | PASS | PASS |
| Day 28 | 2025-11-01 | 3,015 | 3,015 | 3,015 | PASS | PASS | PASS |
| Day 29 | 2025-11-02 | 3,348 | 3,348 | 3,348 | PASS | PASS | PASS |
| Day 30 | 2025-11-03 | 3,490 | 3,490 | 3,490 | PASS | PASS | PASS |
| Day 31 | 2025-11-04 | 3,601 | 3,601 | 3,601 | PASS | PASS | PASS |
| Day 32 | 2025-11-05 | 2,887 | 2,887 | 2,887 | PASS | PASS | PASS |
| Day 33 | 2025-11-06 | 3,710 | 3,710 | 3,710 | PASS | PASS | PASS |
| Day 34 | 2025-11-07 | 3,572 | 3,572 | 3,572 | PASS | PASS | PASS |
| Day 35 | 2025-11-08 | 3,257 | 3,257 | 3,257 | PASS | PASS | PASS |
| Day 36 | 2025-11-09 | 3,147 | 3,147 | 3,147 | PASS | PASS | PASS |
| Day 37 | 2025-11-10 | 3,315 | 3,315 | 3,315 | PASS | PASS | PASS |
| Day 38 | 2025-11-11 | 3,107 | 3,107 | 3,107 | PASS | PASS | PASS |
| Day 39 | 2025-11-12 | 3,316 | 3,316 | 3,316 | PASS | PASS | PASS |
| Day 40 | 2025-11-13 | 3,587 | 3,587 | 3,587 | PASS | PASS | PASS |
| Day 41 | 2025-11-14 | 3,721 | 3,721 | 3,721 | PASS | PASS | PASS |
| Day 42 | 2025-11-15 | 3,423 | 3,423 | 3,423 | PASS | PASS | PASS |
| Day 43 | 2025-11-16 | 3,637 | 3,637 | 3,637 | PASS | PASS | PASS |
| Day 44 | 2025-11-17 | 3,936 | 3,936 | 3,936 | PASS | PASS | PASS |
| Day 45 | 2025-11-18 | 4,255 | 4,255 | 4,255 | PASS | PASS | PASS |
| Day 46 | 2025-11-19 | 4,588 | 4,588 | 4,588 | PASS | PASS | PASS |
| Day 47 | 2025-11-20 | 4,024 | 4,024 | 4,024 | PASS | PASS | PASS |
| Day 48 | 2025-11-21 | 4,352 | 4,352 | 4,352 | PASS | PASS | PASS |
| Day 49 | 2025-11-22 | 3,977 | 3,977 | 3,977 | PASS | PASS | PASS |
| Day 50 | 2025-11-23 | 3,935 | 3,935 | 3,935 | PASS | PASS | PASS |
| Day 51 | 2025-11-24 | 4,571 | 4,571 | 4,571 | PASS | PASS | PASS |
| Day 52 | 2025-11-25 | 4,670 | 4,670 | 4,670 | PASS | PASS | PASS |
| Day 53 | 2025-11-26 | 4,852 | 4,852 | 4,852 | PASS | PASS | PASS |
| Day 54 | 2025-11-27 | 4,961 | 4,961 | 4,961 | PASS | PASS | PASS |
| Day 55 | 2025-11-28 | 5,723 | 5,723 | 5,723 | PASS | PASS | PASS |
| Day 56 | 2025-11-29 | 6,164 | 6,164 | 6,164 | PASS | PASS | PASS |
| Day 57 | 2025-11-30 | 6,258 | 6,258 | 6,258 | PASS | PASS | PASS |
| Day 58 | 2025-12-01 | 6,310 | 6,310 | 6,310 | PASS | PASS | PASS |
| Day 59 | 2025-12-02 | 6,605 | 6,605 | 6,605 | PASS | PASS | PASS |
| Day 60 | 2025-12-03 | 6,957 | 6,957 | 6,957 | PASS | PASS | PASS |
| Day 61 | 2025-12-04 | 7,064 | 7,064 | 7,064 | PASS | PASS | PASS |
| Day 62 | 2025-12-05 | 5,754 | 5,754 | 5,754 | PASS | PASS | PASS |
| Day 63 | 2025-12-06 | 6,925 | 6,925 | 6,925 | PASS | PASS | PASS |
| Day 64 | 2025-12-07 | 7,357 | 7,357 | 7,357 | PASS | PASS | PASS |
| Day 65 | 2025-12-08 | 7,399 | 7,399 | 7,399 | PASS | PASS | PASS |
| Day 66 | 2025-12-09 | 7,146 | 7,146 | 7,146 | PASS | PASS | PASS |
| Day 67 | 2025-12-10 | 7,154 | 7,154 | 7,154 | PASS | PASS | PASS |
| Day 68 | 2025-12-11 | 7,308 | 7,308 | 7,308 | PASS | PASS | PASS |
| Day 69 | 2025-12-12 | 6,859 | 6,859 | 6,859 | PASS | PASS | PASS |
| Day 70 | 2025-12-13 | 6,926 | 6,926 | 6,926 | PASS | PASS | PASS |
| Day 71 | 2025-12-14 | 6,997 | 6,997 | 6,997 | PASS | PASS | PASS |
| Day 72 | 2025-12-15 | 6,898 | 6,898 | 6,898 | PASS | PASS | PASS |
| Day 73 | 2025-12-16 | 6,722 | 6,722 | 6,722 | PASS | PASS | PASS |
| Day 74 | 2025-12-17 | 6,435 | 6,435 | 6,435 | PASS | PASS | PASS |
| Day 75 | 2025-12-18 | 6,670 | 6,670 | 6,670 | PASS | PASS | PASS |
| Day 76 | 2025-12-19 | 6,787 | 6,787 | 6,787 | PASS | PASS | PASS |
| Day 77 | 2025-12-20 | 6,387 | 6,387 | 6,387 | PASS | PASS | PASS |
| Day 78 | 2025-12-21 | 7,045 | 7,045 | 7,045 | PASS | PASS | PASS |
| Day 79 | 2025-12-22 | 7,047 | 7,047 | 7,047 | PASS | PASS | PASS |
| Day 80 | 2025-12-23 | 7,484 | 7,484 | 7,484 | PASS | PASS | PASS |
| Day 81 | 2025-12-24 | 7,999 | 7,999 | 7,999 | PASS | PASS | PASS |
| Day 82 | 2025-12-25 | 7,862 | 7,862 | 7,862 | PASS | PASS | PASS |
| Day 83 | 2025-12-26 | 7,658 | 7,658 | 7,658 | PASS | PASS | PASS |
| Day 84 | 2025-12-27 | 7,388 | 7,388 | 7,388 | PASS | PASS | PASS |
| Day 85 | 2025-12-28 | 7,658 | 7,658 | 7,658 | PASS | PASS | PASS |
| Day 86 | 2025-12-29 | 8,127 | 8,127 | 8,127 | PASS | PASS | PASS |
| Day 87 | 2025-12-30 | 8,087 | 8,087 | 8,087 | PASS | PASS | PASS |
| **合计** | | **513,212** | **513,212** | **513,212** | **87/87天PASS** | **87/87天PASS** | **87/87天PASS** |

#### 库存日志 (StockOnHandLog)

- **总行数 (全87天)**: Dev=72,471, Src=72,471, DB=72,471
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-10-05 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 02 | 2025-10-06 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 03 | 2025-10-07 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 04 | 2025-10-08 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 05 | 2025-10-09 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 06 | 2025-10-10 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 07 | 2025-10-11 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 08 | 2025-10-12 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 09 | 2025-10-13 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 10 | 2025-10-14 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 11 | 2025-10-15 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 12 | 2025-10-16 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 13 | 2025-10-17 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 14 | 2025-10-18 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 15 | 2025-10-19 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 16 | 2025-10-20 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 17 | 2025-10-21 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 18 | 2025-10-22 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 19 | 2025-10-23 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 20 | 2025-10-24 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 21 | 2025-10-25 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 22 | 2025-10-26 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 23 | 2025-10-27 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 24 | 2025-10-28 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 25 | 2025-10-29 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 26 | 2025-10-30 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 27 | 2025-10-31 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 28 | 2025-11-01 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 29 | 2025-11-02 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 30 | 2025-11-03 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 31 | 2025-11-04 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 32 | 2025-11-05 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 33 | 2025-11-06 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 34 | 2025-11-07 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 35 | 2025-11-08 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 36 | 2025-11-09 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 37 | 2025-11-10 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 38 | 2025-11-11 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 39 | 2025-11-12 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 40 | 2025-11-13 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 41 | 2025-11-14 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 42 | 2025-11-15 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 43 | 2025-11-16 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 44 | 2025-11-17 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 45 | 2025-11-18 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 46 | 2025-11-19 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 47 | 2025-11-20 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 48 | 2025-11-21 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 49 | 2025-11-22 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 50 | 2025-11-23 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 51 | 2025-11-24 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 52 | 2025-11-25 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 53 | 2025-11-26 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 54 | 2025-11-27 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 55 | 2025-11-28 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 56 | 2025-11-29 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 57 | 2025-11-30 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 58 | 2025-12-01 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 59 | 2025-12-02 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 60 | 2025-12-03 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 61 | 2025-12-04 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 62 | 2025-12-05 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 63 | 2025-12-06 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 64 | 2025-12-07 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 65 | 2025-12-08 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 66 | 2025-12-09 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 67 | 2025-12-10 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 68 | 2025-12-11 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 69 | 2025-12-12 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 70 | 2025-12-13 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 71 | 2025-12-14 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 72 | 2025-12-15 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 73 | 2025-12-16 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 74 | 2025-12-17 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 75 | 2025-12-18 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 76 | 2025-12-19 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 77 | 2025-12-20 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 78 | 2025-12-21 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 79 | 2025-12-22 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 80 | 2025-12-23 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 81 | 2025-12-24 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 82 | 2025-12-25 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 83 | 2025-12-26 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 84 | 2025-12-27 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 85 | 2025-12-28 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 86 | 2025-12-29 | 833 | 833 | 833 | PASS | PASS | PASS |
| Day 87 | 2025-12-30 | 833 | 833 | 833 | PASS | PASS | PASS |
| **合计** | | **72,471** | **72,471** | **72,471** | **87/87天PASS** | **87/87天PASS** | **87/87天PASS** |

#### 验证 (Validation)

- **总行数 (全87天)**: Dev=40,368, Src=40,368, DB=40,368
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-10-05 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 02 | 2025-10-06 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 03 | 2025-10-07 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 04 | 2025-10-08 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 05 | 2025-10-09 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 06 | 2025-10-10 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 07 | 2025-10-11 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 08 | 2025-10-12 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 09 | 2025-10-13 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 10 | 2025-10-14 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 11 | 2025-10-15 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 12 | 2025-10-16 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 13 | 2025-10-17 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 14 | 2025-10-18 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 15 | 2025-10-19 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 16 | 2025-10-20 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 17 | 2025-10-21 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 18 | 2025-10-22 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 19 | 2025-10-23 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 20 | 2025-10-24 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 21 | 2025-10-25 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 22 | 2025-10-26 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 23 | 2025-10-27 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 24 | 2025-10-28 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 25 | 2025-10-29 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 26 | 2025-10-30 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 27 | 2025-10-31 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 28 | 2025-11-01 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 29 | 2025-11-02 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 30 | 2025-11-03 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 31 | 2025-11-04 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 32 | 2025-11-05 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 33 | 2025-11-06 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 34 | 2025-11-07 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 35 | 2025-11-08 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 36 | 2025-11-09 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 37 | 2025-11-10 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 38 | 2025-11-11 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 39 | 2025-11-12 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 40 | 2025-11-13 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 41 | 2025-11-14 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 42 | 2025-11-15 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 43 | 2025-11-16 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 44 | 2025-11-17 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 45 | 2025-11-18 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 46 | 2025-11-19 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 47 | 2025-11-20 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 48 | 2025-11-21 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 49 | 2025-11-22 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 50 | 2025-11-23 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 51 | 2025-11-24 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 52 | 2025-11-25 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 53 | 2025-11-26 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 54 | 2025-11-27 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 55 | 2025-11-28 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 56 | 2025-11-29 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 57 | 2025-11-30 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 58 | 2025-12-01 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 59 | 2025-12-02 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 60 | 2025-12-03 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 61 | 2025-12-04 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 62 | 2025-12-05 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 63 | 2025-12-06 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 64 | 2025-12-07 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 65 | 2025-12-08 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 66 | 2025-12-09 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 67 | 2025-12-10 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 68 | 2025-12-11 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 69 | 2025-12-12 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 70 | 2025-12-13 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 71 | 2025-12-14 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 72 | 2025-12-15 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 73 | 2025-12-16 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 74 | 2025-12-17 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 75 | 2025-12-18 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 76 | 2025-12-19 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 77 | 2025-12-20 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 78 | 2025-12-21 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 79 | 2025-12-22 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 80 | 2025-12-23 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 81 | 2025-12-24 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 82 | 2025-12-25 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 83 | 2025-12-26 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 84 | 2025-12-27 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 85 | 2025-12-28 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 86 | 2025-12-29 | 464 | 464 | 464 | PASS | PASS | PASS |
| Day 87 | 2025-12-30 | 464 | 464 | 464 | PASS | PASS | PASS |
| **合计** | | **40,368** | **40,368** | **40,368** | **87/87天PASS** | **87/87天PASS** | **87/87天PASS** |

### 3.6 Module6 - 物流执行模块

#### 交付计划 (DeliveryPlan)

- **总行数 (全87天)**: Dev=43,055, Src=43,055, DB=43,055
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-10-05 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 02 | 2025-10-06 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 03 | 2025-10-07 | 187 | 187 | 187 | PASS | PASS | PASS |
| Day 04 | 2025-10-08 | 350 | 350 | 350 | PASS | PASS | PASS |
| Day 05 | 2025-10-09 | 178 | 178 | 178 | PASS | PASS | PASS |
| Day 06 | 2025-10-10 | 327 | 327 | 327 | PASS | PASS | PASS |
| Day 07 | 2025-10-11 | 617 | 617 | 617 | PASS | PASS | PASS |
| Day 08 | 2025-10-12 | 704 | 704 | 704 | PASS | PASS | PASS |
| Day 09 | 2025-10-13 | 295 | 295 | 295 | PASS | PASS | PASS |
| Day 10 | 2025-10-14 | 204 | 204 | 204 | PASS | PASS | PASS |
| Day 11 | 2025-10-15 | 664 | 664 | 664 | PASS | PASS | PASS |
| Day 12 | 2025-10-16 | 854 | 854 | 854 | PASS | PASS | PASS |
| Day 13 | 2025-10-17 | 709 | 709 | 709 | PASS | PASS | PASS |
| Day 14 | 2025-10-18 | 1,149 | 1,149 | 1,149 | PASS | PASS | PASS |
| Day 15 | 2025-10-19 | 512 | 512 | 512 | PASS | PASS | PASS |
| Day 16 | 2025-10-20 | 1,222 | 1,222 | 1,222 | PASS | PASS | PASS |
| Day 17 | 2025-10-21 | 321 | 321 | 321 | PASS | PASS | PASS |
| Day 18 | 2025-10-22 | 969 | 969 | 969 | PASS | PASS | PASS |
| Day 19 | 2025-10-23 | 806 | 806 | 806 | PASS | PASS | PASS |
| Day 20 | 2025-10-24 | 1,052 | 1,052 | 1,052 | PASS | PASS | PASS |
| Day 21 | 2025-10-25 | 845 | 845 | 845 | PASS | PASS | PASS |
| Day 22 | 2025-10-26 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 23 | 2025-10-27 | 557 | 557 | 557 | PASS | PASS | PASS |
| Day 24 | 2025-10-28 | 181 | 181 | 181 | PASS | PASS | PASS |
| Day 25 | 2025-10-29 | 398 | 398 | 398 | PASS | PASS | PASS |
| Day 26 | 2025-10-30 | 482 | 482 | 482 | PASS | PASS | PASS |
| Day 27 | 2025-10-31 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 28 | 2025-11-01 | 640 | 640 | 640 | PASS | PASS | PASS |
| Day 29 | 2025-11-02 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 30 | 2025-11-03 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 31 | 2025-11-04 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 32 | 2025-11-05 | 51 | 51 | 51 | PASS | PASS | PASS |
| Day 33 | 2025-11-06 | 792 | 792 | 792 | PASS | PASS | PASS |
| Day 34 | 2025-11-07 | 1,669 | 1,669 | 1,669 | PASS | PASS | PASS |
| Day 35 | 2025-11-08 | 610 | 610 | 610 | PASS | PASS | PASS |
| Day 36 | 2025-11-09 | 475 | 475 | 475 | PASS | PASS | PASS |
| Day 37 | 2025-11-10 | 220 | 220 | 220 | PASS | PASS | PASS |
| Day 38 | 2025-11-11 | 496 | 496 | 496 | PASS | PASS | PASS |
| Day 39 | 2025-11-12 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 40 | 2025-11-13 | 552 | 552 | 552 | PASS | PASS | PASS |
| Day 41 | 2025-11-14 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 42 | 2025-11-15 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 43 | 2025-11-16 | 611 | 611 | 611 | PASS | PASS | PASS |
| Day 44 | 2025-11-17 | 122 | 122 | 122 | PASS | PASS | PASS |
| Day 45 | 2025-11-18 | 268 | 268 | 268 | PASS | PASS | PASS |
| Day 46 | 2025-11-19 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 47 | 2025-11-20 | 943 | 943 | 943 | PASS | PASS | PASS |
| Day 48 | 2025-11-21 | 1,572 | 1,572 | 1,572 | PASS | PASS | PASS |
| Day 49 | 2025-11-22 | 835 | 835 | 835 | PASS | PASS | PASS |
| Day 50 | 2025-11-23 | 312 | 312 | 312 | PASS | PASS | PASS |
| Day 51 | 2025-11-24 | 315 | 315 | 315 | PASS | PASS | PASS |
| Day 52 | 2025-11-25 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 53 | 2025-11-26 | 586 | 586 | 586 | PASS | PASS | PASS |
| Day 54 | 2025-11-27 | 356 | 356 | 356 | PASS | PASS | PASS |
| Day 55 | 2025-11-28 | 915 | 915 | 915 | PASS | PASS | PASS |
| Day 56 | 2025-11-29 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 57 | 2025-11-30 | 199 | 199 | 199 | PASS | PASS | PASS |
| Day 58 | 2025-12-01 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 59 | 2025-12-02 | 881 | 881 | 881 | PASS | PASS | PASS |
| Day 60 | 2025-12-03 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 61 | 2025-12-04 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 62 | 2025-12-05 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 63 | 2025-12-06 | 822 | 822 | 822 | PASS | PASS | PASS |
| Day 64 | 2025-12-07 | 650 | 650 | 650 | PASS | PASS | PASS |
| Day 65 | 2025-12-08 | 1,191 | 1,191 | 1,191 | PASS | PASS | PASS |
| Day 66 | 2025-12-09 | 695 | 695 | 695 | PASS | PASS | PASS |
| Day 67 | 2025-12-10 | 918 | 918 | 918 | PASS | PASS | PASS |
| Day 68 | 2025-12-11 | 1,284 | 1,284 | 1,284 | PASS | PASS | PASS |
| Day 69 | 2025-12-12 | 815 | 815 | 815 | PASS | PASS | PASS |
| Day 70 | 2025-12-13 | 1,759 | 1,759 | 1,759 | PASS | PASS | PASS |
| Day 71 | 2025-12-14 | 351 | 351 | 351 | PASS | PASS | PASS |
| Day 72 | 2025-12-15 | 84 | 84 | 84 | PASS | PASS | PASS |
| Day 73 | 2025-12-16 | 713 | 713 | 713 | PASS | PASS | PASS |
| Day 74 | 2025-12-17 | 224 | 224 | 224 | PASS | PASS | PASS |
| Day 75 | 2025-12-18 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 76 | 2025-12-19 | 227 | 227 | 227 | PASS | PASS | PASS |
| Day 77 | 2025-12-20 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 78 | 2025-12-21 | 302 | 302 | 302 | PASS | PASS | PASS |
| Day 79 | 2025-12-22 | 651 | 651 | 651 | PASS | PASS | PASS |
| Day 80 | 2025-12-23 | 782 | 782 | 782 | PASS | PASS | PASS |
| Day 81 | 2025-12-24 | 103 | 103 | 103 | PASS | PASS | PASS |
| Day 82 | 2025-12-25 | 238 | 238 | 238 | PASS | PASS | PASS |
| Day 83 | 2025-12-26 | 476 | 476 | 476 | PASS | PASS | PASS |
| Day 84 | 2025-12-27 | 1,443 | 1,443 | 1,443 | PASS | PASS | PASS |
| Day 85 | 2025-12-28 | 1,607 | 1,607 | 1,607 | PASS | PASS | PASS |
| Day 86 | 2025-12-29 | 298 | 298 | 298 | PASS | PASS | PASS |
| Day 87 | 2025-12-30 | 419 | 419 | 419 | PASS | PASS | PASS |
| **合计** | | **43,055** | **43,055** | **43,055** | **87/87天PASS** | **87/87天PASS** | **87/87天PASS** |

#### 车辆日志 (VehicleLog)

- **总行数 (全87天)**: Dev=258, Src=258, DB=258
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-10-05 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 02 | 2025-10-06 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 03 | 2025-10-07 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 04 | 2025-10-08 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 05 | 2025-10-09 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 06 | 2025-10-10 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 07 | 2025-10-11 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 08 | 2025-10-12 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 09 | 2025-10-13 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 10 | 2025-10-14 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 11 | 2025-10-15 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 12 | 2025-10-16 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 13 | 2025-10-17 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 14 | 2025-10-18 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 15 | 2025-10-19 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 16 | 2025-10-20 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 17 | 2025-10-21 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 18 | 2025-10-22 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 19 | 2025-10-23 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 20 | 2025-10-24 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 21 | 2025-10-25 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 22 | 2025-10-26 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 23 | 2025-10-27 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 24 | 2025-10-28 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 25 | 2025-10-29 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 26 | 2025-10-30 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 27 | 2025-10-31 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 28 | 2025-11-01 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 29 | 2025-11-02 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 30 | 2025-11-03 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 31 | 2025-11-04 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 32 | 2025-11-05 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 33 | 2025-11-06 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 34 | 2025-11-07 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 35 | 2025-11-08 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 36 | 2025-11-09 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 37 | 2025-11-10 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 38 | 2025-11-11 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 39 | 2025-11-12 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 40 | 2025-11-13 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 41 | 2025-11-14 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 42 | 2025-11-15 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 43 | 2025-11-16 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 44 | 2025-11-17 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 45 | 2025-11-18 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 46 | 2025-11-19 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 47 | 2025-11-20 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 48 | 2025-11-21 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 49 | 2025-11-22 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 50 | 2025-11-23 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 51 | 2025-11-24 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 52 | 2025-11-25 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 53 | 2025-11-26 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 54 | 2025-11-27 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 55 | 2025-11-28 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 56 | 2025-11-29 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 57 | 2025-11-30 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 58 | 2025-12-01 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 59 | 2025-12-02 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 60 | 2025-12-03 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 61 | 2025-12-04 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 62 | 2025-12-05 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 63 | 2025-12-06 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 64 | 2025-12-07 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 65 | 2025-12-08 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 66 | 2025-12-09 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 67 | 2025-12-10 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 68 | 2025-12-11 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 69 | 2025-12-12 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 70 | 2025-12-13 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 71 | 2025-12-14 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 72 | 2025-12-15 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 73 | 2025-12-16 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 74 | 2025-12-17 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 75 | 2025-12-18 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 76 | 2025-12-19 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 77 | 2025-12-20 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 78 | 2025-12-21 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 79 | 2025-12-22 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 80 | 2025-12-23 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 81 | 2025-12-24 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 82 | 2025-12-25 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 83 | 2025-12-26 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 84 | 2025-12-27 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 85 | 2025-12-28 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 86 | 2025-12-29 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 87 | 2025-12-30 | 4 | 4 | 4 | PASS | PASS | PASS |
| **合计** | | **258** | **258** | **258** | **87/87天PASS** | **87/87天PASS** | **87/87天PASS** |

#### 卡车使用日志 (TruckUsageLog)

- **总行数 (全87天)**: Dev=192, Src=192, DB=192
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-10-05 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 02 | 2025-10-06 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 03 | 2025-10-07 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 04 | 2025-10-08 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 05 | 2025-10-09 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 06 | 2025-10-10 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 07 | 2025-10-11 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 08 | 2025-10-12 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 09 | 2025-10-13 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 10 | 2025-10-14 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 11 | 2025-10-15 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 12 | 2025-10-16 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 13 | 2025-10-17 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 14 | 2025-10-18 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 15 | 2025-10-19 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 16 | 2025-10-20 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 17 | 2025-10-21 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 18 | 2025-10-22 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 19 | 2025-10-23 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 20 | 2025-10-24 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 21 | 2025-10-25 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 22 | 2025-10-26 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 23 | 2025-10-27 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 24 | 2025-10-28 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 25 | 2025-10-29 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 26 | 2025-10-30 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 27 | 2025-10-31 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 28 | 2025-11-01 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 29 | 2025-11-02 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 30 | 2025-11-03 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 31 | 2025-11-04 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 32 | 2025-11-05 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 33 | 2025-11-06 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 34 | 2025-11-07 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 35 | 2025-11-08 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 36 | 2025-11-09 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 37 | 2025-11-10 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 38 | 2025-11-11 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 39 | 2025-11-12 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 40 | 2025-11-13 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 41 | 2025-11-14 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 42 | 2025-11-15 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 43 | 2025-11-16 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 44 | 2025-11-17 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 45 | 2025-11-18 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 46 | 2025-11-19 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 47 | 2025-11-20 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 48 | 2025-11-21 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 49 | 2025-11-22 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 50 | 2025-11-23 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 51 | 2025-11-24 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 52 | 2025-11-25 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 53 | 2025-11-26 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 54 | 2025-11-27 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 55 | 2025-11-28 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 56 | 2025-11-29 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 57 | 2025-11-30 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 58 | 2025-12-01 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 59 | 2025-12-02 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 60 | 2025-12-03 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 61 | 2025-12-04 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 62 | 2025-12-05 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 63 | 2025-12-06 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 64 | 2025-12-07 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 65 | 2025-12-08 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 66 | 2025-12-09 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 67 | 2025-12-10 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 68 | 2025-12-11 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 69 | 2025-12-12 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 70 | 2025-12-13 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 71 | 2025-12-14 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 72 | 2025-12-15 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 73 | 2025-12-16 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 74 | 2025-12-17 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 75 | 2025-12-18 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 76 | 2025-12-19 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 77 | 2025-12-20 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 78 | 2025-12-21 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 79 | 2025-12-22 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 80 | 2025-12-23 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 81 | 2025-12-24 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 82 | 2025-12-25 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 83 | 2025-12-26 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 84 | 2025-12-27 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 85 | 2025-12-28 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 86 | 2025-12-29 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 87 | 2025-12-30 | 3 | 3 | 3 | PASS | PASS | PASS |
| **合计** | | **192** | **192** | **192** | **87/87天PASS** | **87/87天PASS** | **87/87天PASS** |

---

## 4. 每日库存一致性验证（全周期）

| 天数 | 日期 | Dev库存量 | Src库存量 | DB库存量 | 一致性 |
|------|------|-----------|-----------|----------|--------|
| Day 01 | 2025-10-05 | 348,191 | 348,191 | 348,191 | PASS |
| Day 02 | 2025-10-06 | 334,671 | 334,671 | 334,671 | PASS |
| Day 03 | 2025-10-07 | 346,343 | 346,343 | 346,343 | PASS |
| Day 04 | 2025-10-08 | 326,865 | 326,865 | 326,865 | PASS |
| Day 05 | 2025-10-09 | 324,868 | 324,868 | 324,868 | PASS |
| Day 06 | 2025-10-10 | 317,868 | 317,868 | 317,868 | PASS |
| Day 07 | 2025-10-11 | 308,512 | 308,512 | 308,512 | PASS |
| Day 08 | 2025-10-12 | 306,005 | 306,005 | 306,005 | PASS |
| Day 09 | 2025-10-13 | 317,058 | 317,058 | 317,058 | PASS |
| Day 10 | 2025-10-14 | 336,025 | 336,025 | 336,025 | PASS |
| Day 11 | 2025-10-15 | 339,113 | 339,113 | 339,113 | PASS |
| Day 12 | 2025-10-16 | 348,022 | 348,022 | 348,022 | PASS |
| Day 13 | 2025-10-17 | 356,510 | 356,510 | 356,510 | PASS |
| Day 14 | 2025-10-18 | 363,975 | 363,975 | 363,975 | PASS |
| Day 15 | 2025-10-19 | 379,425 | 379,425 | 379,425 | PASS |
| Day 16 | 2025-10-20 | 388,137 | 388,137 | 388,137 | PASS |
| Day 17 | 2025-10-21 | 401,027 | 401,027 | 401,027 | PASS |
| Day 18 | 2025-10-22 | 435,270 | 435,270 | 435,270 | PASS |
| Day 19 | 2025-10-23 | 445,257 | 445,257 | 445,257 | PASS |
| Day 20 | 2025-10-24 | 457,432 | 457,432 | 457,432 | PASS |
| Day 21 | 2025-10-25 | 457,565 | 457,565 | 457,565 | PASS |
| Day 22 | 2025-10-26 | 469,826 | 469,826 | 469,826 | PASS |
| Day 23 | 2025-10-27 | 474,664 | 474,664 | 474,664 | PASS |
| Day 24 | 2025-10-28 | 491,058 | 491,058 | 491,058 | PASS |
| Day 25 | 2025-10-29 | 490,737 | 490,737 | 490,737 | PASS |
| Day 26 | 2025-10-30 | 473,222 | 473,222 | 473,222 | PASS |
| Day 27 | 2025-10-31 | 465,147 | 465,147 | 465,147 | PASS |
| Day 28 | 2025-11-01 | 463,162 | 463,162 | 463,162 | PASS |
| Day 29 | 2025-11-02 | 457,507 | 457,507 | 457,507 | PASS |
| Day 30 | 2025-11-03 | 449,633 | 449,633 | 449,633 | PASS |
| Day 31 | 2025-11-04 | 437,257 | 437,257 | 437,257 | PASS |
| Day 32 | 2025-11-05 | 425,870 | 425,870 | 425,870 | PASS |
| Day 33 | 2025-11-06 | 424,042 | 424,042 | 424,042 | PASS |
| Day 34 | 2025-11-07 | 416,057 | 416,057 | 416,057 | PASS |
| Day 35 | 2025-11-08 | 415,971 | 415,971 | 415,971 | PASS |
| Day 36 | 2025-11-09 | 406,074 | 406,074 | 406,074 | PASS |
| Day 37 | 2025-11-10 | 407,842 | 407,842 | 407,842 | PASS |
| Day 38 | 2025-11-11 | 416,842 | 416,842 | 416,842 | PASS |
| Day 39 | 2025-11-12 | 416,623 | 416,623 | 416,623 | PASS |
| Day 40 | 2025-11-13 | 415,318 | 415,318 | 415,318 | PASS |
| Day 41 | 2025-11-14 | 405,141 | 405,141 | 405,141 | PASS |
| Day 42 | 2025-11-15 | 393,956 | 393,956 | 393,956 | PASS |
| Day 43 | 2025-11-16 | 392,344 | 392,344 | 392,344 | PASS |
| Day 44 | 2025-11-17 | 390,859 | 390,859 | 390,859 | PASS |
| Day 45 | 2025-11-18 | 383,227 | 383,227 | 383,227 | PASS |
| Day 46 | 2025-11-19 | 379,808 | 379,808 | 379,808 | PASS |
| Day 47 | 2025-11-20 | 383,694 | 383,694 | 383,694 | PASS |
| Day 48 | 2025-11-21 | 391,830 | 391,830 | 391,830 | PASS |
| Day 49 | 2025-11-22 | 399,873 | 399,873 | 399,873 | PASS |
| Day 50 | 2025-11-23 | 397,320 | 397,320 | 397,320 | PASS |
| Day 51 | 2025-11-24 | 388,785 | 388,785 | 388,785 | PASS |
| Day 52 | 2025-11-25 | 393,379 | 393,379 | 393,379 | PASS |
| Day 53 | 2025-11-26 | 410,198 | 410,198 | 410,198 | PASS |
| Day 54 | 2025-11-27 | 405,817 | 405,817 | 405,817 | PASS |
| Day 55 | 2025-11-28 | 388,014 | 388,014 | 388,014 | PASS |
| Day 56 | 2025-11-29 | 377,930 | 377,930 | 377,930 | PASS |
| Day 57 | 2025-11-30 | 378,649 | 378,649 | 378,649 | PASS |
| Day 58 | 2025-12-01 | 376,339 | 376,339 | 376,339 | PASS |
| Day 59 | 2025-12-02 | 362,306 | 362,306 | 362,306 | PASS |
| Day 60 | 2025-12-03 | 350,317 | 350,317 | 350,317 | PASS |
| Day 61 | 2025-12-04 | 337,858 | 337,858 | 337,858 | PASS |
| Day 62 | 2025-12-05 | 327,969 | 327,969 | 327,969 | PASS |
| Day 63 | 2025-12-06 | 324,743 | 324,743 | 324,743 | PASS |
| Day 64 | 2025-12-07 | 315,156 | 315,156 | 315,156 | PASS |
| Day 65 | 2025-12-08 | 309,126 | 309,126 | 309,126 | PASS |
| Day 66 | 2025-12-09 | 305,948 | 305,948 | 305,948 | PASS |
| Day 67 | 2025-12-10 | 309,614 | 309,614 | 309,614 | PASS |
| Day 68 | 2025-12-11 | 322,915 | 322,915 | 322,915 | PASS |
| Day 69 | 2025-12-12 | 332,835 | 332,835 | 332,835 | PASS |
| Day 70 | 2025-12-13 | 355,132 | 355,132 | 355,132 | PASS |
| Day 71 | 2025-12-14 | 360,503 | 360,503 | 360,503 | PASS |
| Day 72 | 2025-12-15 | 367,842 | 367,842 | 367,842 | PASS |
| Day 73 | 2025-12-16 | 389,175 | 389,175 | 389,175 | PASS |
| Day 74 | 2025-12-17 | 389,102 | 389,102 | 389,102 | PASS |
| Day 75 | 2025-12-18 | 382,857 | 382,857 | 382,857 | PASS |
| Day 76 | 2025-12-19 | 372,640 | 372,640 | 372,640 | PASS |
| Day 77 | 2025-12-20 | 368,498 | 368,498 | 368,498 | PASS |
| Day 78 | 2025-12-21 | 377,322 | 377,322 | 377,322 | PASS |
| Day 79 | 2025-12-22 | 372,493 | 372,493 | 372,493 | PASS |
| Day 80 | 2025-12-23 | 360,677 | 360,677 | 360,677 | PASS |
| Day 81 | 2025-12-24 | 359,640 | 359,640 | 359,640 | PASS |
| Day 82 | 2025-12-25 | 364,001 | 364,001 | 364,001 | PASS |
| Day 83 | 2025-12-26 | 381,214 | 381,214 | 381,214 | PASS |
| Day 84 | 2025-12-27 | 385,492 | 385,492 | 385,492 | PASS |
| Day 85 | 2025-12-28 | 385,881 | 385,881 | 385,881 | PASS |
| Day 86 | 2025-12-29 | 395,115 | 395,115 | 395,115 | PASS |
| Day 87 | 2025-12-30 | 396,489 | 396,489 | 396,489 | PASS |

### 4.1 全周期一致性总结
| 验证项 | 结果 |
|--------|------|
| **仿真总天数** | 87天 |
| **库存不一致天数** | 0天 |
| **一致性比例** | 100% |
| **起始库存** | 348,191 |
| **结束库存** | 396,489 |

---

## 5. 测试结论

### 5.1 功能验证结论

**重构版本(Src/DB)与原始版本(Dev)业务数据100%一致**

验证覆盖（全87天）：
- Module1 / 订单日志 (OrderLog): 424,848行 (Dev vs Src PASS, Dev vs DB PASS)
- Module1 / 发货日志 (ShipmentLog): 24,252行 (Dev vs Src PASS, Dev vs DB PASS)
- Module1 / 削减日志 (CutLog): 24,252行 (Dev vs Src PASS, Dev vs DB PASS)
- Module1 / 供需日志 (SupplyDemandLog): 2,084,295行 (Dev vs Src PASS, Dev vs DB PASS)
- Module1 / 每日汇总 (Summary): 87行 (Dev vs Src PASS, Dev vs DB PASS)
- Module3 / 净需求 (NetDemand): 30,702行 (Dev vs Src PASS, Dev vs DB PASS)
- Module4 / 生产计划 (ProductionPlan): 379行 (Dev vs Src PASS, Dev vs DB PASS)
- Module4 / 超容量报告 (CapacityExceed): 15行 (Dev vs Src PASS, Dev vs DB PASS)
- Module4 / 换型日志 (ChangeoverLog): 146行 (Dev vs Src PASS, Dev vs DB PASS)
- Module5 / 部署计划 (DeploymentPlan): 1,149,959行 (Dev vs Src PASS, Dev vs DB PASS)
- Module5 / 未满足日志 (UnfulfilledLog): 513,212行 (Dev vs Src PASS, Dev vs DB PASS)
- Module5 / 库存日志 (StockOnHandLog): 72,471行 (Dev vs Src PASS, Dev vs DB PASS)
- Module5 / 验证 (Validation): 40,368行 (Dev vs Src PASS, Dev vs DB PASS)
- Module6 / 交付计划 (DeliveryPlan): 43,055行 (Dev vs Src PASS, Dev vs DB PASS)
- Module6 / 车辆日志 (VehicleLog): 258行 (Dev vs Src PASS, Dev vs DB PASS)
- Module6 / 卡车使用日志 (TruckUsageLog): 192行 (Dev vs Src PASS, Dev vs DB PASS)

### 5.2 全周期性能总结
| 指标 | Dev | Src | DB | 最佳提升 |
|------|-----|-----|-----|----------|
| 总运行时间 | 159.0分钟 | 48.3分钟 | 26.1分钟 | 83.6% |
| M1 订单生成平均耗时/天 | ~18.3秒 | ~5.7秒 | ~5.6秒 | 69.6% |
| M3 净需求计算平均耗时/天 | ~24.1秒 | ~5.8秒 | ~2.3秒 | 90.7% |
| M5 部署规划平均耗时/天 | ~60.6秒 | ~18.4秒 | ~7.1秒 | 88.3% |

---

**报告生成时间**: 2026-02-26  
**测试执行人**: chenxianyue002@chinasofti.com  
**版本**: v8.0 (完整版)
