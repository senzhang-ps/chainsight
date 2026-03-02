# ChainSight 供应链仿真系统 OC算法优化测试报告 (完整版 Day 01-76)

## 1. 测试概述

### 1.1 测试目的
本报告对 ChainSight 供应链仿真系统全仿真周期 Day 01-76 (2025-12-15 至 2026-02-28) 的仿真结果进行三版本对比测试，验证代码重构后的功能正确性和性能提升效果。

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
- **测试数据**: OC_Paste_S1_20251224.xlsx 配置文件
- **仿真周期**: 76天 (2025-12-15 至 2026-02-28)
- **随机种子**: 42（确保可重复性）
- **测试时间**: 2026-02-12

### 1.4 测试输出目录
| 版本 | 输出路径 |
|------|----------|
| Dev | `outputs/dev_output/OC_Paste_S1_20251224/run_20260127_142402/` |
| Src | `outputs/OC_Paste_S1_20251224/run_20260301_125122/` |
| DB | `outputs/db_OC_Paste_S1_20251224_20260301_222907/` |

---

## 2. 性能对比分析

### 2.1 全周期运行时间 (共76天)
| 版本 | 总时间 | 平均每天 |
|------|--------|----------|
| **Dev** | 97611秒 (1626.8分钟) | 1284.4秒 |
| **Src** | 14854秒 (247.6分钟) | 195.4秒 |
| **DB** | 7850秒 (130.8分钟) | 103.3秒 |

### 2.2 性能提升比例
| 对比项 | 加速倍数 | 时间减少百分比 |
|--------|----------|----------------|
| **Src vs Dev** | **6.57x** | 84.8% |
| **DB vs Dev**  | **12.43x**  | 92.0% |
| **DB vs Src**  | **1.89x**  | 47.2% |

### 2.3 各模块平均耗时对比（秒/天，全76天均值）
| 模块 | Dev | Src | DB | Src加速 | DB加速 |
|------|-----|-----|-----|---------|--------|
| **M1 订单生成** | ~377.2 | ~38.0 | ~36.0 | 9.93x | 10.48x |
| **M3 净需求计算** | ~342.8 | ~44.9 | ~20.4 | 7.64x | 16.84x |
| **M5 部署规划** | ~544.6 | ~103.9 | ~39.0 | 5.24x | 13.96x |

---

## 3. 数据一致性验证

### 3.1 总体验证结果

| 模块 | 表名 | Dev总行数 | Src总行数 | DB总行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|-----------|-----------|----------|------------|-----------|-----------|
| Module1 | 订单日志 (OrderLog) | 351,100 | 351,100 | 351,100 | PASS | PASS | PASS |
| Module1 | 发货日志 (ShipmentLog) | 157,433 | 157,433 | 157,433 | PASS | PASS | PASS |
| Module1 | 削减日志 (CutLog) | 157,433 | 157,433 | 157,433 | PASS | PASS | PASS |
| Module1 | 每日汇总 (Summary) | 76 | 76 | 76 | PASS | PASS | PASS |
| Module3 | 净需求 (NetDemand) | 93,693 | 93,693 | 93,693 | PASS | PASS | PASS |
| Module4 | 生产计划 (ProductionPlan) | 685 | 685 | 685 | PASS | PASS | PASS |
| Module4 | 超容量报告 (CapacityExceed) | 3,018 | 3,018 | 3,018 | PASS | PASS | PASS |
| Module4 | 换型日志 (ChangeoverLog) | 515 | 515 | 515 | PASS | PASS | PASS |
| Module5 | 部署计划 (DeploymentPlan) | 3,336,330 | 3,336,330 | 3,336,330 | PASS | PASS | PASS |
| Module5 | 未满足日志 (UnfulfilledLog) | 527,164 | 527,164 | 527,164 | PASS | PASS | PASS |
| Module5 | 库存日志 (StockOnHandLog) | 387,730 | 387,730 | 387,730 | PASS | PASS | PASS |
| Module5 | 验证 (Validation) | 623,200 | 623,200 | 623,200 | PASS | PASS | PASS |
| Module6 | 交付计划 (DeliveryPlan) | 71,145 | 71,145 | 71,145 | PASS | PASS | PASS |
| Module6 | 车辆日志 (VehicleLog) | 823 | 823 | 823 | PASS | PASS | PASS |
| Module6 | 卡车使用日志 (TruckUsageLog) | 803 | 803 | 803 | PASS | PASS | PASS |

> **Dev vs Src**: 15/15 表全部PASS  
> **Dev vs DB**:  15/15 表PASS  
> **Src vs DB**:  15/15 表PASS

---

### 3.2 Module1 - 订单生成模块

#### 订单日志 (OrderLog)

- **总行数 (全76天)**: Dev=351,100, Src=351,100, DB=351,100
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-12-15 | 4,532 | 4,532 | 4,532 | PASS | PASS | PASS |
| Day 02 | 2025-12-16 | 4,682 | 4,682 | 4,682 | PASS | PASS | PASS |
| Day 03 | 2025-12-17 | 4,680 | 4,680 | 4,680 | PASS | PASS | PASS |
| Day 04 | 2025-12-18 | 4,677 | 4,677 | 4,677 | PASS | PASS | PASS |
| Day 05 | 2025-12-19 | 4,676 | 4,676 | 4,676 | PASS | PASS | PASS |
| Day 06 | 2025-12-20 | 4,676 | 4,676 | 4,676 | PASS | PASS | PASS |
| Day 07 | 2025-12-21 | 4,676 | 4,676 | 4,676 | PASS | PASS | PASS |
| Day 08 | 2025-12-22 | 4,673 | 4,673 | 4,673 | PASS | PASS | PASS |
| Day 09 | 2025-12-23 | 4,636 | 4,636 | 4,636 | PASS | PASS | PASS |
| Day 10 | 2025-12-24 | 4,632 | 4,632 | 4,632 | PASS | PASS | PASS |
| Day 11 | 2025-12-25 | 4,631 | 4,631 | 4,631 | PASS | PASS | PASS |
| Day 12 | 2025-12-26 | 4,631 | 4,631 | 4,631 | PASS | PASS | PASS |
| Day 13 | 2025-12-27 | 4,631 | 4,631 | 4,631 | PASS | PASS | PASS |
| Day 14 | 2025-12-28 | 4,631 | 4,631 | 4,631 | PASS | PASS | PASS |
| Day 15 | 2025-12-29 | 4,630 | 4,630 | 4,630 | PASS | PASS | PASS |
| Day 16 | 2025-12-30 | 4,644 | 4,644 | 4,644 | PASS | PASS | PASS |
| Day 17 | 2025-12-31 | 4,641 | 4,641 | 4,641 | PASS | PASS | PASS |
| Day 18 | 2026-01-01 | 4,638 | 4,638 | 4,638 | PASS | PASS | PASS |
| Day 19 | 2026-01-02 | 4,636 | 4,636 | 4,636 | PASS | PASS | PASS |
| Day 20 | 2026-01-03 | 4,633 | 4,633 | 4,633 | PASS | PASS | PASS |
| Day 21 | 2026-01-04 | 4,630 | 4,630 | 4,630 | PASS | PASS | PASS |
| Day 22 | 2026-01-05 | 4,580 | 4,580 | 4,580 | PASS | PASS | PASS |
| Day 23 | 2026-01-06 | 4,634 | 4,634 | 4,634 | PASS | PASS | PASS |
| Day 24 | 2026-01-07 | 4,634 | 4,634 | 4,634 | PASS | PASS | PASS |
| Day 25 | 2026-01-08 | 4,634 | 4,634 | 4,634 | PASS | PASS | PASS |
| Day 26 | 2026-01-09 | 4,634 | 4,634 | 4,634 | PASS | PASS | PASS |
| Day 27 | 2026-01-10 | 4,634 | 4,634 | 4,634 | PASS | PASS | PASS |
| Day 28 | 2026-01-11 | 4,634 | 4,634 | 4,634 | PASS | PASS | PASS |
| Day 29 | 2026-01-12 | 4,634 | 4,634 | 4,634 | PASS | PASS | PASS |
| Day 30 | 2026-01-13 | 4,636 | 4,636 | 4,636 | PASS | PASS | PASS |
| Day 31 | 2026-01-14 | 4,636 | 4,636 | 4,636 | PASS | PASS | PASS |
| Day 32 | 2026-01-15 | 4,636 | 4,636 | 4,636 | PASS | PASS | PASS |
| Day 33 | 2026-01-16 | 4,636 | 4,636 | 4,636 | PASS | PASS | PASS |
| Day 34 | 2026-01-17 | 4,636 | 4,636 | 4,636 | PASS | PASS | PASS |
| Day 35 | 2026-01-18 | 4,636 | 4,636 | 4,636 | PASS | PASS | PASS |
| Day 36 | 2026-01-19 | 4,636 | 4,636 | 4,636 | PASS | PASS | PASS |
| Day 37 | 2026-01-20 | 4,634 | 4,634 | 4,634 | PASS | PASS | PASS |
| Day 38 | 2026-01-21 | 4,634 | 4,634 | 4,634 | PASS | PASS | PASS |
| Day 39 | 2026-01-22 | 4,634 | 4,634 | 4,634 | PASS | PASS | PASS |
| Day 40 | 2026-01-23 | 4,634 | 4,634 | 4,634 | PASS | PASS | PASS |
| Day 41 | 2026-01-24 | 4,634 | 4,634 | 4,634 | PASS | PASS | PASS |
| Day 42 | 2026-01-25 | 4,634 | 4,634 | 4,634 | PASS | PASS | PASS |
| Day 43 | 2026-01-26 | 4,634 | 4,634 | 4,634 | PASS | PASS | PASS |
| Day 44 | 2026-01-27 | 4,626 | 4,626 | 4,626 | PASS | PASS | PASS |
| Day 45 | 2026-01-28 | 4,626 | 4,626 | 4,626 | PASS | PASS | PASS |
| Day 46 | 2026-01-29 | 4,626 | 4,626 | 4,626 | PASS | PASS | PASS |
| Day 47 | 2026-01-30 | 4,626 | 4,626 | 4,626 | PASS | PASS | PASS |
| Day 48 | 2026-01-31 | 4,626 | 4,626 | 4,626 | PASS | PASS | PASS |
| Day 49 | 2026-02-01 | 4,626 | 4,626 | 4,626 | PASS | PASS | PASS |
| Day 50 | 2026-02-02 | 4,626 | 4,626 | 4,626 | PASS | PASS | PASS |
| Day 51 | 2026-02-03 | 4,611 | 4,611 | 4,611 | PASS | PASS | PASS |
| Day 52 | 2026-02-04 | 4,610 | 4,610 | 4,610 | PASS | PASS | PASS |
| Day 53 | 2026-02-05 | 4,610 | 4,610 | 4,610 | PASS | PASS | PASS |
| Day 54 | 2026-02-06 | 4,610 | 4,610 | 4,610 | PASS | PASS | PASS |
| Day 55 | 2026-02-07 | 4,610 | 4,610 | 4,610 | PASS | PASS | PASS |
| Day 56 | 2026-02-08 | 4,610 | 4,610 | 4,610 | PASS | PASS | PASS |
| Day 57 | 2026-02-09 | 4,610 | 4,610 | 4,610 | PASS | PASS | PASS |
| Day 58 | 2026-02-10 | 4,543 | 4,543 | 4,543 | PASS | PASS | PASS |
| Day 59 | 2026-02-11 | 4,517 | 4,517 | 4,517 | PASS | PASS | PASS |
| Day 60 | 2026-02-12 | 4,505 | 4,505 | 4,505 | PASS | PASS | PASS |
| Day 61 | 2026-02-13 | 4,501 | 4,501 | 4,501 | PASS | PASS | PASS |
| Day 62 | 2026-02-14 | 4,500 | 4,500 | 4,500 | PASS | PASS | PASS |
| Day 63 | 2026-02-15 | 4,500 | 4,500 | 4,500 | PASS | PASS | PASS |
| Day 64 | 2026-02-16 | 4,500 | 4,500 | 4,500 | PASS | PASS | PASS |
| Day 65 | 2026-02-17 | 4,632 | 4,632 | 4,632 | PASS | PASS | PASS |
| Day 66 | 2026-02-18 | 4,632 | 4,632 | 4,632 | PASS | PASS | PASS |
| Day 67 | 2026-02-19 | 4,632 | 4,632 | 4,632 | PASS | PASS | PASS |
| Day 68 | 2026-02-20 | 4,632 | 4,632 | 4,632 | PASS | PASS | PASS |
| Day 69 | 2026-02-21 | 4,632 | 4,632 | 4,632 | PASS | PASS | PASS |
| Day 70 | 2026-02-22 | 4,632 | 4,632 | 4,632 | PASS | PASS | PASS |
| Day 71 | 2026-02-23 | 4,632 | 4,632 | 4,632 | PASS | PASS | PASS |
| Day 72 | 2026-02-24 | 4,617 | 4,617 | 4,617 | PASS | PASS | PASS |
| Day 73 | 2026-02-25 | 4,609 | 4,609 | 4,609 | PASS | PASS | PASS |
| Day 74 | 2026-02-26 | 4,605 | 4,605 | 4,605 | PASS | PASS | PASS |
| Day 75 | 2026-02-27 | 4,603 | 4,603 | 4,603 | PASS | PASS | PASS |
| Day 76 | 2026-02-28 | 4,599 | 4,599 | 4,599 | PASS | PASS | PASS |
| **合计** | | **351,100** | **351,100** | **351,100** | **76/76天PASS** | **76/76天PASS** | **76/76天PASS** |

#### 发货日志 (ShipmentLog)

- **总行数 (全76天)**: Dev=157,433, Src=157,433, DB=157,433
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-12-15 | 1,314 | 1,314 | 1,314 | PASS | PASS | PASS |
| Day 02 | 2025-12-16 | 1,780 | 1,780 | 1,780 | PASS | PASS | PASS |
| Day 03 | 2025-12-17 | 2,070 | 2,070 | 2,070 | PASS | PASS | PASS |
| Day 04 | 2025-12-18 | 2,120 | 2,120 | 2,120 | PASS | PASS | PASS |
| Day 05 | 2025-12-19 | 2,144 | 2,144 | 2,144 | PASS | PASS | PASS |
| Day 06 | 2025-12-20 | 2,145 | 2,145 | 2,145 | PASS | PASS | PASS |
| Day 07 | 2025-12-21 | 2,145 | 2,145 | 2,145 | PASS | PASS | PASS |
| Day 08 | 2025-12-22 | 2,143 | 2,143 | 2,143 | PASS | PASS | PASS |
| Day 09 | 2025-12-23 | 2,106 | 2,106 | 2,106 | PASS | PASS | PASS |
| Day 10 | 2025-12-24 | 2,102 | 2,102 | 2,102 | PASS | PASS | PASS |
| Day 11 | 2025-12-25 | 2,101 | 2,101 | 2,101 | PASS | PASS | PASS |
| Day 12 | 2025-12-26 | 2,101 | 2,101 | 2,101 | PASS | PASS | PASS |
| Day 13 | 2025-12-27 | 2,101 | 2,101 | 2,101 | PASS | PASS | PASS |
| Day 14 | 2025-12-28 | 2,101 | 2,101 | 2,101 | PASS | PASS | PASS |
| Day 15 | 2025-12-29 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 16 | 2025-12-30 | 2,114 | 2,114 | 2,114 | PASS | PASS | PASS |
| Day 17 | 2025-12-31 | 2,111 | 2,111 | 2,111 | PASS | PASS | PASS |
| Day 18 | 2026-01-01 | 2,108 | 2,108 | 2,108 | PASS | PASS | PASS |
| Day 19 | 2026-01-02 | 2,106 | 2,106 | 2,106 | PASS | PASS | PASS |
| Day 20 | 2026-01-03 | 2,103 | 2,103 | 2,103 | PASS | PASS | PASS |
| Day 21 | 2026-01-04 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 22 | 2026-01-05 | 2,050 | 2,050 | 2,050 | PASS | PASS | PASS |
| Day 23 | 2026-01-06 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 24 | 2026-01-07 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 25 | 2026-01-08 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 26 | 2026-01-09 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 27 | 2026-01-10 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 28 | 2026-01-11 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 29 | 2026-01-12 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 30 | 2026-01-13 | 2,102 | 2,102 | 2,102 | PASS | PASS | PASS |
| Day 31 | 2026-01-14 | 2,102 | 2,102 | 2,102 | PASS | PASS | PASS |
| Day 32 | 2026-01-15 | 2,102 | 2,102 | 2,102 | PASS | PASS | PASS |
| Day 33 | 2026-01-16 | 2,102 | 2,102 | 2,102 | PASS | PASS | PASS |
| Day 34 | 2026-01-17 | 2,102 | 2,102 | 2,102 | PASS | PASS | PASS |
| Day 35 | 2026-01-18 | 2,102 | 2,102 | 2,102 | PASS | PASS | PASS |
| Day 36 | 2026-01-19 | 2,102 | 2,102 | 2,102 | PASS | PASS | PASS |
| Day 37 | 2026-01-20 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 38 | 2026-01-21 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 39 | 2026-01-22 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 40 | 2026-01-23 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 41 | 2026-01-24 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 42 | 2026-01-25 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 43 | 2026-01-26 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 44 | 2026-01-27 | 2,092 | 2,092 | 2,092 | PASS | PASS | PASS |
| Day 45 | 2026-01-28 | 2,092 | 2,092 | 2,092 | PASS | PASS | PASS |
| Day 46 | 2026-01-29 | 2,092 | 2,092 | 2,092 | PASS | PASS | PASS |
| Day 47 | 2026-01-30 | 2,092 | 2,092 | 2,092 | PASS | PASS | PASS |
| Day 48 | 2026-01-31 | 2,092 | 2,092 | 2,092 | PASS | PASS | PASS |
| Day 49 | 2026-02-01 | 2,092 | 2,092 | 2,092 | PASS | PASS | PASS |
| Day 50 | 2026-02-02 | 2,092 | 2,092 | 2,092 | PASS | PASS | PASS |
| Day 51 | 2026-02-03 | 2,077 | 2,077 | 2,077 | PASS | PASS | PASS |
| Day 52 | 2026-02-04 | 2,076 | 2,076 | 2,076 | PASS | PASS | PASS |
| Day 53 | 2026-02-05 | 2,076 | 2,076 | 2,076 | PASS | PASS | PASS |
| Day 54 | 2026-02-06 | 2,076 | 2,076 | 2,076 | PASS | PASS | PASS |
| Day 55 | 2026-02-07 | 2,076 | 2,076 | 2,076 | PASS | PASS | PASS |
| Day 56 | 2026-02-08 | 2,076 | 2,076 | 2,076 | PASS | PASS | PASS |
| Day 57 | 2026-02-09 | 2,076 | 2,076 | 2,076 | PASS | PASS | PASS |
| Day 58 | 2026-02-10 | 2,009 | 2,009 | 2,009 | PASS | PASS | PASS |
| Day 59 | 2026-02-11 | 1,983 | 1,983 | 1,983 | PASS | PASS | PASS |
| Day 60 | 2026-02-12 | 1,971 | 1,971 | 1,971 | PASS | PASS | PASS |
| Day 61 | 2026-02-13 | 1,967 | 1,967 | 1,967 | PASS | PASS | PASS |
| Day 62 | 2026-02-14 | 1,966 | 1,966 | 1,966 | PASS | PASS | PASS |
| Day 63 | 2026-02-15 | 1,966 | 1,966 | 1,966 | PASS | PASS | PASS |
| Day 64 | 2026-02-16 | 1,966 | 1,966 | 1,966 | PASS | PASS | PASS |
| Day 65 | 2026-02-17 | 2,098 | 2,098 | 2,098 | PASS | PASS | PASS |
| Day 66 | 2026-02-18 | 2,098 | 2,098 | 2,098 | PASS | PASS | PASS |
| Day 67 | 2026-02-19 | 2,098 | 2,098 | 2,098 | PASS | PASS | PASS |
| Day 68 | 2026-02-20 | 2,098 | 2,098 | 2,098 | PASS | PASS | PASS |
| Day 69 | 2026-02-21 | 2,098 | 2,098 | 2,098 | PASS | PASS | PASS |
| Day 70 | 2026-02-22 | 2,098 | 2,098 | 2,098 | PASS | PASS | PASS |
| Day 71 | 2026-02-23 | 2,098 | 2,098 | 2,098 | PASS | PASS | PASS |
| Day 72 | 2026-02-24 | 2,083 | 2,083 | 2,083 | PASS | PASS | PASS |
| Day 73 | 2026-02-25 | 2,075 | 2,075 | 2,075 | PASS | PASS | PASS |
| Day 74 | 2026-02-26 | 2,071 | 2,071 | 2,071 | PASS | PASS | PASS |
| Day 75 | 2026-02-27 | 2,069 | 2,069 | 2,069 | PASS | PASS | PASS |
| Day 76 | 2026-02-28 | 2,065 | 2,065 | 2,065 | PASS | PASS | PASS |
| **合计** | | **157,433** | **157,433** | **157,433** | **76/76天PASS** | **76/76天PASS** | **76/76天PASS** |

#### 削减日志 (CutLog)

- **总行数 (全76天)**: Dev=157,433, Src=157,433, DB=157,433
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-12-15 | 1,314 | 1,314 | 1,314 | PASS | PASS | PASS |
| Day 02 | 2025-12-16 | 1,780 | 1,780 | 1,780 | PASS | PASS | PASS |
| Day 03 | 2025-12-17 | 2,070 | 2,070 | 2,070 | PASS | PASS | PASS |
| Day 04 | 2025-12-18 | 2,120 | 2,120 | 2,120 | PASS | PASS | PASS |
| Day 05 | 2025-12-19 | 2,144 | 2,144 | 2,144 | PASS | PASS | PASS |
| Day 06 | 2025-12-20 | 2,145 | 2,145 | 2,145 | PASS | PASS | PASS |
| Day 07 | 2025-12-21 | 2,145 | 2,145 | 2,145 | PASS | PASS | PASS |
| Day 08 | 2025-12-22 | 2,143 | 2,143 | 2,143 | PASS | PASS | PASS |
| Day 09 | 2025-12-23 | 2,106 | 2,106 | 2,106 | PASS | PASS | PASS |
| Day 10 | 2025-12-24 | 2,102 | 2,102 | 2,102 | PASS | PASS | PASS |
| Day 11 | 2025-12-25 | 2,101 | 2,101 | 2,101 | PASS | PASS | PASS |
| Day 12 | 2025-12-26 | 2,101 | 2,101 | 2,101 | PASS | PASS | PASS |
| Day 13 | 2025-12-27 | 2,101 | 2,101 | 2,101 | PASS | PASS | PASS |
| Day 14 | 2025-12-28 | 2,101 | 2,101 | 2,101 | PASS | PASS | PASS |
| Day 15 | 2025-12-29 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 16 | 2025-12-30 | 2,114 | 2,114 | 2,114 | PASS | PASS | PASS |
| Day 17 | 2025-12-31 | 2,111 | 2,111 | 2,111 | PASS | PASS | PASS |
| Day 18 | 2026-01-01 | 2,108 | 2,108 | 2,108 | PASS | PASS | PASS |
| Day 19 | 2026-01-02 | 2,106 | 2,106 | 2,106 | PASS | PASS | PASS |
| Day 20 | 2026-01-03 | 2,103 | 2,103 | 2,103 | PASS | PASS | PASS |
| Day 21 | 2026-01-04 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 22 | 2026-01-05 | 2,050 | 2,050 | 2,050 | PASS | PASS | PASS |
| Day 23 | 2026-01-06 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 24 | 2026-01-07 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 25 | 2026-01-08 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 26 | 2026-01-09 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 27 | 2026-01-10 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 28 | 2026-01-11 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 29 | 2026-01-12 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 30 | 2026-01-13 | 2,102 | 2,102 | 2,102 | PASS | PASS | PASS |
| Day 31 | 2026-01-14 | 2,102 | 2,102 | 2,102 | PASS | PASS | PASS |
| Day 32 | 2026-01-15 | 2,102 | 2,102 | 2,102 | PASS | PASS | PASS |
| Day 33 | 2026-01-16 | 2,102 | 2,102 | 2,102 | PASS | PASS | PASS |
| Day 34 | 2026-01-17 | 2,102 | 2,102 | 2,102 | PASS | PASS | PASS |
| Day 35 | 2026-01-18 | 2,102 | 2,102 | 2,102 | PASS | PASS | PASS |
| Day 36 | 2026-01-19 | 2,102 | 2,102 | 2,102 | PASS | PASS | PASS |
| Day 37 | 2026-01-20 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 38 | 2026-01-21 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 39 | 2026-01-22 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 40 | 2026-01-23 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 41 | 2026-01-24 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 42 | 2026-01-25 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 43 | 2026-01-26 | 2,100 | 2,100 | 2,100 | PASS | PASS | PASS |
| Day 44 | 2026-01-27 | 2,092 | 2,092 | 2,092 | PASS | PASS | PASS |
| Day 45 | 2026-01-28 | 2,092 | 2,092 | 2,092 | PASS | PASS | PASS |
| Day 46 | 2026-01-29 | 2,092 | 2,092 | 2,092 | PASS | PASS | PASS |
| Day 47 | 2026-01-30 | 2,092 | 2,092 | 2,092 | PASS | PASS | PASS |
| Day 48 | 2026-01-31 | 2,092 | 2,092 | 2,092 | PASS | PASS | PASS |
| Day 49 | 2026-02-01 | 2,092 | 2,092 | 2,092 | PASS | PASS | PASS |
| Day 50 | 2026-02-02 | 2,092 | 2,092 | 2,092 | PASS | PASS | PASS |
| Day 51 | 2026-02-03 | 2,077 | 2,077 | 2,077 | PASS | PASS | PASS |
| Day 52 | 2026-02-04 | 2,076 | 2,076 | 2,076 | PASS | PASS | PASS |
| Day 53 | 2026-02-05 | 2,076 | 2,076 | 2,076 | PASS | PASS | PASS |
| Day 54 | 2026-02-06 | 2,076 | 2,076 | 2,076 | PASS | PASS | PASS |
| Day 55 | 2026-02-07 | 2,076 | 2,076 | 2,076 | PASS | PASS | PASS |
| Day 56 | 2026-02-08 | 2,076 | 2,076 | 2,076 | PASS | PASS | PASS |
| Day 57 | 2026-02-09 | 2,076 | 2,076 | 2,076 | PASS | PASS | PASS |
| Day 58 | 2026-02-10 | 2,009 | 2,009 | 2,009 | PASS | PASS | PASS |
| Day 59 | 2026-02-11 | 1,983 | 1,983 | 1,983 | PASS | PASS | PASS |
| Day 60 | 2026-02-12 | 1,971 | 1,971 | 1,971 | PASS | PASS | PASS |
| Day 61 | 2026-02-13 | 1,967 | 1,967 | 1,967 | PASS | PASS | PASS |
| Day 62 | 2026-02-14 | 1,966 | 1,966 | 1,966 | PASS | PASS | PASS |
| Day 63 | 2026-02-15 | 1,966 | 1,966 | 1,966 | PASS | PASS | PASS |
| Day 64 | 2026-02-16 | 1,966 | 1,966 | 1,966 | PASS | PASS | PASS |
| Day 65 | 2026-02-17 | 2,098 | 2,098 | 2,098 | PASS | PASS | PASS |
| Day 66 | 2026-02-18 | 2,098 | 2,098 | 2,098 | PASS | PASS | PASS |
| Day 67 | 2026-02-19 | 2,098 | 2,098 | 2,098 | PASS | PASS | PASS |
| Day 68 | 2026-02-20 | 2,098 | 2,098 | 2,098 | PASS | PASS | PASS |
| Day 69 | 2026-02-21 | 2,098 | 2,098 | 2,098 | PASS | PASS | PASS |
| Day 70 | 2026-02-22 | 2,098 | 2,098 | 2,098 | PASS | PASS | PASS |
| Day 71 | 2026-02-23 | 2,098 | 2,098 | 2,098 | PASS | PASS | PASS |
| Day 72 | 2026-02-24 | 2,083 | 2,083 | 2,083 | PASS | PASS | PASS |
| Day 73 | 2026-02-25 | 2,075 | 2,075 | 2,075 | PASS | PASS | PASS |
| Day 74 | 2026-02-26 | 2,071 | 2,071 | 2,071 | PASS | PASS | PASS |
| Day 75 | 2026-02-27 | 2,069 | 2,069 | 2,069 | PASS | PASS | PASS |
| Day 76 | 2026-02-28 | 2,065 | 2,065 | 2,065 | PASS | PASS | PASS |
| **合计** | | **157,433** | **157,433** | **157,433** | **76/76天PASS** | **76/76天PASS** | **76/76天PASS** |

#### 每日汇总 (Summary)

- **总行数 (全76天)**: Dev=76, Src=76, DB=76
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-12-15 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 02 | 2025-12-16 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 03 | 2025-12-17 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 04 | 2025-12-18 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 05 | 2025-12-19 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 06 | 2025-12-20 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 07 | 2025-12-21 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 08 | 2025-12-22 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 09 | 2025-12-23 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 10 | 2025-12-24 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 11 | 2025-12-25 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 12 | 2025-12-26 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 13 | 2025-12-27 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 14 | 2025-12-28 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 15 | 2025-12-29 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 16 | 2025-12-30 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 17 | 2025-12-31 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 18 | 2026-01-01 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 19 | 2026-01-02 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 20 | 2026-01-03 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 21 | 2026-01-04 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 22 | 2026-01-05 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 23 | 2026-01-06 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 24 | 2026-01-07 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 25 | 2026-01-08 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 26 | 2026-01-09 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 27 | 2026-01-10 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 28 | 2026-01-11 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 29 | 2026-01-12 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 30 | 2026-01-13 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 31 | 2026-01-14 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 32 | 2026-01-15 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 33 | 2026-01-16 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 34 | 2026-01-17 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 35 | 2026-01-18 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 36 | 2026-01-19 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 37 | 2026-01-20 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 38 | 2026-01-21 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 39 | 2026-01-22 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 40 | 2026-01-23 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 41 | 2026-01-24 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 42 | 2026-01-25 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 43 | 2026-01-26 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 44 | 2026-01-27 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 45 | 2026-01-28 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 46 | 2026-01-29 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 47 | 2026-01-30 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 48 | 2026-01-31 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 49 | 2026-02-01 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 50 | 2026-02-02 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 51 | 2026-02-03 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 52 | 2026-02-04 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 53 | 2026-02-05 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 54 | 2026-02-06 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 55 | 2026-02-07 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 56 | 2026-02-08 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 57 | 2026-02-09 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 58 | 2026-02-10 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 59 | 2026-02-11 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 60 | 2026-02-12 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 61 | 2026-02-13 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 62 | 2026-02-14 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 63 | 2026-02-15 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 64 | 2026-02-16 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 65 | 2026-02-17 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 66 | 2026-02-18 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 67 | 2026-02-19 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 68 | 2026-02-20 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 69 | 2026-02-21 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 70 | 2026-02-22 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 71 | 2026-02-23 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 72 | 2026-02-24 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 73 | 2026-02-25 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 74 | 2026-02-26 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 75 | 2026-02-27 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 76 | 2026-02-28 | 1 | 1 | 1 | PASS | PASS | PASS |
| **合计** | | **76** | **76** | **76** | **76/76天PASS** | **76/76天PASS** | **76/76天PASS** |

### 3.3 Module3 - 净需求计算模块

#### 净需求 (NetDemand)

- **总行数 (全76天)**: Dev=93,693, Src=93,693, DB=93,693
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-12-15 | 1,227 | 1,227 | 1,227 | PASS | PASS | PASS |
| Day 02 | 2025-12-16 | 1,331 | 1,331 | 1,331 | PASS | PASS | PASS |
| Day 03 | 2025-12-17 | 1,492 | 1,492 | 1,492 | PASS | PASS | PASS |
| Day 04 | 2025-12-18 | 1,644 | 1,644 | 1,644 | PASS | PASS | PASS |
| Day 05 | 2025-12-19 | 1,743 | 1,743 | 1,743 | PASS | PASS | PASS |
| Day 06 | 2025-12-20 | 1,888 | 1,888 | 1,888 | PASS | PASS | PASS |
| Day 07 | 2025-12-21 | 2,001 | 2,001 | 2,001 | PASS | PASS | PASS |
| Day 08 | 2025-12-22 | 2,032 | 2,032 | 2,032 | PASS | PASS | PASS |
| Day 09 | 2025-12-23 | 2,020 | 2,020 | 2,020 | PASS | PASS | PASS |
| Day 10 | 2025-12-24 | 2,005 | 2,005 | 2,005 | PASS | PASS | PASS |
| Day 11 | 2025-12-25 | 1,958 | 1,958 | 1,958 | PASS | PASS | PASS |
| Day 12 | 2025-12-26 | 1,920 | 1,920 | 1,920 | PASS | PASS | PASS |
| Day 13 | 2025-12-27 | 1,886 | 1,886 | 1,886 | PASS | PASS | PASS |
| Day 14 | 2025-12-28 | 1,845 | 1,845 | 1,845 | PASS | PASS | PASS |
| Day 15 | 2025-12-29 | 1,819 | 1,819 | 1,819 | PASS | PASS | PASS |
| Day 16 | 2025-12-30 | 1,782 | 1,782 | 1,782 | PASS | PASS | PASS |
| Day 17 | 2025-12-31 | 1,784 | 1,784 | 1,784 | PASS | PASS | PASS |
| Day 18 | 2026-01-01 | 1,835 | 1,835 | 1,835 | PASS | PASS | PASS |
| Day 19 | 2026-01-02 | 1,816 | 1,816 | 1,816 | PASS | PASS | PASS |
| Day 20 | 2026-01-03 | 1,799 | 1,799 | 1,799 | PASS | PASS | PASS |
| Day 21 | 2026-01-04 | 1,815 | 1,815 | 1,815 | PASS | PASS | PASS |
| Day 22 | 2026-01-05 | 1,796 | 1,796 | 1,796 | PASS | PASS | PASS |
| Day 23 | 2026-01-06 | 1,753 | 1,753 | 1,753 | PASS | PASS | PASS |
| Day 24 | 2026-01-07 | 1,717 | 1,717 | 1,717 | PASS | PASS | PASS |
| Day 25 | 2026-01-08 | 1,694 | 1,694 | 1,694 | PASS | PASS | PASS |
| Day 26 | 2026-01-09 | 1,689 | 1,689 | 1,689 | PASS | PASS | PASS |
| Day 27 | 2026-01-10 | 1,726 | 1,726 | 1,726 | PASS | PASS | PASS |
| Day 28 | 2026-01-11 | 1,664 | 1,664 | 1,664 | PASS | PASS | PASS |
| Day 29 | 2026-01-12 | 1,515 | 1,515 | 1,515 | PASS | PASS | PASS |
| Day 30 | 2026-01-13 | 1,490 | 1,490 | 1,490 | PASS | PASS | PASS |
| Day 31 | 2026-01-14 | 1,379 | 1,379 | 1,379 | PASS | PASS | PASS |
| Day 32 | 2026-01-15 | 1,344 | 1,344 | 1,344 | PASS | PASS | PASS |
| Day 33 | 2026-01-16 | 1,376 | 1,376 | 1,376 | PASS | PASS | PASS |
| Day 34 | 2026-01-17 | 1,284 | 1,284 | 1,284 | PASS | PASS | PASS |
| Day 35 | 2026-01-18 | 1,286 | 1,286 | 1,286 | PASS | PASS | PASS |
| Day 36 | 2026-01-19 | 1,159 | 1,159 | 1,159 | PASS | PASS | PASS |
| Day 37 | 2026-01-20 | 1,111 | 1,111 | 1,111 | PASS | PASS | PASS |
| Day 38 | 2026-01-21 | 1,122 | 1,122 | 1,122 | PASS | PASS | PASS |
| Day 39 | 2026-01-22 | 1,088 | 1,088 | 1,088 | PASS | PASS | PASS |
| Day 40 | 2026-01-23 | 1,184 | 1,184 | 1,184 | PASS | PASS | PASS |
| Day 41 | 2026-01-24 | 1,188 | 1,188 | 1,188 | PASS | PASS | PASS |
| Day 42 | 2026-01-25 | 1,113 | 1,113 | 1,113 | PASS | PASS | PASS |
| Day 43 | 2026-01-26 | 1,051 | 1,051 | 1,051 | PASS | PASS | PASS |
| Day 44 | 2026-01-27 | 1,000 | 1,000 | 1,000 | PASS | PASS | PASS |
| Day 45 | 2026-01-28 | 997 | 997 | 997 | PASS | PASS | PASS |
| Day 46 | 2026-01-29 | 1,012 | 1,012 | 1,012 | PASS | PASS | PASS |
| Day 47 | 2026-01-30 | 1,060 | 1,060 | 1,060 | PASS | PASS | PASS |
| Day 48 | 2026-01-31 | 1,014 | 1,014 | 1,014 | PASS | PASS | PASS |
| Day 49 | 2026-02-01 | 1,022 | 1,022 | 1,022 | PASS | PASS | PASS |
| Day 50 | 2026-02-02 | 986 | 986 | 986 | PASS | PASS | PASS |
| Day 51 | 2026-02-03 | 953 | 953 | 953 | PASS | PASS | PASS |
| Day 52 | 2026-02-04 | 956 | 956 | 956 | PASS | PASS | PASS |
| Day 53 | 2026-02-05 | 930 | 930 | 930 | PASS | PASS | PASS |
| Day 54 | 2026-02-06 | 919 | 919 | 919 | PASS | PASS | PASS |
| Day 55 | 2026-02-07 | 924 | 924 | 924 | PASS | PASS | PASS |
| Day 56 | 2026-02-08 | 866 | 866 | 866 | PASS | PASS | PASS |
| Day 57 | 2026-02-09 | 843 | 843 | 843 | PASS | PASS | PASS |
| Day 58 | 2026-02-10 | 769 | 769 | 769 | PASS | PASS | PASS |
| Day 59 | 2026-02-11 | 730 | 730 | 730 | PASS | PASS | PASS |
| Day 60 | 2026-02-12 | 715 | 715 | 715 | PASS | PASS | PASS |
| Day 61 | 2026-02-13 | 699 | 699 | 699 | PASS | PASS | PASS |
| Day 62 | 2026-02-14 | 671 | 671 | 671 | PASS | PASS | PASS |
| Day 63 | 2026-02-15 | 683 | 683 | 683 | PASS | PASS | PASS |
| Day 64 | 2026-02-16 | 788 | 788 | 788 | PASS | PASS | PASS |
| Day 65 | 2026-02-17 | 794 | 794 | 794 | PASS | PASS | PASS |
| Day 66 | 2026-02-18 | 845 | 845 | 845 | PASS | PASS | PASS |
| Day 67 | 2026-02-19 | 922 | 922 | 922 | PASS | PASS | PASS |
| Day 68 | 2026-02-20 | 921 | 921 | 921 | PASS | PASS | PASS |
| Day 69 | 2026-02-21 | 847 | 847 | 847 | PASS | PASS | PASS |
| Day 70 | 2026-02-22 | 527 | 527 | 527 | PASS | PASS | PASS |
| Day 71 | 2026-02-23 | 476 | 476 | 476 | PASS | PASS | PASS |
| Day 72 | 2026-02-24 | 394 | 394 | 394 | PASS | PASS | PASS |
| Day 73 | 2026-02-25 | 225 | 225 | 225 | PASS | PASS | PASS |
| Day 74 | 2026-02-26 | 238 | 238 | 238 | PASS | PASS | PASS |
| Day 75 | 2026-02-27 | 277 | 277 | 277 | PASS | PASS | PASS |
| Day 76 | 2026-02-28 | 319 | 319 | 319 | PASS | PASS | PASS |
| **合计** | | **93,693** | **93,693** | **93,693** | **76/76天PASS** | **76/76天PASS** | **76/76天PASS** |

### 3.4 Module4 - 生产计划模块

#### 生产计划 (ProductionPlan)

- **总行数 (全76天)**: Dev=685, Src=685, DB=685
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-12-15 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 02 | 2025-12-16 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 03 | 2025-12-17 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 04 | 2025-12-18 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 05 | 2025-12-19 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 06 | 2025-12-20 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 07 | 2025-12-21 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 08 | 2025-12-22 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 09 | 2025-12-23 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 10 | 2025-12-24 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 11 | 2025-12-25 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 12 | 2025-12-26 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 13 | 2025-12-27 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 14 | 2025-12-28 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 15 | 2025-12-29 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 16 | 2025-12-30 | 14 | 14 | 14 | PASS | PASS | PASS |
| Day 17 | 2025-12-31 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 18 | 2026-01-01 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 19 | 2026-01-02 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 20 | 2026-01-03 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 21 | 2026-01-04 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 22 | 2026-01-05 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 23 | 2026-01-06 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 24 | 2026-01-07 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 25 | 2026-01-08 | 14 | 14 | 14 | PASS | PASS | PASS |
| Day 26 | 2026-01-09 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 27 | 2026-01-10 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 28 | 2026-01-11 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 29 | 2026-01-12 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 30 | 2026-01-13 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 31 | 2026-01-14 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 32 | 2026-01-15 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 33 | 2026-01-16 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 34 | 2026-01-17 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 35 | 2026-01-18 | 14 | 14 | 14 | PASS | PASS | PASS |
| Day 36 | 2026-01-19 | 14 | 14 | 14 | PASS | PASS | PASS |
| Day 37 | 2026-01-20 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 38 | 2026-01-21 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 39 | 2026-01-22 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 40 | 2026-01-23 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 41 | 2026-01-24 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 42 | 2026-01-25 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 43 | 2026-01-26 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 44 | 2026-01-27 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 45 | 2026-01-28 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 46 | 2026-01-29 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 47 | 2026-01-30 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 48 | 2026-01-31 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 49 | 2026-02-01 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 50 | 2026-02-02 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 51 | 2026-02-03 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 52 | 2026-02-04 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 53 | 2026-02-05 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 54 | 2026-02-06 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 55 | 2026-02-07 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 56 | 2026-02-08 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 57 | 2026-02-09 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 58 | 2026-02-10 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 59 | 2026-02-11 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 60 | 2026-02-12 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 61 | 2026-02-13 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 62 | 2026-02-14 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 63 | 2026-02-15 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 64 | 2026-02-16 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 65 | 2026-02-17 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 66 | 2026-02-18 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 67 | 2026-02-19 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 68 | 2026-02-20 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 69 | 2026-02-21 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 70 | 2026-02-22 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 71 | 2026-02-23 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 72 | 2026-02-24 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 73 | 2026-02-25 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 74 | 2026-02-26 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 75 | 2026-02-27 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 76 | 2026-02-28 | 5 | 5 | 5 | PASS | PASS | PASS |
| **合计** | | **685** | **685** | **685** | **76/76天PASS** | **76/76天PASS** | **76/76天PASS** |

#### 超容量报告 (CapacityExceed)

- **总行数 (全76天)**: Dev=3,018, Src=3,018, DB=3,018
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-12-15 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 02 | 2025-12-16 | 155 | 155 | 155 | PASS | PASS | PASS |
| Day 03 | 2025-12-17 | 149 | 149 | 149 | PASS | PASS | PASS |
| Day 04 | 2025-12-18 | 147 | 147 | 147 | PASS | PASS | PASS |
| Day 05 | 2025-12-19 | 142 | 142 | 142 | PASS | PASS | PASS |
| Day 06 | 2025-12-20 | 131 | 131 | 131 | PASS | PASS | PASS |
| Day 07 | 2025-12-21 | 126 | 126 | 126 | PASS | PASS | PASS |
| Day 08 | 2025-12-22 | 125 | 125 | 125 | PASS | PASS | PASS |
| Day 09 | 2025-12-23 | 117 | 117 | 117 | PASS | PASS | PASS |
| Day 10 | 2025-12-24 | 110 | 110 | 110 | PASS | PASS | PASS |
| Day 11 | 2025-12-25 | 104 | 104 | 104 | PASS | PASS | PASS |
| Day 12 | 2025-12-26 | 99 | 99 | 99 | PASS | PASS | PASS |
| Day 13 | 2025-12-27 | 88 | 88 | 88 | PASS | PASS | PASS |
| Day 14 | 2025-12-28 | 83 | 83 | 83 | PASS | PASS | PASS |
| Day 15 | 2025-12-29 | 78 | 78 | 78 | PASS | PASS | PASS |
| Day 16 | 2025-12-30 | 74 | 74 | 74 | PASS | PASS | PASS |
| Day 17 | 2025-12-31 | 73 | 73 | 73 | PASS | PASS | PASS |
| Day 18 | 2026-01-01 | 69 | 69 | 69 | PASS | PASS | PASS |
| Day 19 | 2026-01-02 | 72 | 72 | 72 | PASS | PASS | PASS |
| Day 20 | 2026-01-03 | 68 | 68 | 68 | PASS | PASS | PASS |
| Day 21 | 2026-01-04 | 57 | 57 | 57 | PASS | PASS | PASS |
| Day 22 | 2026-01-05 | 55 | 55 | 55 | PASS | PASS | PASS |
| Day 23 | 2026-01-06 | 55 | 55 | 55 | PASS | PASS | PASS |
| Day 24 | 2026-01-07 | 54 | 54 | 54 | PASS | PASS | PASS |
| Day 25 | 2026-01-08 | 44 | 44 | 44 | PASS | PASS | PASS |
| Day 26 | 2026-01-09 | 40 | 40 | 40 | PASS | PASS | PASS |
| Day 27 | 2026-01-10 | 44 | 44 | 44 | PASS | PASS | PASS |
| Day 28 | 2026-01-11 | 34 | 34 | 34 | PASS | PASS | PASS |
| Day 29 | 2026-01-12 | 34 | 34 | 34 | PASS | PASS | PASS |
| Day 30 | 2026-01-13 | 32 | 32 | 32 | PASS | PASS | PASS |
| Day 31 | 2026-01-14 | 31 | 31 | 31 | PASS | PASS | PASS |
| Day 32 | 2026-01-15 | 33 | 33 | 33 | PASS | PASS | PASS |
| Day 33 | 2026-01-16 | 39 | 39 | 39 | PASS | PASS | PASS |
| Day 34 | 2026-01-17 | 35 | 35 | 35 | PASS | PASS | PASS |
| Day 35 | 2026-01-18 | 28 | 28 | 28 | PASS | PASS | PASS |
| Day 36 | 2026-01-19 | 22 | 22 | 22 | PASS | PASS | PASS |
| Day 37 | 2026-01-20 | 19 | 19 | 19 | PASS | PASS | PASS |
| Day 38 | 2026-01-21 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 39 | 2026-01-22 | 17 | 17 | 17 | PASS | PASS | PASS |
| Day 40 | 2026-01-23 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 41 | 2026-01-24 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 42 | 2026-01-25 | 18 | 18 | 18 | PASS | PASS | PASS |
| Day 43 | 2026-01-26 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 44 | 2026-01-27 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 45 | 2026-01-28 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 46 | 2026-01-29 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 47 | 2026-01-30 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 48 | 2026-01-31 | 17 | 17 | 17 | PASS | PASS | PASS |
| Day 49 | 2026-02-01 | 17 | 17 | 17 | PASS | PASS | PASS |
| Day 50 | 2026-02-02 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 51 | 2026-02-03 | 20 | 20 | 20 | PASS | PASS | PASS |
| Day 52 | 2026-02-04 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 53 | 2026-02-05 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 54 | 2026-02-06 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 55 | 2026-02-07 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 56 | 2026-02-08 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 57 | 2026-02-09 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 58 | 2026-02-10 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 59 | 2026-02-11 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 60 | 2026-02-12 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 61 | 2026-02-13 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 62 | 2026-02-14 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 63 | 2026-02-15 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 64 | 2026-02-16 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 65 | 2026-02-17 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 66 | 2026-02-18 | 16 | 16 | 16 | PASS | PASS | PASS |
| Day 67 | 2026-02-19 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 68 | 2026-02-20 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 69 | 2026-02-21 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 70 | 2026-02-22 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 71 | 2026-02-23 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 72 | 2026-02-24 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 73 | 2026-02-25 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 74 | 2026-02-26 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 75 | 2026-02-27 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 76 | 2026-02-28 | 0 | 0 | 0 | PASS | PASS | PASS |
| **合计** | | **3,018** | **3,018** | **3,018** | **76/76天PASS** | **76/76天PASS** | **76/76天PASS** |

#### 换型日志 (ChangeoverLog)

- **总行数 (全76天)**: Dev=515, Src=515, DB=515
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-12-15 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 02 | 2025-12-16 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 03 | 2025-12-17 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 04 | 2025-12-18 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 05 | 2025-12-19 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 06 | 2025-12-20 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 07 | 2025-12-21 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 08 | 2025-12-22 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 09 | 2025-12-23 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 10 | 2025-12-24 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 11 | 2025-12-25 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 12 | 2025-12-26 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 13 | 2025-12-27 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 14 | 2025-12-28 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 15 | 2025-12-29 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 16 | 2025-12-30 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 17 | 2025-12-31 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 18 | 2026-01-01 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 19 | 2026-01-02 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 20 | 2026-01-03 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 21 | 2026-01-04 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 22 | 2026-01-05 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 23 | 2026-01-06 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 24 | 2026-01-07 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 25 | 2026-01-08 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 26 | 2026-01-09 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 27 | 2026-01-10 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 28 | 2026-01-11 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 29 | 2026-01-12 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 30 | 2026-01-13 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 31 | 2026-01-14 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 32 | 2026-01-15 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 33 | 2026-01-16 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 34 | 2026-01-17 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 35 | 2026-01-18 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 36 | 2026-01-19 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 37 | 2026-01-20 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 38 | 2026-01-21 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 39 | 2026-01-22 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 40 | 2026-01-23 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 41 | 2026-01-24 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 42 | 2026-01-25 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 43 | 2026-01-26 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 44 | 2026-01-27 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 45 | 2026-01-28 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 46 | 2026-01-29 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 47 | 2026-01-30 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 48 | 2026-01-31 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 49 | 2026-02-01 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 50 | 2026-02-02 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 51 | 2026-02-03 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 52 | 2026-02-04 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 53 | 2026-02-05 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 54 | 2026-02-06 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 55 | 2026-02-07 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 56 | 2026-02-08 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 57 | 2026-02-09 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 58 | 2026-02-10 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 59 | 2026-02-11 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 60 | 2026-02-12 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 61 | 2026-02-13 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 62 | 2026-02-14 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 63 | 2026-02-15 | 2 | 2 | 2 | PASS | PASS | PASS |
| Day 64 | 2026-02-16 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 65 | 2026-02-17 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 66 | 2026-02-18 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 67 | 2026-02-19 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 68 | 2026-02-20 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 69 | 2026-02-21 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 70 | 2026-02-22 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 71 | 2026-02-23 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 72 | 2026-02-24 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 73 | 2026-02-25 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 74 | 2026-02-26 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 75 | 2026-02-27 | 3 | 3 | 3 | PASS | PASS | PASS |
| Day 76 | 2026-02-28 | 4 | 4 | 4 | PASS | PASS | PASS |
| **合计** | | **515** | **515** | **515** | **76/76天PASS** | **76/76天PASS** | **76/76天PASS** |

### 3.5 Module5 - 部署规划模块

#### 部署计划 (DeploymentPlan)

- **总行数 (全76天)**: Dev=3,336,330, Src=3,336,330, DB=3,336,330
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-12-15 | 31,196 | 31,196 | 31,196 | PASS | PASS | PASS |
| Day 02 | 2025-12-16 | 34,775 | 34,775 | 34,775 | PASS | PASS | PASS |
| Day 03 | 2025-12-17 | 38,074 | 38,074 | 38,074 | PASS | PASS | PASS |
| Day 04 | 2025-12-18 | 40,845 | 40,845 | 40,845 | PASS | PASS | PASS |
| Day 05 | 2025-12-19 | 43,045 | 43,045 | 43,045 | PASS | PASS | PASS |
| Day 06 | 2025-12-20 | 44,803 | 44,803 | 44,803 | PASS | PASS | PASS |
| Day 07 | 2025-12-21 | 46,231 | 46,231 | 46,231 | PASS | PASS | PASS |
| Day 08 | 2025-12-22 | 47,252 | 47,252 | 47,252 | PASS | PASS | PASS |
| Day 09 | 2025-12-23 | 47,667 | 47,667 | 47,667 | PASS | PASS | PASS |
| Day 10 | 2025-12-24 | 47,936 | 47,936 | 47,936 | PASS | PASS | PASS |
| Day 11 | 2025-12-25 | 47,977 | 47,977 | 47,977 | PASS | PASS | PASS |
| Day 12 | 2025-12-26 | 47,670 | 47,670 | 47,670 | PASS | PASS | PASS |
| Day 13 | 2025-12-27 | 47,443 | 47,443 | 47,443 | PASS | PASS | PASS |
| Day 14 | 2025-12-28 | 47,278 | 47,278 | 47,278 | PASS | PASS | PASS |
| Day 15 | 2025-12-29 | 47,100 | 47,100 | 47,100 | PASS | PASS | PASS |
| Day 16 | 2025-12-30 | 47,257 | 47,257 | 47,257 | PASS | PASS | PASS |
| Day 17 | 2025-12-31 | 47,228 | 47,228 | 47,228 | PASS | PASS | PASS |
| Day 18 | 2026-01-01 | 47,224 | 47,224 | 47,224 | PASS | PASS | PASS |
| Day 19 | 2026-01-02 | 47,224 | 47,224 | 47,224 | PASS | PASS | PASS |
| Day 20 | 2026-01-03 | 47,421 | 47,421 | 47,421 | PASS | PASS | PASS |
| Day 21 | 2026-01-04 | 47,700 | 47,700 | 47,700 | PASS | PASS | PASS |
| Day 22 | 2026-01-05 | 47,750 | 47,750 | 47,750 | PASS | PASS | PASS |
| Day 23 | 2026-01-06 | 47,594 | 47,594 | 47,594 | PASS | PASS | PASS |
| Day 24 | 2026-01-07 | 47,367 | 47,367 | 47,367 | PASS | PASS | PASS |
| Day 25 | 2026-01-08 | 47,595 | 47,595 | 47,595 | PASS | PASS | PASS |
| Day 26 | 2026-01-09 | 47,704 | 47,704 | 47,704 | PASS | PASS | PASS |
| Day 27 | 2026-01-10 | 47,906 | 47,906 | 47,906 | PASS | PASS | PASS |
| Day 28 | 2026-01-11 | 47,893 | 47,893 | 47,893 | PASS | PASS | PASS |
| Day 29 | 2026-01-12 | 47,102 | 47,102 | 47,102 | PASS | PASS | PASS |
| Day 30 | 2026-01-13 | 46,598 | 46,598 | 46,598 | PASS | PASS | PASS |
| Day 31 | 2026-01-14 | 46,261 | 46,261 | 46,261 | PASS | PASS | PASS |
| Day 32 | 2026-01-15 | 45,934 | 45,934 | 45,934 | PASS | PASS | PASS |
| Day 33 | 2026-01-16 | 45,756 | 45,756 | 45,756 | PASS | PASS | PASS |
| Day 34 | 2026-01-17 | 45,665 | 45,665 | 45,665 | PASS | PASS | PASS |
| Day 35 | 2026-01-18 | 45,294 | 45,294 | 45,294 | PASS | PASS | PASS |
| Day 36 | 2026-01-19 | 44,779 | 44,779 | 44,779 | PASS | PASS | PASS |
| Day 37 | 2026-01-20 | 44,495 | 44,495 | 44,495 | PASS | PASS | PASS |
| Day 38 | 2026-01-21 | 44,052 | 44,052 | 44,052 | PASS | PASS | PASS |
| Day 39 | 2026-01-22 | 43,990 | 43,990 | 43,990 | PASS | PASS | PASS |
| Day 40 | 2026-01-23 | 44,183 | 44,183 | 44,183 | PASS | PASS | PASS |
| Day 41 | 2026-01-24 | 44,240 | 44,240 | 44,240 | PASS | PASS | PASS |
| Day 42 | 2026-01-25 | 44,044 | 44,044 | 44,044 | PASS | PASS | PASS |
| Day 43 | 2026-01-26 | 43,756 | 43,756 | 43,756 | PASS | PASS | PASS |
| Day 44 | 2026-01-27 | 43,686 | 43,686 | 43,686 | PASS | PASS | PASS |
| Day 45 | 2026-01-28 | 43,709 | 43,709 | 43,709 | PASS | PASS | PASS |
| Day 46 | 2026-01-29 | 43,859 | 43,859 | 43,859 | PASS | PASS | PASS |
| Day 47 | 2026-01-30 | 43,794 | 43,794 | 43,794 | PASS | PASS | PASS |
| Day 48 | 2026-01-31 | 43,880 | 43,880 | 43,880 | PASS | PASS | PASS |
| Day 49 | 2026-02-01 | 43,687 | 43,687 | 43,687 | PASS | PASS | PASS |
| Day 50 | 2026-02-02 | 43,316 | 43,316 | 43,316 | PASS | PASS | PASS |
| Day 51 | 2026-02-03 | 43,006 | 43,006 | 43,006 | PASS | PASS | PASS |
| Day 52 | 2026-02-04 | 42,995 | 42,995 | 42,995 | PASS | PASS | PASS |
| Day 53 | 2026-02-05 | 42,780 | 42,780 | 42,780 | PASS | PASS | PASS |
| Day 54 | 2026-02-06 | 42,646 | 42,646 | 42,646 | PASS | PASS | PASS |
| Day 55 | 2026-02-07 | 42,518 | 42,518 | 42,518 | PASS | PASS | PASS |
| Day 56 | 2026-02-08 | 42,378 | 42,378 | 42,378 | PASS | PASS | PASS |
| Day 57 | 2026-02-09 | 42,241 | 42,241 | 42,241 | PASS | PASS | PASS |
| Day 58 | 2026-02-10 | 41,750 | 41,750 | 41,750 | PASS | PASS | PASS |
| Day 59 | 2026-02-11 | 41,560 | 41,560 | 41,560 | PASS | PASS | PASS |
| Day 60 | 2026-02-12 | 41,339 | 41,339 | 41,339 | PASS | PASS | PASS |
| Day 61 | 2026-02-13 | 41,184 | 41,184 | 41,184 | PASS | PASS | PASS |
| Day 62 | 2026-02-14 | 40,849 | 40,849 | 40,849 | PASS | PASS | PASS |
| Day 63 | 2026-02-15 | 40,958 | 40,958 | 40,958 | PASS | PASS | PASS |
| Day 64 | 2026-02-16 | 41,226 | 41,226 | 41,226 | PASS | PASS | PASS |
| Day 65 | 2026-02-17 | 41,117 | 41,117 | 41,117 | PASS | PASS | PASS |
| Day 66 | 2026-02-18 | 41,256 | 41,256 | 41,256 | PASS | PASS | PASS |
| Day 67 | 2026-02-19 | 41,070 | 41,070 | 41,070 | PASS | PASS | PASS |
| Day 68 | 2026-02-20 | 41,108 | 41,108 | 41,108 | PASS | PASS | PASS |
| Day 69 | 2026-02-21 | 41,077 | 41,077 | 41,077 | PASS | PASS | PASS |
| Day 70 | 2026-02-22 | 40,334 | 40,334 | 40,334 | PASS | PASS | PASS |
| Day 71 | 2026-02-23 | 40,444 | 40,444 | 40,444 | PASS | PASS | PASS |
| Day 72 | 2026-02-24 | 40,445 | 40,445 | 40,445 | PASS | PASS | PASS |
| Day 73 | 2026-02-25 | 40,258 | 40,258 | 40,258 | PASS | PASS | PASS |
| Day 74 | 2026-02-26 | 40,483 | 40,483 | 40,483 | PASS | PASS | PASS |
| Day 75 | 2026-02-27 | 40,833 | 40,833 | 40,833 | PASS | PASS | PASS |
| Day 76 | 2026-02-28 | 41,240 | 41,240 | 41,240 | PASS | PASS | PASS |
| **合计** | | **3,336,330** | **3,336,330** | **3,336,330** | **76/76天PASS** | **76/76天PASS** | **76/76天PASS** |

#### 未满足日志 (UnfulfilledLog)

- **总行数 (全76天)**: Dev=527,164, Src=527,164, DB=527,164
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-12-15 | 6,179 | 6,179 | 6,179 | PASS | PASS | PASS |
| Day 02 | 2025-12-16 | 6,917 | 6,917 | 6,917 | PASS | PASS | PASS |
| Day 03 | 2025-12-17 | 8,019 | 8,019 | 8,019 | PASS | PASS | PASS |
| Day 04 | 2025-12-18 | 9,205 | 9,205 | 9,205 | PASS | PASS | PASS |
| Day 05 | 2025-12-19 | 10,038 | 10,038 | 10,038 | PASS | PASS | PASS |
| Day 06 | 2025-12-20 | 11,372 | 11,372 | 11,372 | PASS | PASS | PASS |
| Day 07 | 2025-12-21 | 12,689 | 12,689 | 12,689 | PASS | PASS | PASS |
| Day 08 | 2025-12-22 | 13,476 | 13,476 | 13,476 | PASS | PASS | PASS |
| Day 09 | 2025-12-23 | 13,813 | 13,813 | 13,813 | PASS | PASS | PASS |
| Day 10 | 2025-12-24 | 13,629 | 13,629 | 13,629 | PASS | PASS | PASS |
| Day 11 | 2025-12-25 | 13,127 | 13,127 | 13,127 | PASS | PASS | PASS |
| Day 12 | 2025-12-26 | 12,648 | 12,648 | 12,648 | PASS | PASS | PASS |
| Day 13 | 2025-12-27 | 12,307 | 12,307 | 12,307 | PASS | PASS | PASS |
| Day 14 | 2025-12-28 | 11,839 | 11,839 | 11,839 | PASS | PASS | PASS |
| Day 15 | 2025-12-29 | 11,827 | 11,827 | 11,827 | PASS | PASS | PASS |
| Day 16 | 2025-12-30 | 11,759 | 11,759 | 11,759 | PASS | PASS | PASS |
| Day 17 | 2025-12-31 | 11,651 | 11,651 | 11,651 | PASS | PASS | PASS |
| Day 18 | 2026-01-01 | 11,790 | 11,790 | 11,790 | PASS | PASS | PASS |
| Day 19 | 2026-01-02 | 11,841 | 11,841 | 11,841 | PASS | PASS | PASS |
| Day 20 | 2026-01-03 | 12,152 | 12,152 | 12,152 | PASS | PASS | PASS |
| Day 21 | 2026-01-04 | 12,450 | 12,450 | 12,450 | PASS | PASS | PASS |
| Day 22 | 2026-01-05 | 11,974 | 11,974 | 11,974 | PASS | PASS | PASS |
| Day 23 | 2026-01-06 | 11,716 | 11,716 | 11,716 | PASS | PASS | PASS |
| Day 24 | 2026-01-07 | 11,563 | 11,563 | 11,563 | PASS | PASS | PASS |
| Day 25 | 2026-01-08 | 11,543 | 11,543 | 11,543 | PASS | PASS | PASS |
| Day 26 | 2026-01-09 | 11,940 | 11,940 | 11,940 | PASS | PASS | PASS |
| Day 27 | 2026-01-10 | 12,090 | 12,090 | 12,090 | PASS | PASS | PASS |
| Day 28 | 2026-01-11 | 11,652 | 11,652 | 11,652 | PASS | PASS | PASS |
| Day 29 | 2026-01-12 | 10,595 | 10,595 | 10,595 | PASS | PASS | PASS |
| Day 30 | 2026-01-13 | 9,529 | 9,529 | 9,529 | PASS | PASS | PASS |
| Day 31 | 2026-01-14 | 8,880 | 8,880 | 8,880 | PASS | PASS | PASS |
| Day 32 | 2026-01-15 | 8,525 | 8,525 | 8,525 | PASS | PASS | PASS |
| Day 33 | 2026-01-16 | 8,377 | 8,377 | 8,377 | PASS | PASS | PASS |
| Day 34 | 2026-01-17 | 8,066 | 8,066 | 8,066 | PASS | PASS | PASS |
| Day 35 | 2026-01-18 | 7,602 | 7,602 | 7,602 | PASS | PASS | PASS |
| Day 36 | 2026-01-19 | 6,810 | 6,810 | 6,810 | PASS | PASS | PASS |
| Day 37 | 2026-01-20 | 6,103 | 6,103 | 6,103 | PASS | PASS | PASS |
| Day 38 | 2026-01-21 | 5,702 | 5,702 | 5,702 | PASS | PASS | PASS |
| Day 39 | 2026-01-22 | 5,479 | 5,479 | 5,479 | PASS | PASS | PASS |
| Day 40 | 2026-01-23 | 5,840 | 5,840 | 5,840 | PASS | PASS | PASS |
| Day 41 | 2026-01-24 | 5,623 | 5,623 | 5,623 | PASS | PASS | PASS |
| Day 42 | 2026-01-25 | 5,331 | 5,331 | 5,331 | PASS | PASS | PASS |
| Day 43 | 2026-01-26 | 4,983 | 4,983 | 4,983 | PASS | PASS | PASS |
| Day 44 | 2026-01-27 | 4,585 | 4,585 | 4,585 | PASS | PASS | PASS |
| Day 45 | 2026-01-28 | 5,037 | 5,037 | 5,037 | PASS | PASS | PASS |
| Day 46 | 2026-01-29 | 4,901 | 4,901 | 4,901 | PASS | PASS | PASS |
| Day 47 | 2026-01-30 | 4,882 | 4,882 | 4,882 | PASS | PASS | PASS |
| Day 48 | 2026-01-31 | 4,846 | 4,846 | 4,846 | PASS | PASS | PASS |
| Day 49 | 2026-02-01 | 4,474 | 4,474 | 4,474 | PASS | PASS | PASS |
| Day 50 | 2026-02-02 | 4,265 | 4,265 | 4,265 | PASS | PASS | PASS |
| Day 51 | 2026-02-03 | 3,639 | 3,639 | 3,639 | PASS | PASS | PASS |
| Day 52 | 2026-02-04 | 3,806 | 3,806 | 3,806 | PASS | PASS | PASS |
| Day 53 | 2026-02-05 | 3,282 | 3,282 | 3,282 | PASS | PASS | PASS |
| Day 54 | 2026-02-06 | 3,294 | 3,294 | 3,294 | PASS | PASS | PASS |
| Day 55 | 2026-02-07 | 3,025 | 3,025 | 3,025 | PASS | PASS | PASS |
| Day 56 | 2026-02-08 | 2,719 | 2,719 | 2,719 | PASS | PASS | PASS |
| Day 57 | 2026-02-09 | 2,612 | 2,612 | 2,612 | PASS | PASS | PASS |
| Day 58 | 2026-02-10 | 2,080 | 2,080 | 2,080 | PASS | PASS | PASS |
| Day 59 | 2026-02-11 | 1,905 | 1,905 | 1,905 | PASS | PASS | PASS |
| Day 60 | 2026-02-12 | 1,739 | 1,739 | 1,739 | PASS | PASS | PASS |
| Day 61 | 2026-02-13 | 1,600 | 1,600 | 1,600 | PASS | PASS | PASS |
| Day 62 | 2026-02-14 | 1,348 | 1,348 | 1,348 | PASS | PASS | PASS |
| Day 63 | 2026-02-15 | 1,560 | 1,560 | 1,560 | PASS | PASS | PASS |
| Day 64 | 2026-02-16 | 1,815 | 1,815 | 1,815 | PASS | PASS | PASS |
| Day 65 | 2026-02-17 | 1,802 | 1,802 | 1,802 | PASS | PASS | PASS |
| Day 66 | 2026-02-18 | 1,991 | 1,991 | 1,991 | PASS | PASS | PASS |
| Day 67 | 2026-02-19 | 2,012 | 2,012 | 2,012 | PASS | PASS | PASS |
| Day 68 | 2026-02-20 | 2,161 | 2,161 | 2,161 | PASS | PASS | PASS |
| Day 69 | 2026-02-21 | 2,304 | 2,304 | 2,304 | PASS | PASS | PASS |
| Day 70 | 2026-02-22 | 2,206 | 2,206 | 2,206 | PASS | PASS | PASS |
| Day 71 | 2026-02-23 | 2,679 | 2,679 | 2,679 | PASS | PASS | PASS |
| Day 72 | 2026-02-24 | 2,585 | 2,585 | 2,585 | PASS | PASS | PASS |
| Day 73 | 2026-02-25 | 2,697 | 2,697 | 2,697 | PASS | PASS | PASS |
| Day 74 | 2026-02-26 | 3,178 | 3,178 | 3,178 | PASS | PASS | PASS |
| Day 75 | 2026-02-27 | 3,494 | 3,494 | 3,494 | PASS | PASS | PASS |
| Day 76 | 2026-02-28 | 3,990 | 3,990 | 3,990 | PASS | PASS | PASS |
| **合计** | | **527,164** | **527,164** | **527,164** | **76/76天PASS** | **76/76天PASS** | **76/76天PASS** |

#### 库存日志 (StockOnHandLog)

- **总行数 (全76天)**: Dev=387,730, Src=387,730, DB=387,730
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-12-15 | 5,079 | 5,079 | 5,079 | PASS | PASS | PASS |
| Day 02 | 2025-12-16 | 5,078 | 5,078 | 5,078 | PASS | PASS | PASS |
| Day 03 | 2025-12-17 | 5,081 | 5,081 | 5,081 | PASS | PASS | PASS |
| Day 04 | 2025-12-18 | 5,088 | 5,088 | 5,088 | PASS | PASS | PASS |
| Day 05 | 2025-12-19 | 5,091 | 5,091 | 5,091 | PASS | PASS | PASS |
| Day 06 | 2025-12-20 | 5,091 | 5,091 | 5,091 | PASS | PASS | PASS |
| Day 07 | 2025-12-21 | 5,096 | 5,096 | 5,096 | PASS | PASS | PASS |
| Day 08 | 2025-12-22 | 5,096 | 5,096 | 5,096 | PASS | PASS | PASS |
| Day 09 | 2025-12-23 | 5,095 | 5,095 | 5,095 | PASS | PASS | PASS |
| Day 10 | 2025-12-24 | 5,096 | 5,096 | 5,096 | PASS | PASS | PASS |
| Day 11 | 2025-12-25 | 5,096 | 5,096 | 5,096 | PASS | PASS | PASS |
| Day 12 | 2025-12-26 | 5,097 | 5,097 | 5,097 | PASS | PASS | PASS |
| Day 13 | 2025-12-27 | 5,096 | 5,096 | 5,096 | PASS | PASS | PASS |
| Day 14 | 2025-12-28 | 5,096 | 5,096 | 5,096 | PASS | PASS | PASS |
| Day 15 | 2025-12-29 | 5,094 | 5,094 | 5,094 | PASS | PASS | PASS |
| Day 16 | 2025-12-30 | 5,096 | 5,096 | 5,096 | PASS | PASS | PASS |
| Day 17 | 2025-12-31 | 5,097 | 5,097 | 5,097 | PASS | PASS | PASS |
| Day 18 | 2026-01-01 | 5,097 | 5,097 | 5,097 | PASS | PASS | PASS |
| Day 19 | 2026-01-02 | 5,097 | 5,097 | 5,097 | PASS | PASS | PASS |
| Day 20 | 2026-01-03 | 5,097 | 5,097 | 5,097 | PASS | PASS | PASS |
| Day 21 | 2026-01-04 | 5,097 | 5,097 | 5,097 | PASS | PASS | PASS |
| Day 22 | 2026-01-05 | 5,097 | 5,097 | 5,097 | PASS | PASS | PASS |
| Day 23 | 2026-01-06 | 5,098 | 5,098 | 5,098 | PASS | PASS | PASS |
| Day 24 | 2026-01-07 | 5,098 | 5,098 | 5,098 | PASS | PASS | PASS |
| Day 25 | 2026-01-08 | 5,098 | 5,098 | 5,098 | PASS | PASS | PASS |
| Day 26 | 2026-01-09 | 5,098 | 5,098 | 5,098 | PASS | PASS | PASS |
| Day 27 | 2026-01-10 | 5,098 | 5,098 | 5,098 | PASS | PASS | PASS |
| Day 28 | 2026-01-11 | 5,098 | 5,098 | 5,098 | PASS | PASS | PASS |
| Day 29 | 2026-01-12 | 5,098 | 5,098 | 5,098 | PASS | PASS | PASS |
| Day 30 | 2026-01-13 | 5,098 | 5,098 | 5,098 | PASS | PASS | PASS |
| Day 31 | 2026-01-14 | 5,098 | 5,098 | 5,098 | PASS | PASS | PASS |
| Day 32 | 2026-01-15 | 5,098 | 5,098 | 5,098 | PASS | PASS | PASS |
| Day 33 | 2026-01-16 | 5,099 | 5,099 | 5,099 | PASS | PASS | PASS |
| Day 34 | 2026-01-17 | 5,099 | 5,099 | 5,099 | PASS | PASS | PASS |
| Day 35 | 2026-01-18 | 5,099 | 5,099 | 5,099 | PASS | PASS | PASS |
| Day 36 | 2026-01-19 | 5,099 | 5,099 | 5,099 | PASS | PASS | PASS |
| Day 37 | 2026-01-20 | 5,099 | 5,099 | 5,099 | PASS | PASS | PASS |
| Day 38 | 2026-01-21 | 5,100 | 5,100 | 5,100 | PASS | PASS | PASS |
| Day 39 | 2026-01-22 | 5,102 | 5,102 | 5,102 | PASS | PASS | PASS |
| Day 40 | 2026-01-23 | 5,102 | 5,102 | 5,102 | PASS | PASS | PASS |
| Day 41 | 2026-01-24 | 5,102 | 5,102 | 5,102 | PASS | PASS | PASS |
| Day 42 | 2026-01-25 | 5,104 | 5,104 | 5,104 | PASS | PASS | PASS |
| Day 43 | 2026-01-26 | 5,105 | 5,105 | 5,105 | PASS | PASS | PASS |
| Day 44 | 2026-01-27 | 5,107 | 5,107 | 5,107 | PASS | PASS | PASS |
| Day 45 | 2026-01-28 | 5,108 | 5,108 | 5,108 | PASS | PASS | PASS |
| Day 46 | 2026-01-29 | 5,108 | 5,108 | 5,108 | PASS | PASS | PASS |
| Day 47 | 2026-01-30 | 5,108 | 5,108 | 5,108 | PASS | PASS | PASS |
| Day 48 | 2026-01-31 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 49 | 2026-02-01 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 50 | 2026-02-02 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 51 | 2026-02-03 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 52 | 2026-02-04 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 53 | 2026-02-05 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 54 | 2026-02-06 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 55 | 2026-02-07 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 56 | 2026-02-08 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 57 | 2026-02-09 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 58 | 2026-02-10 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 59 | 2026-02-11 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 60 | 2026-02-12 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 61 | 2026-02-13 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 62 | 2026-02-14 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 63 | 2026-02-15 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 64 | 2026-02-16 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 65 | 2026-02-17 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 66 | 2026-02-18 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 67 | 2026-02-19 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 68 | 2026-02-20 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 69 | 2026-02-21 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 70 | 2026-02-22 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 71 | 2026-02-23 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 72 | 2026-02-24 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 73 | 2026-02-25 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 74 | 2026-02-26 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 75 | 2026-02-27 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| Day 76 | 2026-02-28 | 5,109 | 5,109 | 5,109 | PASS | PASS | PASS |
| **合计** | | **387,730** | **387,730** | **387,730** | **76/76天PASS** | **76/76天PASS** | **76/76天PASS** |

#### 验证 (Validation)

- **总行数 (全76天)**: Dev=623,200, Src=623,200, DB=623,200
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-12-15 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 02 | 2025-12-16 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 03 | 2025-12-17 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 04 | 2025-12-18 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 05 | 2025-12-19 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 06 | 2025-12-20 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 07 | 2025-12-21 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 08 | 2025-12-22 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 09 | 2025-12-23 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 10 | 2025-12-24 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 11 | 2025-12-25 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 12 | 2025-12-26 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 13 | 2025-12-27 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 14 | 2025-12-28 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 15 | 2025-12-29 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 16 | 2025-12-30 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 17 | 2025-12-31 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 18 | 2026-01-01 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 19 | 2026-01-02 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 20 | 2026-01-03 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 21 | 2026-01-04 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 22 | 2026-01-05 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 23 | 2026-01-06 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 24 | 2026-01-07 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 25 | 2026-01-08 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 26 | 2026-01-09 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 27 | 2026-01-10 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 28 | 2026-01-11 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 29 | 2026-01-12 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 30 | 2026-01-13 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 31 | 2026-01-14 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 32 | 2026-01-15 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 33 | 2026-01-16 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 34 | 2026-01-17 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 35 | 2026-01-18 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 36 | 2026-01-19 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 37 | 2026-01-20 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 38 | 2026-01-21 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 39 | 2026-01-22 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 40 | 2026-01-23 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 41 | 2026-01-24 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 42 | 2026-01-25 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 43 | 2026-01-26 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 44 | 2026-01-27 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 45 | 2026-01-28 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 46 | 2026-01-29 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 47 | 2026-01-30 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 48 | 2026-01-31 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 49 | 2026-02-01 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 50 | 2026-02-02 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 51 | 2026-02-03 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 52 | 2026-02-04 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 53 | 2026-02-05 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 54 | 2026-02-06 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 55 | 2026-02-07 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 56 | 2026-02-08 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 57 | 2026-02-09 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 58 | 2026-02-10 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 59 | 2026-02-11 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 60 | 2026-02-12 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 61 | 2026-02-13 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 62 | 2026-02-14 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 63 | 2026-02-15 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 64 | 2026-02-16 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 65 | 2026-02-17 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 66 | 2026-02-18 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 67 | 2026-02-19 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 68 | 2026-02-20 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 69 | 2026-02-21 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 70 | 2026-02-22 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 71 | 2026-02-23 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 72 | 2026-02-24 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 73 | 2026-02-25 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 74 | 2026-02-26 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 75 | 2026-02-27 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| Day 76 | 2026-02-28 | 8,200 | 8,200 | 8,200 | PASS | PASS | PASS |
| **合计** | | **623,200** | **623,200** | **623,200** | **76/76天PASS** | **76/76天PASS** | **76/76天PASS** |

### 3.6 Module6 - 物流执行模块

#### 交付计划 (DeliveryPlan)

- **总行数 (全76天)**: Dev=71,145, Src=71,145, DB=71,145
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-12-15 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 02 | 2025-12-16 | 39 | 39 | 39 | PASS | PASS | PASS |
| Day 03 | 2025-12-17 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 04 | 2025-12-18 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 05 | 2025-12-19 | 512 | 512 | 512 | PASS | PASS | PASS |
| Day 06 | 2025-12-20 | 247 | 247 | 247 | PASS | PASS | PASS |
| Day 07 | 2025-12-21 | 409 | 409 | 409 | PASS | PASS | PASS |
| Day 08 | 2025-12-22 | 682 | 682 | 682 | PASS | PASS | PASS |
| Day 09 | 2025-12-23 | 611 | 611 | 611 | PASS | PASS | PASS |
| Day 10 | 2025-12-24 | 694 | 694 | 694 | PASS | PASS | PASS |
| Day 11 | 2025-12-25 | 917 | 917 | 917 | PASS | PASS | PASS |
| Day 12 | 2025-12-26 | 660 | 660 | 660 | PASS | PASS | PASS |
| Day 13 | 2025-12-27 | 661 | 661 | 661 | PASS | PASS | PASS |
| Day 14 | 2025-12-28 | 919 | 919 | 919 | PASS | PASS | PASS |
| Day 15 | 2025-12-29 | 834 | 834 | 834 | PASS | PASS | PASS |
| Day 16 | 2025-12-30 | 955 | 955 | 955 | PASS | PASS | PASS |
| Day 17 | 2025-12-31 | 958 | 958 | 958 | PASS | PASS | PASS |
| Day 18 | 2026-01-01 | 374 | 374 | 374 | PASS | PASS | PASS |
| Day 19 | 2026-01-02 | 977 | 977 | 977 | PASS | PASS | PASS |
| Day 20 | 2026-01-03 | 1,053 | 1,053 | 1,053 | PASS | PASS | PASS |
| Day 21 | 2026-01-04 | 1,064 | 1,064 | 1,064 | PASS | PASS | PASS |
| Day 22 | 2026-01-05 | 1,298 | 1,298 | 1,298 | PASS | PASS | PASS |
| Day 23 | 2026-01-06 | 1,231 | 1,231 | 1,231 | PASS | PASS | PASS |
| Day 24 | 2026-01-07 | 1,517 | 1,517 | 1,517 | PASS | PASS | PASS |
| Day 25 | 2026-01-08 | 1,418 | 1,418 | 1,418 | PASS | PASS | PASS |
| Day 26 | 2026-01-09 | 1,241 | 1,241 | 1,241 | PASS | PASS | PASS |
| Day 27 | 2026-01-10 | 992 | 992 | 992 | PASS | PASS | PASS |
| Day 28 | 2026-01-11 | 1,835 | 1,835 | 1,835 | PASS | PASS | PASS |
| Day 29 | 2026-01-12 | 1,782 | 1,782 | 1,782 | PASS | PASS | PASS |
| Day 30 | 2026-01-13 | 1,685 | 1,685 | 1,685 | PASS | PASS | PASS |
| Day 31 | 2026-01-14 | 1,883 | 1,883 | 1,883 | PASS | PASS | PASS |
| Day 32 | 2026-01-15 | 1,254 | 1,254 | 1,254 | PASS | PASS | PASS |
| Day 33 | 2026-01-16 | 1,082 | 1,082 | 1,082 | PASS | PASS | PASS |
| Day 34 | 2026-01-17 | 1,727 | 1,727 | 1,727 | PASS | PASS | PASS |
| Day 35 | 2026-01-18 | 935 | 935 | 935 | PASS | PASS | PASS |
| Day 36 | 2026-01-19 | 1,113 | 1,113 | 1,113 | PASS | PASS | PASS |
| Day 37 | 2026-01-20 | 1,377 | 1,377 | 1,377 | PASS | PASS | PASS |
| Day 38 | 2026-01-21 | 927 | 927 | 927 | PASS | PASS | PASS |
| Day 39 | 2026-01-22 | 1,060 | 1,060 | 1,060 | PASS | PASS | PASS |
| Day 40 | 2026-01-23 | 579 | 579 | 579 | PASS | PASS | PASS |
| Day 41 | 2026-01-24 | 1,084 | 1,084 | 1,084 | PASS | PASS | PASS |
| Day 42 | 2026-01-25 | 1,959 | 1,959 | 1,959 | PASS | PASS | PASS |
| Day 43 | 2026-01-26 | 954 | 954 | 954 | PASS | PASS | PASS |
| Day 44 | 2026-01-27 | 1,703 | 1,703 | 1,703 | PASS | PASS | PASS |
| Day 45 | 2026-01-28 | 1,252 | 1,252 | 1,252 | PASS | PASS | PASS |
| Day 46 | 2026-01-29 | 808 | 808 | 808 | PASS | PASS | PASS |
| Day 47 | 2026-01-30 | 925 | 925 | 925 | PASS | PASS | PASS |
| Day 48 | 2026-01-31 | 1,931 | 1,931 | 1,931 | PASS | PASS | PASS |
| Day 49 | 2026-02-01 | 1,119 | 1,119 | 1,119 | PASS | PASS | PASS |
| Day 50 | 2026-02-02 | 1,179 | 1,179 | 1,179 | PASS | PASS | PASS |
| Day 51 | 2026-02-03 | 1,011 | 1,011 | 1,011 | PASS | PASS | PASS |
| Day 52 | 2026-02-04 | 1,142 | 1,142 | 1,142 | PASS | PASS | PASS |
| Day 53 | 2026-02-05 | 838 | 838 | 838 | PASS | PASS | PASS |
| Day 54 | 2026-02-06 | 1,039 | 1,039 | 1,039 | PASS | PASS | PASS |
| Day 55 | 2026-02-07 | 616 | 616 | 616 | PASS | PASS | PASS |
| Day 56 | 2026-02-08 | 1,280 | 1,280 | 1,280 | PASS | PASS | PASS |
| Day 57 | 2026-02-09 | 1,027 | 1,027 | 1,027 | PASS | PASS | PASS |
| Day 58 | 2026-02-10 | 699 | 699 | 699 | PASS | PASS | PASS |
| Day 59 | 2026-02-11 | 635 | 635 | 635 | PASS | PASS | PASS |
| Day 60 | 2026-02-12 | 542 | 542 | 542 | PASS | PASS | PASS |
| Day 61 | 2026-02-13 | 74 | 74 | 74 | PASS | PASS | PASS |
| Day 62 | 2026-02-14 | 224 | 224 | 224 | PASS | PASS | PASS |
| Day 63 | 2026-02-15 | 561 | 561 | 561 | PASS | PASS | PASS |
| Day 64 | 2026-02-16 | 641 | 641 | 641 | PASS | PASS | PASS |
| Day 65 | 2026-02-17 | 824 | 824 | 824 | PASS | PASS | PASS |
| Day 66 | 2026-02-18 | 938 | 938 | 938 | PASS | PASS | PASS |
| Day 67 | 2026-02-19 | 454 | 454 | 454 | PASS | PASS | PASS |
| Day 68 | 2026-02-20 | 1,073 | 1,073 | 1,073 | PASS | PASS | PASS |
| Day 69 | 2026-02-21 | 995 | 995 | 995 | PASS | PASS | PASS |
| Day 70 | 2026-02-22 | 664 | 664 | 664 | PASS | PASS | PASS |
| Day 71 | 2026-02-23 | 429 | 429 | 429 | PASS | PASS | PASS |
| Day 72 | 2026-02-24 | 776 | 776 | 776 | PASS | PASS | PASS |
| Day 73 | 2026-02-25 | 666 | 666 | 666 | PASS | PASS | PASS |
| Day 74 | 2026-02-26 | 1,180 | 1,180 | 1,180 | PASS | PASS | PASS |
| Day 75 | 2026-02-27 | 754 | 754 | 754 | PASS | PASS | PASS |
| Day 76 | 2026-02-28 | 1,647 | 1,647 | 1,647 | PASS | PASS | PASS |
| **合计** | | **71,145** | **71,145** | **71,145** | **76/76天PASS** | **76/76天PASS** | **76/76天PASS** |

#### 车辆日志 (VehicleLog)

- **总行数 (全76天)**: Dev=823, Src=823, DB=823
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-12-15 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 02 | 2025-12-16 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 03 | 2025-12-17 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 04 | 2025-12-18 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 05 | 2025-12-19 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 06 | 2025-12-20 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 07 | 2025-12-21 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 08 | 2025-12-22 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 09 | 2025-12-23 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 10 | 2025-12-24 | 14 | 14 | 14 | PASS | PASS | PASS |
| Day 11 | 2025-12-25 | 16 | 16 | 16 | PASS | PASS | PASS |
| Day 12 | 2025-12-26 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 13 | 2025-12-27 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 14 | 2025-12-28 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 15 | 2025-12-29 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 16 | 2025-12-30 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 17 | 2025-12-31 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 18 | 2026-01-01 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 19 | 2026-01-02 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 20 | 2026-01-03 | 16 | 16 | 16 | PASS | PASS | PASS |
| Day 21 | 2026-01-04 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 22 | 2026-01-05 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 23 | 2026-01-06 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 24 | 2026-01-07 | 16 | 16 | 16 | PASS | PASS | PASS |
| Day 25 | 2026-01-08 | 14 | 14 | 14 | PASS | PASS | PASS |
| Day 26 | 2026-01-09 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 27 | 2026-01-10 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 28 | 2026-01-11 | 14 | 14 | 14 | PASS | PASS | PASS |
| Day 29 | 2026-01-12 | 16 | 16 | 16 | PASS | PASS | PASS |
| Day 30 | 2026-01-13 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 31 | 2026-01-14 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 32 | 2026-01-15 | 14 | 14 | 14 | PASS | PASS | PASS |
| Day 33 | 2026-01-16 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 34 | 2026-01-17 | 16 | 16 | 16 | PASS | PASS | PASS |
| Day 35 | 2026-01-18 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 36 | 2026-01-19 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 37 | 2026-01-20 | 14 | 14 | 14 | PASS | PASS | PASS |
| Day 38 | 2026-01-21 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 39 | 2026-01-22 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 40 | 2026-01-23 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 41 | 2026-01-24 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 42 | 2026-01-25 | 16 | 16 | 16 | PASS | PASS | PASS |
| Day 43 | 2026-01-26 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 44 | 2026-01-27 | 16 | 16 | 16 | PASS | PASS | PASS |
| Day 45 | 2026-01-28 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 46 | 2026-01-29 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 47 | 2026-01-30 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 48 | 2026-01-31 | 16 | 16 | 16 | PASS | PASS | PASS |
| Day 49 | 2026-02-01 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 50 | 2026-02-02 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 51 | 2026-02-03 | 14 | 14 | 14 | PASS | PASS | PASS |
| Day 52 | 2026-02-04 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 53 | 2026-02-05 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 54 | 2026-02-06 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 55 | 2026-02-07 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 56 | 2026-02-08 | 14 | 14 | 14 | PASS | PASS | PASS |
| Day 57 | 2026-02-09 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 58 | 2026-02-10 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 59 | 2026-02-11 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 60 | 2026-02-12 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 61 | 2026-02-13 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 62 | 2026-02-14 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 63 | 2026-02-15 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 64 | 2026-02-16 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 65 | 2026-02-17 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 66 | 2026-02-18 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 67 | 2026-02-19 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 68 | 2026-02-20 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 69 | 2026-02-21 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 70 | 2026-02-22 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 71 | 2026-02-23 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 72 | 2026-02-24 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 73 | 2026-02-25 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 74 | 2026-02-26 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 75 | 2026-02-27 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 76 | 2026-02-28 | 8 | 8 | 8 | PASS | PASS | PASS |
| **合计** | | **823** | **823** | **823** | **76/76天PASS** | **76/76天PASS** | **76/76天PASS** |

#### 卡车使用日志 (TruckUsageLog)

- **总行数 (全76天)**: Dev=803, Src=803, DB=803
- **全局一致性**: Dev vs Src PASS, Dev vs DB PASS, Src vs DB PASS

| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |
|------|------|---------|---------|--------|------------|-----------|-----------|
| Day 01 | 2025-12-15 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 02 | 2025-12-16 | 1 | 1 | 1 | PASS | PASS | PASS |
| Day 03 | 2025-12-17 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 04 | 2025-12-18 | 0 | 0 | 0 | PASS | PASS | PASS |
| Day 05 | 2025-12-19 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 06 | 2025-12-20 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 07 | 2025-12-21 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 08 | 2025-12-22 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 09 | 2025-12-23 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 10 | 2025-12-24 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 11 | 2025-12-25 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 12 | 2025-12-26 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 13 | 2025-12-27 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 14 | 2025-12-28 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 15 | 2025-12-29 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 16 | 2025-12-30 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 17 | 2025-12-31 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 18 | 2026-01-01 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 19 | 2026-01-02 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 20 | 2026-01-03 | 16 | 16 | 16 | PASS | PASS | PASS |
| Day 21 | 2026-01-04 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 22 | 2026-01-05 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 23 | 2026-01-06 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 24 | 2026-01-07 | 16 | 16 | 16 | PASS | PASS | PASS |
| Day 25 | 2026-01-08 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 26 | 2026-01-09 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 27 | 2026-01-10 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 28 | 2026-01-11 | 14 | 14 | 14 | PASS | PASS | PASS |
| Day 29 | 2026-01-12 | 16 | 16 | 16 | PASS | PASS | PASS |
| Day 30 | 2026-01-13 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 31 | 2026-01-14 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 32 | 2026-01-15 | 14 | 14 | 14 | PASS | PASS | PASS |
| Day 33 | 2026-01-16 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 34 | 2026-01-17 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 35 | 2026-01-18 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 36 | 2026-01-19 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 37 | 2026-01-20 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 38 | 2026-01-21 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 39 | 2026-01-22 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 40 | 2026-01-23 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 41 | 2026-01-24 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 42 | 2026-01-25 | 16 | 16 | 16 | PASS | PASS | PASS |
| Day 43 | 2026-01-26 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 44 | 2026-01-27 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 45 | 2026-01-28 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 46 | 2026-01-29 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 47 | 2026-01-30 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 48 | 2026-01-31 | 16 | 16 | 16 | PASS | PASS | PASS |
| Day 49 | 2026-02-01 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 50 | 2026-02-02 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 51 | 2026-02-03 | 14 | 14 | 14 | PASS | PASS | PASS |
| Day 52 | 2026-02-04 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 53 | 2026-02-05 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 54 | 2026-02-06 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 55 | 2026-02-07 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 56 | 2026-02-08 | 14 | 14 | 14 | PASS | PASS | PASS |
| Day 57 | 2026-02-09 | 15 | 15 | 15 | PASS | PASS | PASS |
| Day 58 | 2026-02-10 | 11 | 11 | 11 | PASS | PASS | PASS |
| Day 59 | 2026-02-11 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 60 | 2026-02-12 | 13 | 13 | 13 | PASS | PASS | PASS |
| Day 61 | 2026-02-13 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 62 | 2026-02-14 | 8 | 8 | 8 | PASS | PASS | PASS |
| Day 63 | 2026-02-15 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 64 | 2026-02-16 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 65 | 2026-02-17 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 66 | 2026-02-18 | 12 | 12 | 12 | PASS | PASS | PASS |
| Day 67 | 2026-02-19 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 68 | 2026-02-20 | 9 | 9 | 9 | PASS | PASS | PASS |
| Day 69 | 2026-02-21 | 10 | 10 | 10 | PASS | PASS | PASS |
| Day 70 | 2026-02-22 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 71 | 2026-02-23 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 72 | 2026-02-24 | 7 | 7 | 7 | PASS | PASS | PASS |
| Day 73 | 2026-02-25 | 5 | 5 | 5 | PASS | PASS | PASS |
| Day 74 | 2026-02-26 | 6 | 6 | 6 | PASS | PASS | PASS |
| Day 75 | 2026-02-27 | 4 | 4 | 4 | PASS | PASS | PASS |
| Day 76 | 2026-02-28 | 8 | 8 | 8 | PASS | PASS | PASS |
| **合计** | | **803** | **803** | **803** | **76/76天PASS** | **76/76天PASS** | **76/76天PASS** |

---

## 4. 每日库存一致性验证（全周期）

| 天数 | 日期 | Dev库存量 | Src库存量 | DB库存量 | 一致性 |
|------|------|-----------|-----------|----------|--------|
| Day 01 | 2025-12-15 | 460,603 | 460,603 | 460,603 | PASS |
| Day 02 | 2025-12-16 | 444,753 | 444,753 | 444,753 | PASS |
| Day 03 | 2025-12-17 | 420,554 | 420,554 | 420,554 | PASS |
| Day 04 | 2025-12-18 | 393,855 | 393,855 | 393,855 | PASS |
| Day 05 | 2025-12-19 | 367,221 | 367,221 | 367,221 | PASS |
| Day 06 | 2025-12-20 | 342,599 | 342,599 | 342,599 | PASS |
| Day 07 | 2025-12-21 | 318,850 | 318,850 | 318,850 | PASS |
| Day 08 | 2025-12-22 | 293,645 | 293,645 | 293,645 | PASS |
| Day 09 | 2025-12-23 | 272,189 | 272,189 | 272,189 | PASS |
| Day 10 | 2025-12-24 | 260,852 | 260,852 | 260,852 | PASS |
| Day 11 | 2025-12-25 | 256,017 | 256,017 | 256,017 | PASS |
| Day 12 | 2025-12-26 | 253,148 | 253,148 | 253,148 | PASS |
| Day 13 | 2025-12-27 | 249,265 | 249,265 | 249,265 | PASS |
| Day 14 | 2025-12-28 | 244,446 | 244,446 | 244,446 | PASS |
| Day 15 | 2025-12-29 | 252,867 | 252,867 | 252,867 | PASS |
| Day 16 | 2025-12-30 | 258,399 | 258,399 | 258,399 | PASS |
| Day 17 | 2025-12-31 | 265,209 | 265,209 | 265,209 | PASS |
| Day 18 | 2026-01-01 | 269,519 | 269,519 | 269,519 | PASS |
| Day 19 | 2026-01-02 | 273,441 | 273,441 | 273,441 | PASS |
| Day 20 | 2026-01-03 | 273,267 | 273,267 | 273,267 | PASS |
| Day 21 | 2026-01-04 | 278,174 | 278,174 | 278,174 | PASS |
| Day 22 | 2026-01-05 | 284,744 | 284,744 | 284,744 | PASS |
| Day 23 | 2026-01-06 | 288,225 | 288,225 | 288,225 | PASS |
| Day 24 | 2026-01-07 | 294,068 | 294,068 | 294,068 | PASS |
| Day 25 | 2026-01-08 | 299,739 | 299,739 | 299,739 | PASS |
| Day 26 | 2026-01-09 | 302,335 | 302,335 | 302,335 | PASS |
| Day 27 | 2026-01-10 | 307,091 | 307,091 | 307,091 | PASS |
| Day 28 | 2026-01-11 | 312,168 | 312,168 | 312,168 | PASS |
| Day 29 | 2026-01-12 | 323,752 | 323,752 | 323,752 | PASS |
| Day 30 | 2026-01-13 | 329,480 | 329,480 | 329,480 | PASS |
| Day 31 | 2026-01-14 | 332,086 | 332,086 | 332,086 | PASS |
| Day 32 | 2026-01-15 | 340,563 | 340,563 | 340,563 | PASS |
| Day 33 | 2026-01-16 | 348,201 | 348,201 | 348,201 | PASS |
| Day 34 | 2026-01-17 | 348,029 | 348,029 | 348,029 | PASS |
| Day 35 | 2026-01-18 | 358,046 | 358,046 | 358,046 | PASS |
| Day 36 | 2026-01-19 | 366,215 | 366,215 | 366,215 | PASS |
| Day 37 | 2026-01-20 | 373,775 | 373,775 | 373,775 | PASS |
| Day 38 | 2026-01-21 | 379,545 | 379,545 | 379,545 | PASS |
| Day 39 | 2026-01-22 | 378,268 | 378,268 | 378,268 | PASS |
| Day 40 | 2026-01-23 | 377,564 | 377,564 | 377,564 | PASS |
| Day 41 | 2026-01-24 | 384,656 | 384,656 | 384,656 | PASS |
| Day 42 | 2026-01-25 | 386,299 | 386,299 | 386,299 | PASS |
| Day 43 | 2026-01-26 | 393,294 | 393,294 | 393,294 | PASS |
| Day 44 | 2026-01-27 | 402,583 | 402,583 | 402,583 | PASS |
| Day 45 | 2026-01-28 | 400,485 | 400,485 | 400,485 | PASS |
| Day 46 | 2026-01-29 | 404,042 | 404,042 | 404,042 | PASS |
| Day 47 | 2026-01-30 | 400,844 | 400,844 | 400,844 | PASS |
| Day 48 | 2026-01-31 | 404,682 | 404,682 | 404,682 | PASS |
| Day 49 | 2026-02-01 | 411,075 | 411,075 | 411,075 | PASS |
| Day 50 | 2026-02-02 | 415,538 | 415,538 | 415,538 | PASS |
| Day 51 | 2026-02-03 | 413,942 | 413,942 | 413,942 | PASS |
| Day 52 | 2026-02-04 | 410,042 | 410,042 | 410,042 | PASS |
| Day 53 | 2026-02-05 | 412,911 | 412,911 | 412,911 | PASS |
| Day 54 | 2026-02-06 | 418,333 | 418,333 | 418,333 | PASS |
| Day 55 | 2026-02-07 | 420,383 | 420,383 | 420,383 | PASS |
| Day 56 | 2026-02-08 | 426,469 | 426,469 | 426,469 | PASS |
| Day 57 | 2026-02-09 | 418,210 | 418,210 | 418,210 | PASS |
| Day 58 | 2026-02-10 | 424,511 | 424,511 | 424,511 | PASS |
| Day 59 | 2026-02-11 | 439,734 | 439,734 | 439,734 | PASS |
| Day 60 | 2026-02-12 | 446,413 | 446,413 | 446,413 | PASS |
| Day 61 | 2026-02-13 | 448,554 | 448,554 | 448,554 | PASS |
| Day 62 | 2026-02-14 | 452,247 | 452,247 | 452,247 | PASS |
| Day 63 | 2026-02-15 | 462,525 | 462,525 | 462,525 | PASS |
| Day 64 | 2026-02-16 | 462,800 | 462,800 | 462,800 | PASS |
| Day 65 | 2026-02-17 | 477,249 | 477,249 | 477,249 | PASS |
| Day 66 | 2026-02-18 | 485,520 | 485,520 | 485,520 | PASS |
| Day 67 | 2026-02-19 | 488,837 | 488,837 | 488,837 | PASS |
| Day 68 | 2026-02-20 | 489,397 | 489,397 | 489,397 | PASS |
| Day 69 | 2026-02-21 | 506,813 | 506,813 | 506,813 | PASS |
| Day 70 | 2026-02-22 | 522,086 | 522,086 | 522,086 | PASS |
| Day 71 | 2026-02-23 | 527,922 | 527,922 | 527,922 | PASS |
| Day 72 | 2026-02-24 | 536,539 | 536,539 | 536,539 | PASS |
| Day 73 | 2026-02-25 | 533,542 | 533,542 | 533,542 | PASS |
| Day 74 | 2026-02-26 | 527,925 | 527,925 | 527,925 | PASS |
| Day 75 | 2026-02-27 | 527,623 | 527,623 | 527,623 | PASS |
| Day 76 | 2026-02-28 | 521,171 | 521,171 | 521,171 | PASS |

### 4.1 全周期一致性总结
| 验证项 | 结果 |
|--------|------|
| **仿真总天数** | 76天 |
| **库存不一致天数** | 0天 |
| **一致性比例** | 100% |
| **起始库存** | 460,603 |
| **结束库存** | 521,171 |

---

## 5. 测试结论

### 5.1 功能验证结论

**重构版本(Src/DB)与原始版本(Dev)业务数据100%一致**

验证覆盖（全76天）：
- Module1 / 订单日志 (OrderLog): 351,100行 (Dev vs Src PASS, Dev vs DB PASS)
- Module1 / 发货日志 (ShipmentLog): 157,433行 (Dev vs Src PASS, Dev vs DB PASS)
- Module1 / 削减日志 (CutLog): 157,433行 (Dev vs Src PASS, Dev vs DB PASS)
- Module1 / 每日汇总 (Summary): 76行 (Dev vs Src PASS, Dev vs DB PASS)
- Module3 / 净需求 (NetDemand): 93,693行 (Dev vs Src PASS, Dev vs DB PASS)
- Module4 / 生产计划 (ProductionPlan): 685行 (Dev vs Src PASS, Dev vs DB PASS)
- Module4 / 超容量报告 (CapacityExceed): 3,018行 (Dev vs Src PASS, Dev vs DB PASS)
- Module4 / 换型日志 (ChangeoverLog): 515行 (Dev vs Src PASS, Dev vs DB PASS)
- Module5 / 部署计划 (DeploymentPlan): 3,336,330行 (Dev vs Src PASS, Dev vs DB PASS)
- Module5 / 未满足日志 (UnfulfilledLog): 527,164行 (Dev vs Src PASS, Dev vs DB PASS)
- Module5 / 库存日志 (StockOnHandLog): 387,730行 (Dev vs Src PASS, Dev vs DB PASS)
- Module5 / 验证 (Validation): 623,200行 (Dev vs Src PASS, Dev vs DB PASS)
- Module6 / 交付计划 (DeliveryPlan): 71,145行 (Dev vs Src PASS, Dev vs DB PASS)
- Module6 / 车辆日志 (VehicleLog): 823行 (Dev vs Src PASS, Dev vs DB PASS)
- Module6 / 卡车使用日志 (TruckUsageLog): 803行 (Dev vs Src PASS, Dev vs DB PASS)

### 5.2 全周期性能总结
| 指标 | Dev | Src | DB | 最佳提升 |
|------|-----|-----|-----|----------|
| 总运行时间 | 1626.8分钟 | 247.6分钟 | 130.8分钟 | 92.0% |
| M1 订单生成平均耗时/天 | ~377.2秒 | ~38.0秒 | ~36.0秒 | 90.5% |
| M3 净需求计算平均耗时/天 | ~342.8秒 | ~44.9秒 | ~20.4秒 | 94.1% |
| M5 部署规划平均耗时/天 | ~544.6秒 | ~103.9秒 | ~39.0秒 | 92.8% |

---

**报告生成时间**: 2026-02-26  
**测试执行人**: chenxianyue002@chinasofti.com  
**版本**: v8.0 (完整版)
