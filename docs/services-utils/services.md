# `src/services` — 服务层模块文档

## 文档信息

| 项 | 内容 |
|---|---|
| 文档版本 | v1.1 |
| 最后更新 | 2026-04-10 |
| 编写人 | 陈显跃 |
| 适用范围 | `src/services/` 目录（共 3 个文件：`__init__.py`、`performance_profiler.py`、`summary_report_generator.py`） |
| 目标读者 | 算法工程师、测试工程师、运维人员 |

本文档覆盖 `src/services/` 目录下的所有文件：`performance_profiler.py` 和 `summary_report_generator.py`。

---

## 目录

1. [模块概述](#模块概述)
2. [performance_profiler.py — 性能分析器](#performance_profilerpy--性能分析器)
3. [summary_report_generator.py — 摘要报告生成器](#summary_report_generatorpy--摘要报告生成器)
4. [数据流](#数据流)
5. [依赖关系](#依赖关系)

---

## 模块概述

`src/services/` 目录包含仿真系统的横切服务，与业务逻辑模块解耦：

| 文件 | 职责 |
|------|------|
| `performance_profiler.py` | 运行时性能分析（cProfile 包装，生成文本报告） |
| `summary_report_generator.py` | 跨日期汇总报告生成（从各模块输出 Excel/CSV 聚合） |

---

## performance_profiler.py — 性能分析器

**文件路径**：`src/services/performance_profiler.py`  
**行数**：137 行

### 模块概述

封装 `cProfile` 提供两种使用方式：
1. **上下文管理器**（`PerformanceProfiler` 类）：对代码块进行整体分析
2. **装饰器**（`profile_function`）：对单个函数进行分析

---

### `PerformanceProfiler` 类

```python
class PerformanceProfiler:
    def __init__(self, module_name: str, output_dir: str = "logs/profiles")
```

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `module_name` | `str` | 模块名称，用于生成报告文件名 |
| `output_dir` | `str` | 报告输出目录，默认 `"logs/profiles"` |

**用法**：

```python
with PerformanceProfiler("module6", output_dir="logs/profiles"):
    result = run_physical_flow_module(config)
```

#### `__enter__(self)` → `PerformanceProfiler`

启动 `cProfile.Profile()` 进行性能采样。

#### `__exit__(self, exc_type, exc_val, exc_tb)` → `None`

停止采样，调用 `_save_report()` 写入报告文件。即使代码块抛出异常，报告仍会被保存。

#### `_save_report(self)` → `None`

生成并保存性能报告文本文件。

**报告内容**：
- 按 `cumulative`（累计时间）排序：前 30 个函数
- 按 `calls`（调用次数）排序：前 30 个函数
- 按 `time`（自身时间）排序：前 30 个函数
- 同时通过日志输出 **Top 5 耗时函数**

**文件命名格式**：

```
performance_profile_{module_name}_{YYYYMMDD_HHMMSS}.txt
```

**示例**：

```
performance_profile_module6_20240315_143022.txt
```

---

### `profile_function` 装饰器

```python
def profile_function(module_name: str, output_dir: str = "logs/profiles"):
    def decorator(func):
        ...
    return decorator
```

**功能**：对单个函数进行性能分析，函数调用结束后自动写入报告，并通过日志输出前 10 个函数统计。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `module_name` | `str` | 模块名称标识 |
| `output_dir` | `str` | 报告输出目录 |

**用法**：

```python
@profile_function("module4", output_dir="logs/profiles")
def run_production_planning(config):
    ...
```

---

### 报告文件结构示例

```
================================================================================
Performance Profile: module6
Generated: 2024-03-15 14:30:22
================================================================================

--- Sort by: cumulative (Top 30) ---
         12345 function calls in 3.456 seconds

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      100    0.234    0.002    2.345    0.023 module6.py:887(_process_truck_type)
      ...

--- Sort by: calls (Top 30) ---
...

--- Sort by: time (Top 30) ---
...
```

---

## summary_report_generator.py — 摘要报告生成器

**文件路径**：`src/services/summary_report_generator.py`  
**行数**：833 行

### 模块概述

从各模块每日输出的 Excel 文件和 Orchestrator 保存的 CSV 文件中，跨日期聚合生成 8 份汇总报告。

---

### `SummaryReportGenerator` 类

```python
class SummaryReportGenerator:
    def __init__(
        self,
        output_base_dir: str,
        sim_start_date: str,
        sim_end_date: str
    )
```

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `output_base_dir` | `str` | 仿真输出根目录 |
| `sim_start_date` | `str` | 仿真开始日期（`YYYY-MM-DD`） |
| `sim_end_date` | `str` | 仿真结束日期（`YYYY-MM-DD`） |

---

### 主入口方法

#### `generate_all_reports(self)` → `None`

依次调用所有 8 个报告生成方法，生成全套汇总报告。

```python
generator = SummaryReportGenerator(
    output_base_dir="output/",
    sim_start_date="2024-01-01",
    sim_end_date="2024-03-31"
)
generator.generate_all_reports()
```

---

### 8 个报告方法详解

#### 1. `_generate_order_shipment_cut_report()`

**数据来源**：每日 `module1_output_{YYYYMMDD}.xlsx`（Sheet：`OrderLog`、`ShipmentLog`、`CutLog`）

**输出文件**：`summary_order_shipment_cut.xlsx`

**功能**：汇总订单生成、发货执行、削减记录，用于需求满足率分析。

**关键列**：来自 OrderLog 的订单数量、ShipmentLog 的实际发货量、CutLog 的削减量和原因。

---

#### 2. `_generate_delivery_report()`

**数据来源**：每日 `module6_output_{YYYYMMDD}.xlsx`（Sheet：`FullDeliveryPlan`）

**输出文件**：`summary_delivery.xlsx`

**功能**：汇总物流执行的实际交付计划，包含发运日期、到达日期、数量、路线等。

---

#### 3. `_generate_truck_usage_report()`

**数据来源**：每日 `module6_output_{YYYYMMDD}.xlsx`（Sheet：`TruckUsageLog`）

**输出文件**：`summary_truck_usage.xlsx`

**功能**：汇总车辆使用统计，用于运输资源利用率分析。

---

#### 4. `_generate_capacity_exceed_report()`

**数据来源**：每日 `module4_output_{YYYYMMDD}.xlsx`（Sheet：`CapacityExceed`）

**输出文件**：`summary_capacity_exceed.xlsx`

**功能**：汇总产能超出事件，标识生产瓶颈日期和产线。

---

#### 5. `_generate_changeover_report()`

**数据来源**：每日 `module4_output_{YYYYMMDD}.xlsx`（Sheet：`ChangeoverLog`）

**输出文件**：`summary_changeover.xlsx`

**功能**：汇总换线日志。使用 `changeover_end_date` 列进行日期范围过滤（仅保留仿真区间内的换线记录）。

**注意**：过滤基于 `changeover_end_date`，而非 `changeover_start_date`。

---

#### 6. `_generate_deployment_report()`

**数据来源**：每日 `module5_output_{YYYYMMDD}.xlsx`（Sheet：`DeploymentPlan`）

**输出文件**：`summary_deployment.xlsx`（或 `.csv`）

**功能**：汇总部署规划结果。

**特殊处理**：当合并后的数据量超过 **1,048,575 行**（Excel 行数上限）时，自动改用 CSV 格式输出。

---

#### 7. `_generate_production_report()`

**数据来源**：每日 `module4_output_{YYYYMMDD}.xlsx`（Sheet：`ProductionPlan`）

**输出文件**：`summary_production.xlsx`

**功能**：汇总生产计划，按 `available_date` 过滤仿真区间内的记录。

---

#### 8. `_generate_historical_inventory_report()`

**数据来源**（7 种文件 + 1 种模块输出）：

| 文件类型 | 文件名模式 | 来源 |
|---------|---------|------|
| 非限制库存 | `unrestricted_inventory_{YYYYMMDD}.csv` | Orchestrator 目录 |
| 在途库存 | `planning_intransit_{YYYYMMDD}.csv` | Orchestrator 目录 |
| 生产收货 | `production_gr_{YYYYMMDD}.csv` | Orchestrator 目录 |
| 交付收货 | `delivery_gr_{YYYYMMDD}.csv` | Orchestrator 目录 |
| 发货日志 | `shipment_log_{YYYYMMDD}.csv` | Orchestrator 目录 |
| 交付发货日志 | `delivery_shipment_log_{YYYYMMDD}.csv` | Orchestrator 目录 |
| Module1 输出 | `module1_output_{YYYYMMDD}.xlsx` | Module1 目录 |

**输出文件**：`summary_historical_inventory.xlsx`

**输出列（12 列）**：

| 列名 | 类型 | 说明 |
|------|------|------|
| `date` | `str` | 日期 |
| `material` | `str` | 物料编号 |
| `location` | `str` | 地点编号 |
| `ending_inventory` | `float` | 期末库存 |
| `in_transit` | `float` | 在途库存 |
| `production_gr` | `float` | 生产收货量 |
| `delivery_gr` | `float` | 交付收货量 |
| `order` | `float` | 订单量 |
| `shipment` | `float` | 发货量 |
| `delivery_ship` | `float` | 交付发货量 |
| `supply_demand` | `float` | 供需差 |
| `safety_stock` | `float` | 安全库存 |

---

### 辅助方法

#### `_extract_date_from_filename(filename)` → `Optional[str]`

从文件名中提取 8 位日期字符串。

**实现**：使用正则表达式 `\d{8}` 匹配。

**返回**：`YYYYMMDD` 格式字符串，未找到则返回 `None`。

**示例**：

```python
_extract_date_from_filename("module6_output_20240315.xlsx")  
# → "20240315"
```

---

### 通用数据处理规则

所有 8 个报告方法遵循以下统一规则：

1. **日期过滤**：过滤掉超出仿真结束日期的数据（防止遗留数据污染）
2. **空 DataFrame 处理**：在 `pd.concat()` 前过滤空 DataFrame，避免 `FutureWarning`
3. **空数据处理**：当合并结果为空时，创建带完整列名的空文件（而非跳过），确保下游处理不报错
4. **大数据集处理**：`_generate_deployment_report()` 超过 Excel 行数上限时自动切换 CSV 输出

---

## 数据流

```
每日模块输出（Excel/CSV）
        │
        ▼
SummaryReportGenerator.generate_all_reports()
        │
        ├─ _generate_order_shipment_cut_report()  ← module1_output_{date}.xlsx
        ├─ _generate_delivery_report()            ← module6_output_{date}.xlsx
        ├─ _generate_truck_usage_report()         ← module6_output_{date}.xlsx
        ├─ _generate_capacity_exceed_report()     ← module4_output_{date}.xlsx
        ├─ _generate_changeover_report()          ← module4_output_{date}.xlsx
        ├─ _generate_deployment_report()          ← module5_output_{date}.xlsx
        ├─ _generate_production_report()          ← module4_output_{date}.xlsx
        └─ _generate_historical_inventory_report()← orchestrator CSVs + module1 xlsx
                │
                ▼
        summary_*.xlsx / summary_*.csv
```

---

## 依赖关系

```
src/services/
├── performance_profiler.py
│   ├── cProfile (Python 标准库)
│   ├── pstats (Python 标准库)
│   └── src/utils/logger_config.py
│
└── summary_report_generator.py
    ├── pandas
    ├── openpyxl
    ├── os, re, glob (Python 标准库)
    └── src/utils/logger_config.py
```

---

## `src/services/__init__.py` 导出内容

```python
# 公开导出
from .performance_profiler import PerformanceProfiler, profile_function
from .summary_report_generator import SummaryReportGenerator
```
