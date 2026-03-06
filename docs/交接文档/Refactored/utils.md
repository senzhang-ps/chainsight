# utils — 工具层模块文档

本文档覆盖 `src/utils/` 目录下的所有 19 个文件。

---

## 目录

1. [模块概述](#模块概述)
2. [包导出（__init__.py）](#包导出__initpy)
3. [日志配置（logger_config.py）](#日志配置logger_configpy)
4. [配置验证（config_validator.py）](#配置验证config_validatorpy)
5. [时间管理（time_manager.py）](#时间管理time_managerpy)
6. [资源配置（resource_config.py）](#资源配置resource_configpy)
7. [优化配置（optimization_config.py）](#优化配置optimization_configpy)
8. [验证管理（validation_manager.py）](#验证管理validation_managerpy)
9. [内存数据存储（memory_data_store.py）](#内存数据存储memory_data_storepy)
10. [仿真缓存（simulation_cache.py）](#仿真缓存simulation_cachepy)
11. [库存平衡检查（inventory_balance_checker.py）](#库存平衡检查inventory_balance_checkerpy)
12. [性能工具（performance.py）](#性能工具performancepy)
13. [DuckDB SQL 包装（duckdb_sql_wrapper.py）](#duckdb-sql-包装duckdb_sql_wrapperpy)
14. [DuckDB 优化器（duckdb_optimizer.py）](#duckdb-优化器duckdb_optimizerpy)
15. [DuckDB 加速器（duckdb_accelerator.py）](#duckdb-加速器duckdb_acceleratorpy)
16. [CPU 配置（cpu_config.py）](#cpu-配置cpu_configpy)
17. [并行优化（parallel_optimizer.py）](#并行优化parallel_optimizerpy)
18. [进程池执行器（process_pool_executor.py）](#进程池执行器process_pool_executorpy)
19. [高性能执行器（high_perf_executor.py）](#高性能执行器high_perf_executorpy)
20. [多进程执行器（multiprocess_executor.py）](#多进程执行器multiprocess_executorpy)
21. [依赖关系总览](#依赖关系总览)

---

## 模块概述

`src/utils/` 是 ChainSight 的横切基础设施层，提供以下能力：

| 能力分类 | 文件 |
|---------|------|
| 日志 | `logger_config.py` |
| 配置验证 | `config_validator.py`, `validation_manager.py` |
| 时间管理 | `time_manager.py` |
| 资源与优化配置 | `resource_config.py`, `optimization_config.py`, `cpu_config.py` |
| 数据存储与缓存 | `memory_data_store.py`, `simulation_cache.py` |
| 库存验证 | `inventory_balance_checker.py` |
| 性能与向量化 | `performance.py` |
| DuckDB 集成 | `duckdb_sql_wrapper.py`, `duckdb_optimizer.py`, `duckdb_accelerator.py` |
| 并行执行 | `parallel_optimizer.py`, `process_pool_executor.py`, `high_perf_executor.py`, `multiprocess_executor.py` |

---

## 包导出（`__init__.py`）

**文件路径**：`src/utils/__init__.py`  
**行数**：28 行

### 公开导出列表

```python
from .config_validator import run_pre_simulation_validation
from .logger_config import setup_logging
from .validation_manager import ValidationManager
from .inventory_balance_checker import InventoryBalanceChecker
from .time_manager import SimulationTimeManager, initialize_time_manager
from .simulation_cache import (
    SimulationCache,
    initialize_simulation_cache,
    get_simulation_cache,
    clear_simulation_cache,
)
```

**`__all__`**：

| 名称 | 来源文件 |
|------|---------|
| `run_pre_simulation_validation` | `config_validator.py` |
| `setup_logging` | `logger_config.py` |
| `ValidationManager` | `validation_manager.py` |
| `InventoryBalanceChecker` | `inventory_balance_checker.py` |
| `SimulationTimeManager` | `time_manager.py` |
| `initialize_time_manager` | `time_manager.py` |
| `SimulationCache` | `simulation_cache.py` |
| `initialize_simulation_cache` | `simulation_cache.py` |
| `get_simulation_cache` | `simulation_cache.py` |
| `clear_simulation_cache` | `simulation_cache.py` |

---

## 日志配置（`logger_config.py`）

**文件路径**：`src/utils/logger_config.py`  
**行数**：218 行

### `DualLogger` 类

同时向文件和控制台输出日志的双输出日志器。

```python
class DualLogger:
    def __init__(self, name: str, log_file: str, level: int = logging.INFO)
```

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | Logger 名称 |
| `log_file` | `str` | 日志文件路径 |
| `level` | `int` | 日志级别，默认 `INFO` |

---

### `PrintRedirector` 类

将 `print()` 输出重定向到日志系统。

```python
class PrintRedirector:
    def __init__(self, logger: logging.Logger)
```

`write(message)` 方法将 `print` 的内容转发至 `logger.info()`。

---

### `setup_logging(log_dir, module_name, level)` → `logging.Logger`

创建并配置双输出 Logger（文件 + 控制台）。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `log_dir` | `str` | 日志目录 |
| `module_name` | `str` | 模块名称（用于日志文件命名） |
| `level` | `int` | 日志级别 |

**文件命名**：`{module_name}_{YYYYMMDD_HHMMSS}.log`

---

### `create_simple_file_logger(name, log_file, level)` → `logging.Logger`

创建仅写入文件的简单 Logger（不输出控制台）。

---

## 配置验证（`config_validator.py`）

**文件路径**：`src/utils/config_validator.py`  
**行数**：720 行

### `ConfigValidator` 类

对仿真配置进行仿真前完整验证，包含 7 个验证方法。

```python
class ConfigValidator:
    def __init__(self, config: dict)
```

#### 验证方法

| 方法 | 说明 |
|------|------|
| `validate_global()` | 验证全局配置（仿真日期、输出路径等） |
| `validate_module1()` | 验证 M1 需求规划配置 |
| `validate_module3()` | 验证 M3 MRP 规划配置 |
| `validate_module4()` | 验证 M4 生产规划配置 |
| `validate_module5()` | 验证 M5 部署规划配置 |
| `validate_module6()` | 验证 M6 物流执行配置 |
| `validate_cross_module()` | 跨模块一致性验证（如日期范围对齐） |

**已知设计问题**：M6 的 `TruckReleaseCon` 路线 OTD 检查代码位于 `return` 语句之后（死代码，永远不会执行）。

---

### `run_pre_simulation_validation(config)` → `ValidationResult`

仿真启动前的统一验证入口。依次执行所有 7 个验证方法，返回汇总验证结果。

**参数**：`config: dict`  
**返回值**：`ValidationResult`（含 errors、warnings、is_valid 字段）

---

## 时间管理（`time_manager.py`）

**文件路径**：`src/utils/time_manager.py`  
**行数**：278 行

### `SimulationTimeManager` 类

管理仿真时间序列，提供日期迭代、PTF 窗口计算、节假日处理等能力。

```python
class SimulationTimeManager:
    def __init__(
        self,
        sim_start: str,
        sim_end: str,
        calendar_config: Optional[dict] = None
    )
```

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `sim_start` | `str` | 仿真开始日期（`YYYY-MM-DD`） |
| `sim_end` | `str` | 仿真结束日期（`YYYY-MM-DD`） |
| `calendar_config` | `Optional[dict]` | 日历配置（节假日等） |

#### 13 个实例方法

| 方法 | 说明 |
|------|------|
| `get_simulation_dates()` | 返回仿真日期列表 |
| `get_current_date()` | 返回当前仿真日期 |
| `advance_date()` | 推进到下一个仿真日期 |
| `is_simulation_complete()` | 检查仿真是否结束 |
| `get_ptf_window(date, ptf_days)` | 获取给定日期的 PTF 窗口范围 |
| `get_working_days(start, end)` | 获取工作日列表 |
| `add_working_days(date, n)` | 在工作日基础上加 n 天 |
| `is_working_day(date)` | 判断是否为工作日 |
| `get_week_number(date)` | 获取周数 |
| `get_month_end(date)` | 获取月末日期 |
| `format_date(date)` | 格式化日期为字符串 |
| `parse_date(date_str)` | 解析日期字符串 |
| `days_between(date1, date2)` | 计算两日期间隔天数 |

---

### 全局实例管理函数

```python
def initialize_time_manager(sim_start: str, sim_end: str, calendar_config: Optional[dict] = None) -> SimulationTimeManager
def get_time_manager() -> SimulationTimeManager
def reset_time_manager() -> None
```

**使用模式**：

```python
# 初始化（仿真启动时调用一次）
initialize_time_manager("2024-01-01", "2024-03-31")

# 在任意模块中获取实例
tm = get_time_manager()
dates = tm.get_simulation_dates()
```

---

## 资源配置（`resource_config.py`）

**文件路径**：`src/utils/resource_config.py`  
**行数**：259 行

### 模块概述

动态计算系统资源配置，目标 CPU 利用率 **90%**。通过 `__getattr__` 支持常量名访问。

### 核心函数

| 函数 | 返回类型 | 说明 |
|------|---------|------|
| `get_optimal_threads()` | `int` | 最优线程数（`cpu_count * 0.9`，向上取整） |
| `get_optimal_memory()` | `int` | 最优内存限制（字节），默认系统可用内存的 75% |
| `get_duckdb_config()` | `dict` | DuckDB 运行时配置字典（threads、memory_limit 等） |
| `get_resource_config()` | `dict` | 完整资源配置字典 |

### `get_duckdb_config()` 返回结构

```python
{
    "threads": int,           # 线程数
    "memory_limit": str,      # 如 "8GB"
    "enable_progress_bar": bool,
    "temp_directory": str
}
```

### 常量访问

```python
from src.utils.resource_config import OPTIMAL_THREADS, MEMORY_LIMIT_GB
```

通过模块级 `__getattr__` 实现，动态计算后缓存。

---

## 优化配置（`optimization_config.py`）

**文件路径**：`src/utils/optimization_config.py`  
**行数**：304 行

### `_OptimizationConfig` Dataclass

全局优化策略开关。

```python
@dataclass
class _OptimizationConfig:
    USE_DUCKDB: bool = False          # 默认不使用 DuckDB（Pandas 在大多数场景更快）
    USE_PARALLEL: bool = True
    DUCKDB_MIN_ROWS: int = 10_000     # 触发 DuckDB 的最小行数阈值
    ...
```

**单例访问**：`OptimizationConfig`（全局单例实例）

**重要设计决定**：`USE_DUCKDB` 默认为 `False`。经测试，Pandas 在大多数仿真场景下比 DuckDB 快，仅在超大数据集（>10 万行）时 DuckDB 有优势。

### 便捷函数

```python
def use_duckdb_for_merge(df_size: int) -> bool
def use_duckdb_for_groupby(df_size: int) -> bool
def use_duckdb_for_filter(df_size: int) -> bool
```

每个函数根据 `df_size` 和对应阈值返回是否应使用 DuckDB。

---

### `PerformanceStats` 类

单例性能统计收集器，记录各操作的耗时分布。

```python
class PerformanceStats:
    def record(self, operation: str, duration: float) -> None
    def get_summary(self) -> dict
    def reset(self) -> None
```

---

## 验证管理（`validation_manager.py`）

**文件路径**：`src/utils/validation_manager.py`  
**行数**：335 行

### `ValidationManager` 类

管理仿真运行期间的验证消息，分三个级别。

```python
class ValidationManager:
    def __init__(self, module_name: str)
```

**三级消息列表**：`errors`（错误）、`warnings`（警告）、`infos`（信息）

#### 实例方法

| 方法 | 说明 |
|------|------|
| `add_error(message, location)` | 添加错误（会影响仿真继续执行） |
| `add_warning(message, location)` | 添加警告（记录但不停止） |
| `add_info(message, location)` | 添加信息性日志 |
| `has_errors()` → `bool` | 是否存在错误 |
| `write_report(output_path)` | 将所有消息写入 `validation_report.txt` |
| `validate_required_columns(df, required_cols, df_name)` | 检查 DataFrame 必要列是否存在 |
| `validate_positive_numbers(df, cols, df_name)` | 检查列值是否为正数 |
| `validate_date_ranges(df, date_col, start, end, df_name)` | 检查日期是否在范围内 |
| `validate_safe_date_conversion(series, col_name)` | 安全日期类型转换，记录转换失败 |

---

### `execute_with_validation(func)` 装饰器

包装函数，自动捕获异常并记录为 ValidationManager 错误。

```python
@execute_with_validation
def my_function(vm: ValidationManager, ...):
    ...
```

---

### 辅助函数

```python
def is_critical_error(error_message: str) -> bool
def get_default_result(result_type: str) -> Any
```

---

## 内存数据存储（`memory_data_store.py`）

**文件路径**：`src/utils/memory_data_store.py`  
**行数**：531 行

### `MemoryDataStore` 类

基于 **DuckDB 内存数据库**的模块间数据共享单例，线程安全。

```python
class MemoryDataStore:
    # 单例，不直接实例化
```

#### 生命周期方法

| 方法 | 说明 |
|------|------|
| `enable()` | 启用内存存储（创建 DuckDB 内存连接） |
| `disable()` | 禁用并关闭连接，释放内存 |
| `is_enabled()` → `bool` | 是否已启用 |

#### 数据操作方法

| 方法 | 参数 | 说明 |
|------|------|------|
| `write_module_output(module, sheet, date, df)` | `str, str, str, DataFrame` | 写入模块输出数据 |
| `read_module_output(module, sheet, date)` | `str, str, str` | 读取模块输出数据 |
| `query(sql)` | `str` | 执行任意 SQL 查询 |
| `export_to_parquet(table_name, path)` | `str, str` | 导出为 Parquet 文件 |

**表名格式**：`{module}_{sheet}_{date}`（如 `module4_ProductionPlan_20240315`）

#### 模块级便捷函数

```python
# 写入
def write_module4_output(sheet: str, date: str, df: pd.DataFrame) -> None
def write_module3_output(sheet: str, date: str, df: pd.DataFrame) -> None
def write_module5_output(sheet: str, date: str, df: pd.DataFrame) -> None
def write_module6_output(sheet: str, date: str, df: pd.DataFrame) -> None
def write_module1_output(sheet: str, date: str, df: pd.DataFrame) -> None

# 读取（常用快捷方式）
def read_module4_production_plan(date: str) -> pd.DataFrame
def read_module3_net_demand(date: str) -> pd.DataFrame
```

---

## 仿真缓存（`simulation_cache.py`）

**文件路径**：`src/utils/simulation_cache.py`  
**行数**：364 行

### `SimulationCache` 类

预构建 5 种高频查找索引，避免仿真循环中重复构建。

```python
class SimulationCache:
    def __init__(self, config: dict)
```

#### 预构建的 5 种缓存

| 缓存名 | 键结构 | 说明 |
|--------|--------|------|
| `network_cache` | `(sending, receiving)` → 路线配置 | 物流网络路线信息 |
| `ptf_lsk_cache` | `(material, location)` → PTF/LSK 值 | 计划时间围栏和安全库存键 |
| `lead_time_cache` | `(material, sending, receiving)` → `int` | 交货提前期（天数） |
| `deploy_config_cache` | `(material, location)` → 部署配置 | 部署规则配置 |
| `safety_stock_cache` | `(material, location)` → `float` | 安全库存量 |

#### 访问方法

```python
def get_network(sending: str, receiving: str) -> Optional[dict]
def get_ptf_lsk(material: str, location: str) -> Optional[tuple]
def get_lead_time(material: str, sending: str, receiving: str) -> Optional[int]
def get_deploy_config(material: str, location: str) -> Optional[dict]
def get_safety_stock(material: str, location: str) -> Optional[float]
```

#### 依赖

从 `deployment_planning.cache_utils` 调用 `assign_location_layers()` 构建层级结构。

---

### 全局实例管理函数

```python
def initialize_simulation_cache(config: dict) -> SimulationCache
def get_simulation_cache() -> SimulationCache
def clear_simulation_cache() -> None
```

---

## 库存平衡检查（`inventory_balance_checker.py`）

**文件路径**：`src/utils/inventory_balance_checker.py`  
**行数**：803 行

### `InventoryBalanceChecker` 类

验证仿真日度库存平衡公式的正确性。

```python
class InventoryBalanceChecker:
    def __init__(self, tolerance: float = 0.01)
```

### 库存平衡公式

```
期末库存 = 期初库存 + 生产GR + 交付GR - 发货量 - delivery_plan出库量
```

**负库存处理**：负库存重置为 0（与 Orchestrator 行为一致）。

#### 核心方法

| 方法 | 说明 |
|------|------|
| `check_daily_balance(date, inventory_state, module_outputs)` | 检查单日库存平衡 |
| `check_period_balance(start, end, daily_states)` | 检查期间累计平衡 |
| `generate_balance_report(output_path)` | 生成平衡验证报告 |
| `get_violations(tolerance)` | 获取超出容差的违规记录 |

**调试代码**：`_output_detailed_comparison()` 方法中的详细对比输出全部已注释掉。

---

## 性能工具（`performance.py`）

**文件路径**：`src/utils/performance.py`  
**行数**：293 行

### 向量化规范化函数

高性能批量规范化，比逐行应用快 10-100x。

```python
def normalize_material_vectorized(series: pd.Series) -> pd.Series
def normalize_location_vectorized(series: pd.Series) -> pd.Series
def normalize_identifiers_vectorized(df: pd.DataFrame) -> pd.DataFrame
```

**`normalize_identifiers_vectorized(df)`**：同时规范化 `material` 和 `location` 列，返回处理后的 DataFrame。

---

### `ConfigCache` 类

配置数据查询字典缓存，避免重复的 DataFrame 查找操作。

```python
class ConfigCache:
    def __init__(self)
    def build_index(self, df: pd.DataFrame, key_cols: List[str], value_cols: List[str]) -> None
    def lookup(self, key: tuple) -> Optional[dict]
    def clear(self) -> None
```

**全局实例**：`_global_config_cache`

---

### 高效 DataFrame 操作

```python
def efficient_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: List[str],
    how: str = "left"
) -> pd.DataFrame
```

根据数据量自动选择最优 merge 策略（Pandas / DuckDB）。

```python
def batch_groupby_apply(
    df: pd.DataFrame,
    groupby_cols: List[str],
    apply_func: Callable,
    n_workers: int = 1
) -> List[Any]
```

批量分组并行应用函数。

---

## DuckDB SQL 包装（`duckdb_sql_wrapper.py`）

**文件路径**：`src/utils/duckdb_sql_wrapper.py`  
**行数**：603 行

### `DuckDBSQL` 类

提供静态方法，将常见 Pandas 操作替换为 DuckDB SQL 实现。

#### 静态方法

```python
@staticmethod
def merge(left: pd.DataFrame, right: pd.DataFrame, on: List[str], how: str = "inner") -> pd.DataFrame

@staticmethod
def groupby_agg(df: pd.DataFrame, groupby: List[str], aggs: dict) -> pd.DataFrame
# aggs 格式: {"output_col": ("input_col", "agg_func")}

@staticmethod
def filter(df: pd.DataFrame, condition: str) -> pd.DataFrame
# condition 为 SQL WHERE 子句字符串

@staticmethod
def batch_filter_by_keys(df: pd.DataFrame, key_cols: List[str], keys: List[tuple]) -> pd.DataFrame

@staticmethod
def sort(df: pd.DataFrame, by: List[str], ascending: bool = True) -> pd.DataFrame
```

### 自动引擎选择

所有方法根据 `OptimizationConfig` 和数据量自动选择 Pandas 或 DuckDB 引擎。

### 便捷函数

```python
def smart_merge(left, right, on, how="inner") -> pd.DataFrame
def smart_groupby_agg(df, groupby, aggs) -> pd.DataFrame
def smart_filter(df, condition) -> pd.DataFrame
```

---

## DuckDB 优化器（`duckdb_optimizer.py`）

**文件路径**：`src/utils/duckdb_optimizer.py`  
**行数**：489 行

### `DuckDBOptimizer` 类

专为 **M5（部署规划）** 和 **M3（MRP 规划）** 优化的 DuckDB 单例。

```python
class DuckDBOptimizer:
    # 单例，通过 get_instance() 访问
```

#### 3 个批量索引构建方法

| 方法 | 说明 | 用途 |
|------|------|------|
| `batch_build_sdl_index(df)` | 构建 SDL（Supply/Demand/Lead-time）索引 | M3/M5 |
| `batch_build_ss_index(df)` | 构建安全库存索引 | M3/M5 |
| `batch_build_order_index(df)` | 构建订单索引 | M3 |

#### 批量净需求计算

```python
def batch_calculate_net_demand(
    demand_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    ss_df: pd.DataFrame
) -> pd.DataFrame
```

使用 DuckDB SQL 一次性计算所有物料地点组合的净需求，替代逐行循环。

#### 一键构建

```python
def build_indexes_with_duckdb(sdl_df, ss_df, order_df) -> dict
```

同时构建全部 3 个索引，返回索引字典。

**线程配置**：使用 `CPU_COUNT` 配置 DuckDB 线程数。

---

## DuckDB 加速器（`duckdb_accelerator.py`）

**文件路径**：`src/utils/duckdb_accelerator.py`  
**行数**：275 行

### `DuckDBAccelerator` 类

通用 DuckDB 加速器单例，提供高频 DataFrame 操作的 DuckDB 优化版本。

#### 5 个方法

```python
def batch_filter_by_material_location(
    df: pd.DataFrame,
    material: str,
    location: str
) -> pd.DataFrame
```
按物料和地点过滤。

```python
def aggregate_by_groups(
    df: pd.DataFrame,
    group_cols: List[str],
    agg_specs: dict
) -> pd.DataFrame
```
分组聚合。

```python
def filter_date_range(
    df: pd.DataFrame,
    date_col: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame
```
日期范围过滤。

```python
def build_material_location_index(df: pd.DataFrame) -> dict
```
构建物料-地点复合键索引。

```python
def batch_filter_all_pairs(
    df: pd.DataFrame,
    pairs: List[Tuple[str, str]]
) -> pd.DataFrame
```
**单次 JOIN 获取所有（物料, 地点）组合的结果**，比逐对过滤快 10-100x。

---

## CPU 配置（`cpu_config.py`）

**文件路径**：`src/utils/cpu_config.py`  
**行数**：99 行

### 模块概述

动态 CPU 配置，目标利用率 **90%**。通过 `__getattr__` 支持常量名直接访问。

### 核心函数

```python
def get_cpu_count() -> int
# 返回总 CPU 核心数（os.cpu_count()）

def get_max_workers() -> int
# 返回推荐最大工作进程数（ceil(cpu_count * 0.9)，最小 1）

def get_optimal_workers(task_count: int) -> int
# 根据任务数量返回最优工作数（min(task_count, max_workers)）
```

### 常量访问

```python
from src.utils.cpu_config import CPU_COUNT, MAX_WORKERS
```

通过模块级 `__getattr__` 实现，动态计算后缓存：

| 常量 | 说明 |
|------|------|
| `CPU_COUNT` | 总 CPU 核心数 |
| `MAX_WORKERS` | 推荐最大工作数（≈ CPU_COUNT * 0.9） |

---

## 并行优化（`parallel_optimizer.py`）

**文件路径**：`src/utils/parallel_optimizer.py`  
**行数**：307 行

### 核心函数

#### `parallel_batch_process(func, items, use_threads, n_workers)` → `List[Any]`

使用线程池或进程池批量并行处理。

| 参数 | 类型 | 说明 |
|------|------|------|
| `func` | `Callable` | 处理函数 |
| `items` | `List[Any]` | 待处理项列表 |
| `use_threads` | `bool` | `True` 使用线程，`False` 使用进程 |
| `n_workers` | `Optional[int]` | 工作数，默认 `MAX_WORKERS` |

#### `batch_dataframe_filter(df, key_cols, filter_keys)` → `Dict[tuple, pd.DataFrame]`

**一次性分组过滤**，按复合键分组后直接映射，比逐个过滤快 **10-100x**。

| 参数 | 类型 | 说明 |
|------|------|------|
| `df` | `pd.DataFrame` | 源 DataFrame |
| `key_cols` | `List[str]` | 复合键列名 |
| `filter_keys` | `List[tuple]` | 目标键值元组列表 |

**返回**：`{key_tuple: sub_dataframe}` 字典

#### `vectorized_demand_aggregation(demand_df, group_cols, agg_col)` → `pd.DataFrame`

向量化需求聚合，使用 NumPy 加速 groupby 操作。

---

### `ParallelBatchProcessor` 类

封装并行批处理器，支持任务队列管理。

```python
class ParallelBatchProcessor:
    def __init__(self, n_workers: int = MAX_WORKERS, use_processes: bool = False)
    def process(self, func: Callable, items: List[Any]) -> List[Any]
    def process_dataframe_groups(self, df, groupby_cols, apply_func) -> List[Any]
```

---

### 预计算函数

```python
def precompute_all_indices(config_data: dict) -> dict
```
预构建所有常用查找索引，减少仿真循环中的重复构建。

```python
def warmup_cache(config_data: dict) -> None
```
预热缓存：构建索引 + 预加载常用数据。

---

## 进程池执行器（`process_pool_executor.py`）

**文件路径**：`src/utils/process_pool_executor.py`  
**行数**：267 行

### 模块概述

使用 `ProcessPoolExecutor` 突破 GIL 限制，针对 **MRP（M3）** 和**部署规划（M5）**的批量计算优化。Windows 兼容（使用 `spawn` 方式创建子进程）。

### `_worker_init()` → `None`

工作进程初始化函数。在 Windows 下将子进程优先级设置为 `ABOVE_NORMAL_PRIORITY_CLASS`（`0x00008000`）。

### 核心函数

#### `batch_process_parallel(func, items, n_workers, batch_size)` → `List[Any]`

批量并行处理，将任务分批提交到进程池。

| 参数 | 类型 | 说明 |
|------|------|------|
| `func` | `Callable` | 处理函数（必须可序列化） |
| `items` | `List[Any]` | 待处理项列表 |
| `n_workers` | `Optional[int]` | 工作进程数 |
| `batch_size` | `int` | 批处理大小，默认 50 |

**失败处理**：单个任务失败时结果置 `None`，全体并行失败时自动回退到串行执行。

#### `parallel_dataframe_groupby_apply(df, groupby_cols, apply_func, n_workers)` → `List[Any]`

对 DataFrame 分组并行应用函数。`apply_func` 签名为 `(group_key, group_df) -> result`。

#### `parallel_layer_process(layer_items, process_func, context_data, n_workers)` → `List[Any]`

专为 M3/M5 层级处理优化。`layer_items` 为 `(material, location)` 元组列表，上下文数据自动序列化/反序列化。

---

### 序列化辅助函数

```python
def _serialize_context(ctx: dict) -> dict
# DataFrame → dict of records

def _deserialize_context(ctx: dict) -> dict
# dict of records → DataFrame
```

---

### `ProcessPoolManager` 类

进程池管理器单例，用于复用进程池，避免重复创建开销。

```python
class ProcessPoolManager:
    @classmethod
    def get_instance(cls, max_workers: Optional[int] = None) -> 'ProcessPoolManager'
    def start_pool(self) -> None
    def shutdown_pool(self) -> None
    def submit(self, func: Callable, *args, **kwargs) -> Future
    def map(self, func: Callable, items: List[Any]) -> List[Any]
```

### 便捷函数

```python
def get_pool_manager() -> ProcessPoolManager
def get_optimal_workers() -> int
def print_cpu_info() -> None
```

---

## 高性能执行器（`high_perf_executor.py`）

**文件路径**：`src/utils/high_perf_executor.py`  
**行数**：336 行  
**版本**：v3.1

### 模块概述

进程/线程混合并行执行器，支持 IO 密集型（线程池）和 CPU 密集型（进程池）两种模式。目标 90% CPU 利用率。

### `HighPerformanceExecutor` 类

```python
class HighPerformanceExecutor:
    def __init__(self, n_workers: Optional[int] = None)
    
    @classmethod
    def get_instance(cls, n_workers: Optional[int] = None) -> 'HighPerformanceExecutor'
```

**兼容变量**：`OPTIMAL_WORKERS = MAX_WORKERS`（向后兼容别名）

#### 池管理

| 方法 | 说明 |
|------|------|
| `get_process_pool()` | 获取或创建进程池（使用 `spawn` 上下文，Windows 兼容） |
| `get_thread_pool()` | 获取或创建线程池（线程数为 `n_workers * 4`，IO 密集型） |
| `shutdown()` | 关闭所有池 |

#### 执行方法

```python
def parallel_map_threads(func, items, timeout=None) -> List[Any]
# 使用线程池并行映射（适合 IO 密集型，如文件读写、网络请求）

def parallel_map_processes(func, items, timeout=None) -> List[Any]
# 使用进程池并行映射（适合 CPU 密集型，如数值计算）
# 注：items < 4 时自动退化为串行

def batch_process(func, items, batch_size=100, use_processes=False) -> List[Any]
# 分批并行处理，减少内存压力

def chunked_dataframe_apply(df, func, n_chunks=None) -> pd.DataFrame
# 将 DataFrame 分块并行处理，合并结果
```

---

### 模块级函数

```python
def get_executor() -> HighPerformanceExecutor
# 获取全局执行器单例

def parallel_process_layers(layer_data, process_func, ctx) -> List[Any]
# 多层级处理：层间串行（存在依赖），层内并行
# layer_data: {layer_num: [(material, location), ...]}

def optimize_cpu_bound_loop(items, func, use_multiprocessing=False) -> List[Any]
# 优化 CPU 密集型循环

def get_optimal_workers() -> int
def print_executor_stats() -> None
```

---

## 多进程执行器（`multiprocess_executor.py`）

**文件路径**：`src/utils/multiprocess_executor.py`  
**行数**：178 行

### 模块概述

最简化的多进程执行器，侧重简单易用。在模块加载时自动设置 `spawn` 启动方式（Windows 兼容）：

```python
if os.name == 'nt':
    mp.set_start_method('spawn', force=True)
```

### `MultiProcessExecutor` 类

```python
class MultiProcessExecutor:
    def __init__(self, max_workers: Optional[int] = None)
    
    @classmethod
    def get_instance(cls, max_workers: Optional[int] = None) -> 'MultiProcessExecutor'
    
    def map_parallel(self, func: Callable, items: List[Any], chunk_size: int = 10) -> List[Any]
```

`map_parallel()` 特点：
- 保持原始顺序（通过 `(idx, result)` 元组排序）
- 失败时自动回退到串行执行

---

### 模块级函数

#### `parallel_dataframe_apply(df, func, axis=1, n_workers=None)` → `pd.DataFrame`

并行应用函数到 DataFrame 的每行（`axis=1`）或每列（`axis=0`）。

内部实现：
1. 调用 `_split_dataframe(df, n_partitions)` 分割 DataFrame
2. 并行执行 `_apply_to_chunk(chunk, func, axis)`
3. `pd.concat()` 合并结果

#### `get_optimal_workers(task_count, max_workers=None)` → `int`

根据任务数量和系统配置返回最优工作进程数：`min(task_count, max_workers, CPU_COUNT)`

#### `get_executor()` → `MultiProcessExecutor`

获取全局多进程执行器单例。

---

## 依赖关系总览

```
src/utils/
│
├── cpu_config.py               ← 基础配置（无依赖）
├── resource_config.py          ← 基础配置（依赖 cpu_config）
├── optimization_config.py      ← 基础配置（无外部依赖）
│
├── logger_config.py            ← 基础工具（无项目依赖）
│
├── validation_manager.py       ← 依赖 logger_config
├── config_validator.py         ← 依赖 validation_manager, logger_config
│
├── time_manager.py             ← 依赖 logger_config
│
├── performance.py              ← 依赖 optimization_config, logger_config
│
├── duckdb_sql_wrapper.py       ← 依赖 optimization_config, resource_config
├── duckdb_optimizer.py         ← 依赖 cpu_config, duckdb_sql_wrapper
├── duckdb_accelerator.py       ← 依赖 duckdb_optimizer
│
├── parallel_optimizer.py       ← 依赖 cpu_config
├── process_pool_executor.py    ← 依赖 cpu_config
├── high_perf_executor.py       ← 依赖 cpu_config
├── multiprocess_executor.py    ← 依赖 cpu_config
│
├── memory_data_store.py        ← 依赖 logger_config, duckdb_sql_wrapper
├── simulation_cache.py         ← 依赖 logger_config, deployment_planning.cache_utils
│
└── inventory_balance_checker.py← 依赖 logger_config, validation_manager
```

### 三类并行执行器对比

| 文件 | 特点 | 适用场景 |
|------|------|---------|
| `parallel_optimizer.py` | 线程/进程自选，含向量化工具 | 通用并行，M3/M5 索引构建 |
| `process_pool_executor.py` | 进程池，含上下文序列化 | M3/M5 层级节点批量处理 |
| `high_perf_executor.py` | 进程+线程混合，v3.1 | IO/CPU 混合场景，层级处理 |
| `multiprocess_executor.py` | 最简多进程 | 简单 CPU 密集型任务 |
