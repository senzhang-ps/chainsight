# DuckDB 优化使用指南

## 概述

ChainSight 支持使用 DuckDB 作为可选的数据处理引擎。本文档说明何时应该启用/禁用 DuckDB，以及如何配置。

## 性能测试结论

经过实际性能测试（`tools/benchmark_duckdb_vs_pandas.py`），我们发现：

| 操作 | 数据量 | Pandas | DuckDB | 推荐 |
|------|--------|--------|--------|------|
| MERGE | 1K-500K | ✅ 更快 | 较慢 | Pandas |
| GROUPBY | 1K-100K | ✅ 更快 | 较慢 | Pandas |
| GROUPBY | 500K+ | 较慢 | ✅ 更快 | DuckDB |
| FILTER | 任意 | ✅ 更快 | 较慢 | Pandas |
| SORT | 任意 | ✅ 更快 | 较慢 | Pandas |

### 关键发现

⚠️ **在内存数据处理场景下，Pandas 通常比 DuckDB 更快！**

原因：
1. DuckDB 每次查询有启动开销（连接、注册表、编译SQL）
2. Pandas + NumPy 的向量化操作已经非常高效
3. 数据已在内存中时，DuckDB 的磁盘优化优势不明显

## DuckDB 适用场景

✅ **推荐使用 DuckDB**：
- 超大数据集（500K+ 行）
- 复杂多表 JOIN
- 需要 SQL 语法便利性
- 数据在磁盘上（直接查询 Parquet/CSV）

❌ **不推荐使用 DuckDB**：
- 中小数据量（<100K 行）
- 简单的过滤、排序操作
- 数据已加载到内存中

## 配置方法

### 1. 配置文件方式

编辑 `src/utils/optimization_config.py`：

```python
# 主开关
USE_DUCKDB: bool = False  # 默认关闭

# 阈值设置
DUCKDB_MIN_ROWS: int = 100000      # 低于此值使用 Pandas
DUCKDB_LARGE_TABLE_ROWS: int = 500000  # 超过此值使用 DuckDB
```

### 2. 环境变量方式

```powershell
# Windows PowerShell - 启用 DuckDB
$env:CHAINSIGHT_USE_DUCKDB = "true"

# Windows PowerShell - 禁用 DuckDB
$env:CHAINSIGHT_USE_DUCKDB = "false"
```

```bash
# Linux/macOS
export CHAINSIGHT_USE_DUCKDB=true   # 启用
export CHAINSIGHT_USE_DUCKDB=false  # 禁用
```

### 3. 代码中动态配置

```python
from src.utils.optimization_config import OptimizationConfig

# 禁用所有 DuckDB 优化
OptimizationConfig.set_duckdb_enabled(False)

# 启用所有 DuckDB 优化
OptimizationConfig.set_duckdb_enabled(True)

# 查看当前配置
OptimizationConfig.print_status()
```

## 模块级优化开关

### Module1（订单生成）
```python
USE_VECTORIZED_CONSUMPTION: bool = True  # 向量化消耗计算 ✅推荐
USE_HISTORY_FILE_LIMIT: bool = True      # 历史文件范围限制 ✅推荐
```

### Module3（净需求计算）
```python
USE_MRP_CACHE: bool = True           # MRP缓存优化 ✅推荐
USE_DUCKDB_NET_DEMAND: bool = False  # DuckDB净需求计算 ❌不推荐
```

### Module5（部署规划）
```python
USE_DUCKDB_DEMAND_COLLECTION: bool = False  # DuckDB需求收集 ❌不推荐
USE_VECTORIZED_DEMAND: bool = False         # 向量化需求收集（有bug）
USE_HORIZON_CACHE: bool = True              # Horizon预计算缓存 ✅推荐
USE_DATA_INDEXER: bool = True               # 数据索引器 ✅推荐
```

## 使用智能包装器

`src/utils/duckdb_sql_wrapper.py` 提供了智能包装器，自动根据数据量选择最优引擎：

```python
from src.utils.duckdb_sql_wrapper import DuckDBSQL, smart_merge, smart_groupby_agg

# 智能 merge（自动选择引擎）
result = smart_merge(df1, df2, on='key', how='left')

# 智能 groupby（自动选择引擎）
result = smart_groupby_agg(df, ['col1'], {'col2': 'sum'})

# 强制使用特定引擎
result = DuckDBSQL.merge(df1, df2, on='key', force_pandas=True)
result = DuckDBSQL.merge(df1, df2, on='key', force_duckdb=True)
```

## 性能统计

启用性能统计收集：

```python
from src.utils.optimization_config import OptimizationConfig, perf_stats

# 启用统计收集
OptimizationConfig.COLLECT_STATS = True

# 运行您的代码...

# 查看统计
perf_stats.print_summary()
```

## 运行性能测试

```powershell
# 运行 DuckDB vs Pandas 性能对比测试
python tools/benchmark_duckdb_vs_pandas.py
```

## 当前默认配置

基于性能测试结果，当前默认配置为：

- **DuckDB**: ❌ 关闭
- **向量化消耗**: ✅ 启用
- **历史文件限制**: ✅ 启用
- **MRP缓存**: ✅ 启用
- **Horizon缓存**: ✅ 启用
- **数据索引器**: ✅ 启用
- **并行处理**: ✅ 启用

这个配置在大多数场景下提供最佳性能。

## 总结

| 场景 | 推荐配置 |
|------|----------|
| 一般使用 | `USE_DUCKDB=False`（默认） |
| 超大数据集 | `USE_DUCKDB=True` |
| 调试/开发 | `DEBUG_MODE=True` |
| 性能分析 | `COLLECT_STATS=True` |
