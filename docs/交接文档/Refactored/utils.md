# `src/utils` — 工具层模块文档

**编写人**：陈显跃  
**更新时间**：2026-04-15

本文档覆盖当前 `src/utils/` 目录中实际存在的工具模块，并同步第三阶段后续清理结果。

---

## 1. 当前结论

`src/utils/` 当前是项目的横切基础设施层，目录中实际存在 **15 个文件**：

- `__init__.py`
- `config_validator.py`
- `date_helpers.py`
- `defaults.py`
- `duckdb_accelerator.py`
- `duckdb_optimizer.py`
- `inventory_balance_checker.py`
- `logger_config.py`
- `memory_data_store.py`
- `normalization.py`
- `resource_config.py`
- `simulation_cache.py`
- `time_manager.py`
- `validation_manager.py`

已删除且不应再引用的旧兼容文件：

- `runtime_defaults.py`
- `normalization_common.py`
- `cpu_config.py`

---

## 2. 模块分层概览

| 能力分类 | 当前文件 |
|---|---|
| 配置与默认值 | `defaults.py`, `config_validator.py`, `resource_config.py` |
| 标识符规范化 | `normalization.py` |
| 日期与时间 | `date_helpers.py`, `time_manager.py` |
| 日志与校验 | `logger_config.py`, `validation_manager.py`, `inventory_balance_checker.py` |
| 缓存与内存存储 | `simulation_cache.py`, `memory_data_store.py` |
| DuckDB 加速 | `duckdb_optimizer.py`, `duckdb_accelerator.py` |

---

## 3. 包入口（`__init__.py`）

**文件路径**：`src/utils/__init__.py`

当前公开导出：

```python
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

`run_pre_simulation_validation` 仍通过 `__getattr__` 延迟导入，以避免循环引用。

以下模块不经由包入口自动暴露，需显式导入：

- `src.utils.defaults`
- `src.utils.normalization`
- `src.utils.resource_config`
- `src.utils.date_helpers`

---

## 4. 默认值与配置

### 4.1 `defaults.py`

**文件路径**：`src/utils/defaults.py`  
**配置来源**：`config/defaults.yaml`

用途：集中管理跨模块默认参数，避免硬编码散落在各模块。

当前重要常量包括：

- `DEFAULT_MOQ`
- `DEFAULT_RV`
- `DEFAULT_PTF`
- `DEFAULT_LSK`
- `DEFAULT_LEAD_TIME`
- `DEFAULT_HORIZON`
- `DEFAULT_CHANGEOVER_TIME`
- `DEFAULT_PUSH_LEVELS`
- `RESOURCE_UTILIZATION`
- `M1_FUTURE_CUTOFF_DAYS`
- `M1_DEFAULT_MAX_ADVANCE_DAYS`
- `M6_MAX_WAIT_DAYS`
- `M6_RANDOM_SEED`

推荐导入：

```python
from src.utils.defaults import M6_MAX_WAIT_DAYS, M6_RANDOM_SEED
```

### 4.2 `config_validator.py`

用于仿真启动前检查配置完整性与结构合法性，是 `run.py` 主链路预校验的一部分。

---

## 5. 标识符归一化

### `normalization.py`

**文件路径**：`src/utils/normalization.py`

用途：作为全仓库标识符归一化的单一真源。

关键函数：

- `normalize_material()`
- `normalize_location()`
- `normalize_identifiers()`
- `cast_identifier_columns()`

调用位置包括：

- `src/core/main_integration/production_runner.py`
- `src/core/main_integration/simulation_file.py`
- `src/core/main_integration/simulation_db.py`
- `src/core/orchestrator/views.py`
- `src/core/orchestrator/processors.py`
- `src/core/orchestrator/persistence.py`
- `src/core/orchestrator/orchestrator_main.py`
- `src/core/orchestrator/daily_ops.py`
- `src/modules/deployment_planning/data_loader.py`
- `src/modules/deployment_planning/validation.py`
- `src/modules/*` 的部分工具函数

> 说明：`runtime_defaults.py`、`normalization_common.py`、`src/core/main_integration/normalize.py`、`src/modules/demand_planning/normalization.py`、`src/core/orchestrator/normalize.py`、`src/modules/deployment_planning/normalizer.py` 已删除，不再作为兼容层保留。

---

## 6. 日期、时间与资源

### 6.1 `date_helpers.py`

提供共享的规划窗口、review day 与 lead time 计算 helper。

关键函数：

- `compute_planning_window()`
- `calculate_transport_lead_time()`

### 6.2 `time_manager.py`

提供 `SimulationTimeManager` 与 `initialize_time_manager()`，用于统一仿真日期管理。

### 6.3 `resource_config.py`

当前资源配置单一入口，替代已删除的 `cpu_config.py`。

关键能力：

- `get_optimal_threads()`
- `get_optimal_workers()`
- `get_optimal_memory()`
- `get_resource_config()`
- `get_duckdb_config()`

实现特点：

- 优先使用 `psutil` 获取更准确的内存信息
- 若 `psutil` 不可用，则回退到环境变量/默认值
- `RESOURCE_UTILIZATION` 从 `config/defaults.yaml` 加载

---

## 7. 日志、验证与状态

### `logger_config.py`

提供 `setup_logging()`，为文件模式和数据库模式建立统一日志输出。

### `validation_manager.py`

用于收集和管理仿真过程中的校验结果。

### `inventory_balance_checker.py`

用于最终库存平衡检查，是主仿真流程结束前的重要验证步骤。

### `simulation_cache.py`

提供运行期缓存，用于减少重复计算。

### `memory_data_store.py`

提供内存级数据存储，用于数据库模式和部分高性能链路。

---

## 8. DuckDB 相关模块

### `duckdb_optimizer.py`

提供 DuckDB 批量计算优化能力，并改为从 `resource_config.py` 读取线程/内存配置。

### `duckdb_accelerator.py`

提供更底层的 DuckDB 加速封装，为部分模块的批处理路径提供支撑。

---

## 9. 已删除模块与迁移规则

以下旧文件已从 `src/utils/` 物理删除：

| 已删除文件 | 替代方案 |
|---|---|
| `runtime_defaults.py` | `defaults.py` |
| `normalization_common.py` | `normalization.py` |
| `cpu_config.py` | `resource_config.py` |

推荐迁移方式：

```python
# 默认值
from src.utils.defaults import DEFAULT_MOQ

# 归一化
from src.utils.normalization import normalize_identifiers

# 资源配置
from src.utils.resource_config import get_optimal_threads
```

不要再使用：

```python
from src.utils.runtime_defaults import ...
from src.utils.normalization_common import ...
from src.utils.cpu_config import ...
```

---

## 10. 当前依赖关系总览

```text
src/utils/
├── __init__.py
├── defaults.py              ← 从 config/defaults.yaml 加载共享默认值
├── normalization.py         ← 统一归一化单一真源
├── date_helpers.py          ← 日期与前置期 helper
├── resource_config.py       ← 统一资源配置入口（替代 cpu_config）
├── logger_config.py         ← 日志配置
├── config_validator.py      ← 配置校验
├── validation_manager.py    ← 校验结果管理
├── time_manager.py          ← 仿真时间管理
├── simulation_cache.py      ← 运行缓存
├── memory_data_store.py     ← 内存存储
├── inventory_balance_checker.py ← 库存平衡检查
├── duckdb_optimizer.py      ← DuckDB 优化
└── duckdb_accelerator.py    ← DuckDB 加速
```

---

## 11. 维护建议

1. 新的跨模块默认参数只放在 `config/defaults.yaml` + `src/utils/defaults.py`
2. 新的归一化逻辑只改 `src/utils/normalization.py`
3. 新的资源相关逻辑只改 `src/utils/resource_config.py`
4. 不要重新引入兼容层文件，避免再次形成多份真源
