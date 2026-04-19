# src/modules/deployment_planning 模块详细文档

## 文档信息

| 项 | 内容 |
|---|---|
| 文档版本 | v1.1 |
| 最后更新 | 2026-04-10 |
| 编写人 | 陈显跃 |
| 适用范围 | `src/modules/deployment_planning/` 目录（共 16 个文件） |
| 目标读者 | 算法工程师、测试工程师、业务分析师 |

> **当前目录完整文件清单**（16 个）：  
> `__init__.py`、`allocation.py`、`batch_optimizer.py`、`cache_utils.py`、`constants.py`、`data_loader.py`、`demand_collector.py`、`demand_collector_vectorized.py`、`duckdb_batch_calculator.py`、`horizon_batch_calculator.py`、`inventory.py`、`main.py`、`multiprocess_optimizer.py`、`normalizer.py`、`push_allocation.py`、`validation.py`

---

## 目录

1. [模块概述](#1-模块概述)
2. [主要类与函数列表](#2-主要类与函数列表)
3. [核心数据结构](#3-核心数据结构)
4. [依赖关系](#4-依赖关系)

---

## 1. 模块概述

**模块路径**: `src/modules/deployment_planning/`

**主要职责**:
- Module 5 调拨规划与分配
- 节点需求收集与聚合
- 调拨计划生成（Push 逻辑、优先级分配）
- 库存平衡检查与约束验证
- 优化计算（批量优化、DuckDB 批处理）
- 缓存管理

**核心功能点**:
1. **需求收集**: 收集各节点的需求（SDL + 安全库存 + 订单 + 上游 Gap）
2. **调拨规划**: 应用供给策略，生成调拨计划
3. **分配策略**: 按优先级分配可用库存
4. **批量优化**: 多进程/多线程批量计算
5. **DuckDB 加速**: 向量化批量计算
6. **库存管理**: 跟踪库存状态
7. **验证**: 调拨计划一致性验证

---

## 2. 主要类与函数列表

### 2.1 allocation.py - 调拨分配

**主要类**:
- 无独立类定义，主要为函数集合

**核心函数**:

| 函数名 | 功能 | 关键参数 |
|---|---|---|
| `apply_priority_allocation_vectorized()` | 按优先级向量分配库存 | demand_rows, adjusted_qtys, current_stock, priority_map |
| `push_softpush_allocation()` | Push/SoftPush 补货 | deployment_plan_rows, config, dynamic_soh |
| `collect_node_demands()` | 收集节点需求 | material, location, config, caches |

### 2.2 batch_optimizer.py - 批量优化

**主要类**:
- `BatchOptimizer`

**核心功能**:
- 多进程/多线程批量计算
- 并行优化调拨计算

### 2.3 cache_utils.py - 缓存管理

**主要函数**:

| 函数名 | 功能 |
|---|---|
| `get_or_create_cache()` | 获取或创建缓存 |
| `invalidate_cache()` | 失效缓存 |

### 2.4 constants.py - 常量定义

**常量**:
```python
# 优先级常量
PRIORITY_HIGH = 1
PRIORITY_MEDIUM = 2
PRIORITY_LOW = 3

# 调拨模式
MODE_PUSH = "push"
MODE_PULL = "pull"
```

### 2.5 data_loader.py - 数据加载

**主要函数**:
- 加载配置数据
- 加载输入数据

### 2.6 demand_collector.py - 需求收集

**主要函数**:

| 函数名 | 功能 | 关键参数 |
|---|---|---|
| `collect_node_demands()` | 收集节点需求 | material, location, config, ptf_lsk_cache, lead_time_cache, sdl_index, ss_index, order_index, deploy_config_index, horizon |

### 2.7 demand_collector_vectorized.py - 向量化需求收集

**主要函数**:
- 向量化批量收集需求
- 使用 DuckDB 加速

### 2.8 duckdb_batch_calculator.py - DuckDB 批量计算

**主要功能**:
- DuckDB 批量查询计算
- 高性能数据处理

### 2.9 horizon_batch_calculator.py - 时间窗口批量计算

**主要功能**:
- 按时间窗口批量计算
- 支持多日并行

### 2.10 horizon_batch_calculator.py - 实际时间窗口计算

**主要函数**:
- 计算调拨时间窗口
- 产能和约束检查

### 2.11 inventory.py - 库存管理

**主要函数**:

| 函数名 | 功能 | 关键参数 |
|---|---|---|
| `check_stock_availability()` | 检查库存可用性 | material, location, qty |
| `update_stock()` | 更新库存 | material, location, qty_delta |

### 2.12 main.py - 主入口

**函数签名**:
```python
def main(
    input_path: str = None,
    output_path: str = None,
    sim_start: str = None,
    sim_end: str = None,
    config_dict: dict = None,
    module1_output_dir: str = None,
    module4_output_path: str = None,
    orchestrator: object = None,
    current_date: str = None,
    skip_file_output: bool = False,
    module1_result: dict = None,
    module4_result: dict = None
)
```

**参数说明**:

| 参数名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `input_path` | `str` | 否 | 输入文件路径 |
| `output_path` | `str` | 否 | 输出文件路径 |
| `sim_start` | `str` | 否 | 仿真开始日期 |
| `sim_end` | `str` | 否 | 仿真结束日期 |
| `config_dict` | `dict` | 否 | 配置数据字典 |
| `module1_output_dir` | `str` | 否 | M1 输出目录 |
| `module4_output_path` | `str` | 否 | M4 输出路径 |
| `orchestrator` | `object` | 否 | 编排器对象 |
| `current_date` | `str` | 否 | 当前仿真日期 |
| `skip_file_output` | `bool` | 否 | 是否跳过文件输出 |
| `module1_result` | `dict` | 否 | M1 运行结果 |
| `module4_result` | `dict` | 否 | M4 运行结果 |

**返回值**:
```python
{
    'deployment_plan': pd.DataFrame,    # 调拨计划
    'stock_on_hand_log': pd.DataFrame,  # 库存日志
    'unfulfilled_log': pd.DataFrame,     # 未满足日志
    'validation_log': pd.DataFrame          # 验证日志
}
```

**使用示例**:
```python
result = main(
    input_path="./config/BC_S5.xlsx",
    output_path="./outputs/deployment",
    sim_start="2025-01-01",
    sim_end="2025-01-31",
    skip_file_output=True
)
```

### 2.13 multiprocess_optimizer.py - 多进程优化

**主要类**:
- `MultiprocessOptimizer`

**核心功能**:
- 多进程批量计算
- 性能优化

### 2.14 normalizer.py - 标准化

**主要函数**:
- 标识符标准化
- 数据格式转换

---

## 3. 核心数据结构

### 3.1 DeploymentPlan 数据结构

```python
{
    'material': str,          # 物料编码
    'sending': str,             # 发送地
    'receiving': str,          # 接收地
    'planned_deploy_date': str,  # 计划部署日期
    'deployed_qty': int,        # 部署数量
    'demand_type': str,        # 需求类型
    'priority': int,            # 优先级
    'demand_element': str        # 需求元素
}
```

### 3.2 StockBalance 数据结构

```python
{
    'material': str,          # 物料编码
    'location': str,           # 地点编号
    'quantity': int,            # 可用数量
}
```

### 3.3 NodeDemand 数据结构

```python
{
    'material': str,             # 物料编码
    'location': str,            # 地点编号
    'sdl': float,              # 安全库存水平（SDL）
    'ss': float,                # 安全库存（SS）
    'order_qty': float,          # 订单需求
    'up_gap': float,            # 上游缺口
    'demand_type': str           # 需求类型
    'priority': int              # 优先级
}
```

---

## 4. 依赖关系

```mermaid
flowchart LR
    A["配置加载"] --> B["需求收集"] --> C["调拨分配"] --> D["批量优化"] --> E["验证"] --> F["输出"]
```

**模块依赖**:
- demand_planning 依赖 M1 的需求预测
- 依赖 orchestrator 的库存状态
- 依赖 configuration 模块

**依赖的外部模块**:
- `src/core/orchestrator/`（包） - 库存状态管理
- `src/core/parallel_executor/`（包） - 并行执行（可选）

---

## 附录：相关文档

- [../core.md](core.md) - Core 模块文档
- [ARCHITECTURE.md](ARCHITECTURE.md) - 架构设计文档
- [API.md](API.md) - API 接口文档

---
