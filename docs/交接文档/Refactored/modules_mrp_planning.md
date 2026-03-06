# modules/mrp_planning — MRP计划模块文档

## 模块概述

`src/modules/mrp_planning/` 是 ChainSight 的 **物料需求计划 (MRP) 仿真引擎**，负责对多级供应链网络进行逐日分层净需求计算。核心设计目标：

- **逐日仿真**：每天运行一次，基于当日 Module1 输出推算每个节点的净需求（GAP）
- **分层处理**：BFS 层级分配后从 layer=0 向上逐层传递 GAP
- **双模式计算**：节点数 ≥ `BATCH_CALCULATION_THRESHOLD(50)` 时用 DuckDB 批量，否则 ThreadPoolExecutor 并行
- **预索引优化**：`DataIndexer` 将 O(n×m) 查找降至 O(1)

---

## 文件结构

```
src/modules/mrp_planning/
├── __init__.py               # 公开 API 导出
├── constants.py              # 全局常量
├── config_loader.py          # 配置与输入数据加载
├── data_indexer.py           # 预索引加速结构
├── layer_assignment.py       # BFS 层级分配
├── lead_time.py              # 前置时间与水平期推算
├── utils.py                  # 通用工具函数
├── net_demand.py             # 净需求计算（原始版 + 索引版）
├── node_processor.py         # 单节点处理器（gap 传递）
├── mrp_simulation.py         # 主仿真入口（分层 + 批量/并行）
├── integration.py            # DuckDB 内存模式集成接口
└── duckdb_batch_calculator.py # DuckDB/Pandas 批量净需求计算
```

---

## 主要文件说明

### `constants.py`

定义全局常量：

| 常量 | 类型 | 值 | 说明 |
|---|---|---|---|
| `USE_DUCKDB_BATCH_CALCULATION` | `bool` | `True` | 是否启用 DuckDB 批量计算开关 |
| `BATCH_CALCULATION_THRESHOLD` | `int` | `50` | 节点数阈值：>= 此值时切换至 DuckDB 批量模式 |

---

### `config_loader.py`

负责从 Excel 配置文件和目录加载所有输入数据。

#### `load_config(config_file: str) -> dict`

加载 Module3 配置 Excel 文件，返回各工作表的 DataFrame 字典。

| 参数 | 类型 | 说明 |
|---|---|---|
| `config_file` | `str` | Excel 配置文件路径 |

**返回值：** `dict`，key 为工作表名，value 为 `pd.DataFrame`。常用 key：
- `"BOM"` — 物料清单
- `"LocationMaster"` — 地点主数据
- `"InventoryBalance"` — 库存余额
- `"OpenOrders"` — 未结订单
- `"MCT"` — 最小周期时间
- `"PDT"` — 计划交货时间
- `"PTF"` — 计划时间围栏
- `"LSK"` — 周期数
- `"MOQ"` / `"RV"` — 最小订购量 / 舍入值

#### `load_module1_daily_outputs(module1_output_dir: str, simulation_date: pd.Timestamp) -> dict`

扫描 Module1 日度输出目录，加载仿真日期对应的各类需求文件（AO/FC/SS）。

| 参数 | 类型 | 说明 |
|---|---|---|
| `module1_output_dir` | `str` | Module1 输出目录路径 |
| `simulation_date` | `pd.Timestamp` | 仿真日期 |

**返回值：** `dict`，包含 `"ao"`, `"fc"`, `"ss"` 三类 DataFrame。

#### `load_excel_with_sheets(file_path: str, sheet_names: list) -> dict`

通用 Excel 多工作表加载函数。

---

### `data_indexer.py`

预构建查找索引，将热点数据访问从 O(n×m) 优化为 O(1)。

#### 类 `DataIndexer`

```python
class DataIndexer:
    ml_index: Dict[tuple, Any]   # (material, location) 级别索引
    mr_index: Dict[tuple, Any]   # (material, receiving) 级别索引
    ms_index: Dict[str, Any]     # (material, sending) 级别索引
```

**构造方法：** `DataIndexer()`（空索引），通过 `create_simulation_indexer()` 工厂函数填充。

##### `create_simulation_indexer(config: dict) -> DataIndexer`

工厂函数，一键从配置 dict 构建包含 **10 个数据索引** 的完整 `DataIndexer` 对象。

| 参数 | 类型 | 说明 |
|---|---|---|
| `config` | `dict` | `load_config()` 返回的配置字典 |

**内部构建的 10 个索引：**

| 索引名 | 键 | 值 | 用途 |
|---|---|---|---|
| `inventory_ml` | `(material, location)` | 库存量 | 快速查库存 |
| `bom_parent` | `(material, location)` | 父节点列表 | BOM 树向上遍历 |
| `bom_children` | `(material, location)` | 子节点列表 | BOM 树向下遍历 |
| `open_orders_ml` | `(material, location)` | 已排期订单 | 订单供给查询 |
| `mct_ml` | `(material, location)` | MCT 值 | 前置时间计算 |
| `pdt_ml` | `(material, location)` | PDT 值 | 前置时间计算 |
| `ptf_ml` | `(material, location)` | PTF 值 | 水平期计算 |
| `lsk_ml` | `(material, location)` | LSK 值 | 水平期计算 |
| `moq_three` | `(material, sending, receiving)` | MOQ 值 | 三键 MOQ 查找 |
| `moq_two` | `(material, sending)` | MOQ 值 | 二键 MOQ 查找（fallback） |

---

### `layer_assignment.py`

#### `assign_location_layers(bom_df: pd.DataFrame, demand_locations: list) -> dict`

使用 **BFS（广度优先搜索）** 对供应链网络节点进行层级分配。

| 参数 | 类型 | 说明 |
|---|---|---|
| `bom_df` | `pd.DataFrame` | BOM 数据（含 sending_location / receiving_location） |
| `demand_locations` | `list` | 有需求的地点列表（作为 layer=0 起点） |

**返回值：** `dict`，key 为 `(material, location)` 元组，value 为层级整数（0 = 最终需求点，越大越上游）。

**算法：**
1. 从 demand_locations 构建 layer=0 集合
2. BFS 向上游遍历 BOM，每层 +1
3. 若节点在多条路径上取最大层级

---

### `lead_time.py`

前置时间与计划水平期推算。

#### `compute_root_horizon(material: str, location: str, config: dict, indexer: Optional[DataIndexer]) -> int`

计算 Plant 节点的计划水平期（horizon）。

**公式：** `max(MCT, PDT + GR) + PTF + LSK - 1`

| 参数 | 类型 | 说明 |
|---|---|---|
| `material` | `str` | 物料编号 |
| `location` | `str` | 地点编号 |
| `config` | `dict` | 配置字典 |
| `indexer` | `Optional[DataIndexer]` | 预索引（可选，有则 O(1) 查询） |

**返回值：** `int`，水平期天数。

#### `determine_lead_time(material: str, sending: str, receiving: str, config: dict, indexer: Optional[DataIndexer]) -> int`

确定从 sending 到 receiving 的前置时间。优先使用 PDT，fallback 到 MCT。

#### `infer_sending_location_type(location: str, config: dict) -> str`

从 LocationMaster 推断地点类型（`"Plant"` / `"DC"` / `"Customer"` 等）。

---

### `utils.py`

通用工具函数集合。

#### `apply_moq_rv(quantity: float, moq: float, rv: float) -> int`

对数量应用 MOQ（最小订购量）和 RV（舍入值）约束。

| 参数 | 类型 | 说明 |
|---|---|---|
| `quantity` | `float` | 原始需求量 |
| `moq` | `float` | 最小订购量 |
| `rv` | `float` | 舍入值 |

**返回值：** `int`，经 MOQ/RV 约束后的整数数量。  
**公式：** `max(moq, ceil(max(quantity, moq) / rv) * rv)`

#### `normalize_location(location: Any) -> str`

标准化地点编号（去空格、统一大小写）。

#### `normalize_material(material: Any) -> str`

标准化物料编号（去空格、统一大小写）。

#### `normalize_identifiers(df: pd.DataFrame, cols: list) -> pd.DataFrame`

批量标准化 DataFrame 中指定列的标识符。

#### `apportion_largest_remainder(total: int, weights: Dict[str, float]) -> Dict[str, int]`

**最大余数法**，将总整数按权重比例分配，保证各部分之和恰好等于 total。

| 参数 | 类型 | 说明 |
|---|---|---|
| `total` | `int` | 总数量（待分配） |
| `weights` | `Dict[str, float]` | 各类型的权重（如 `{"ao": 0.5, "fc": 0.3, "ss": 0.2}`） |

**返回值：** `Dict[str, int]`，各类型分配到的整数量。  
**用途：** 将层级间传递的 gap 按 AO/FC/SS 比例精确分配，避免浮点舍入导致总量不守恒。

#### `build_ptf_lsk_cache(config: dict) -> dict`

预构建 PTF/LSK 缓存字典，加速后续 horizon 计算。

#### `get_ptf_lsk(material: str, location: str, cache: dict) -> Tuple[int, int]`

从缓存中获取指定物料/地点的 PTF 和 LSK 值。

#### `lookup_moq_rv_three_keys(material: str, sending: str, receiving: str, moq_df: pd.DataFrame, rv_df: pd.DataFrame) -> Tuple[float, float]`

MOQ/RV 三键查找，优先级：

1. 三键 `(material, sending, receiving)` 精确匹配
2. 二键 `(material, sending)` 匹配
3. 默认值 `(1, 1)`

---

### `net_demand.py`

净需求计算核心逻辑（含两个版本）。

#### `calculate_daily_net_demand(node: dict, date: pd.Timestamp, config: dict) -> dict`

**原始版本**，基于 Pandas 逐行查找计算单节点、单日净需求。

| 参数 | 类型 | 说明 |
|---|---|---|
| `node` | `dict` | 节点信息 `{"material": str, "location": str, "layer": int}` |
| `date` | `pd.Timestamp` | 仿真日期 |
| `config` | `dict` | 配置字典 |

**返回值：** `dict`，包含：
```python
{
    "material": str,
    "location": str,
    "date": pd.Timestamp,
    "ao_gap": float,        # 实际订单缺口
    "fc_gap": float,        # 预测缺口
    "ss_gap": float,        # 安全库存缺口
    "total_gap": float,     # 总缺口
    "inventory": float,     # 当日库存
    "supply": float,        # 当日到货
    "demand_ao": float,     # AO 需求
    "demand_fc": float,     # FC 需求
    "demand_ss": float,     # SS 需求
}
```

**GAP 优先级顺序：** AO gap → FC gap → SS gap（从总供给中依次扣除）

#### `calculate_daily_net_demand_indexed(node: dict, date: pd.Timestamp, config: dict, indexer: DataIndexer) -> dict`

**索引优化版本**，接口与原始版完全相同，但使用 `DataIndexer` 进行 O(1) 数据查找，大幅减少 DataFrame 查询开销。

---

### `node_processor.py`

#### 类 `NodeProcessor`

单节点处理器，封装节点的完整处理逻辑（含 gap 向上传递）。

```python
class NodeProcessor:
    node: dict
    config: dict
    indexer: Optional[DataIndexer]
    parent_gap_records: List[dict]   # 待传递给父节点的 gap 记录
```

##### `process(date: pd.Timestamp, parent_gaps: Dict[tuple, dict]) -> dict`

处理指定节点在指定日期的净需求，并将自身的需求 gap 传递给父节点。

| 参数 | 类型 | 说明 |
|---|---|---|
| `date` | `pd.Timestamp` | 仿真日期 |
| `parent_gaps` | `Dict[tuple, dict]` | 父节点接收 gap 的累积字典（会被修改） |

**处理逻辑：**
1. 调用 `calculate_daily_net_demand_indexed()`（若有索引）或原始版
2. 用 `apportion_largest_remainder()` 按 AO/FC/SS 权重将 total_gap 分配
3. 根据 BOM 关系，将各类型 gap 乘以 usage_rate 传递给上游父节点
4. 返回当日净需求记录

---

### `mrp_simulation.py`

主仿真入口，协调分层处理流程。

#### `run_mrp_layered_simulation_daily(config: dict, simulation_date: pd.Timestamp, indexer: Optional[DataIndexer]) -> pd.DataFrame`

执行一次逐日 MRP 分层仿真。

| 参数 | 类型 | 说明 |
|---|---|---|
| `config` | `dict` | 配置字典 |
| `simulation_date` | `pd.Timestamp` | 仿真日期 |
| `indexer` | `Optional[DataIndexer]` | 预索引（可选） |

**返回值：** `pd.DataFrame`，所有节点的净需求结果，含列：
`material, location, layer, date, ao_gap, fc_gap, ss_gap, total_gap, inventory, supply, demand_ao, demand_fc, demand_ss`

**执行流程：**
```
1. assign_location_layers()       → 计算各节点层级
2. create_simulation_indexer()    → 构建预索引（若启用）
3. for layer in sorted(layers):   → 从 layer=0 逐层向上
4.     _process_layer()           → 批量或并行处理本层所有节点
5. 汇总所有层结果返回
```

#### `_process_layer(nodes: list, layer: int, date: pd.Timestamp, config: dict, indexer: Optional[DataIndexer], parent_gaps: dict) -> List[dict]`

处理单层的所有节点，根据节点数量自动选择计算模式：

| 条件 | 计算模式 |
|---|---|
| `len(nodes) >= BATCH_CALCULATION_THRESHOLD (50)` | DuckDB 批量计算（`batch_calculate_net_demand_duckdb()`） |
| `len(nodes) < 50` | `ThreadPoolExecutor` 并行处理 |

DuckDB 批量失败时自动 fallback 到 Pandas 版本。

---

### `integration.py`

DuckDB 内存模式集成接口，支持整个仿真在内存中端到端运行（跳过磁盘 I/O）。

#### `run_integrated_mode(config: dict, simulation_date: pd.Timestamp, memory_store=None) -> pd.DataFrame`

在 DuckDB 内存模式下运行完整 MRP 仿真：
1. 从内存存储读取 Module1 输出（替代文件读取）
2. 调用 `run_mrp_layered_simulation_daily()`
3. 将结果写回内存存储（替代文件写出）

| 参数 | 类型 | 说明 |
|---|---|---|
| `config` | `dict` | 配置字典 |
| `simulation_date` | `pd.Timestamp` | 仿真日期 |
| `memory_store` | `optional` | DuckDB 内存存储对象 |

---

### `duckdb_batch_calculator.py`

DuckDB 批量净需求计算的核心实现文件。

#### `batch_calculate_net_demand_duckdb(nodes: list, date: pd.Timestamp, config: dict, indexer: DataIndexer, parent_gaps: dict) -> List[dict]`

使用 DuckDB SQL 引擎对一批节点执行向量化净需求计算。

| 参数 | 类型 | 说明 |
|---|---|---|
| `nodes` | `list` | 节点列表（每个为 `{"material", "location", "layer"}` 字典） |
| `date` | `pd.Timestamp` | 仿真日期 |
| `config` | `dict` | 配置字典 |
| `indexer` | `DataIndexer` | 预构建索引 |
| `parent_gaps` | `dict` | 父节点 gap 累积字典（会被更新） |

**执行流程：**
1. 将所有节点数据转换为 DuckDB 临时表
2. 通过 SQL JOIN 获取库存、供给、需求数据
3. 使用 DuckDB 向量化计算 ao_gap / fc_gap / ss_gap
4. 将 gap 按 BOM 关系传递给父节点
5. 失败时调用 `_batch_calculate_pandas()` fallback

#### `_batch_calculate_pandas(nodes: list, date: pd.Timestamp, config: dict, indexer: DataIndexer, parent_gaps: dict) -> List[dict]`

DuckDB 失败时的 Pandas fallback 版本，逐节点调用 `calculate_daily_net_demand_indexed()`。

---

## 数据结构

### 节点 (Node)

```python
{
    "material": str,      # 物料编号
    "location": str,      # 地点编号
    "layer": int,         # BFS 层级（0=最终需求点）
}
```

### 净需求记录 (Net Demand Record)

```python
{
    "material": str,
    "location": str,
    "layer": int,
    "date": pd.Timestamp,
    "ao_gap": float,         # 实际订单缺口（负值 = 超额供给）
    "fc_gap": float,         # 预测缺口
    "ss_gap": float,         # 安全库存缺口
    "total_gap": float,      # 三类缺口之和
    "inventory": float,      # 当日期初库存
    "supply": float,         # 当日到货量（来自订单/上层传递）
    "demand_ao": float,      # AO 需求量
    "demand_fc": float,      # FC 需求量
    "demand_ss": float,      # SS 需求量
}
```

### 父节点 Gap 传递 (Parent Gap)

```python
# parent_gaps 字典的 value 格式
{
    "ao": float,    # 传递给父节点的 AO gap
    "fc": float,    # 传递给父节点的 FC gap
    "ss": float,    # 传递给父节点的 SS gap
}
```

---

## 依赖关系

```
mrp_simulation.py
    ├── layer_assignment.py     (BFS 层级分配)
    ├── data_indexer.py         (预索引构建)
    ├── duckdb_batch_calculator.py  (批量计算)
    │       └── net_demand.py   (fallback)
    └── node_processor.py
            ├── net_demand.py   (净需求计算)
            ├── utils.py        (apportion_largest_remainder)
            └── lead_time.py    (前置时间)

integration.py
    └── mrp_simulation.py
        └── (+ memory_data_store from utils/)
```

---

## 数据流图

```
Module1 输出 (AO/FC/SS 文件)
        │
        ▼
config_loader.load_module1_daily_outputs()
        │
        ▼
data_indexer.create_simulation_indexer()  ─── 构建 10 个 O(1) 索引
        │
        ▼
layer_assignment.assign_location_layers()  ─── BFS 分层
        │
        ▼
mrp_simulation._process_layer()  (按层级从 0 到 max)
    ├── [节点数 ≥ 50]  duckdb_batch_calculator  ─── SQL 向量化
    │       └── [失败] _batch_calculate_pandas  ─── fallback
    └── [节点数 < 50]  ThreadPoolExecutor
            └── node_processor.NodeProcessor.process()
                    └── net_demand.calculate_daily_net_demand_indexed()
        │
        ▼
GAP 通过 apportion_largest_remainder() 按 AO/FC/SS 比例分配
        │
        ▼
GAP 乘以 usage_rate 传递给上游父节点（下一层输入）
        │
        ▼
最终汇总 pd.DataFrame  ─→  Module4 输入 (layer=0 净需求)
```

---

## 关键设计决策

| 决策 | 原因 |
|---|---|
| DataIndexer 预构建 10 个索引 | 避免在仿真循环中重复 DataFrame 查找，热路径 O(n×m) → O(1) |
| BATCH_CALCULATION_THRESHOLD=50 | 经验阈值：节点数少时线程调度开销 > 并行收益，多时 DuckDB SQL 向量化优势明显 |
| DuckDB 失败自动 fallback | 保证仿真不因 DuckDB 问题中断，生产环境稳定性优先 |
| apportion_largest_remainder | 整数分配时浮点舍入会导致传递总量不守恒，最大余数法保证精确 |
| GAP 优先级 AO → FC → SS | 实际订单优先于预测需求，预测优先于安全库存补充 |
| MOQ/RV 三键 > 二键 > 默认 | 精确匹配供应商路由，兼容不完整主数据 |
