# ChainSight 架构设计文档

**版本**: 2.1.0  
**最后更新**: 2026-01-29

## 概述

ChainSight 是一个生产级的供应链规划仿真系统，采用标准分层架构设计，确保代码的可维护性、可扩展性和可测试性。系统支持本地文件模式和数据库模式，并集成了 Cython 性能优化内核和 DuckDB 高性能数据处理引擎。

## 技术栈

| 组件 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 编程语言 | Python | 3.10+ | 主要开发语言 |
| 性能优化 | Cython | 3.0+ | 计算密集型操作加速 |
| 数据处理 | DuckDB | 1.1.3 | 高性能分析型数据处理 |
| 数据操作 | pandas | 2.2.3 | 数据分析和转换 |
| 数据持久化 | PostgreSQL | 14+ | 生产环境数据存储 |
| 文件读写 | openpyxl | 3.1.5 | Excel 配置和输出 |
| 数值计算 | numpy | 2.0+ | 数组和矩阵运算 |

## 架构分层

### 1. CLI 层 (Entry Point)

**位置**: `run.py` (项目root)

**职责**:
- 解析命令行参数
- 验证输入的有效性
- 调用Core层的main函数
- 处理顶级异常和退出码

**特点**:
- 无业务逻辑
- 只关心参数解析和调度
- 所有实现都代理给src/core/

### 2. Core 层 (`src/core/`)

核心执行引擎，负责整个仿真流程的编排和状态管理。

#### 2.1 Orchestrator (`orchestrator.py`)

**设计模式**: 中央集权模式

**数据结构**:
```python
class Orchestrator:
    # 全局状态维护
    physical_inventory: Dict[date, Dict[material, location, quantity]]
    open_deployment: Dict[date, List[Deployment]]
    in_transit_inventory: Dict[date, Dict[shipment_id, Inventory]]
    production_gr: Dict[date, List[ProductionReceipt]]
    delivery_gr: Dict[date, List[DeliveryReceipt]]
    space_capacity: Dict[location, capacity]
    
    # 持久化
    save_snapshot(date, output_dir)
    load_snapshot(date, output_dir)
```

**职责**:
- 维护全局物理库存状态（来自M4生产、M6交付）
- 跟踪部署计划和在途库存
- 管理生产收货和交付收货
- 提供日级快照和审计日志
- 库存守恒验证

**关键方法**:
- `update_state()`: 接收各模块更新，同步全局状态
- `get_inventory()`: 查询特定时点的库存
- `validate_consistency()`: 验证库存守恒

#### 2.2 Main Integration (`main_integration.py`)

**设计模式**: 编排器模式

**职责**:
- 实现日循环执行逻辑
- 模块序列调度：M1 → M4 → M5 → M6 → M3
- 断点续跑能力
- 数据流转和一致性保证

**执行流程**:
```
Load Config & Validate
    ↓
Initialize Orchestrator & Database (if --use-db)
    ↓
Check Resume Capability
    ↓
FOR each_day in [start_date, end_date]:
    │
    ├─ M1 (Demand Planning)
    │   - 需求预测 (Demand Forecast)
    │   - 订单生成 (Order Log)
    │   - 发货日志 (Shipment Log)
    ├─ Orchestrator.update_state()
    │
    ├─ M4 (Production Planning)
    │   - 生产计划 (Production Plan)
    │   - 产能分配 (Capacity Allocation)
    │   - 生产收货 (Production Receipt)
    ├─ Orchestrator.update_state()
    │
    ├─ M5 (Deployment Planning)
    │   - 部署计划 (Deployment Plan)
    │   - 库存推送 (Push Plan)
    │   - 库存预测 (Inventory Projection)
    ├─ Orchestrator.update_state()
    │
    ├─ M6 (Logistics Execution)
    │   - 交付计划 (Delivery Plan)
    │   - 车辆装载 (Vehicle Packing)
    │   - 运输日志 (Transport Log)
    ├─ Orchestrator.update_state()
    │
    ├─ M3 (MRP Planning)
    │   - 净需求计算 (Net Demand)
    │   - MRP 仿真 (MRP Simulation)
    │   - 建议采购单 (Suggested PO)
    ├─ Orchestrator.update_state()
    │
    └─ Generate Daily Summary & Snapshot
    
Generate Final Report & Validation
Write to Database (if --use-db) or Excel Files
```

#### 2.3 CLI Runner (`src/core/run.py`)

**职责**:
- 运行管理（断点续跑、目录选择）
- 参数校验和转换
- 日志初始化

### 3. Modules 层 (`src/modules/`)

6个独立的业务规划模块，按供应链流程阶段划分。

#### 3.1 模块接口约定

```python
def execute(date: str, 
           config: Dict,
           orchestrator: Orchestrator,
           historical_data: Dict) -> Dict:
    """
    Args:
        date: 仿真日期 (YYYY-MM-DD)
        config: 配置参数
        orchestrator: 全局状态管理器
        historical_data: 历史数据（订单、库存等）
    
    Returns:
        {
            'module_outputs': {...},
            'state_updates': {...},  # 返回需要更新到orchestrator的状态
            'metrics': {...},        # 关键指标
            'errors': [...]          # 错误信息
        }
    """
```

#### 3.2 各模块说明

| 模块 | 名称 | 职责 | 主要输入 | 主要输出 |
|-----|------|------|---------|---------|
| **M1** | Demand Planning | 需求规划 | 订单、预测、配置 | DemandForecast, OrderLog, ShipmentLog |
| **M3** | MRP Planning | 物料需求计划 | 净需求、BOM、库存 | NetDemand, SuggestedPO |
| **M4** | Production Planning | 生产计划 | 生产需求、产能、物料 | ProductionPlan, CapacityUtilization, ProductionReceipt |
| **M5** | Deployment Planning | 部署规划 | 库存、需求、运输 | DeploymentPlan, InventoryProjection, PushPlan |
| **M6** | Logistics Execution | 物流执行 | 部署计划、车辆、路线 | DeliveryPlan, VehicleLoading, TransportLog |

#### 3.3 子模块组织

每个模块进一步细分为子包，实现模块化和单一职责原则：

```
modules/
├── demand_planning/          # M1 子模块
│   ├── forecast_processor.py      # 需求预测处理
│   ├── order_generator.py         # 订单生成
│   └── shipment_tracker.py        # 发货跟踪
│
├── mrp_planning/             # M3 子模块
│   ├── net_demand_calculator.py   # 净需求计算
│   └── mrp_simulator.py           # MRP 仿真引擎
│
├── production_planning/      # M4 子模块
│   ├── plan_builder.py            # 计划构建
│   └── capacity_allocator.py      # 产能分配
│
├── deployment_planning/      # M5 子模块
│   ├── allocation_optimizer.py    # 分配优化
│   ├── inventory_manager.py       # 库存管理
│   └── push_planner.py            # 推送计划
│
└── logistics_execution/      # M6 子模块
    ├── vehicle_packer.py          # 车辆装载
    └── delivery_executor.py       # 交付执行
```

### 4. Utils 层 (`src/utils/`)

通用工具和跨模块的支持服务。

#### 4.1 Config Validator

**职责**:
- 验证配置文件结构
- 标准化标识符字段（物料、地点）
- 去重和修复常见错误

#### 4.2 Logger Config

**职责**:
- 统一的日志配置
- 同时输出到文件和终端
- 日志级别管理

#### 4.3 Validation Manager

**职责**:
- 数据有效性检查
- 业务规则验证
- 异常数据报告

#### 4.4 Inventory Balance Checker

**职责**:
- 验证库存守恒原理
- 追踪入库、出库、库存变化
- 识别数据不一致

#### 4.5 Time Manager

**职责**:
- 日期、时间计算
- 工作日/非工作日判断
- 时间相关配置

### 5. Services 层 (`src/services/`)

高级业务服务，提供跨模块的复杂功能。

#### 5.1 Summary Report Generator

**职责**:
- 生成各类汇总报告
- 整合多个模块的输出
- 生成可视化图表
- 历史库存记录 (Historical Inventory Record)
- KPI 汇总报告

#### 5.2 Performance Profiler

**职责**:
- 性能分析和监测
- 识别瓶颈和热点
- 性能优化建议
- 执行时间统计

### 6. 性能优化层

#### 6.1 Cython 内核 (`src/cython_kernels/`)

使用 Cython 编写的高性能计算内核，提供数量级的性能提升。

**编译后的内核模块**:
```python
# production_kernels.pyx - 生产计算内核
- 批量生产计划计算
- 产能分配算法
- BOM 展开计算

# logistics_kernels.pyx - 物流计算内核  
- 车辆装载优化
- 路径规划算法
- 运输成本计算

# aggregation_kernels.pyx - 聚合计算内核
- 大规模数据聚合
- 多维度汇总
- 统计指标计算
```

**编译方式**:
```bash
# 使用 setup.py 编译 Cython 扩展
python setup.py build_ext --inplace

# 生成文件（Windows）:
# - production_kernels.cp314-win_amd64.pyd
# - logistics_kernels.cp314-win_amd64.pyd
# - aggregation_kernels.cp314-win_amd64.pyd
```

**性能提升**:
- 数值密集计算：50-100x 加速
- 循环优化：20-50x 加速
- 并行计算（OpenMP）：4-8x 额外加速

#### 6.2 DuckDB 数据处理 (`pgsql_db/duckdb_*.py`)

集成 DuckDB 进行高性能数据处理，特别适合分析型查询。

**核心组件**:
```python
# duckdb_processor.py - DuckDB 处理器
- 批量数据加载（零拷贝）
- SQL 分析查询优化
- 向量化计算

# duckdb_integration.py - DuckDB 集成
- 与 PostgreSQL 双向同步
- Parquet 格式缓存
- 增量更新策略

# optimized_processor.py - 优化处理器
- 智能查询计划
- 内存管理优化
- 并行执行引擎
```

**性能特点**:
- 列式存储：比 pandas 快 10-100x
- 零拷贝集成：无需数据复制
- 并行查询：自动多线程执行
- Parquet 缓存：快速持久化

#### 6.3 数据库层 (`pgsql_db/`)

完整的 PostgreSQL 数据库支持模块。

**核心组件**:
- `db_connection.py`: 连接池管理
- `db_initializer.py`: 自动建库建表
- `table_schemas.py`: 统一表结构定义
- `table_mapping.py`: Excel-DB 表名映射
- `module_data_writer.py`: 批量数据写入
- `excel_importer.py`: Excel 配置导入

**设计模式**:
- 统一表名：同类配置共用一张表，用 `config_name` 字段区分
- 自动初始化：检测并创建缺失的数据库和表
- 批量写入：使用 `COPY` 命令高效写入
- 索引优化：关键字段自动创建索引

### 7. 工具层 (`tools/`)

独立的辅助工具脚本，不依赖于核心代码。

**主要工具**:
```bash
# 数据库初始化
python tools/init_database.py

# 输出对比
python tools/compare_outputs.py --run1 outputs/run1 --run2 outputs/run2

# 性能基准测试
python tools/benchmark_duckdb_vs_pandas.py

# 生成 Word 报告
python tools/generate_docx_report.py --output report.docx

# 配置表迁移
python tools/migrate_config_tables.py --from BC_S5 --to BC_S9
```

### 8. Tests 层 (`tests/`, `test_files/`)

自动化测试框架和对比工具。

#### 8.1 单元测试和集成测试 (`tests/`)

**组织**:
```
tests/
├── e2e_integration_test.py    # 端到端集成测试
├── test_logger.py             # 日志系统测试
└── (未来扩展)
    ├── unit/                  # 单元测试（模块级）
    ├── integration/           # 集成测试（跨模块）
    └── conftest.py            # pytest 配置和 fixtures
```

#### 8.2 测试和对比工具 (`test_files/`)

**测试配置**:
- `BC_S5.xlsx`: 主测试配置
- `BC_S9.xlsx`: 备用测试配置

**对比工具**:
```python
# 主对比工具 - 全面输出对比
python test_files/compare_all_outputs.py --run1 dir1 --run2 dir2

# 数据库 vs 本地文件对比
python test_files/compare_db_vs_local.py

# 与 ChainSight_Dev 对比
python test_files/compare_db_vs_chainsight_dev.py

# Orchestrator 状态对比
python test_files/compare_orchestrator.py

# M3 差异调试
python test_files/debug_m3_difference.py

# 快速检查
python test_files/quick_check.py
```

**测试文档**:
- `TESTING_GUIDE.md`: 测试指南
- `DATA_COMPARISON_TOOLS_GUIDE.md`: 对比工具使用指南
- `Data_Type.md`: 数据类型规范
- `Python_former.md`: Python 编码规范

## 依赖关系

### 推荐的导入方向

```
CLI Layer (run.py)
    ↓
Core Layer (orchestrator, main_integration, run.py)
    ├→ Modules Layer (module1-6 及其子模块)
    ├→ Utils Layer (validators, loggers, time_manager 等)
    ├→ Services Layer (generators, profilers)
    └→ Cython Kernels (性能优化内核)

Core Layer → Database Layer (pgsql_db/*)
    ├→ DuckDB Processing (duckdb_*.py)
    ├→ PostgreSQL (db_*.py, table_*.py)
    └→ Excel Import/Export (excel_importer.py, module_data_writer.py)

Modules ↔ Utils (可相互调用)
Modules → Cython Kernels (调用优化内核)
Utils → Services (可单向依赖)
Tests → 所有层 (可导入测试)
Tools → 独立运行 (最小依赖)
```

### 避免的依赖

```
❌ Tools → Core (工具应独立于核心代码)
❌ Modules → Modules (模块间通过 Orchestrator 交互，不直接依赖)
❌ Services → Modules (服务应保持通用化，不依赖具体模块)
❌ Cython Kernels → Python Modules (内核应是纯计算，不依赖业务逻辑)
❌ 循环依赖 (任何方向都不允许)
```

### 性能优化策略

**三层优化体系**:
```
1. Python 层（业务逻辑）
   - 清晰的业务代码
   - 易于维护和扩展
   ↓
2. Cython 层（计算内核）
   - 数值密集计算
   - 循环优化
   - OpenMP 并行
   ↓
3. DuckDB 层（数据处理）
   - 大规模数据聚合
   - SQL 分析查询
   - 列式存储优化
```

**何时使用何种优化**:
- **Pandas**: 小规模数据（< 10万行），原型开发
- **DuckDB**: 大规模数据分析（> 10万行），复杂聚合查询
- **Cython**: 数值密集计算，嵌套循环，实时性要求高
- **PostgreSQL**: 数据持久化，事务一致性，多用户并发

## 数据流

### 日内数据流

```
历史数据 (订单、库存、参数)
    ↓
[Module1] 库存规划 → 需求计划
    ↓
[Orchestrator] 状态更新 → 物理库存快照
    ↓
[Module4] 生产排程 → 生产计划 + 收货
    ↓
[Orchestrator] 状态更新
    ↓
[Module5] 库存优化 → 安全库存建议
    ↓
[Orchestrator] 状态更新
    ↓
[Module6] 物流执行 → 出货执行
    ↓
[Orchestrator] 状态更新
    ↓
[Module3] 交付规划 → 发货计划
    ↓
[Orchestrator] 状态更新 → 日度快照
    ↓
[日输出] 各模块输出文件、日志、指标
```

### 跨日数据流（断点续跑）

```
Last Complete Date ← Orchestrator Snapshot
    ↓
Load Snapshot (T-1)
    ↓
Resume from Date T
    ↓
M1 → M4 → M5 → M6 → M3 (for T...End)
    ↓
生成增量报告
```

## 状态管理策略

### Orchestrator 的三层缓存

1. **内存缓存** (当前日期的状态)
   - 快速访问
   - 各模块更新

2. **快照** (日级持久化)
   - 保存到磁盘
   - 用于续跑

3. **完整历史** (可选)
   - 存档
   - 分析查询

## 扩展点

### 添加新模块

1. 在 `src/modules/` 创建新文件
2. 实现标准接口
3. 在 `src/modules/__init__.py` 导出
4. 在 `main_integration.py` 的日循环中调用

### 添加新验证器

1. 在 `src/utils/` 创建新文件
2. 继承 `ValidationManager` 或创建新类
3. 在 `src/utils/__init__.py` 导出
4. 在需要的地方导入使用

### 添加新报告

1. 在 `src/services/` 创建新文件
2. 遵循 `*Generator` 的命名约定
3. 实现 `generate()` 方法
4. 从 `main_integration.py` 调用

## 配置管理

### 配置来源（优先级从高到低）

1. 命令行参数 (`--start-date`, `--end-date`, etc.)
2. 配置文件 (`config.xlsx`)
3. 环境变量 (可选)
4. 默认常量 (代码中)

### 配置验证

所有配置在执行前需通过 `config_validator.run_pre_simulation_validation()`：
- 检查必要字段
- 标准化标识符
- 验证日期范围
- 去重处理

## 错误处理

### 错误分类

1. **配置错误** → 启动时fail
2. **数据错误** → 记录日志，可选跳过
3. **逻辑错误** → 生成错误报告，可选中止
4. **系统错误** → 保存状态后fail

### 恢复策略

- **配置错误**: 修复配置重新运行
- **数据错误**: 修复数据重新运行
- **逻辑错误**: 使用 `--check-resume` 检查状态
- **系统错误**: 使用 `--resume` 自动续跑

## 性能考虑

### 优化点

1. **并行化** (module1.py中)
   - AO消耗的并行分组计算
   - 文件读取并行化

2. **缓存** (orchestrator.py中)
   - 库存快照缓存
   - 配置参数缓存

3. **增量处理** (main_integration.py中)
   - 只处理新日期
   - 增量更新状态

### 监控指标

通过 `PerformanceProfiler` 收集：
- 模块执行时间
- 内存使用
- I/O操作
- 瓶颈识别

## 部署建议

### 开发环境

```bash
pip install -r config/requirements.txt
python run.py --config config/sample.xlsx --start-date 2024-01-01 --end-date 2024-01-31
```

### 生产环境

```bash
# 使用系统级Python或虚拟环境
python -m venv venv
venv/Scripts/activate
pip install -r config/requirements.txt

# 使用非交互模式
python run.py --config config/prod.xlsx --end-date 2024-12-31 --resume --non-interactive

# 使用cron/scheduler定期运行
```

### Docker化（可选）

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r config/requirements.txt
ENTRYPOINT ["python", "run.py"]
```

---

**版本**: 1.0.0  
**最后更新**: 2026-01-05
