# ChainSight 架构设计文档

## 概述

ChainSight是一个生产级的供应链规划仿真系统，采用标准分层架构设计，确保代码的可维护性、可扩展性和可测试性。

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
Check Resume Capability
    ↓
FOR each_day:
    │
    ├─ M1 (Inventory Planning)
    ├─ Orchestrator.update_state()
    │
    ├─ M4 (Production Scheduling)
    ├─ Orchestrator.update_state()
    │
    ├─ M5 (Inventory Optimization)
    ├─ Orchestrator.update_state()
    │
    ├─ M6 (Logistics Execution)
    ├─ Orchestrator.update_state()
    │
    ├─ M3 (Delivery Planning)
    ├─ Orchestrator.update_state()
    │
    └─ Generate Daily Summary & Snapshot
    
Generate Final Report & Validation
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

| 模块 | 职责 | 输入 | 输出 |
|-----|------|------|------|
| M1 | 库存需求规划 | 订单、库存、BOM | 需求计划、采购需求 |
| M3 | 交付规划 | 已承诺订单、库存、运输 | 发货计划 |
| M4 | 生产排程 | 生产需求、产能、物料 | 生产排程、生产收货 |
| M5 | 库存优化 | 库存目标、库存成本 | 优化建议、安全库存 |
| M6 | 物流执行 | 采购订单、交付 | 入库计划、发货执行 |

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

#### 5.2 Performance Profiler

**职责**:
- 性能分析和监测
- 识别瓶颈
- 性能优化建议

### 6. Tests 层 (`tests/`)

自动化测试框架。

**组织**:
```
tests/
├── unit/              # 单元测试（模块级）
├── integration/       # 集成测试（跨模块）
└── conftest.py        # pytest配置和fixtures
```

## 依赖关系

### 推荐的导入方向

```
CLI Layer (run.py)
    ↓
Core Layer (orchestrator, main_integration, run.py)
    ├→ Modules Layer (module1-6)
    ├→ Utils Layer (validators, loggers, etc.)
    └→ Services Layer (generators, profilers)

Modules ↔ Utils (可相互调用)
Utils → Services (可单向依赖)
Tests → 所有层 (可导入测试)
```

### 避免的依赖

```
❌ Tools → Core (工具是独立脚本)
❌ Modules → Modules (模块通过Orchestrator交互)
❌ Services → Modules (服务应通用化)
❌ 循环依赖 (任何方向)
```

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
