# ChainSight 架构可视化图

**版本**: 2.1.0  
**最后更新**: 2026-01-29

## 1. 整体架构分层图

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer (Entry)                        │
│                            run.py                                │
│                     (命令行参数解析与调度)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Core Layer (核心层)                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ main_integration │  │  orchestrator    │  │  run.py       │ │
│  │  (主编排调度器)   │  │  (全局状态管理)   │  │  (CLI管理)    │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐
│  Modules Layer   │  │ Utils Layer  │  │ Services Layer   │
│   (业务模块层)    │  │  (工具层)    │  │   (服务层)        │
├──────────────────┤  ├──────────────┤  ├──────────────────┤
│ • module1 (M1)   │  │ • config     │  │ • report_gen     │
│ • module3 (M3)   │  │ • logger     │  │ • profiler       │
│ • module4 (M4)   │  │ • validator  │  │                  │
│ • module5 (M5)   │  │ • time_mgr   │  │                  │
│ • module6 (M6)   │  │ • inv_check  │  │                  │
└──────────────────┘  └──────────────┘  └──────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │                                     │
          ▼                                     ▼
┌──────────────────────────┐      ┌──────────────────────────┐
│  Performance Layer       │      │  Database Layer          │
│   (性能优化层)            │      │   (数据库层)              │
├──────────────────────────┤      ├──────────────────────────┤
│ • Cython Kernels         │      │ • PostgreSQL             │
│   - production_kernels   │      │   - db_connection        │
│   - logistics_kernels    │      │   - db_initializer       │
│   - aggregation_kernels  │      │   - table_schemas        │
│                          │      │                          │
│ • DuckDB Processing      │      │ • Excel I/O              │
│   - duckdb_processor     │      │   - excel_importer       │
│   - duckdb_integration   │      │   - module_data_writer   │
│   - optimized_processor  │      │                          │
└──────────────────────────┘      └──────────────────────────┘
```

## 2. 模块执行流程图

```
┌─────────── 日循环开始 ───────────┐
│                                 │
│   FOR each day in [start, end]  │
│                                 │
└────────────┬────────────────────┘
             │
             ▼
    ┌────────────────┐
    │   Module 1     │ ◄───────── 需求规划 (Demand Planning)
    │ Demand Planning│
    └────────┬───────┘
             │ 输出: DemandForecast, OrderLog, ShipmentLog
             ▼
    ┌────────────────┐
    │  Orchestrator  │ ◄───────── 更新全局状态
    │ update_state() │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │   Module 4     │ ◄───────── 生产计划 (Production Planning)
    │ Production Plan│
    └────────┬───────┘
             │ 输出: ProductionPlan, ProductionReceipt
             ▼
    ┌────────────────┐
    │  Orchestrator  │ ◄───────── 更新库存状态
    │ update_state() │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │   Module 5     │ ◄───────── 部署规划 (Deployment Planning)
    │ Deployment Plan│
    └────────┬───────┘
             │ 输出: DeploymentPlan, InventoryProjection
             ▼
    ┌────────────────┐
    │  Orchestrator  │ ◄───────── 更新部署状态
    │ update_state() │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │   Module 6     │ ◄───────── 物流执行 (Logistics Execution)
    │ Logistics Exec │
    └────────┬───────┘
             │ 输出: DeliveryPlan, TransportLog
             ▼
    ┌────────────────┐
    │  Orchestrator  │ ◄───────── 更新在途库存
    │ update_state() │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │   Module 3     │ ◄───────── MRP 计划 (MRP Planning)
    │   MRP Planning │
    └────────┬───────┘
             │ 输出: NetDemand, SuggestedPO
             ▼
    ┌────────────────┐
    │  Orchestrator  │ ◄───────── 最终状态更新
    │ update_state() │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │  Daily Summary │ ◄───────── 生成每日汇总
    │   & Snapshot   │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │  Next Day      │ ◄───────── 继续下一天
    │  or End        │
    └────────────────┘
```

## 3. 数据流向图

```
┌──────────────┐
│ Config Files │ ◄────── Excel / PostgreSQL 配置
│  (.xlsx)     │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│      Config Validator & Loader       │
│   (验证、标准化、加载配置数据)          │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│         Main Integration Loop        │
│       (日循环主控制逻辑)               │
└──────┬───────────────────────────────┘
       │
       ├─────► Module 1 ────► OrderLog, DemandForecast
       │                             │
       │                             ▼
       │                    ┌────────────────┐
       │                    │  Orchestrator  │
       │                    │ (全局状态管理)  │
       │                    └────────────────┘
       │                             │
       ├─────► Module 4 ────► ProductionPlan, ProductionReceipt
       │                             │
       │                             ▼
       │                    ┌────────────────┐
       │                    │  Orchestrator  │
       │                    └────────────────┘
       │                             │
       ├─────► Module 5 ────► DeploymentPlan, PushPlan
       │                             │
       │                             ▼
       │                    ┌────────────────┐
       │                    │  Orchestrator  │
       │                    └────────────────┘
       │                             │
       ├─────► Module 6 ────► DeliveryPlan, TransportLog
       │                             │
       │                             ▼
       │                    ┌────────────────┐
       │                    │  Orchestrator  │
       │                    └────────────────┘
       │                             │
       └─────► Module 3 ────► NetDemand, SuggestedPO
                                     │
                                     ▼
                            ┌────────────────┐
                            │  Orchestrator  │
                            │ (最终状态快照)  │
                            └────────┬───────┘
                                     │
                                     ▼
                            ┌────────────────┐
                            │ Summary Report │
                            │   Generator    │
                            └────────┬───────┘
                                     │
       ┌─────────────────────────────┼─────────────────────────────┐
       │                             │                             │
       ▼                             ▼                             ▼
┌─────────────┐           ┌─────────────────┐         ┌─────────────┐
│ Excel Files │           │   PostgreSQL    │         │   Reports   │
│  (本地模式)  │           │   (数据库模式)   │         │  & Metrics  │
└─────────────┘           └─────────────────┘         └─────────────┘
```

## 4. 性能优化架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      业务逻辑层 (Python)                         │
│                   清晰、可维护、易扩展                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Pandas     │  │   Cython     │  │   DuckDB     │
│  (原型开发)   │  │  (计算加速)   │  │  (数据处理)   │
├──────────────┤  ├──────────────┤  ├──────────────┤
│ • 小数据量    │  │ • 数值密集   │  │ • 大数据量    │
│ • 灵活便捷    │  │ • 嵌套循环   │  │ • SQL查询     │
│ • 快速迭代    │  │ • OpenMP并行 │  │ • 列式存储    │
│              │  │ • 50-100x 🚀 │  │ • 10-100x 🚀  │
└──────────────┘  └──────────────┘  └──────────────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │  PostgreSQL DB   │
              │   (数据持久化)    │
              └──────────────────┘
```

## 5. 配置与输出结构

```
project_root/
│
├── config/                    ◄── 配置输入
│   ├── BC_S5.xlsx                 (本地文件模式)
│   ├── BC_S9.xlsx
│   └── OC_Paste_S1_20251224/      (数据库配置名)
│
├── src/                       ◄── 源代码
│   ├── core/                      (核心引擎)
│   ├── modules/                   (业务模块)
│   ├── cython_kernels/            (Cython优化)
│   ├── utils/                     (工具函数)
│   └── services/                  (高级服务)
│
├── pgsql_db/                  ◄── 数据库层
│   ├── db_*.py                    (PostgreSQL)
│   ├── duckdb_*.py                (DuckDB处理)
│   └── table_*.py                 (表结构)
│
└── outputs/                   ◄── 运行输出
    ├── {config_name}/             (本地文件模式)
    │   └── run_YYYYMMDD_HHMMSS/
    │       ├── module1/
    │       ├── module3/
    │       ├── module4/
    │       ├── module5/
    │       ├── module6/
    │       ├── orchestrator/
    │       ├── summary/
    │       └── performance/
    │
    └── db_config/                 (数据库模式)
        └── {config}_YYYYMMDD_HHMMSS/
            └── simulation_log.txt
```

## 6. 关键设计模式

### 6.1 中央集权模式 (Orchestrator)
```
所有模块 ──► Orchestrator ◄── 查询全局状态
             (单一真相源)
                  │
                  └──► 维护库存、在途、生产等全局状态
```

### 6.2 编排器模式 (Main Integration)
```
Main Integration
     │
     ├──► 按序调度各模块
     ├──► 管理模块间数据流
     ├──► 处理异常和重试
     └──► 生成汇总报告
```

### 6.3 策略模式 (运行模式)
```
CLI ──► 选择运行策略
         │
         ├──► 本地文件模式 (Excel I/O)
         │
         └──► 数据库模式 (PostgreSQL I/O)
```

## 7. 技术栈选型原则

| 场景 | 技术选型 | 原因 |
|------|---------|------|
| 配置管理 | Excel / PostgreSQL | 业务人员熟悉，易于维护 |
| 数据处理 | pandas → DuckDB | 从原型到生产，性能递增 |
| 计算加速 | Python → Cython | 热点优化，保持可读性 |
| 数据持久化 | PostgreSQL | 成熟、可靠、事务支持 |
| 日志管理 | logging | Python标准库，功能完善 |
| 并行处理 | ThreadPoolExecutor / OpenMP | 轻量级并行，适合I/O和CPU密集 |

---

**维护说明**:  
本文档应与实际代码保持同步，每次架构调整后及时更新。

**参考文档**:
- [ARCHITECTURE.md](ARCHITECTURE.md) - 详细架构设计
- [README_CN.md](../README_CN.md) - 项目说明
- [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) - 优化总结
