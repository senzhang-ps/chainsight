# ChainSight - 供应链规划仿真系统

**中文** | [English](README_EN.md)

## 📋 目录

- [环境要求](#-环境要求)
- [快速开始](#-快速开始)
- [运行模式](#-运行模式)
- [项目架构](#-项目架构)
- [模块说明](#-模块说明)
- [数据库配置](#-数据库配置)
- [故障排除](#-故障排除)

---

## 💻 环境要求

- **Python**: 3.12（推荐使用 `.venv` 虚拟环境，已在 3.12.9 验证）
- **操作系统**: Windows / Linux / macOS
- **数据库** (可选): PostgreSQL 14+ (用于数据库模式)

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd chainsight
```

### 2. 创建虚拟环境

**Windows (PowerShell):**
```powershell
# 使用 Python 3.12 创建虚拟环境
py -3.12 -m venv .venv

# 激活虚拟环境
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force; . .\.venv\Scripts\Activate.ps1
--- 
.\.venv\Scripts\Activate.ps1
--- 
```

**Windows (CMD):**
```cmd
# 使用 Python 3.12 创建虚拟环境
py -3.12 -m venv .venv

# 激活虚拟环境
.\.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
# 使用 Python 3.12 创建虚拟环境
python3.12 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate
```

### 3. 安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt

# 编译 Cython 扩展模块（可选，用于性能优化）
python setup.py build_ext --inplace
```

**核心依赖说明:**
| 依赖 | 版本 | 用途 |
|------|------|------|
| pandas | 3.0.1 | 数据处理 |
| openpyxl | 3.1.5 | Excel 读写 |
| xlsxwriter | 3.2.9 | Excel 写入（M6 输出）|
| duckdb | 1.4.4 | 高性能数据处理 |
| psycopg[binary] | 3.3.3 | PostgreSQL 连接 |
| numpy | 2.4.2 | 数值计算 |
| scipy | 1.17.1 | 统计计算 |
| matplotlib | 3.10.8 | 图表生成 |

### 4. 运行仿真

**本地文件模式 (默认):**
```bash
# 首次运行（需指定起始日期）
python run.py --config config/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10

# 或使用测试配置
python run.py --config test_files/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10

# 续跑模式
python run.py --config config/BC_S5.xlsx --end-date 2025-10-15 --resume
```

**数据库模式:**
```bash
# 使用数据库读写配置和输出
python run.py --config config/OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2026-02-28 --use-db

# 指定数据库参数
python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-10 \
  --use-db --db-host localhost --db-port 5432 --db-name test_db \
  --db-user postgres --db-password 123456
```

### 5. 导出表映射关系

如果您需要查看 Excel 配置表、模块输出与数据库表名之间的详细对应关系，可以运行以下命令生成映射表：

```bash
python tools/export_mapping.py
```
该命令会在项目根目录下生成 `database_table_mapping.xlsx` 文件。

---

## 🔄 运行模式

### 本地文件模式

- 从 Excel 配置文件读取配置
- 输出保存到本地文件系统
- 适合开发和测试

```bash
python run.py --config test_files/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10
```

### 数据库模式 (`--use-db`)

- 从 PostgreSQL 读取配置
- 输出写入 PostgreSQL 数据库
- 自动检测数据库是否存在，不存在则创建
- 自动检测配置表是否存在，不存在则从 Excel 导入
- 适合生产环境和数据持久化

```bash
python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-10 --use-db
```

**数据库自动初始化流程:**
```
🔍 检测数据库 'test_db'...
   └─ 不存在 → ✅ 自动创建数据库
🔍 测试数据库连接...
   └─ ✅ 连接成功
🔍 检测配置表 'BC_S5'...
   └─ 不存在 → 📁 查找 Excel 文件 → ✅ 自动导入配置表
```

---

## 📦 项目架构

该项目已按标准分层架构进行重组织，便于维护和扩展。

### 目录结构

```
chainsight_cpython/
├── run.py                              # 🚀 主入口 CLI
├── run.ps1                             # PowerShell 运行脚本  
├── setup.py                            # Cython 扩展编译配置
├── requirements.txt                    # 依赖声明
├── README_CN.md / README_EN.md         # 中英文文档
├── CYTHON_OPTIMIZATION_REPORT.md       # Cython 优化报告
│
├── src/                                # 📦 核心源代码包
│   ├── core/                           #    编排引擎
│   │   ├── main_integration.py         #    主集成调度器（日循环 M1→M4→M5→M6→M3）
│   │   ├── orchestrator.py             #    统一状态管理中枢
│   │   ├── parallel_executor.py        #    并行执行框架（ThreadPoolExecutor）
│   │   └── run.py                      #    CLI 解析（被根 run.py 调用）
│   │
│   ├── modules/                        #    业务模块层
│   │   ├── module1.py                  #    M1: 需求规划 (Demand Planning)
│   │   ├── module3.py                  #    M3: MRP计划 (MRP Planning)
│   │   ├── module4.py                  #    M4: 生产计划 (Production Planning)
│   │   ├── module5.py                  #    M5: 部署规划 (Deployment Planning)
│   │   ├── module6.py                  #    M6: 物流执行 (Logistics Execution)
│   │   ├── demand_planning/            #    M1 子模块（forecast, order, shipment）
│   │   ├── mrp_planning/               #    M3 子模块（net_demand, mrp_simulation）
│   │   ├── production_planning/        #    M4 子模块（plan_builder, capacity_allocator）
│   │   ├── deployment_planning/        #    M5 子模块（allocation, inventory, push）
│   │   └── logistics_execution/        #    M6 子模块（vehicle_packer, delivery）
│   │
│   ├── cython_kernels/                 #    Cython 性能优化内核（编译后生成 .pyd/.so）
│   │   ├── production_kernels.pyx      #    生产计算内核
│   │   ├── logistics_kernels.pyx       #    物流计算内核
│   │   └── aggregation_kernels.pyx     #    聚合计算内核
│   │
│   ├── utils/                          #    工具库
│   │   ├── config_validator.py         #    配置校验
│   │   ├── logger_config.py            #    日志配置
│   │   ├── validation_manager.py       #    数据验证
│   │   ├── inventory_balance_checker.py#    库存平衡检查
│   │   └── time_manager.py             #    时间管理
│   │
│   └── services/                       #    业务服务
│       ├── summary_report_generator.py #    汇总报告生成
│       └── performance_profiler.py     #    性能分析器
│
├── pgsql_db/                           # 🗄️ 数据库支持模块
│   ├── db_connection.py                #    连接管理
│   ├── db_initializer.py               #    数据库初始化
│   ├── excel_importer.py               #    Excel 导入
│   ├── module_data_writer.py           #    模块输出写入
│   ├── table_mapping.py                #    表名映射
│   ├── table_schemas.py                #    表结构定义
│   ├── duckdb_processor.py             #    DuckDB 高性能处理
│   ├── duckdb_integration.py           #    DuckDB 集成
│   ├── optimized_processor.py          #    优化处理器
│   ├── optimized_simulation.py         #    优化仿真引擎
│   ├── high_performance_engine.py      #    高性能执行引擎
│   ├── data_pipeline.py                #    数据管道（DuckDB + PostgreSQL）
│   ├── module_engine.py                #    模块执行引擎
│   ├── module_optimizers.py            #    模块优化器
│   ├── incremental_processor.py        #    增量处理器
│   ├── test_engine.py                  #    测试引擎
│   ├── test_optimization_validation.py #    优化验证测试
│   └── performance_dashboard.py        #    性能监控面板
│
├── tools/                              # 🔧 辅助工具脚本
│   ├── init_database.py                #    数据库初始化
│   ├── compare_outputs.py              #    输出对比工具
│   ├── compare_all_versions.py         #    版本对比工具
│   ├── benchmark_duckdb_vs_pandas.py   #    性能基准测试
│   ├── performance_benchmark.py        #    性能基准测试
│   ├── generate_docx_report.py         #    生成 Word 报告
│   ├── generate_report_charts.py       #    生成报告图表
│   └── migrate_config_tables.py        #    配置表迁移工具
│
├── tests/                              # 🧪 测试模块
│   ├── e2e_integration_test.py         #    端到端集成测试
│   └── test_logger.py                  #    日志测试
│
├── test_files/                         # 📋 测试数据与对比工具
│   ├── BC_S5.xlsx                      #    主测试配置
│   ├── BC_S9.xlsx                      #    备用测试配置
│   ├── compare_all_outputs.py          #    输出对比工具（主工具）
│   ├── compare_db_vs_local.py          #    数据库与本地对比
│   ├── compare_db_vs_chainsight_dev.py #    与 ChainSight_Dev 对比
│   ├── compare_dev_refactored.py       #    开发版与重构版对比
│   ├── compare_orchestrator.py         #    Orchestrator 状态对比
│   ├── compare_outputs_detail.py       #    详细输出对比
│   ├── debug_m3_difference.py          #    M3 差异调试工具
│   ├── quick_check.py                  #    快速检查工具
│   ├── test_config_consistency.py      #    配置一致性测试
│   ├── test_duckdb_performance.py      #    DuckDB 性能测试
│   ├── TESTING_GUIDE.md                #    测试指南
│   ├── DATA_COMPARISON_TOOLS_GUIDE.md  #    对比工具指南
│   ├── Data_Type.md                    #    类型说明
│   └── Python_former.md                #    Python 编码规范
│
├── config/                             # ⚙️ 配置文件
│   ├── BC_S5.xlsx                      #    测试配置 S5
│   ├── BC_S9.xlsx                      #    测试配置 S9
│   ├── OC_Paste_S1_20251224/           #    生产配置 OC Paste S1
│   ├── ChainSight 1st SIT.xlsx         #    SIT 样例配置
│   └── config_guide.xlsx               #    配置指南
│
├── docs/                               # 📚 设计文档
│   ├── ARCHITECTURE.md                 #    架构设计
│   ├── MODULE*_DESIGN.md               #    模块设计文档（M1, M3, M5）
│   ├── OPTIMIZATION_*.md               #    优化相关文档
│   ├── MIGRATION.md                    #    迁移指南
│   ├── REFACTORING_*.md                #    重构相关文档
│   ├── PERFORMANCE_*.md                #    性能优化报告
│   ├── DUCKDB_OPTIMIZATION_GUIDE.md    #    DuckDB 优化指南
│   ├── 算法优化测试报告.md              #    算法优化测试报告
│   └── 20260127变更版本与修复.md        #    最新变更记录
│
├── ChainSight_Dev/                     # 🔬 开发版本（保留用于对比和参考）
│   ├── module*.py                      #    开发版模块实现
│   ├── orchestrator.py                 #    开发版编排器
│   ├── main_integration.py             #    开发版主集成
│   ├── BC_S5/                          #    开发版测试输出
│   └── *.md                            #    开发文档
│
├── build/                              # 🏗️ Cython 编译输出（自动生成）
│   ├── lib.win-amd64-cpython-314/      #    编译的库文件
│   └── temp.win-amd64-cpython-314/     #    临时编译文件
│
└── outputs/                            # 📤 运行输出（自动生成，已加入 .gitignore）
    ├── {config_name}/                  #    本地文件模式输出
    │   └── run_YYYYMMDD_HHMMSS/        #    按运行时间戳组织
    └── db_config/                      #    数据库模式输出
        └── {config_name}_YYYYMMDD_HHMMSS/  #    按配置和时间戳组织
```

### CLI 参数说明

| 参数 | 必需 | 说明 |
|------|------|------|
| `--config` | ✅ | 配置文件路径（文件模式）或配置名称（数据库模式） |
| `--start-date` | 首次运行 | 仿真开始日期 (YYYY-MM-DD) |
| `--end-date` | ✅ | 仿真结束日期 (YYYY-MM-DD) |
| `--use-db` | ❌ | 启用数据库模式 |
| `--db-host` | ❌ | 数据库主机 (默认: localhost) |
| `--db-port` | ❌ | 数据库端口 (默认: 5432) |
| `--db-name` | ❌ | 数据库名称 (默认: test_db) |
| `--db-user` | ❌ | 数据库用户 (默认: postgres) |
| `--db-password` | ❌ | 数据库密码 (默认: 123456) |
| `--resume` | ❌ | 启用断点续跑 |
| `--force-restart` | ❌ | 强制重新开始 |
| `--list-runs` | ❌ | 列出可用的运行目录 |

---

## 📤 输出目录说明

### outputs/ 目录结构

所有仿真运行输出统一管理在 `outputs/` 目录中：

```
outputs/
├── BC_S5/                              # 本地文件模式输出
│   └── run_YYYYMMDD_HHMMSS/           # 单次运行目录（时间戳格式）
│       ├── module1/                    # M1: 需求规划输出
│       │   ├── DemandForecast.xlsx
│       │   ├── OrderLog.xlsx
│       │   └── ShipmentLog.xlsx
│       ├── module3/                    # M3: MRP计划输出
│       │   ├── NetDemand.xlsx
│       │   └── SuggestedPO.xlsx
│       ├── module4/                    # M4: 生产计划输出
│       │   ├── ProductionPlan.xlsx
│       │   └── CapacityUtilization.xlsx
│       ├── module5/                    # M5: 部署规划输出
│       │   ├── DeploymentPlan.xlsx
│       │   └── InventoryProjection.xlsx
│       ├── module6/                    # M6: 物流执行输出
│       │   ├── DeliveryPlan.xlsx
│       │   └── TransportLog.xlsx
│       ├── orchestrator/               # 状态管理输出
│       │   ├── PhysicalInventory.xlsx
│       │   ├── InTransitInventory.xlsx
│       │   └── daily_logs/
│       ├── summary/                    # 汇总报告
│       │   ├── HistoricalInventoryRecord.xlsx
│       │   └── KPISummary.xlsx
│       ├── performance/                # 性能分析
│       │   └── performance_report.txt
│       └── validation_report.txt       # 数据一致性验证
│
└── db_config/                          # 数据库模式输出（--use-db）
    └── {config_name}_YYYYMMDD_HHMMSS/  # 按配置和时间戳组织
        ├── simulation_log_YYYYMMDD_HHMMSS.txt  # 运行日志
        └── [其他日志文件]
```

**注意事项:**
- 本地文件模式：输出保存为 Excel 文件，便于查看和分析
- 数据库模式：数据写入 PostgreSQL，日志保存在 outputs/db_config/ 目录
- 所有输出目录都已加入 .gitignore，不会提交到版本控制

---

## 🗄️ 数据库配置

### PostgreSQL 安装

**Windows:**
1. 下载 PostgreSQL: https://www.postgresql.org/download/windows/
2. 安装时记住设置的密码
3. 默认端口: 5432

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

### 数据库模块 (pgsql_db)

项目包含完整的 PostgreSQL 数据库支持模块：

```python
from pgsql_db import DatabaseInitializer, initialize_database

# 方式1: 使用便捷函数
result = initialize_database(
    config_name='BC_S5',
    database='test_db',
    auto_import=True
)

# 方式2: 使用初始化器类
initializer = DatabaseInitializer(database='test_db')
result = initializer.initialize(config_name='BC_S5')
print(initializer.get_status_report('BC_S5'))
```

### 数据库表结构

#### 配置表命名规则（统一表名）

配置表采用"同结构同表"规则，所有配置文件的相同类型数据存储在同一张表中，通过 `config_name` 字段区分不同的配置：

| 表类型 | 命名格式 | 示例 | 说明 |
|--------|----------|------|------|
| 配置表 | `cfg_*` | `cfg_m1_demandforecast` | 所有配置共用，通过 config_name 区分 |
| Module1输出 | `module1_output_*` | `module1_output_orderlog` | 模块输出表 |
| Module3输出 | `module3_output_*` | `module3_output_netdemand` | 模块输出表 |
| Module4输出 | `module4_output_*` | `module4_output_productionplan` | 模块输出表 |
| Module5输出 | `module5_output_*` | `module5_output_deploymentplan` | 模块输出表 |
| Module6输出 | `module6_output_*` | `module6_output_deliveryplan` | 模块输出表 |
| Orchestrator | `orchestrator_*` | `orchestrator_daily_logs` | 编排器日志表 |
| 汇总报告 | `summary_*` | `summary_historical_inventory_record` | 汇总报告表 |

**配置表特殊说明：**
- 不再为每个配置创建独立的表（如 ~~bc_s5_m1_demandforecast~~, ~~bc_s9_m1_demandforecast~~）
- 所有配置数据写入同一张表（如 `cfg_m1_demandforecast`）
- 通过 `config_name` 字段（如 'BC_S5', 'BC_S9'）区分不同配置的数据
- 这种设计便于跨配置查询和管理

---

## 📁 模块说明

### Core 层 (`src/core/`)

- **orchestrator.py**: 统一的状态管理中枢
  - 管理物理库存、部署计划、在途库存、生产收货、交付收货等全局状态
  - 提供日度粒度的快照和审计日志

- **main_integration.py**: 主集成编排器
  - 实现日循环执行：M1 → M4 → M5 → M6 → M3
  - 断点续跑能力
  - 数据一致性验证

- **run.py**: CLI管理 (被root level的run.py调用)

### Modules 层 (`src/modules/`)

5个业务模块，按供应链流程组织：

| 模块 | 入口文件 | 子包 | 说明 |
|------|---------|------|------|
| **M1** | module1.py | `demand_planning/` | 需求规划 (Demand Planning) |
| **M3** | module3.py | `mrp_planning/` | MRP计划 (MRP Planning) |
| **M4** | module4.py | `production_planning/` | 生产计划 (Production Planning) |
| **M5** | module5.py | `deployment_planning/` | 部署规划 (Deployment Planning) |
| **M6** | module6.py | `logistics_execution/` | 物流执行 (Logistics Execution) |

> 每个模块的入口文件 (moduleX.py) 作为向后兼容层，实际业务逻辑位于对应子包中。

### Utils 层 (`src/utils/`)

通用工具和验证器：
- 配置验证
- 日志管理
- 数据验证
- 库存平衡检查
- 时间管理

### Services 层 (`src/services/`)

高级业务服务：
- 汇总报告生成
- 性能分析和优化建议

## 🔄 数据流

```
CLI (run.py)
  ↓
main_integration.run_integrated_simulation()
  ├→ 加载并验证配置
  ├→ 检查断点续跑能力
  ├→ FOR each_day in [start_date, end_date]:
  │   ├→ Module1 (需求规划)
  │   ├→ Orchestrator.update_state()
  │   ├→ Module4 (生产计划)
  │   ├→ Orchestrator.update_state()
  │   ├→ Module5 (部署规划)
  │   ├→ Orchestrator.update_state()
  │   ├→ Module6 (物流执行)
  │   ├→ Orchestrator.update_state()
  │   ├→ Module3 (MRP计划)
  │   ├→ Orchestrator.update_state()
  │   └→ 生成每日汇总与快照
  └→ 生成最终报告与一致性检查
```

## 🔧 配置管理

配置文件位于 `config/` 目录：
- 所有Excel配置文件 (.xlsx)
- 依赖声明 (requirements.txt)
- 可选：config.yaml (待添加)

## 📊 输出目录

每次运行生成一个唯一的输出目录：
```
outputs/runs/
└── run_20240115_143022/
    ├── module1/          # 各模块输出
    ├── module3/
    ├── module4/
    ├── module5/
    ├── module6/
    ├── orchestrator/     # 中枢状态输出
    ├── summary/          # 汇总报告
    ├── logs/             # 日志文件
    └── validation_report.txt
```

## 🧪 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/e2e_integration_test.py

# 显示详细输出
pytest -v tests/
```

## 📚 文档

详细文档位于 `docs/` 目录：

### 核心文档
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - 完整架构设计文档
- [ARCHITECTURE_DIAGRAM.md](docs/ARCHITECTURE_DIAGRAM.md) - 架构可视化图谱
- [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - 快速参考指南（命令速查）

### 模块设计
- [MODULE1_DESIGN.md](docs/MODULE1_DESIGN.md) - M1: 需求规划模块设计
- [MODULE3_DESIGN.md](docs/MODULE3_DESIGN.md) - M3: MRP 计划模块设计
- [MODULE5_DESIGN.md](docs/MODULE5_DESIGN.md) - M5: 部署规划模块设计

### 优化文档
- [OPTIMIZATION_SUMMARY.md](docs/OPTIMIZATION_SUMMARY.md) - 优化总结
- [DUCKDB_OPTIMIZATION_GUIDE.md](docs/DUCKDB_OPTIMIZATION_GUIDE.md) - DuckDB 优化指南
- [CYTHON_OPTIMIZATION_REPORT.md](CYTHON_OPTIMIZATION_REPORT.md) - Cython 优化报告
- [算法优化测试报告.md](docs/算法优化测试报告.md) - 算法优化测试报告

### 其他文档
- [MIGRATION.md](docs/MIGRATION.md) - 版本迁移指南
- [REFACTORING_SUMMARY.md](docs/REFACTORING_SUMMARY.md) - 重构总结
- [20260127变更版本与修复.md](docs/20260127变更版本与修复.md) - 最新变更日志

## 🔗 导入约定

### 内部导入

从任意模块导入其他模块时，使用相对导入：

```python
# 在 src/core/main_integration.py 中
from .orchestrator import create_orchestrator
from ..utils.validation_manager import ValidationManager
from ..modules import module1
from ..services.summary_report_generator import SummaryReportGenerator
```

### 外部导入

从root level脚本导入时：

```python
# 在 run.py 中
from src.core.run import main
from src.core.main_integration import run_integrated_simulation
```

## 🏗️ 架构优势

1. **清晰的关注点分离**: 按功能层分组，易于定位和修改
2. **可维护性**: 明确的依赖关系，减少循环导入
3. **可扩展性**: 新模块/服务可轻松添加到对应目录
4. **可测试性**: 各层独立，便于单元测试和集成测试
5. **生产就绪**: 遵循Python项目最佳实践

## 📝 约定

- 所有模块内导入采用相对导入
- 配置文件统一放在 `config/` 目录
- 输出文件自动组织到 `outputs/` 目录
- 日志同时输出到终端和文件
- 所有Python包都包含 `__init__.py`

## 🐛 故障排除

### ImportError: No module named 'psycopg'

**原因**: 未安装 PostgreSQL 驱动
**解决**:
```bash
pip install psycopg[binary]
```

### ImportError: No module named 'module1'

**原因**: 在项目外直接运行代码
**解决**: 确保从root目录运行，使用 `python run.py` 或 `sys.path.insert(0, '.')`

### 找不到配置文件

**原因**: 路径相对于当前工作目录
**解决**: 使用绝对路径或从项目root目录运行

### 数据库连接失败

**原因**: PostgreSQL 未启动或连接参数错误
**解决**:
```bash
# 检查 PostgreSQL 是否运行
# Windows
net start postgresql-x64-14

# Linux
sudo systemctl status postgresql

# 测试连接
python -c "from pgsql_db import DatabaseConnection; db = DatabaseConnection(); print(db.test_connection())"
```

### 断点续跑不工作

**原因**: 运行目录结构不完整
**解决**: 使用 `--check-resume` 检查状态，必要时使用 `--force-restart`

### bigint 类型错误

**原因**: 浮点数写入整数列
**解决**: 已在代码中自动处理，确保使用最新版本

---

## 📞 支持

遇到问题？检查：
1. `docs/` 目录中的设计文档
2. 各模块的代码注释
3. 运行日志（保存在outputs目录中）

---

## 🗂️ 项目维护

### 目录清洁度
- ✅ 根目录仅保留必要文件（run.py, README.md, requirements.txt 等）
- ✅ 临时文档已移至 docs/ 或删除
- ✅ 工具脚本已整理至 tools/
- ✅ 样例配置已归档至 config/
- ✅ 运行输出自动存储在 outputs/ (本地和数据库模式统一)

### 代码风格
- 遵循 [test_files/Python_former.md](test_files/Python_former.md) 编码规范
- 类型说明参考 [test_files/Data_Type.md](test_files/Data_Type.md)

---

**版本**: 2.1.1  
**最后更新**: 2026-03-03

## 📝 更新日志

### v2.1.1 (2026-03-03)
- ✅ **Python 3.12 支持**: 创建 `.venv` 虚拟环境（Python 3.12.9），确认所有依赖兼容
- ✅ **依赖升级**: pandas 3.0.1, duckdb 1.4.4, numpy 2.4.2, scipy 1.17.1
- ✅ **requirements.txt**: 按功能分类，附精确版本号
- ✅ **数据处理**: 集成 DuckDB 高性能数据处理引擎
- ✅ **数据库优化**: 完善统一配置表设计
- ✅ **工具增强**: 增加对比、基准测试、报告生成工具

### v2.0.0 (2026-01-09)
- 🎯 标准化分层架构重构
- 📦 模块化业务逻辑（M1-M6 子包）
- 🗄️ 完整的 PostgreSQL 数据库支持
- 🔄 断点续跑能力
- ✅ 数据一致性验证
