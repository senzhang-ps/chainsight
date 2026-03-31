# ChainSight - Supply Chain Planning Simulation System

[中文版](README_CN.md) | **English**

## 📋 Table of Contents

- [Requirements](#-requirements)
- [Quick Start](#-quick-start)
- [Run Modes](#-run-modes)
- [Project Architecture](#-project-architecture)
- [Module Description](#-module-description)
- [Database Configuration](#-database-configuration)
- [Troubleshooting](#-troubleshooting)

---

## 💻 Requirements

- **Python**: 3.12 (Virtual environment `.venv` recommended, validated on 3.12.9)
- **OS**: Windows / Linux / macOS
- **Database** (Optional): PostgreSQL 14+ (for database mode)

---

## 🚀 Quick Start

### 1. Clone the Project

```bash
git clone <repository-url>
cd chainsight
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
# Create virtual environment with Python 3.12
py -3.12 -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
# Create virtual environment with Python 3.12
py -3.12 -m venv .venv
.\.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
# Create virtual environment with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For current performance and optimization notes, use these repository docs instead:
- `docs/CYTHON_OPTIMIZATION_REPORT.md`
- `docs/DUCKDB_OPTIMIZATION_GUIDE.md`
- `docs/PERFORMANCE_OPTIMIZATION_REPORT.md`

**Core Dependencies:**
| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 3.0.1 | Data processing |
| openpyxl | 3.1.5 | Excel I/O |
| xlsxwriter | 3.2.9 | Excel write (M6 output) |
| duckdb | 1.4.4 | High-performance data processing |
| psycopg[binary] | 3.3.3 | PostgreSQL connection |
| numpy | 2.4.2 | Numerical computation |
| scipy | 1.17.1 | Statistical computation |
| matplotlib | 3.10.8 | Chart generation |
| python-docx | 1.2.0 | Word report generation |
| tqdm | 4.67.3 | Progress display |
| Cython | 3.0+ | Performance optimization (optional) |

### 4. Run Simulation

**Local File Mode (default):**
```bash
# First run
python run.py --config config/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10

# Resume mode
python run.py --config config/BC_S5.xlsx --end-date 2025-10-15 --resume
```

**Database Mode:**
```bash
# Use database for config and output
python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-10 --use-db

# With custom database parameters
python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-10 \
  --use-db --db-host localhost --db-port 5432 --db-name test_db \
  --db-user postgres --db-password 123456
```

### 5. Check Table Mapping and Interfaces

The repository no longer ships a standalone `tools/export_mapping.py` script. For current Excel/config/module-output/database mapping and interface details, use:

- `docs/交接文档/Refactored/函数级接口与文件格式总表.md`
- `docs/交接文档/Refactored/API.md`
- `docs/QUICK_REFERENCE.md`

---

## 🔄 Run Modes

### Local File Mode

- Reads configuration from Excel files
- Outputs saved to local filesystem
- Best for development and testing

```bash
# First run
python run.py --config config/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10

# Resume mode
python run.py --config config/BC_S5.xlsx --end-date 2025-10-15 --resume
```

### Database Mode (`--use-db`)

- Reads configuration from PostgreSQL
- Outputs written to PostgreSQL database
- Auto-detects and creates database if not exists
- Auto-imports config tables from Excel if not found
- Best for production and data persistence

```bash
python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-10 --use-db

# With custom database parameters
python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-10 \
  --use-db --db-host localhost --db-port 5432 --db-name test_db \
  --db-user postgres --db-password 123456
```

**Auto-initialization Flow:**
```
🔍 Detecting database 'test_db'...
   └─ Not found → ✅ Auto-create database
🔍 Testing connection...
   └─ ✅ Connected
🔍 Detecting config tables 'BC_S5'...
   └─ Not found → 📁 Find Excel file → ✅ Auto-import config tables
```

---

## 📦 Project Architecture

The project follows a standard layered architecture for maintainability and extensibility.

### Directory Structure

```
chainsight/
├── run.py                              # 🚀 Main CLI entry point
├── run.ps1                             # PowerShell run script
├── requirements.txt                    # Dependency declaration
├── README_CN.md / README_EN.md         # Chinese/English documentation
│
├── src/                                # 📦 Core source code package
│   ├── core/                           #    Orchestration engine
│   │   ├── main_integration.py /       #    Main integration and run orchestration
│   │   ├── orchestrator.py /           #    State management hub
│   │   ├── parallel_executor.py /      #    Parallel execution support
│   │   └── run.py /                    #    CLI parser
│   │
│   ├── modules/                        #    Business module layer
│   │   ├── module1.py                  #    M1: Demand Planning
│   │   ├── module3.py                  #    M3: MRP Planning
│   │   ├── module4.py                  #    M4: Production Planning
│   │   ├── module5.py                  #    M5: Deployment Planning
│   │   ├── module6.py                  #    M6: Logistics Execution
│   │   ├── demand_planning/            #    M1 submodules (forecast, order, shipment)
│   │   ├── mrp_planning/               #    M3 submodules (net_demand, mrp_simulation)
│   │   ├── production_planning/        #    M4 submodules (plan_builder, capacity_allocator)
│   │   ├── deployment_planning/        #    M5 submodules (allocation, inventory, push)
│   │   └── logistics_execution/        #    M6 submodules (vehicle_packer, delivery)
│   │
│   ├── utils/                          #    Utility library
│   │   ├── config_validator.py         #    Configuration validation
│   │   ├── logger_config.py            #    Logging configuration
│   │   ├── validation_manager.py       #    Data validation
│   │   ├── inventory_balance_checker.py#    Inventory balance checking
│   │   └── time_manager.py             #    Time management
│   │
│   └── services/                       #    Business services
│       ├── summary_report_generator.py #    Summary report generation
│       └── performance_profiler.py     #    Performance profiler
│
├── pgsql_db/                           # 🗄️ Database support module
│   ├── db_connection.py                #    Connection management
│   ├── db_initializer.py               #    Database initialization
│   ├── excel_importer.py               #    Excel import
│   ├── module_data_writer.py           #    Module output writer
│   ├── table_mapping.py                #    Table name mapping
│   ├── table_schemas.py                #    Table schema definitions
│   ├── duckdb_processor.py             #    DuckDB high-performance processing
│   ├── duckdb_integration.py           #    DuckDB integration
│   ├── optimized_processor.py          #    Optimized processor
│   ├── optimized_simulation.py         #    Optimized simulation engine
│   ├── high_performance_engine.py      #    High-performance execution engine
│   ├── data_pipeline.py                #    Data pipeline (DuckDB + PostgreSQL)
│   ├── module_engine.py                #    Module execution engine
│   ├── module_optimizers.py            #    Module optimizers
│   ├── incremental_processor.py        #    Incremental processor
│   └── performance_dashboard.py        #    Performance monitoring dashboard
│
├── config/                             # ⚙️ Configuration files
│   ├── BC_S5.xlsx                      #    Test config S5
│   ├── BC_S9.xlsx                      #    Test config S9
│   ├── OC_Paste_S1_20251224.xlsx       #    OC config sample
│   ├── ChainSight 1st SIT.xlsx         #    SIT sample config
│   ├── database.json                   #    Database connection config
│   └── config_guide.xlsx               #    Configuration guide
│
├── docs/                               # 📚 Design documents
│   ├── INDEX.md                        #    Documentation index
│   ├── 用户使用入口说明.md                 #    Operator entry guide
│   ├── ARCHITECTURE_DIAGRAM.md         #    Architecture diagrams
│   ├── MODULE3_DESIGN.md               #    M3 design doc
│   ├── MODULE5_DESIGN.md               #    M5 design doc
│   ├── QUICK_REFERENCE.md              #    Quick reference
│   ├── 交接文档/Refactored/            #    Canonical handoff package
│   └── _archive/                       #    Archived process/stage docs
│
├── ChainSight_Dev/                     # 🔬 Dev version (for comparison & reference)
│   ├── module*.py                      #    Dev version module implementations
│   ├── orchestrator.py                 #    Dev version orchestrator
│   ├── main_integration.py             #    Dev version main integration
│   ├── BC_S5/                          #    Dev version test outputs
│   └── *.md                            #    Dev documentation
│
└── outputs/                            # 📤 Run outputs (auto-generated, in .gitignore)
    └── {config_stem}/                  #    Organized by config name
        └── run_YYYYMMDD_HHMMSS/        #    Organized by run timestamp
```

### CLI Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--config` | ✅ | Config file path (file mode) or config name (database mode) |
| `--start-date` | First run | Simulation start date (YYYY-MM-DD) |
| `--end-date` | ✅ | Simulation end date (YYYY-MM-DD) |
| `--use-db` | ❌ | Enable database mode |
| `--db-host` | ❌ | Database host (default: localhost) |
| `--db-port` | ❌ | Database port (default: 5432) |
| `--db-name` | ❌ | Database name (default: test_db) |
| `--db-user` | ❌ | Database user (default: postgres) |
| `--db-password` | ❌ | Database password (default: 123456) |
| `--resume` | ❌ | Enable checkpoint resume |
| `--force-restart` | ❌ | Force restart from beginning |
| `--list-runs` | ❌ | List available run directories |

---

## 📤 Output Directory Description

### outputs/ Directory Structure

All simulation run outputs are centrally managed in the `outputs/` directory:

```
outputs/
├── BC_S5/                              # Local simulation outputs (organized by config name)
│   └── run_YYYYMMDD_HHMMSS/           # Single run directory
│       ├── module1/                    # M1 outputs
│       ├── module3/                    # M3 outputs
│       ├── module4/                    # M4 outputs
│       ├── module5/                    # M5 outputs
│       ├── module6/                    # M6 outputs
│       ├── orchestrator/               # State management outputs
│       ├── summary/                    # Summary reports
│       ├── performance/                # Performance analysis
│       └── validation_report.txt       # Data consistency validation
```

For database-mode output and log conventions, use `docs/交接文档/Refactored/setup.md` as the source of truth.

---

## �📁 Module Description

### Core Layer (`src/core/`)

- **orchestrator.py**: Unified state management hub
  - Manages physical inventory, deployment plans, in-transit inventory, production receipts, delivery receipts, etc.
  - Provides daily granularity snapshots and audit logs

- **main_integration.py**: Main integration orchestrator
  - Implements daily loop execution: M1 → M4 → M5 → M6 → M3
  - Checkpoint resume capability
  - Data consistency validation

- **parallel_executor.py**: Parallel execution framework
  - ThreadPoolExecutor-based parallel task execution
  - Environment variable control (`CHAINSIGHT_PARALLEL=true/false`)

### Modules Layer (`src/modules/`)

5 business modules organized by supply chain process:

| Module | Entry File | Sub-package | Description |
|--------|------------|-------------|-------------|
| **M1** | module1.py | `demand_planning/` | Demand Planning |
| **M3** | module3.py | `mrp_planning/` | MRP Planning |
| **M4** | module4.py | `production_planning/` | Production Planning |
| **M5** | module5.py | `deployment_planning/` | Deployment Planning |
| **M6** | module6.py | `logistics_execution/` | Logistics Execution |

### Data Flow

```
CLI (run.py)
  ↓
main_integration.run_integrated_simulation()
  ├→ Load and validate configuration
  ├→ Check checkpoint resume capability
  ├→ FOR each_day in [start_date, end_date]:
  │   ├→ Module1 (Demand Planning)
  │   ├→ Orchestrator.update_state()
  │   ├→ Module4 (Production Planning)
  │   ├→ Orchestrator.update_state()
  │   ├→ Module5 (Deployment Planning)
  │   ├→ Orchestrator.update_state()
  │   ├→ Module6 (Logistics Execution)
  │   ├→ Orchestrator.update_state()
  │   ├→ Module3 (MRP Planning)
  │   ├→ Orchestrator.update_state()
  │   └→ Generate daily summary and snapshot
  └→ Generate final report and consistency check
```

---

## 🗄️ Database Configuration

### PostgreSQL Installation

**Windows:**
1. Download PostgreSQL: https://www.postgresql.org/download/windows/
2. Remember the password set during installation
3. Default port: 5432

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

### Database Module (pgsql_db)

```python
from pgsql_db import DatabaseInitializer, initialize_database

# Method 1: Convenience function
result = initialize_database(
    config_name='BC_S5',
    database='test_db',
    auto_import=True
)

# Method 2: Initializer class
initializer = DatabaseInitializer(database='test_db')
result = initializer.initialize(config_name='BC_S5')
print(initializer.get_status_report('BC_S5'))
```

### Database Table Structure

#### Config Table Naming Rules (Unified Tables)

Config tables use a "same-structure-same-table" rule. All configuration files with the same data structure are stored in the same table, distinguished by the `config_name` field:

| Table Type | Naming Format | Example | Description |
|------------|---------------|---------|-------------|
| Config tables | `cfg_*` | `cfg_m1_demandforecast` | Shared by all configs, filtered by config_name |
| Module1 output | `module1_output_*` | `module1_output_orderlog` | Module output tables |
| Module3 output | `module3_output_*` | `module3_output_netdemand` | Module output tables |
| Module4 output | `module4_output_*` | `module4_output_productionplan` | Module output tables |
| Module5 output | `module5_output_*` | `module5_output_deploymentplan` | Module output tables |
| Module6 output | `module6_output_*` | `module6_output_deliveryplan` | Module output tables |
| Orchestrator | `orchestrator_*` | `orchestrator_daily_logs` | Orchestrator log tables |
| Summary | `summary_*` | `summary_historical_inventory_record` | Summary report tables |

**Config Table Notes:**
- No longer creates separate tables for each config (e.g., ~~bc_s5_m1_demandforecast~~, ~~bc_s9_m1_demandforecast~~)
- All config data is written to the same table (e.g., `cfg_m1_demandforecast`)
- Different configs are distinguished by the `config_name` field (e.g., 'BC_S5', 'BC_S9')
- This design facilitates cross-config queries and management

---

## 🧪 Testing

The current repository does not ship a top-level `tests/` automation directory. For validated test and verification artifacts, use:

- `docs/交接文档/Refactored/BC算法优化测试报告.md`
- `docs/交接文档/Refactored/OC算法优化测试报告.md`
- `docs/PERFORMANCE_OPTIMIZATION_REPORT.md`

---

## 📚 Documentation

Detailed documentation is located in the `docs/` directory:

### Core Documentation
- [INDEX.md](docs/INDEX.md) - Central docs navigation
- [用户使用入口说明.md](docs/用户使用入口说明.md) - Operator entry guide
- [项目交接文档.md](docs/交接文档/Refactored/项目交接文档.md) - Handoff overview
- [ARCHITECTURE.md](docs/交接文档/Refactored/ARCHITECTURE.md) - Complete architecture design document
- [ARCHITECTURE_DIAGRAM.md](docs/ARCHITECTURE_DIAGRAM.md) - Architecture visualization diagrams
- [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - Quick reference guide (command cheat sheet)

### Module Design
- [MODULE3_DESIGN.md](docs/MODULE3_DESIGN.md) - M3: MRP Planning module design
- [MODULE5_DESIGN.md](docs/MODULE5_DESIGN.md) - M5: Deployment Planning module design
- [module.md](docs/交接文档/Refactored/module.md) - Module handoff overview

### Optimization Documentation
- [DUCKDB_OPTIMIZATION_GUIDE.md](docs/DUCKDB_OPTIMIZATION_GUIDE.md) - DuckDB optimization guide
- [CYTHON_OPTIMIZATION_REPORT.md](docs/CYTHON_OPTIMIZATION_REPORT.md) - Cython optimization report
- [PERFORMANCE_OPTIMIZATION_REPORT.md](docs/PERFORMANCE_OPTIMIZATION_REPORT.md) - Performance optimization report
- [BC算法优化测试报告.md](docs/交接文档/Refactored/BC算法优化测试报告.md) - BC scenario optimization test report
- [OC算法优化测试报告.md](docs/交接文档/Refactored/OC算法优化测试报告.md) - OC scenario optimization test report

### Other Documentation
- [MIGRATION.md](docs/MIGRATION.md) - Version migration guide
- [README_REFACTORING_MAP.md](docs/README_REFACTORING_MAP.md) - Refactoring map and code locator
- [20260127变更版本与修复.md](docs/20260127变更版本与修复.md) - Latest change log
- Historical process/stage documents are archived under `docs/_archive/`

---

## 🐛 Troubleshooting

### ImportError: No module named 'psycopg'

**Cause**: PostgreSQL driver not installed
**Solution**:
```bash
pip install psycopg[binary]
```

### ImportError: No module named 'module1'

**Cause**: Running code outside project directory
**Solution**: Run from root directory using `python run.py` or add `sys.path.insert(0, '.')`

### Configuration file not found

**Cause**: Path relative to current working directory
**Solution**: Use absolute path or run from project root

### Database connection failed

**Cause**: PostgreSQL not running or incorrect connection parameters
**Solution**:
```bash
# Check if PostgreSQL is running
# Windows
net start postgresql-x64-14

# Linux
sudo systemctl status postgresql

# Test connection
python -c "from pgsql_db import DatabaseConnection; db = DatabaseConnection(); print(db.test_connection())"
```

### Checkpoint resume not working

**Cause**: Incomplete run directory structure
**Solution**: Use `--check-resume` to check status, use `--force-restart` if necessary

---

## 🗂️ Project Maintenance

### Directory Cleanliness
- ✅ Root directory keeps essential entry files (run.py, README_CN.md, README_EN.md, requirements.txt, etc.)
- ✅ Temporary documents moved to docs/ or deleted
- ✅ Sample configs archived in config/
- ✅ Run outputs auto-stored in outputs/ (local and database modes unified)

### Code Style
- Comment and maintenance rules: [中文注释规范与维护约定.md](docs/交接文档/Refactored/中文注释规范与维护约定.md)
- Module/file lookup: [README_REFACTORING_MAP.md](docs/README_REFACTORING_MAP.md)

---

## 📞 Support

Having issues? Check:
1. Design documents in `docs/` directory
2. Code comments in each module
3. Run logs (saved in outputs directory)

---

**Version**: 2.1.1  
**Last Updated**: 2026-03-03

## 📝 Changelog

### v2.1.1 (2026-03-03)
- ✅ **Python 3.12 support**: Created `.venv` virtual environment (Python 3.12.9), all dependencies verified
- ✅ **Dependency upgrade**: pandas 3.0.1, duckdb 1.4.4, numpy 2.4.2, scipy 1.17.1
- ✅ **Added xlsxwriter**: Required for Module6 Excel output
- ✅ **requirements.txt**: Categorized with pinned versions

### v2.1.0 (2026-01-29)
- ✅ **Code cleanup**: Removed 42 unnecessary test/debug scripts
- ✅ **Documentation**: Updated architecture docs
- ✅ **Performance**: Added Cython optimization kernel support
- ✅ **DuckDB integration**: High-performance data processing engine
