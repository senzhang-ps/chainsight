# ChainSight - Supply Chain Planning Simulation System

[中文版](README.md) | **English**

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

- **Python**: 3.10+ (Recommended: 3.11 or 3.13)
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
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.2.3 | Data processing |
| openpyxl | 3.1.5 | Excel I/O |
| duckdb | 1.1.3 | High-performance data processing |
| psycopg[binary] | 3.2.3 | PostgreSQL connection |
| numpy | 2.0+ | Numerical computation |

### 4. Run Simulation

**Local File Mode (Default):**
```bash
# First run (specify start date)
python run.py --config test_files/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10

# Resume mode
python run.py --config test_files/BC_S5.xlsx --end-date 2025-10-15 --resume
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

### 5. Export Table Mapping

```bash
python tools/export_mapping.py
```
This generates `database_table_mapping.xlsx` showing Excel-to-database table mapping.

---

## 🔄 Run Modes

### Local File Mode

- Reads configuration from Excel files
- Outputs saved to local filesystem
- Best for development and testing

```bash
python run.py --config test_files/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10
```

### Database Mode (`--use-db`)

- Reads configuration from PostgreSQL
- Outputs written to PostgreSQL database
- Auto-detects and creates database if not exists
- Auto-imports config tables from Excel if not found
- Best for production and data persistence

```bash
python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-10 --use-db
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
│   │   ├── main_integration.py         #    Main integration scheduler (daily loop M1→M4→M5→M6→M3)
│   │   ├── orchestrator.py             #    Unified state management hub
│   │   ├── parallel_executor.py        #    Parallel execution framework (ThreadPoolExecutor)
│   │   └── run.py                      #    CLI parser (called by root run.py)
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
│   ├── optimized_processor.py          #    Optimized processor
│   ├── optimized_simulation.py         #    Optimized simulation engine
│   ├── data_pipeline.py                #    Data pipeline (DuckDB + PostgreSQL)
│   ├── module_engine.py                #    Module execution engine
│   ├── module_optimizers.py            #    Module optimizers
│   ├── incremental_processor.py        #    Incremental processor
│   └── performance_dashboard.py        #    Performance monitoring dashboard
│
├── tools/                              # 🔧 Utility scripts
│   └── init_database.py                #    Database initialization
│
├── tests/                              # 🧪 Test modules
│   ├── e2e_integration_test.py         #    End-to-end integration test
│   └── test_logger.py                  #    Logger tests
│
├── test_files/                         # 📋 Test data & comparison tools
│   ├── BC_S5.xlsx                      #    Main test configuration
│   ├── BC_S9.xlsx                      #    Alternate test configuration
│   ├── compare_all_outputs.py          #    Output comparison tool
│   ├── compare_db_vs_local.py          #    Database vs local comparison
│   ├── TESTING_GUIDE.md                #    Testing guide
│   ├── DATA_COMPARISON_TOOLS_GUIDE.md  #    Comparison tools guide
│   └── Data_Type.md                    #    Type specifications
│
├── config/                             # ⚙️ Configuration files
│   ├── ChainSight 1st SIT.xlsx         #    SIT sample config
│   └── config_guide.xlsx               #    Configuration guide
│
├── docs/                               # 📚 Design documents
│   ├── ARCHITECTURE.md                 #    Architecture design
│   ├── MODULE*_DESIGN.md               #    Module design documents
│   ├── OPTIMIZATION_SUMMARY.md         #    Optimization summary
│   ├── MIGRATION.md                    #    Migration guide
│   ├── README_REFACTORING_MAP.md       #    Refactoring reference document
│   └── PERFORMANCE_OPTIMIZATION_REPORT.md  #    Performance optimization report
│
└── outputs/                            # 📤 Run outputs (auto-generated, in .gitignore)
    └── {config_name}/                  #    Organized by config name
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

## � Output Directory Description

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
│
├── db_BC_S5_YYYYMMDD_HHMMSS/          # Database mode outputs (--use-db flag)
│   ├── simulation_log_YYYYMMDD_HHMMSS.txt  # Run logs
│   └── [Other txt log files]
│
├── db_duckdb_BC_S5_YYYYMMDD_HHMMSS/  # DuckDB enhanced mode (run_with_duckdb.py)
│   ├── run_log_*.txt                   # Run logs
│   └── [Processed data tables]
│
├── db_optimized/                       # Optimized simulation outputs (run_optimized_example.py)
│   ├── cache/                          # Cached data
│   ├── performance/                    # Performance analysis
│   └── orchestrator/                   # State outputs
│
├── db_optimized_cache/                 # Optimized processing cache (run_optimized.py)
│   └── [Parquet cache files]
│
└── integrated_output/                  # Integrated module outputs (test_write_output.py)
    ├── module1/
    ├── module3/
    ├── module4/
    ├── module5/
    ├── module6/
    └── orchestrator/
```

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

| Table Type | Naming Format | Example |
|------------|---------------|---------|
| Config tables | `{config}_*` | `bc_s5_m1_demandforecast` |
| Module1 output | `module1_output_*` | `module1_output_orderlog` |
| Module3 output | `module3_output_*` | `module3_output_netdemand` |
| Module4 output | `module4_output_*` | `module4_output_productionplan` |
| Module5 output | `module5_output_*` | `module5_output_deploymentplan` |
| Module6 output | `module6_output_*` | `module6_output_deliveryplan` |
| Orchestrator | `orchestrator_*` | `orchestrator_daily_logs` |
| Summary | `summary_*` | `summary_historical_inventory_record` |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/e2e_integration_test.py

# Verbose output
pytest -v tests/
```

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
- ✅ Root directory contains only essential files (run.py, README.md, requirements.txt, etc.)
- ✅ Temporary documents moved to docs/ or deleted
- ✅ Tool scripts organized in tools/
- ✅ Sample configs archived in config/
- ✅ Run outputs auto-stored in outputs/ (local and database modes unified)

### Code Style
- Follow [test_files/Python_former.md](test_files/Python_former.md) coding standards
- Type specifications in [test_files/Data_Type.md](test_files/Data_Type.md)

---

## 📞 Support

Having issues? Check:
1. Design documents in `docs/` directory
2. Code comments in each module
3. Run logs (saved in outputs directory)

---

**Version**: 2.1.0  
**Last Updated**: 2026-01-09
