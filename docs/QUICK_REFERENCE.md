# ChainSight 快速参考指南

**版本**: 2.1.0  
**最后更新**: 2026-01-29

## 📌 常用命令速查

### 🚀 运行仿真

```bash
# 1. 本地文件模式（开发/测试）
python run.py --config config/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10

# 2. 数据库模式（生产环境）
python run.py --config config/OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2026-02-28 --use-db

# 3. 断点续跑
python run.py --config config/BC_S5.xlsx --end-date 2025-10-15 --resume

# 4. 强制重新开始
python run.py --config config/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10 --force-restart

# 5. 列出可用的运行目录
python run.py --config config/BC_S5.xlsx --list-runs
```

### 🔧 环境准备

```bash
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活虚拟环境 (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 3. 激活虚拟环境 (Windows CMD)
.\.venv\Scripts\activate.bat

# 4. 激活虚拟环境 (Linux/macOS)
source .venv/bin/activate

# 5. 安装依赖
pip install -r requirements.txt

# 6. 编译 Cython 扩展（可选，用于性能优化）
python setup.py build_ext --inplace
```

### 🗄️ 数据库操作

```bash
# 1. 初始化数据库
python tools/init_database.py

# 2. 从 Excel 导入配置到数据库
python -c "from pgsql_db import initialize_database; initialize_database('BC_S5', auto_import=True)"

# 3. 测试数据库连接
python -c "from pgsql_db import DatabaseConnection; db = DatabaseConnection(); print(db.test_connection())"

# 4. 查看数据库状态
python -c "from pgsql_db import DatabaseInitializer; init = DatabaseInitializer(); print(init.get_status_report('BC_S5'))"

# 5. 配置表迁移
python tools/migrate_config_tables.py --from BC_S5 --to BC_S9
```

### 📊 输出对比与验证

```bash
# 1. 主输出对比工具
python test_files/compare_all_outputs.py --run1 outputs/BC_S5/run1 --run2 outputs/BC_S5/run2

# 2. 数据库 vs 本地对比
python test_files/compare_db_vs_local.py

# 3. 与 ChainSight_Dev 版本对比
python test_files/compare_db_vs_chainsight_dev.py

# 4. Orchestrator 状态对比
python test_files/compare_orchestrator.py

# 5. M3 差异调试
python test_files/debug_m3_difference.py

# 6. 快速检查
python test_files/quick_check.py

# 7. 配置一致性测试
python test_files/test_config_consistency.py
```

### 📈 性能测试与基准

```bash
# 1. DuckDB vs Pandas 性能对比
python tools/benchmark_duckdb_vs_pandas.py

# 2. 性能基准测试
python tools/performance_benchmark.py

# 3. DuckDB 性能测试
python test_files/test_duckdb_performance.py
```

### 📝 报告生成

```bash
# 1. 生成 Word 报告
python tools/generate_docx_report.py --output report.docx

# 2. 生成报告图表
python tools/generate_report_charts.py

# 3. 导出表映射关系
python tools/export_mapping.py
```

### 🔬 测试运行

```bash
# 1. 端到端集成测试
python tests/e2e_integration_test.py

# 2. 运行所有测试 (pytest)
pytest tests/ -v

# 3. 日志测试
python tests/test_logger.py
```

## 📋 命令行参数说明

### run.py 参数

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | ✅ | - | 配置文件路径（本地）或配置名（数据库） |
| `--start-date` | 首次运行 | - | 仿真开始日期 (YYYY-MM-DD) |
| `--end-date` | ✅ | - | 仿真结束日期 (YYYY-MM-DD) |
| `--use-db` | ❌ | False | 启用数据库模式 |
| `--db-host` | ❌ | localhost | 数据库主机地址 |
| `--db-port` | ❌ | 5432 | 数据库端口 |
| `--db-name` | ❌ | test_db | 数据库名称 |
| `--db-user` | ❌ | postgres | 数据库用户名 |
| `--db-password` | ❌ | 123456 | 数据库密码 |
| `--resume` | ❌ | False | 启用断点续跑 |
| `--force-restart` | ❌ | False | 强制重新开始 |
| `--list-runs` | ❌ | False | 列出可用的运行目录 |

### 示例组合

```bash
# 完整数据库模式命令
python run.py \
  --config OC_Paste_S1_20251224 \
  --start-date 2025-12-15 \
  --end-date 2026-02-28 \
  --use-db \
  --db-host localhost \
  --db-port 5432 \
  --db-name test_db \
  --db-user postgres \
  --db-password 123456
```

## 📂 重要目录与文件

### 配置文件

```
config/
├── BC_S5.xlsx                      # 测试配置 S5
├── BC_S9.xlsx                      # 测试配置 S9
├── OC_Paste_S1_20251224/           # 生产配置（数据库）
├── ChainSight 1st SIT.xlsx         # SIT 样例
└── config_guide.xlsx               # 配置指南
```

### 输出目录

```
outputs/
├── BC_S5/                          # 本地文件模式输出
│   └── run_20260129_145654/        # 按时间戳命名
│       ├── module1/                # 各模块输出 Excel
│       ├── module3/
│       ├── module4/
│       ├── module5/
│       ├── module6/
│       ├── orchestrator/           # 状态快照
│       ├── summary/                # 汇总报告
│       ├── performance/            # 性能分析
│       └── validation_report.txt  # 验证报告
│
└── db_config/                      # 数据库模式输出
    └── OC_Paste_S1_20251224_20260129_145654/
        └── simulation_log_20260129_145654.txt
```

### 文档目录

```
docs/
├── ARCHITECTURE.md                 # 架构设计文档
├── ARCHITECTURE_DIAGRAM.md         # 架构可视化图
├── QUICK_REFERENCE.md              # 快速参考（本文档）
├── MODULE1_DESIGN.md               # M1 模块设计
├── MODULE3_DESIGN.md               # M3 模块设计
├── MODULE5_DESIGN.md               # M5 模块设计
├── OPTIMIZATION_SUMMARY.md         # 优化总结
├── DUCKDB_OPTIMIZATION_GUIDE.md    # DuckDB 优化指南
├── MIGRATION.md                    # 迁移指南
└── 算法优化测试报告.md               # 算法优化报告
```

## 🐍 Python 代码片段

### 1. 程序化运行仿真

```python
from src.core.main_integration import run_integrated_simulation

# 本地文件模式
result = run_integrated_simulation(
    config_path='config/BC_S5.xlsx',
    start_date='2025-10-06',
    end_date='2025-10-10'
)

# 数据库模式
result = run_integrated_simulation(
    config_path='OC_Paste_S1_20251224',
    start_date='2025-12-15',
    end_date='2026-02-28',
    use_db=True,
    db_config={
        'host': 'localhost',
        'port': 5432,
        'database': 'test_db',
        'user': 'postgres',
        'password': '123456'
    }
)
```

### 2. 数据库操作

```python
from pgsql_db import DatabaseConnection, DatabaseInitializer, initialize_database

# 创建数据库连接
db = DatabaseConnection(
    host='localhost',
    port=5432,
    database='test_db',
    user='postgres',
    password='123456'
)

# 测试连接
if db.test_connection():
    print("✅ 数据库连接成功")

# 初始化数据库和配置表
result = initialize_database(
    config_name='BC_S5',
    excel_path='config/BC_S5.xlsx',
    database='test_db',
    auto_import=True
)

# 使用初始化器类
initializer = DatabaseInitializer(database='test_db')
result = initializer.initialize(config_name='BC_S5')
print(initializer.get_status_report('BC_S5'))
```

### 3. 配置验证

```python
from src.utils.config_validator import validate_config
import pandas as pd

# 读取配置
config = {
    'M1_DemandForecast': pd.read_excel('config/BC_S5.xlsx', sheet_name='M1_DemandForecast'),
    'M3_BOM': pd.read_excel('config/BC_S5.xlsx', sheet_name='M3_BOM'),
    # ... 其他配置表
}

# 验证配置
is_valid, errors = validate_config(config)
if is_valid:
    print("✅ 配置验证通过")
else:
    print("❌ 配置验证失败:")
    for error in errors:
        print(f"  - {error}")
```

### 4. 访问 Orchestrator 状态

```python
from src.core.orchestrator import create_orchestrator

# 创建 orchestrator
orchestrator = create_orchestrator()

# 更新状态
orchestrator.update_inventory(date='2025-10-06', material='MAT001', location='DC01', quantity=1000)

# 查询状态
inventory = orchestrator.get_inventory(date='2025-10-06', material='MAT001', location='DC01')
print(f"库存: {inventory}")

# 保存快照
orchestrator.save_snapshot(date='2025-10-06', output_dir='outputs/BC_S5/run_xxx')

# 加载快照
orchestrator.load_snapshot(date='2025-10-06', output_dir='outputs/BC_S5/run_xxx')
```

## 🔍 常见问题速查

### Q1: ImportError: No module named 'psycopg'
```bash
pip install psycopg[binary]
```

### Q2: 数据库连接失败
```bash
# 检查 PostgreSQL 是否运行
# Windows
net start postgresql-x64-14

# Linux
sudo systemctl status postgresql

# 测试连接
python -c "from pgsql_db import DatabaseConnection; db = DatabaseConnection(); print(db.test_connection())"
```

### Q3: Cython 编译错误
```bash
# 确保安装了编译器
# Windows: 安装 Visual Studio Build Tools
# Linux: sudo apt install build-essential

# 重新编译
python setup.py build_ext --inplace --force
```

### Q4: 输出目录找不到
```bash
# 列出所有运行目录
python run.py --config config/BC_S5.xlsx --list-runs

# 输出目录结构
outputs/
└── BC_S5/
    ├── run_20260129_100000/
    ├── run_20260129_110000/
    └── run_20260129_120000/
```

### Q5: 如何清理临时文件
```bash
# 清理 Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# 清理 Cython 编译文件
rm -rf build/
rm src/cython_kernels/*.c
rm src/cython_kernels/*.pyd
rm src/cython_kernels/*.so

# 清理输出（谨慎使用）
# rm -rf outputs/*
```

## 📞 获取帮助

1. **查看文档**: `docs/` 目录下的详细文档
2. **查看代码注释**: 各模块代码中的详细注释
3. **查看日志**: `outputs/` 目录下的日志文件
4. **运行测试**: `python tests/e2e_integration_test.py`

---

**提示**: 
- 使用 `--help` 查看命令行帮助：`python run.py --help`
- 首次运行建议使用测试配置：`config/BC_S5.xlsx`
- 生产环境推荐使用数据库模式：`--use-db`
- 记得定期备份重要的输出结果
