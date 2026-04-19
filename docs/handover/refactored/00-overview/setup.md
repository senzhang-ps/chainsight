# ChainSight 部署与配置指南

## 文档信息

| 项 | 内容 |
|---|---|
| 文档版本 | v1.1 |
| 最后更新 | 2026-04-10 |
| 编写人 | 陈显跃 |
| 适用范围 | 本地文件模式 + 数据库模式 |
| 适用系统 | Windows / Linux / macOS |

---

# 第一部分：环境配置

## 1. 系统要求

### 1.1 最低要求

| 项目 | 最低要求 | 推荐配置 |
|---|---|---|
| Python | 3.12 | 3.12.9 |
| CPU | 4 核 | 8 核及以上 |
| 内存 | 8 GB | 16-32 GB |
| 磁盘 | 10 GB 可用空间 | 50 GB（含输出与日志） |
| PostgreSQL（可选） | 14+ | 15/16 |

### 1.2 操作系统支持

- Windows 10/11（推荐 PowerShell 7+）；
- Linux（Ubuntu/Debian/CentOS 均可）；
- macOS（Intel/Apple Silicon 均可）。

### 1.3 网络与权限要求

1. 能访问项目目录并读写 `outputs/`；
2. 数据库模式下可访问 PostgreSQL 端口（默认 5432）；
3. 具备安装 Python 包权限（或可使用离线镜像源）。

---

## 2. 依赖安装

### 2.1 依赖文件

项目依赖定义在 `requirements.txt`。

### 2.2 一键安装

```bash
pip install -r requirements.txt
```

### 2.3 关键依赖说明

| 包 | 版本 | 用途 |
|---|---|---|
| `pandas` | 3.0.1 | 数据处理 |
| `numpy` | 2.4.2 | 数值计算 |
| `duckdb` | 1.4.4 | 向量化与高速查询 |
| `openpyxl` | 3.1.5 | 读取 Excel 配置 |
| `xlsxwriter` | 3.2.9 | 输出 Excel 报告 |
| `psycopg` / `psycopg-binary` | 3.3.3 | PostgreSQL 连接 |
| `scipy` | 1.17.1 | 统计与分布计算 |
| `matplotlib` | 3.10.8 | 图表生成 |

### 2.4 安装验证

```bash
python -c "import pandas, duckdb, psycopg; print('deps ok')"
```

### 2.5 常用镜像源（可选）

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 3. 虚拟环境设置

### 3.1 Windows（PowerShell）

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
. .\.venv\Scripts\Activate.ps1
python --version
```

### 3.2 Windows（CMD）

```cmd
py -3.12 -m venv .venv
.\.venv\Scripts\activate.bat
python --version
```

### 3.3 Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

### 3.4 退出虚拟环境

```bash
deactivate
```

### 3.5 推荐目录检查

```bash
python -c "import sys; print(sys.executable)"
```

---

## 4. 编译可选扩展（Cython）

### 4.1 当前仓库状态

当前仓库包含 `src/cython_kernels/` 目录结构说明，但根目录未提供可直接执行的 `setup.py` 构建脚本。默认部署可直接使用纯 Python + DuckDB 路径。

### 4.2 可选方案

1. 使用现有 Python 路径直接运行（推荐）；
2. 若团队后续补充 Cython 构建脚本，再开启编译部署流程；
3. 在 CI 中将“是否启用 Cython”作为可配置开关。

### 4.3 性能建议

- 优先先启用 DuckDB 与缓存优化，再评估 Cython 的边际收益。

---

# 第二部分：本地版部署

## 5. 配置文件格式

### 5.1 配置文件位置

- 推荐放置于 `config/` 或 `test_files/`；
- 文件格式：`.xlsx`。

### 5.2 最小必需工作表（核心）

| Sheet | 关键字段 |
|---|---|
| `M1_InitialInventory` | `material, location, quantity` |
| `Global_SpaceCapacity` | `location, eff_from, eff_to, capacity` |
| `Global_Network` | `material, location, sourcing, location_type, eff_from, eff_to` |
| `Global_LeadTime` | `sending, receiving, PDT, OTD, GR, MCT` |
| `Global_DemandPriority` | `demand_element, priority` |

### 5.3 各模块常用工作表

| 模块 | 关键工作表 |
|---|---|
| M1 | `M1_DemandForecast`, `M1_OrderCalendar`, `M1_AOConfig`, `M1_DPSConfig`, `M1_SupplyChoiceConfig` |
| M3 | `M3_SafetyStock` |
| M4 | `M4_MaterialLocationLineCfg`, `M4_LineCapacity`, `M4_ChangeoverMatrix`, `M4_ChangeoverDefinition` |
| M5 | `M5_DeployConfig`, `M5_PushPullModel` |
| M6 | `M6_TruckReleaseCon`, `M6_TruckTypeSpecs`, `M6_DeliveryDelayDistribution`, `M6_TruckCapacityPlan` |

### 5.4 字段规范建议

1. `material` 建议文本格式，避免 Excel 自动转数值；
2. `location/sending/receiving` 建议使用统一编码（如 4 位）；
3. 日期列统一为 `YYYY-MM-DD`；
4. 数量字段避免混入文本。

### 5.5 配置一致性预检查

```bash
python -c "from src.utils.config_validator import run_pre_simulation_validation as f; print(f('test_files/BC_S5.xlsx','./outputs/validation_check'))"
```

---

## 6. 本地版运行步骤

### 6.1 快速开始

```bash
python run.py --config test_files/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10
```

### 6.2 首次运行

首次运行建议显式给出 `--start-date` 与 `--end-date`。

```bash
python run.py --config config/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-20
```

### 6.3 续跑模式

```bash
# 自动续跑
python run.py --config config/BC_S5.xlsx --end-date 2025-10-25 --resume

# 指定续跑目录
python run.py --config config/BC_S5.xlsx --end-date 2025-10-25 --resume-from run_20260305_101530
```

### 6.4 查看可用运行目录

```bash
python run.py --config config/BC_S5.xlsx --end-date 2025-10-25 --list-runs
```

### 6.5 强制重跑

```bash
python run.py --config config/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-25 --force-restart
```

### 6.6 非交互模式

```bash
python run.py --config config/BC_S5.xlsx --end-date 2025-10-25 --resume --non-interactive
```

---

## 7. 性能配置

### 7.1 并行参数

环境变量 `CHAINSIGHT_PARALLEL` 控制并行执行器默认行为。

```bash
# Linux/macOS
export CHAINSIGHT_PARALLEL=true

# Windows PowerShell
$env:CHAINSIGHT_PARALLEL="true"
```

### 7.2 线程参数（可选）

`run.ps1` 示例中使用：

```powershell
$env:OMP_NUM_THREADS=8
$env:NUMEXPR_MAX_THREADS=8
```

### 7.3 DuckDB 内存模式

```python
from src.utils.memory_data_store import enable_memory_mode

ok = enable_memory_mode(memory_limit="8GB", threads=8)
print("memory mode:", ok)
```

### 7.4 缓存策略

```python
from src.utils.simulation_cache import initialize_simulation_cache

cache = initialize_simulation_cache(config_dict, "2025-10-06", "2025-10-20")
cache.print_stats()
```

### 7.5 建议优先级

1. 先开内存模式；
2. 再启用并行；
3. 最后做参数级调优（线程数、数据切分粒度）。

---

## 8. 输出目录结构

### 8.1 目录组织规则

本地模式输出路径由 `src/core/run/output_dir.py::_ensure_output_dir` 统一生成（原 `src/core/run.py` 单体已拆分为 `src/core/run/` 包）：

`outputs/<config_stem>/run_YYYYMMDD_HHMMSS/`

### 8.2 典型目录结构

```text
outputs/
  BC_S5/
    run_20260305_103015/
      module1/
      module3/
      module4/
      module5/
      module6/
      orchestrator/
      summary/
      simulation_log_*.txt
```

### 8.3 Orchestrator 快照文件

| 文件模式 | 说明 |
|---|---|
| `unrestricted_inventory_YYYYMMDD.csv` | 可用库存 |
| `open_deployment_YYYYMMDD.csv` | 开放调拨 |
| `planning_intransit_YYYYMMDD.csv` | 在途 |
| `delivery_gr_YYYYMMDD.csv` | 到货 |
| `production_gr_YYYYMMDD.csv` | 生产收货 |
| `shipment_log_YYYYMMDD.csv` | 客户发货 |
| `delivery_shipment_log_YYYYMMDD.csv` | 调拨发运 |

### 8.4 清理建议

- 长周期输出建议按 `run_` 目录归档；
- 不要手工删除单日快照文件（会影响续跑判断）。

---

# 第三部分：数据库版部署

## 9. PostgreSQL 配置

### 9.1 安装 PostgreSQL

**Windows**：下载官方安装包并安装，默认端口 5432。

**Ubuntu/Debian**：

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl enable postgresql
sudo systemctl start postgresql
```

### 9.2 创建数据库与用户（示例）

```sql
CREATE DATABASE test_db;
CREATE USER chainsight_user WITH ENCRYPTED PASSWORD 'change_me';
GRANT ALL PRIVILEGES ON DATABASE test_db TO chainsight_user;
```

### 9.3 服务检查

```bash
# Linux
sudo systemctl status postgresql

# Windows (服务名可能随版本变化)
net start postgresql-x64-14
```

---

## 10. 初始化数据库

### 10.1 自动初始化（推荐）

```python
from pgsql_db.db_initializer import DatabaseInitializer

initializer = DatabaseInitializer(
    host="localhost",
    port=5432,
    database="test_db",
    user="postgres",
    password="123456",
)
result = initializer.initialize(config_name="BC_S5", auto_import_config=True, verbose=True)
print(result)
```

### 10.2 命令行触发初始化

```bash
python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-10 --use-db
```

首次运行会自动执行：

1. 检测数据库；
2. 缺失时创建数据库；
3. 检测配置数据；
4. 缺失时从 Excel 自动导入。

### 10.3 初始化状态核查

```python
print(initializer.get_status_report("BC_S5"))
```

---

## 11. 连接配置

### 11.1 CLI 连接参数

```bash
python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-10 \
  --use-db --db-host localhost --db-port 5432 --db-name test_db \
  --db-user postgres --db-password 123456
```

### 11.2 连接字符串

格式：

```text
postgresql://<user>:<password>@<host>:<port>/<database>
```

### 11.3 程序化连接测试

```python
from pgsql_db.db_connection import DatabaseConnection

db = DatabaseConnection(host="localhost", database="test_db", user="postgres", password="***")
print(db.test_connection())
```

### 11.4 SSL 配置说明

当前 CLI 未暴露专用 SSL 参数；如需 SSL，可在扩展 `DatabaseConnection.connect()` 时注入 psycopg SSL 配置项。

---

## 12. 数据迁移

### 12.1 本地输出迁移到数据库（推荐路径）

先完成本地运行，再将输出目录写入数据库。

```python
from pgsql_db.module_data_writer import ModuleDataWriter
from pgsql_db.db_connection import DatabaseConnection

db = DatabaseConnection(database="test_db", user="postgres", password="***")
writer = ModuleDataWriter(db, config_name="BC_S5")
writer.write_all_modules("./outputs/BC_S5/run_20260305_103015", run_id="BC_S5_MIG_001", if_exists="replace")
```

### 12.2 运行级迁移（便捷函数）

```python
from pgsql_db.module_data_writer import write_run_data_to_db

ok = write_run_data_to_db(
    run_output_dir="./outputs/BC_S5/run_20260305_103015",
    db_name="test_db",
    db_user="postgres",
    db_password="***",
)
print(ok)
```

> 注意：若便捷函数依赖方法与当前版本不完全匹配，请优先使用 `ModuleDataWriter` 显式流程。

### 12.3 映射规则

| 源 | 目标 |
|---|---|
| Excel 配置 Sheet | `cfg_*` 统一配置表 |
| `moduleX` 输出文件 | `moduleX_output_*` |
| Orchestrator CSV | `orchestrator_*` |
| summary 文件 | `summary_output_*` |

### 12.4 迁移后验证

```python
rows = db.execute_query('SELECT COUNT(*) FROM "module1_output_orderlog" WHERE run_id = %s', ("BC_S5_MIG_001",))
print(rows)
```

---

## 13. 数据库性能配置

### 13.1 PostgreSQL 参数建议（参考）

| 参数 | 建议 |
|---|---|
| `shared_buffers` | 物理内存 20%-25% |
| `work_mem` | 16MB~64MB（按并发调优） |
| `maintenance_work_mem` | 256MB~1GB |
| `max_connections` | 按连接模型设置（避免盲目拉高） |

### 13.2 应用侧优化点

1. 保持 COPY 批量写入路径；
2. 使用 `run_id` 做批次过滤和清理；
3. 对高频列建立索引（material/location/date/run_id）。

### 13.3 DuckDB 优化开关

```python
from pgsql_db.duckdb_integration import DuckDBConfig

DuckDBConfig.enabled = True
DuckDBConfig.min_rows_threshold = 20
DuckDBConfig.fallback_on_error = True
```

---

# 第四部分：故障排查

## 14. 常见问题

### 14.1 Excel 读取错误

**现象**：`Failed to load config file`。

**排查**
1. 路径是否正确；
2. 文件是否被占用；
3. Sheet 名是否符合预期。

### 14.2 编码问题

**现象**：中文路径/字段读取异常。

**处理**
- 保持 Python 与系统编码为 UTF-8；
- 数据库连接中启用 `client_encoding='UTF8'`（项目已默认设置）。

### 14.3 内存不足

**现象**：运行中 OOM 或明显变慢。

**处理**
1. 降低并行度；
2. 关闭不必要本地输出；
3. 减少仿真日期范围做分批运行。

### 14.4 数据库连接失败

**现象**：`连接失败`。

**处理命令**

```bash
python -c "from pgsql_db.db_connection import DatabaseConnection as D; print(D().test_connection())"
```

### 14.5 数据不一致

**现象**：本地与数据库输出不一致。

**处理**
1. 确认运行日期范围一致；
2. 固定随机种子；
3. 对比 `run_id/sim_date` 过滤后的同口径数据。

---

## 15. 日志分析

### 15.1 日志位置

| 模式 | 日志位置 |
|---|---|
| 本地模式 | `outputs/<config>/run_xxx/simulation_log_*.txt` |
| 数据库模式 | `outputs/db_<config>_<timestamp>/simulation_log_*.txt` |

### 15.2 日志格式

`YYYY-MM-DD HH:MM:SS [LEVEL] message`

### 15.3 关键日志关键词

| 关键词 | 说明 |
|---|---|
| `Start` / `Complete` | 阶段开始与结束 |
| `WARN` | 非阻断风险 |
| `ERROR` | 阻断错误 |
| `DB写入耗时` | 数据库落库耗时指标 |
| `balance` | 库存平衡检查结果 |

### 15.4 日志快速过滤（示例）

```bash
# Windows PowerShell
Select-String -Path "outputs\**\simulation_log_*.txt" -Pattern "ERROR|WARN"
```

---

## 16. 性能诊断

### 16.1 本地模式诊断

```python
from src.services.performance_profiler import PerformanceProfiler

with PerformanceProfiler("Module5", enabled=True):
    run_module5()
```

### 16.2 数据库模式诊断

```python
from pgsql_db.performance_dashboard import PerformanceDashboard

dashboard = PerformanceDashboard(output_dir="./performance")
dashboard.start_simulation()
```

### 16.3 DuckDB/Pandas 对比

```python
from pgsql_db.duckdb_integration import run_ab_comparison

result = run_ab_comparison(func_a=duck_func, func_b=pandas_func, test_data=df, iterations=5)
print(result)
```

### 16.4 排障优先顺序

1. 先看日志是否有 ERROR；
2. 再看模块耗时分布；
3. 最后分析 SQL/数据规模与索引策略。

---

# 第五部分：进阶主题

## 17. 自定义配置

### 17.1 添加新参数流程

1. 在 Excel 新增列或新增 Sheet；
2. 在 `load_configuration()`/模块 `load_config` 中读取；
3. 在 `config_validator.py` 增加规则校验；
4. 在模块主逻辑中消费该参数。

### 17.2 示例：新增 M5 参数

```python
# 读取新列
cfg = config_dict["M5_DeployConfig"].copy()
cfg["new_factor"] = cfg.get("new_factor", 1.0)
```

### 17.3 兼容性建议

- 新参数应提供默认值；
- 不破坏既有输出字段；
- 通过 feature flag 控制灰度启用。

---

## 18. 集成指南

### 18.1 与外部调度系统集成

推荐通过命令行触发：

```bash
python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-10 --use-db
```

### 18.2 Python 进程内集成

```python
from src.core.main_integration import run_integrated_simulation

res = run_integrated_simulation(
    config_path="test_files/BC_S5.xlsx",
    start_date="2025-10-06",
    end_date="2025-10-10",
    output_base_dir="./outputs/integration_call",
)
```

### 18.3 数据平台集成建议

1. 将 `summary_output_*` 作为下游 BI 的主事实源；
2. 将 `orchestrator_*` 作为审计与追溯源；
3. 对外只暴露 `run_id` 过滤后的稳定视图。

### 18.4 安全与合规建议

- 不在代码中硬编码数据库密码；
- 使用环境变量或密钥管理系统注入凭据；
- 对生产数据库账号采用最小权限原则。

### 18.5 CI/CD 集成建议

推荐将部署流程拆分为“预检查 -> 执行 -> 验证”三个阶段：

1. **预检查**：Python 版本、依赖安装、数据库连通、配置可读；
2. **执行阶段**：触发 `run.py`（本地或 DB 模式）；
3. **验证阶段**：检查输出目录或关键表记录数、生成汇总报告。

**示例（GitHub Actions 片段）**

```yaml
name: chainsight-run
on: [workflow_dispatch]
jobs:
  simulation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-10 --use-db --non-interactive
```

### 18.6 生产变更发布策略

| 阶段 | 目标 | 推荐动作 |
|---|---|---|
| 预发布（staging） | 验证配置和性能 | 使用真实规模样本运行 7-14 天窗口 |
| 小流量灰度 | 控制风险 | 新参数通过 feature flag 仅对部分配置启用 |
| 全量发布 | 业务切换 | 固定 `run_id` 命名规则并保留回滚点 |
| 发布后观察 | 稳定性确认 | 持续监控模块耗时、缺货率、异常日志 |

---

## 附录 A：部署检查清单

| 检查项 | 本地版 | DB版 |
|---|---|---|
| Python 3.12 环境 | ✅ | ✅ |
| `pip install -r requirements.txt` | ✅ | ✅ |
| 配置校验通过 | ✅ | ✅ |
| PostgreSQL 连通 | - | ✅ |
| 输出目录可写 | ✅ | ✅ |
| 日志可落盘 | ✅ | ✅ |

## 附录 B：最小可运行命令

```bash
# 本地模式
python run.py --config test_files/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10

# 数据库模式
python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-10 --use-db
```

## 附录 C：环境变量模板

建议在本地或服务器上使用环境变量管理敏感信息：

```bash
# PostgreSQL
export CHAINSIGHT_DB_HOST=localhost
export CHAINSIGHT_DB_PORT=5432
export CHAINSIGHT_DB_NAME=test_db
export CHAINSIGHT_DB_USER=postgres
export CHAINSIGHT_DB_PASSWORD=***

# 运行控制
export CHAINSIGHT_LOG_LEVEL=INFO
export CHAINSIGHT_RANDOM_SEED=42
```

Windows PowerShell：

```powershell
$env:CHAINSIGHT_DB_HOST="localhost"
$env:CHAINSIGHT_DB_PORT="5432"
$env:CHAINSIGHT_DB_NAME="test_db"
$env:CHAINSIGHT_DB_USER="postgres"
$env:CHAINSIGHT_DB_PASSWORD="***"
```

## 附录 D：备份与恢复建议

### D.1 PostgreSQL 备份

```bash
pg_dump -h localhost -p 5432 -U postgres -d test_db -F c -f chainsight_test_db.dump
```

### D.2 PostgreSQL 恢复

```bash
pg_restore -h localhost -p 5432 -U postgres -d test_db --clean --if-exists chainsight_test_db.dump
```

### D.3 输出目录备份

至少备份以下目录：

- `outputs/<run_id>/orchestrator/`
- `outputs/<run_id>/summary/`
- `logs/`

### D.4 恢复演练建议

1. 每月执行一次“备份还原 + 3日续跑”演练；
2. 对比恢复前后 `summary` 核心指标（总订单、总发货、总缺货）；
3. 演练结果固化为运维记录，形成可审计闭环。

## 附录 E：上线日 Runbook（值班版）

### E.1 上线前 30 分钟

- 检查代码版本与目标分支一致；
- 校验配置文件版本号与 `config_name` 一致；
- 确认数据库连接可用且磁盘空间充足；
- 预创建输出目录并校验写权限。

### E.2 上线执行窗口

1. 触发目标命令（本地或 DB 模式）；
2. 观察日志中 M1/M4/M5/M6/M3 是否按序执行；
3. 监控模块耗时，若单模块异常放大则暂停并排查；
4. 运行结束后立即执行汇总与一致性检查。

### E.3 上线后 60 分钟观察

- 检查关键输出（或关键输出表）行数是否在预期范围；
- 检查 `summary` 指标是否出现断崖式变化；
- 检查 ERROR/CRITICAL 日志是否新增；
- 若异常，优先走“按日回滚 + 续跑”策略。

### E.4 回滚触发条件（建议）

| 条件 | 建议动作 |
|---|---|
| 主流程中断且 15 分钟内无法恢复 | 立即回滚到上一稳定版本 |
| 关键输出缺失（M5/M6 主输出） | 回滚并补跑当天 |
| 关键 KPI 偏差超过阈值（如 >20%） | 暂停发布，执行差异复盘 |

### E.5 值班记录模板（建议）

```text
[Runbook]
date=2026-03-05
operator=xxx
mode=local|db
config=BC_S5
window=2025-10-06~2025-10-10

precheck=PASS
execute=PASS
summary_check=PASS
inventory_balance=PASS
error_count=0

rollback=no
notes=none
```

建议将该模板持久化到内部运维系统，形成可审计的上线轨迹。

该 Runbook 适用于演练与正式发布双场景。
