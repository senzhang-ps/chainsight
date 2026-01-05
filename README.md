# ChainSight - 供应链规划仿真系统

## 📦 项目架构

该项目已按标准分层架构进行重组织，便于维护和扩展。

### 目录结构

```
chainsight/
├── src/                           # 核心源代码包
│   ├── __init__.py
│   │
│   ├── core/                      # 核心执行引擎
│   │   ├── __init__.py
│   │   ├── orchestrator.py        # 统一状态管理中枢
│   │   ├── main_integration.py    # 主集成编排器
│   │   └── run.py                 # CLI运行管理（被root run.py调用）
│   │
│   ├── modules/                   # 业务模块层
│   │   ├── __init__.py
│   │   ├── module1.py             # 库存需求规划
│   │   ├── module1_optimized.py   # Module1性能优化版
│   │   ├── module3.py             # 交付规划
│   │   ├── module4.py             # 生产排程
│   │   ├── module5.py             # 库存优化
│   │   └── module6.py             # 物流执行
│   │
│   ├── utils/                     # 工具库
│   │   ├── __init__.py
│   │   ├── config_validator.py    # 配置验证
│   │   ├── logger_config.py       # 日志配置
│   │   ├── validation_manager.py  # 数据验证管理
│   │   ├── inventory_balance_checker.py  # 库存平衡检查
│   │   └── time_manager.py        # 时间管理
│   │
│   └── services/                  # 业务服务
│       ├── __init__.py
│       ├── summary_report_generator.py   # 汇总报告生成
│       └── performance_profiler.py       # 性能分析
│
├── tests/                         # 测试模块
│   ├── __init__.py
│   ├── e2e_integration_test.py    # 端到端集成测试
│   ├── test_logger.py             # 日志模块测试
│   └── conftest.py                # pytest配置（可扩展）
│
├── tools/                         # 独立工具脚本
│   ├── __init__.py
│   ├── apply_push_fix.py
│   ├── create_production_config.py
│   ├── diagnose.py
│   ├── order_log_generator.py
│   ├── performance_profiler.py
│   ├── summary_report_generator.py
│   ├── wip_cov_generator.py
│   └── run.ps1
│
├── config/                        # 配置文件目录
│   ├── requirements.txt           # Python依赖
│   ├── config_guide.xlsx          # 配置指南
│   ├── config.xlsx                # 生产配置（示例）
│   └── ChainSight 1st SIT.xlsx    # 样例配置
│
├── docs/                          # 文档目录
│   ├── README.md（本文件）
│   ├── MODULE*.md                 # 模块设计文档
│   ├── OPTIMIZATION*.md           # 优化说明
│   └── *.docx                     # Word文档
│
├── outputs/                       # 仿真输出目录（自动生成）
│   └── runs/
│       └── run_YYYYMMDD_HHMMSS/   # 每次运行的结果
│
├── run.py                         # 主入口脚本（CLI）
├── requirements.txt               # 依赖文件副本（root）
├── setup.py                       # Python包管理（可选）
└── .gitignore
```

## 🚀 快速开始

### 安装依赖

```bash
pip install -r config/requirements.txt
```

### 运行仿真

```bash
# 首次运行（需指定起始日期）
python run.py \
  --config config/config.xlsx \
  --start-date 2024-01-01 \
  --end-date 2024-01-31

# 列出所有可用的运行目录
python run.py \
  --config config/config.xlsx \
  --end-date 2024-01-31 \
  --list-runs

# 继续上次的运行（交互式选择）
python run.py \
  --config config/config.xlsx \
  --end-date 2024-01-31 \
  --resume

# 检查是否可以继续
python run.py \
  --config config/config.xlsx \
  --end-date 2024-01-31 \
  --check-resume
```

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

6个业务模块，按供应链流程组织：
- **M1**: 库存需求规划（Inventory Planning）
- **M3**: 交付规划（Delivery Planning）
- **M4**: 生产排程（Production Scheduling）
- **M5**: 库存优化（Inventory Optimization）
- **M6**: 物流执行（Logistics Execution）

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
  │   ├→ Module1 (库存规划)
  │   ├→ Orchestrator.update_state()
  │   ├→ Module4 (生产排程)
  │   ├→ Orchestrator.update_state()
  │   ├→ Module5 (库存优化)
  │   ├→ Orchestrator.update_state()
  │   ├→ Module6 (物流执行)
  │   ├→ Orchestrator.update_state()
  │   ├→ Module3 (交付规划)
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
- `MODULE*.md`: 各模块的设计和实现说明
- `OPTIMIZATION*.md`: 性能优化和改进说明
- `*.docx`: Word格式设计文档

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

### ImportError: No module named 'module1'

**原因**: 在项目外直接运行代码
**解决**: 确保从root目录运行，使用 `python run.py` 或 `sys.path.insert(0, '.')`

### 找不到配置文件

**原因**: 路径相对于当前工作目录
**解决**: 使用绝对路径或从项目root目录运行

### 断点续跑不工作

**原因**: 运行目录结构不完整
**解决**: 使用 `--check-resume` 检查状态，必要时使用 `--force-restart`

## 📞 支持

遇到问题？检查：
1. `docs/` 目录中的设计文档
2. 各模块的代码注释
3. 运行日志（保存在output目录中）

---

**版本**: 1.0.0  
**最后更新**: 2026-01-05
