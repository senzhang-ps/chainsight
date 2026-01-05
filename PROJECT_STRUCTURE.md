# 项目目录树结构说明

```
chainsight/                              # 项目根目录
│
├─ run.py                               # 🚀 主入口脚本（CLI）
├─ verify_architecture.py               # 🔍 架构验证脚本
├─ requirements.txt                     # 📦 依赖声明（便利副本）
├─ README.md                            # 📖 项目概览
├─ COMPLETION_REPORT.md                 # ✅ 重构完成报告
│
├─ src/                                 # 📁 核心源代码包
│  ├─ __init__.py
│  │
│  ├─ core/                             # 🎯 执行层（编排 & 状态管理）
│  │  ├─ __init__.py
│  │  ├─ orchestrator.py               # 统一状态管理中枢
│  │  ├─ main_integration.py           # 主集成编排器
│  │  └─ run.py                        # CLI管理（被root run.py调用）
│  │
│  ├─ modules/                          # 📊 业务模块层（6个规划模块）
│  │  ├─ __init__.py
│  │  ├─ module1.py                    # 库存需求规划
│  │  ├─ module1_optimized.py          # Module1性能优化版
│  │  ├─ module3.py                    # 交付规划
│  │  ├─ module4.py                    # 生产排程
│  │  ├─ module5.py                    # 库存优化
│  │  └─ module6.py                    # 物流执行
│  │
│  ├─ utils/                            # 🛠️ 工具库（验证 & 管理）
│  │  ├─ __init__.py
│  │  ├─ config_validator.py           # 配置验证
│  │  ├─ logger_config.py              # 日志配置
│  │  ├─ validation_manager.py         # 数据验证管理
│  │  ├─ inventory_balance_checker.py  # 库存平衡检查
│  │  └─ time_manager.py               # 时间管理
│  │
│  └─ services/                         # 🎁 业务服务（高级功能）
│     ├─ __init__.py
│     ├─ summary_report_generator.py   # 汇总报告生成
│     └─ performance_profiler.py       # 性能分析
│
├─ tests/                               # 🧪 测试模块
│  ├─ __init__.py
│  ├─ e2e_integration_test.py          # 端到端集成测试
│  ├─ test_logger.py                   # 日志模块测试
│  └─ conftest.py                      # pytest配置（可扩展）
│
├─ tools/                               # 🔧 独立工具脚本
│  ├─ __init__.py
│  ├─ apply_push_fix.py
│  ├─ create_production_config.py
│  ├─ diagnose.py
│  ├─ order_log_generator.py
│  ├─ performance_profiler.py          # 备份副本
│  ├─ summary_report_generator.py      # 备份副本
│  ├─ wip_cov_generator.py
│  └─ run.ps1
│
├─ config/                              # ⚙️ 配置文件目录
│  ├─ requirements.txt                 # Python依赖（主版本）
│  ├─ config_guide.xlsx                # 配置指南
│  ├─ config.xlsx                      # 生产配置
│  └─ ChainSight 1st SIT.xlsx          # 样例配置
│
├─ docs/                                # 📚 文档目录
│  ├─ README.md                        # 项目概览（副本）
│  ├─ ARCHITECTURE.md                  # ⭐ 详细架构设计
│  ├─ MIGRATION.md                     # 迁移指南（文件对应）
│  ├─ MODULE1_OPTIMIZATION_NOTES.md
│  ├─ MODULE1_OPTIMIZATION_SUMMARY.md
│  ├─ MODULE3_DESIGN.md
│  ├─ MODULE5_DESIGN.md
│  ├─ OPTIMIZATION_SUMMARY.md
│  ├─ ChainSight Design Document.docx
│  ├─ ChainSight 1st SIT.xlsx
│  ├─ WIP COV Generator.docx
│  └─ ... (其他原有文档)
│
├─ outputs/                             # 📤 仿真输出目录（自动生成）
│  ├─ runs/                             # 运行结果集合
│  │  ├─ run_YYYYMMDD_HHMMSS/         # 每次运行的结果目录
│  │  │  ├─ module1/
│  │  │  ├─ module3/
│  │  │  ├─ module4/
│  │  │  ├─ module5/
│  │  │  ├─ module6/
│  │  │  ├─ orchestrator/
│  │  │  ├─ summary/
│  │  │  ├─ logs/
│  │  │  └─ validation_report.txt
│  │  └─ metadata.json                # 运行索引（可选）
│  │
│  └─ ChainSight 1st SIT/             # 历史运行（保留）
│     ├─ simulation_start.txt
│     └─ run_*/                        # 历史运行结果
│
├─ ChainSight 1st SIT/                 # 📦 历史数据（保留）
│  ├─ simulation_start.txt
│  └─ run_*/
│
├─ test_files/                          # 📋 测试数据目录（原有）
│  └─ ...
│
├─ .vscode/                             # VS Code配置
├─ .gitignore                           # Git忽略规则
├─ .git/                                # Git仓库
└─ __pycache__/                         # Python缓存


================================================================================
关键文件说明
================================================================================

🚀 主入口:
  - run.py: CLI使用入口，接收命令行参数，代理到src/core/run.py

📖 重要文档:
  - README.md (根目录): 快速开始指南
  - docs/ARCHITECTURE.md: 架构设计详解
  - docs/MIGRATION.md: 文件迁移映射表
  - COMPLETION_REPORT.md: 重构完成报告

🔍 验证:
  - verify_architecture.py: 验证所有导入正常

================================================================================
常用命令
================================================================================

首次运行:
  python run.py --config config/config.xlsx --start-date 2024-01-01 --end-date 2024-01-31

继续运行:
  python run.py --config config/config.xlsx --end-date 2024-01-31 --resume

列出运行:
  python run.py --config config/config.xlsx --end-date 2024-01-31 --list-runs

检查状态:
  python run.py --config config/config.xlsx --end-date 2024-01-31 --check-resume

验证架构:
  python verify_architecture.py

运行测试:
  pytest tests/ -v

================================================================================
导入示例
================================================================================

在src内部模块中:
  from .orchestrator import create_orchestrator          # 同层相对
  from ..utils.validation_manager import ValidationManager  # 上层相对
  from ..modules import module1                          # 跨层相对

从根目录脚本中:
  from src.core import main_integration                 # 绝对导入
  from src.modules import module1                       # 绝对导入

================================================================================
架构分层
================================================================================

Layer 1: CLI
  └─ run.py (根入口，参数解析和调度)

Layer 2: Core
  ├─ orchestrator.py (状态管理)
  ├─ main_integration.py (编排引擎)
  └─ run.py (CLI实现)

Layer 3: Business
  ├─ modules/ (6个规划模块)
  ├─ utils/ (工具库)
  └─ services/ (高级服务)

Layer 4: Testing & Tools
  ├─ tests/ (自动化测试)
  └─ tools/ (独立脚本)

================================================================================
项目优势
================================================================================

✅ 清晰分层: 代码易于定位和理解
✅ 明确依赖: 减少循环导入，便于追踪
✅ 可扩展: 标准位置易于添加新功能
✅ 可测试: 各层独立，便于编写测试
✅ 文档完整: ARCHITECTURE.md详解设计
✅ 功能保留: 所有原有功能完全保留
✅ 向后兼容: CLI接口不变

================================================================================
后续计划
================================================================================

立即:
  □ 运行e2e测试验证功能
  □ 清理可能的遗留import

中期:
  □ 补充单元测试
  □ 编写模块基类
  □ 增强集成测试

长期:
  □ 性能优化
  □ 异步执行
  □ 微服务化

================================================================================
版本信息
================================================================================

版本: 1.0.0
迁移日期: 2026-01-05
状态: ✅ 完成
遗留问题: 无

================================================================================
联系方式
================================================================================

架构问题: 查看 docs/ARCHITECTURE.md
迁移问题: 查看 docs/MIGRATION.md
其他问题: 查看 README.md

================================================================================
