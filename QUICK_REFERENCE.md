# 🚀 ChainSight 快速参考卡

## 项目概览

```
ChainSight - 供应链规划仿真系统
标准分层架构 (6层) | 41个源文件 | 完全文档化
```

## 核心命令

```bash
# 验证架构
python verify_architecture.py

# 查看CLI帮助
python run.py --help

# 首次运行
python run.py \
  --config config/config.xlsx \
  --start-date 2024-01-01 \
  --end-date 2024-01-31

# 继续运行
python run.py --config config/config.xlsx --end-date 2024-01-31 --resume

# 检查状态
python run.py --config config/config.xlsx --end-date 2024-01-31 --check-resume

# 列出所有运行
python run.py --config config/config.xlsx --end-date 2024-01-31 --list-runs
```

## 目录导航

| 需求 | 位置 | 说明 |
|------|------|------|
| 查看说明 | README.md | 项目概览 |
| 学习架构 | docs/ARCHITECTURE.md ⭐ | 必读 |
| 快速开始 | QUICK_START.md | 5分钟上手 |
| 文件对应 | docs/MIGRATION.md | 迁移指南 |
| 添加模块 | src/modules/ | 新建.py文件 |
| 添加工具 | src/utils/ | 新建.py文件 |
| 添加服务 | src/services/ | 新建.py文件 |
| 编写测试 | tests/ | 新建test_*.py |
| 配置文件 | config/*.xlsx | 配置数据 |
| 依赖管理 | requirements.txt | pip install |

## 项目结构速览

```
src/
├── core/              # 编排 + 状态管理
│   ├── orchestrator.py        (状态中枢)
│   ├── main_integration.py    (编排引擎)
│   └── run.py                 (CLI实现)
│
├── modules/           # 业务规划 (6个模块)
│   ├── module1.py             (库存规划)
│   ├── module3.py             (交付规划)
│   ├── module4.py             (生产排程)
│   ├── module5.py             (库存优化)
│   └── module6.py             (物流执行)
│
├── utils/             # 工具库 (5个工具)
│   ├── config_validator.py    (配置验证)
│   ├── validation_manager.py  (数据验证)
│   ├── inventory_balance_checker.py
│   ├── logger_config.py       (日志)
│   └── time_manager.py        (时间)
│
└── services/          # 高级服务
    ├── summary_report_generator.py
    └── performance_profiler.py

tests/                # 测试
tools/                # 独立脚本
config/               # 配置
docs/                 # 文档
```

## 常见问题速解

**Q: 如何添加新模块?**
```python
# 在 src/modules/module_new.py 中实现
def execute(date, config, orchestrator, historical_data):
    # 业务逻辑
    return {'module_outputs': ..., 'state_updates': ...}
```

**Q: 如何运行测试?**
```bash
pytest tests/ -v
```

**Q: 配置文件在哪?**
```bash
config/*.xlsx  # Excel配置文件
config/requirements.txt  # Python依赖
```

**Q: 输出在哪?**
```bash
outputs/runs/run_YYYYMMDD_HHMMSS/  # 每个运行的输出
```

**Q: 如何修改日志级别?**
```python
# 在 src/utils/logger_config.py 中修改
logger_config.setup_logging(..., log_level="DEBUG")
```

## 导入示例

### 在src内部导入
```python
# src/core/main_integration.py
from .orchestrator import create_orchestrator
from ..utils.validation_manager import ValidationManager
from ..modules import module1
from ..services.summary_report_generator import SummaryReportGenerator
```

### 从根目录导入
```python
# verify_architecture.py 或其他根脚本
from src.core.run import main
from src.core.main_integration import run_integrated_simulation
```

## 性能提示

- 📊 **数据处理**: 使用pandas DataFrame (src/modules/)
- 🔄 **并行化**: Module1 支持并行AO消耗计算
- 💾 **缓存**: Orchestrator管理状态缓存
- 📈 **监控**: 使用 PerformanceProfiler 追踪耗时

## 文件大小概览

- 核心文件: ~50KB (orchestrator + main_integration)
- 模块文件: ~200KB 总计 (6个模块)
- 工具文件: ~50KB 总计
- 测试文件: ~50KB 总计
- 总源代码: ~400KB

## 开发工作流

```
1. 创建新功能分支
   git checkout -b feature/xxx

2. 在合适目录下创建文件
   src/modules/新功能.py

3. 编写代码 + 测试
   tests/test_新功能.py

4. 验证架构
   python verify_architecture.py

5. 运行测试
   pytest tests/

6. 提交PR
   git push origin feature/xxx
```

## 故障排除

| 问题 | 解决方案 |
|------|--------|
| ImportError | 检查 src 路径是否在 sys.path |
| 找不到配置 | 使用相对路径: config/config.xlsx |
| 模块不存在 | 确认文件在 src/modules/ |
| CLI不工作 | 运行 python run.py --help |
| 导入失败 | 运行 python verify_architecture.py |

## 关键概念

- **Orchestrator**: 全局状态管理中枢
- **Module**: 独立的业务规划模块
- **Validator**: 数据验证器
- **Manager**: 资源管理器 (时间、日志等)
- **Generator**: 报告/输出生成器
- **Profiler**: 性能分析工具

## 扩展点

```python
# 1. 添加新验证器
class MyValidator(ValidationManager):
    pass

# 2. 添加新报告
class MyReportGenerator:
    def generate(self, data):
        pass

# 3. 添加新服务
class MyService:
    pass

# 4. 添加新模块
def execute(...):
    pass
```

---

**更新日期**: 2026-01-05  
**版本**: 1.0.0  
**维护者**: AI Assistant
