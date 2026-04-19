# 模块兼容层迁移说明

**编写人**：陈显跃  
**更新时间**：2026-04-15

## 1. 当前结论

截至 2026-04-10，`src/modules` 与 `src/core` 下的所有 legacy wrapper 文件已经被物理删除。  
同时完成了**参数集中化**（YAML 配置）和**归一化统一**（单一真源）的改进。  
本文件描述”旧导入应如何迁移”和”新增的配置管理方式”。

## 2. 已删除的旧文件

### 2.1 模块级平铺入口（`src/modules`）

删除理由：通过在 `src/modules/__init__.py` 中使用包别名导入，可保持向后兼容性。

| 文件 | 行数 | 内容 | 替代方案 |
|---|---|---|---|
| `src/modules/module1.py` | 320 | demand_planning 纯重导出 | `from src.modules import demand_planning as module1` |
| `src/modules/module3.py` | 340 | mrp_planning 纯重导出 | `from src.modules import mrp_planning as module3` |
| `src/modules/module4.py` | 350 | production_planning 纯重导出 | `from src.modules import production_planning as module4` |
| `src/modules/module5.py` | 340 | deployment_planning 纯重导出 | `from src.modules import deployment_planning as module5` |
| `src/modules/module6.py` | 340 | logistics_execution 纯重导出 | `from src.modules import logistics_execution as module6` |

### 2.2 核心层单体文件（`src/core`）

删除理由：对应的包目录（package）已存在。Python 中 package 优先级高于同名 module，因此这些文件是死代码。

| 文件 | 行数 | 原用途 | 当前权威位置 |
|---|---|---|---|
| `src/core/main_integration.py` | 1,500 | 文件/DB 主循环 | `src/core/main_integration/__init__.py` 与其他子文件 |
| `src/core/orchestrator.py` | 1,400 | 仓库状态管理 | `src/core/orchestrator/__init__.py` 与其他子文件 |
| `src/core/run.py` | 600 | 命令行分发 | `src/core/run/__init__.py` 与其他子文件 |
| `src/core/main_integration/module4_runner.py` | 150 | M4 集成适配 | 重命名为 `production_runner.py` |

## 3. 新的权威入口

### 3.1 业务模块

| 旧导入 | 新导入 |
|---|---|
| `src.modules.module1` | `src.modules.demand_planning` |
| `src.modules.module3` | `src.modules.mrp_planning` |
| `src.modules.module4` | `src.modules.production_planning` |
| `src.modules.module5` | `src.modules.deployment_planning` |
| `src.modules.module6` | `src.modules.logistics_execution` |

### 3.2 Core

| 旧导入 | 新导入 |
|---|---|
| `src.core.main_integration` 旧单体文件 | `src.core.main_integration` 包 |
| `src.core.orchestrator` 旧单体文件 | `src.core.orchestrator` 包 |
| `src.core.parallel_executor` 旧单体文件 | `src.core.parallel_executor` 包 |
| `src.core.main_integration.module4_runner` | `src.core.main_integration.production_runner` |

### 3.3 共享工具层【第三阶段更新】

| 需求 | 当前推荐入口 | 备注 |
|---|---|---|
| 跨模块共享默认参数（MOQ、RV、PTF 等） | `src.utils.defaults` | **NEW（YAML 加载，替代了 runtime_defaults）** |
| 统一标识符归一化 | `src.utils.normalization` | **NEW（单一真源，替代了 5 处重复）** |
| 共享日期与前置期 helper | `src.utils.date_helpers` | 保留 |

## 4. 推荐迁移写法

### 4.1 模块别名风格

```python
from src.modules import demand_planning as module1
from src.modules import mrp_planning as module3
from src.modules import production_planning as module4
from src.modules import deployment_planning as module5
from src.modules import logistics_execution as module6
```

### 4.2 M4 集成适配层

```python
from src.core.main_integration.production_runner import run_module4_integrated
from src.core.main_integration.production_runner import load_current_date_production_gr
```

### 4.3 主流程

```python
from src.core.main_integration import run_integrated_simulation
from src.core.main_integration import run_integrated_simulation_from_dict
from src.core.orchestrator import create_orchestrator
```

### 4.4 共享 helper【第三阶段更新】

```python
# 跨模块共享参数（从 YAML 加载）
from src.utils.defaults import DEFAULT_MOQ, DEFAULT_RV, DEFAULT_HORIZON
from src.utils.defaults import DEFAULT_CHANGEOVER_TIME, DEFAULT_PUSH_LEVELS

# 统一的标识符归一化
from src.utils.normalization import normalize_identifiers, normalize_material

# 日期与前置期 helper
from src.utils.date_helpers import compute_planning_window
```

## 5. 为什么要删兼容层

原因有三类：

1. 文件树已经完成从“平铺单体 + 子包并存”到“只保留真实子包”的收口。
2. 继续保留旧文件会让接手人误以为这些文件仍是主要实现位置。
3. 删掉 wrapper 后，结构更清晰，也更符合当前主链路实际导入方式。

## 6. 接手人最容易踩的坑

### 6.1 关于 `from src.modules import module4` 的现状

旧式导入 `from src.modules import module4` **仍然可用**，因为 `src/modules/__init__.py` 通过 `from . import production_planning as module4` 暴露了同名别名（`module1` ~ `module6` 全部如此）。  
但请注意：`src/modules/module4.py` 文件本身已不存在；新代码请直接使用子包名：

```python
# 推荐写法（直接使用子包）
from src.modules import production_planning

# 兼容写法（仍可工作，但仅用于旧代码迁移）
from src.modules import production_planning as module4
```

### 6.2 误以为 `module4_runner.py` 仍在

现在真正的位置是：

```python
src/core/main_integration/production_runner.py
```

### 6.3 把历史文档当成当前结构

如果文档仍描述 `module1.py` 等兼容层存在，应视为历史文档，先以当前文件树为准，再决定是否修订历史资料。

## 7. 建议的迁移检查

做结构改动后，至少检查：

1. 是否还存在对旧文件名的 import
2. `src/modules` 是否只暴露五个真实子包
3. `simulation_file.py` 与 `simulation_db.py` 是否仍直接导向当前包入口
4. `OC_Paste_S1_20251224` 两天 DB 回归是否仍通过
5. 若修改的是公共默认值或 normalize / date helper，是否优先在 `src/utils/` 共享模块落点，而不是重新散落回模块内
