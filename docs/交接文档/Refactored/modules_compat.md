# 模块兼容层迁移说明

更新时间：2026-04-08

## 1. 当前结论

截至 2026-04-08，`src/modules` 与 `src/core` 下的大部分 legacy wrapper 文件已经被物理删除。  
本文件不再描述“兼容层如何保留”，而是描述“旧导入应如何迁移”。

## 2. 已删除的旧文件

### 2.1 `src/modules`

- `src/modules/module1.py`
- `src/modules/module3.py`
- `src/modules/module4.py`
- `src/modules/module5.py`
- `src/modules/module6.py`

### 2.2 `src/core`

- `src/core/main_integration.py`
- `src/core/orchestrator.py`
- `src/core/parallel_executor.py`
- `src/core/main_integration/module4_runner.py`

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
| `src.core.main_integration.module4_runner` | `src.core.main_integration.production_integration` |

### 3.3 共享工具层

| 需求 | 当前推荐入口 |
|---|---|
| 共享小默认值 | `src.utils.runtime_defaults` |
| 共享标识符标准化 helper | `src.utils.normalization_common` |
| 共享窗口 / review day / lead time helper | `src.utils.date_helpers` |

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
from src.core.main_integration.production_integration import run_module4_integrated
from src.core.main_integration.production_integration import load_current_date_production_gr
```

### 4.3 主流程

```python
from src.core.main_integration import run_integrated_simulation
from src.core.main_integration import run_integrated_simulation_from_dict
from src.core.orchestrator import create_orchestrator
```

### 4.4 共享 helper

```python
from src.utils.runtime_defaults import DEFAULT_MOQ, DEFAULT_RV
from src.utils.normalization_common import normalize_identifiers_vectorized
from src.utils.date_helpers import compute_planning_window
```

## 5. 为什么要删兼容层

原因有三类：

1. 文件树已经完成从“平铺单体 + 子包并存”到“只保留真实子包”的收口。
2. 继续保留旧文件会让接手人误以为这些文件仍是主要实现位置。
3. 删掉 wrapper 后，结构更清晰，也更符合当前主链路实际导入方式。

## 6. 接手人最容易踩的坑

### 6.1 误以为 `from src.modules import module4` 仍然可用

现在不再可用，应改为：

```python
from src.modules import production_planning as module4
```

### 6.2 误以为 `module4_runner.py` 仍在

现在真正的位置是：

```python
src/core/main_integration/production_integration.py
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
