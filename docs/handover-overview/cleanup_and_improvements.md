# 零漂移重构清理总结

**更新时间：2026-04-15**

本文档详细说明第三阶段（Legacy Wrapper 清理与配置迁移）的完整执行清单，包括删除的文件、创建的文件、改进的地方，以及验证证据。

---

## 1. 删除的文件清单

### 1.1 模块级平铺入口文件（5 个，共 ~1,690 行）

这些文件在第二阶段被识别为"平铺兼容性入口"，第三阶段直接删除。删除理由：通过在 `src/modules/__init__.py` 中使用包别名导入，可以保持向后兼容性，同时消除代码重复。

| 文件 | 行数 | 内容 | 当前替代方案 |
|---|---|---|---|
| `src/modules/module1.py` | ~320 | `demand_planning` 子包的纯重导出 | `from src.modules import demand_planning as module1` |
| `src/modules/module3.py` | ~340 | `mrp_planning` 子包的纯重导出 | `from src.modules import mrp_planning as module3` |
| `src/modules/module4.py` | ~350 | `production_planning` 子包的纯重导出 | `from src.modules import production_planning as module4` |
| `src/modules/module5.py` | ~340 | `deployment_planning` 子包的纯重导出 | `from src.modules import deployment_planning as module5` |
| `src/modules/module6.py` | ~340 | `logistics_execution` 子包的纯重导出 | `from src.modules import logistics_execution as module6` |

**验证方法**：`git log --oneline -- src/modules/module*.py | head -5`

---

### 1.2 核心层单体兼容文件（3 个，共 ~3,500 行）

这些都是单体文件，其对应的包目录（package）已经存在。在 Python 中，package 优先级高于同名 module，因此这些文件实际已成为死代码。删除理由：包目录已经接管了所有功能。

| 文件 | 行数 | 原用途 | 当前权威位置 | 删除原因 |
|---|---|---|---|---|
| `src/core/main_integration.py` | ~1,500 | 文件模式/数据库模式主循环 | `src/core/main_integration/__init__.py` 与其他子文件 | Package 已存在，module 成为死代码 |
| `src/core/orchestrator.py` | ~1,400 | 仓库状态管理 | `src/core/orchestrator/__init__.py` 与其他子文件 | Package 已存在，module 成为死代码 |
| `src/core/run.py` | ~600 | 命令行分发 | `src/core/run/__init__.py` 与其他子文件 | Package 已存在，module 成为死代码 |

**包优先级验证**：当 `src/core/main_integration/` 目录和 `src/core/main_integration.py` 文件同时存在时，`import src.core.main_integration` 始终导入包目录，`main_integration.py` 文件被完全忽略。

**验证方法**：
```bash
python -c "import src.core.main_integration; print(src.core.main_integration.__file__)"
# 输出：.../src/core/main_integration/__init__.py (不是 main_integration.py)
```

---

### 1.3 产生集成的模块适配器（1 个，~150 行）

| 文件 | 原用途 | 当前替代方案 |
|---|---|---|
| `src/core/main_integration/module4_runner.py` | Module4 集成适配逻辑 | 重命名为 `production_runner.py`，函数名规范化 |

**删除原因**：仅作为中间版本存在，已标准化为 `production_runner.py`。

---

### 1.4 未使用的工具模块（7 个，共 ~2,300 行）

这些文件在重构过程中被识别为"优化尝试、设计阶段或废弃方案"，无当前代码引用该模块的任何函数。

| 文件 | 用途 | 为什么删除 | 影响 |
|---|---|---|---|
| `src/utils/duckdb_sql_wrapper.py` | 早期 DuckDB SQL 包装器 | 代码已迭代，当前模块使用底层 DuckDB API | 无依赖，无反向引用 |
| `src/utils/high_perf_executor.py` | 高性能执行器设计 | 优化方向已更改，代码中未使用 | 无依赖，无反向引用 |
| `src/utils/multiprocess_executor.py` | 多进程执行器 | 改为 DuckDB 加速方案 | 无依赖，无反向引用 |
| `src/utils/parallel_optimizer.py` | 并行优化器 | 需求已变，代码未实现 | 无依赖，无反向引用 |
| `src/utils/performance.py` | 性能监测工具 | 移至 `src/services/performance_profiler.py` | 部分功能迁移 |
| `src/utils/process_pool_executor.py` | 进程池执行器 | 优化方案已变 | 无依赖，无反向引用 |
| `src/utils/optimization_config.py` | 优化配置（仅被 duckdb_sql_wrapper 引用） | 只有 deleted 模块依赖它 | 无有效引用 |

**验证方法**：grep 整个 `src/` 确认无反向引用
```bash
grep -r "from src.utils.duckdb_sql_wrapper" src/
grep -r "from src.utils.high_perf_executor" src/
# 返回空
```

---

## 2. 创建的新文件清单

### 2.1 配置集中化（2 个新文件，共 ~100 行）

这是零漂移重构的重要改进：将默认参数从硬编码常数集中到配置文件，实现"配置即代码"。

| 文件 | 用途 | 创建原因 | 好处 |
|---|---|---|---|
| `config/defaults.yaml` | YAML 形式的默认参数权威来源 | 未来需要无需改 Python 代码即可修改参数 | 参数集中、易修改、版本无关 |
| `src/utils/defaults.py` | YAML 加载器，以 Python 常量形式暴露参数 | 让各模块可以 `from src.utils.defaults import DEFAULT_MOQ` | 单一真源、类型安全、IDE 提示 |

**参数列表**：
- 共享参数：`DEFAULT_MOQ`、`DEFAULT_RV`、`DEFAULT_PTF`、`DEFAULT_LSK`、`DEFAULT_LEAD_TIME`、`DEFAULT_HORIZON`
- 生产规划：`DEFAULT_CHANGEOVER_TIME`
- 部署规划：`DEFAULT_PUSH_LEVELS`

---

### 2.2 归一化统一实现（1 个新文件，~250 行）

在重构前，有 5 处独立的"标识符归一化"实现（demand_planning, mrp_planning, deployment_planning, production_planning, orchestrator）。第三阶段创建了统一实现。

| 文件 | 内容 | 为什么创建 | 维护优势 |
|---|---|---|---|
| `src/utils/normalization.py` | 统一的标识符归一化函数与常数 | 消除 5 处重复，形成单一真源 | 修复 bug 一次生效、避免分叉、版本一致 |

**关键函数**：
- `normalize_material(val)`: 去除 `.0` 后缀
- `normalize_location(val)`: zfill(4) 补零
- `normalize_identifiers(df, extra_columns=None)`: 向量化处理，含 NaN 处理

**关键修复**：`fillna('')` 在 `astype(str)` 之前，避免 None 变成字符串 `'nan'`。

---

### 2.3 模块 6 实现迁移（2 个新文件，~680 行）

Module 6（物流执行）原本没有子包结构，现在正式改为 sub-package。

| 文件 | 行数 | 内容 | 原因 |
|---|---|---|---|
| `src/modules/logistics_execution/main.py` | ~450 | 模块 6 核心仿真循环与路线处理 | 模块主入口，原 `module6.py` 逻辑搬迁 |
| `src/modules/logistics_execution/output_writer.py` | ~230 | Module 6 输出生成（6 个 Excel 工作表） | 输出生成逻辑分离，提高模块内聚 |

---

### 2.4 测试与验证

零漂移验证已完成；仓库内置的回归对比脚本已移除，不再作为当前仓库交付内容。

---

## 3. 改进点汇总

### 3.1 导入路径改进

**之前**（易碎）：
```python
from src.modules import module1, module3, module4, module5, module6
```

**之后**（符合 PEP 8，易维护）：
```python
from src.modules import demand_planning as module1
from src.modules import mrp_planning as module3
from src.modules import production_planning as module4
from src.modules import deployment_planning as module5
from src.modules import logistics_execution as module6
```

**改进点**：
- 导入来源与包名一致（demand_planning → module1），便于代码阅读与维护
- 消除平铺中间件，直接导入子包
- 支持旧代码继续运行（通过 `__init__.py` 别名），零破坏性

---

### 3.2 参数集中化

**之前**：默认值硬编码在各模块
```python
# src/modules/mrp_planning/constants.py
DEFAULT_MOQ = 1  # 硬编码
DEFAULT_RV = 1   # 硬编码
```

**之后**：统一配置文件
```yaml
# config/defaults.yaml
shared:
  default_moq: 1
  default_rv: 1
```

调用方统一从 `src/utils/defaults` 导入，改值只需改 YAML，无需改代码。

---

### 3.3 归一化统一

**之前**：5 处独立实现，存在分叉风险
```python
# demand_planning/normalization.py 中的 normalize_identifiers()
# mrp_planning/normalizer.py 中的 normalize_identifiers()
# ... 三处重复
```

**之后**：单一真源
```python
# src/utils/normalization.py
def normalize_identifiers(df, extra_columns=None):
    # 向量化实现，含 NaN 处理
    # 所有模块共用
```

所有模块统一改为：
```python
# 各模块的 normalize.py 或 normalizer.py
from src.utils.normalization import normalize_identifiers
```

---

### 3.4 Production Runner 重命名

**之前**：
```
src/core/main_integration/module4_runner.py
└─ run_module4_integrated()
```

**之后**：
```
src/core/main_integration/production_runner.py
├─ run_daily_production_planning_integrated()
└─ load_current_date_production_gr()
```

**改进点**：名字更清晰（production 比 module4 更有语义），为后续 module5/6 integration 预留空间。

---

## 4. 代码行数统计

| 分类 | 数量 | 行数 |
|---|---|---|
| **删除** | | |
| 模块级平铺文件 | 5 | 1,690 |
| 核心层单体文件 | 3 | 3,500 |
| 适配器文件 | 1 | 150 |
| 未使用工具模块 | 7 | 2,300 |
| **小计删除** | **16** | **~7,640** |
| | | |
| **创建** | | |
| YAML 配置与加载器 | 2 | 100 |
| 归一化统一实现 | 1 | 250 |
| Module 6 拆分 | 2 | 680 |
| **小计创建** | **6** | **~1,330** |
| | | |
| **修改** | | |
| `src/modules/__init__.py` | 1 | 改导入方式 |
| `src/modules/*/` 各常数文件 | 6 | 改为导入自 defaults |
| `src/modules/*/` 各归一化文件 | 3 | 改为导入自 normalization |
| `src/core/main_integration/` 文件 | 2 | 更新导入路径 |
| `requirements.txt` | 1 | 加 pyyaml |
| **小计修改** | **13** | 多处小改 |

**净代码变化**：删除 7,640 行，创建 1,330 行，减少约 6,310 行重复与死代码。

---

## 5. 零漂移验证

### 5.1 测试场景

使用同一份输入数据（OC_Paste_S1_20251224.xlsx），分别在：
- **基线版本**（重构前，原始代码）
- **清理版本**（删除 module*.py 和 utils 工具）
- **最终版本**（完整第三阶段 + YAML 迁移）

上运行 DB 模式仿真，对比最终状态。

### 5.2 验证结果

| 版本 | 运行模式 | 最终库存 | 开放部署 | 在途 | 发货 | 状态 |
|---|---|---|---|---|---|---|
| 基线（原始代码）| DB | 444,761 | 48 | 39 | 1,780 | ✓ 通过 |
| 清理版本 | DB | 444,761 | 48 | 39 | 1,780 | ✓ 通过 |
| 最终版本 | DB | 444,761 | 48 | 39 | 1,780 | ✓ 通过 |

**结论**：完全零漂移。重构对仿真输出无任何影响。

### 5.3 对比说明

零漂移结论来自当时的对比验证结果；当前仓库不再保留内置回归对比脚本。

---

## 6. 后续维护指南

### 6.1 修改默认参数

**只需改 YAML 文件**：
```yaml
# config/defaults.yaml
shared:
  default_moq: 1  # 改这里
```

Python 代码 **完全不需要改动**，下次导入时自动生效。

### 6.2 修复归一化逻辑

**只改一处**：
```python
# src/utils/normalization.py
def normalize_identifiers(df, extra_columns=None):
    # 修改这个函数，所有模块自动受益
```

### 6.3 添加新模块

1. 在 `src/modules/` 创建新的 sub-package
2. 在 `src/modules/__init__.py` 中用别名导入
3. 无需创建平铺入口文件

### 6.4 查看历史

档案库中保留了所有阶段性文档，可从以下入口查阅：
- `docs/_archive/` —— 优化过程文档
- `archive/handover_docs/dev/` —— 原始代码说明

---

## 7. 相关链接

- [交接文档](./handover.md) —— 整体交接说明
- [API 参考](../api/api.md) —— 当前 `src/` 与 `pgsql_db/` 的接口

---

## 8. 修订历史

| 日期 | 版本 | 内容 | 作者 |
|---|---|---|---|
| 2026-04-10 | v1.0 | 初版，记录第三阶段完整清单 | 陈显跃 |
