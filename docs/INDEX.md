# ChainSight 文档索引

更新时间：2026-04-15

本文档用于给当前仓库提供一份可靠的阅读顺序。以 2026-04-15 的代码状态为准，`src/modules/module1.py` 到 `module6.py`、`src/core/main_integration.py`、`src/core/orchestrator.py` 等旧单体兼容文件已经从代码树中移除。同时完成了参数集中化（YAML 配置）、归一化统一（单一真源）和模块命名一致性改进，当前应以包目录和子包入口为准。

## 推荐阅读顺序

1. `docs/交接文档/Refactored/项目交接文档.md`
2. `docs/交接文档/Refactored/setup.md`
3. `docs/交接文档/Refactored/ARCHITECTURE.md`
4. `docs/交接文档/Refactored/API.md`
5. `docs/交接文档/Refactored/modules_compat.md`
6. `docs/交接文档/Refactored/CLEANUP_AND_IMPROVEMENTS.md`

## 当前最重要的文档

| 文档 | 用途 |
|---|---|
| `docs/交接文档/Refactored/项目交接文档.md` | 当前代码状态、三阶段收口结果、验证证据、后续建议 |
| `docs/交接文档/Refactored/CLEANUP_AND_IMPROVEMENTS.md` | **详细的清理清单** —— 删除了什么文件、创建了什么文件、改进了什么地方、验证证据 |
| `docs/交接文档/Refactored/setup.md` | 环境、依赖、数据库初始化与部署步骤 |
| `docs/交接文档/Refactored/API.md` | 当前 `src/` 与 `pgsql_db/` 的接口参考 |
| `docs/交接文档/Refactored/ARCHITECTURE.md` | 系统架构说明 |

## 结构与架构文档

| 文档 | 用途 |
|---|---|
| `docs/交接文档/Refactored/ARCHITECTURE.md` | 系统架构与模块关系说明 |
| `docs/交接文档/Refactored/CLEANUP_AND_IMPROVEMENTS.md` | 清理详细清单（删除、创建、改进、验证） |
| `docs/交接文档/Refactored/模块级时序图文档.md` | 模块间调用时序图 |
| `docs/交接文档/Refactored/函数级接口与文件格式总表.md` | 函数接口与数据格式参考 |
| `docs/交接文档/Refactored/函数上下游依赖矩阵.md` | 函数依赖关系矩阵 |

### 历史文档与过程材料

过程性文档已清理。`docs/_archive/` 仅保留必要的参考说明。

## 交接材料

### 核心交接文档

| 文档 | 用途 |
|---|---|
| `docs/交接文档/Refactored/项目交接文档.md` | 总体交接说明 |
| `docs/交接文档/Refactored/CLEANUP_AND_IMPROVEMENTS.md` | **清理与改进详细记录** |
| `docs/交接文档/Refactored/setup.md` | 环境与运行交接 |
| `docs/交接文档/Refactored/API.md` | API 与调用边界交接 |
| `docs/交接文档/Refactored/modules_compat.md` | 旧导入路径迁移说明，重点说明"哪些旧文件已经删除" |

### 技术文档

| 文档 | 用途 |
|---|---|
| `docs/交接文档/Refactored/core.md` | Core 层说明 |
| `docs/交接文档/Refactored/module.md` | 业务模块总览 |
| `docs/交接文档/Refactored/services.md` | Services 层说明 |
| `docs/交接文档/Refactored/utils.md` | Utils 层说明（含 defaults.py、normalization.py、resource_config.py） |
| `docs/交接文档/Refactored/modules_demand_planning.md` | 模块 1（需求规划）说明 |
| `docs/交接文档/Refactored/modules_mrp_planning.md` | 模块 3（MRP 规划）说明 |
| `docs/交接文档/Refactored/modules_production_planning.md` | 模块 4（生产计划）说明 |
| `docs/交接文档/Refactored/modules_deployment_planning.md` | 模块 5（部署规划）说明 |
| `docs/交接文档/Refactored/modules_logistics_execution.md` | 模块 6（物流执行）说明 |

### 算法优化报告

| 文档 | 用途 |
|---|---|
| `docs/交接文档/Refactored/OC算法优化测试报告.md` | OC 算法优化测试报告 |
| `docs/交接文档/Refactored/BC算法优化测试报告.md` | BC 算法优化测试报告 |

## 历史与归档

| 目录 | 用途 |
|---|---|
| `docs/_archive/` | 历史过程材料 |
| `docs/交接文档/dev/` | Dev 基线说明 |

## 当前代码树的权威入口

- CLI 入口：`run.py`
- 运行分发：`src/core/run/`（包含 `run_main.py`、`db_runner.py`、`local_writer.py` 等）
- 主集成流程：`src/core/main_integration/`
- 共享状态：`src/core/orchestrator/`
- 业务模块：`src/modules/demand_planning/`、`src/modules/mrp_planning/`、`src/modules/production_planning/`、`src/modules/deployment_planning/`、`src/modules/logistics_execution/`
- 数据库边界：`pgsql_db/`

## 重要提醒

- 若文档仍提到 `src/modules/module1.py` 之类的文件名，应视为历史资料，除非文档明确写明"历史兼容层"。
- 数据库模式下，`--config` 推荐直接传配置名，不带引号，除非参数值本身含空格。
- 当前已验证通过的两天数据库回归命令为：

```powershell
.\.venv\Scripts\python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db
```

---

## 文档信息

**编写与整理**：陈显跃  
**更新时间**：2026-04-15  
**版本**：Phase-3 Complete + Refactor Cleanup
