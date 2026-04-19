# ChainSight 文档索引

更新时间：2026-04-19

本文档给当前仓库提供一份可靠的阅读顺序。以当前代码状态为准，`src/modules/module1.py` 到 `module6.py`、`src/core/main_integration.py`、`src/core/orchestrator.py` 等旧单体兼容文件已经从代码树中移除。同时完成了参数集中化（YAML 配置）、归一化统一（单一真源）和模块命名一致性改进，当前应以包目录和子包入口为准。

## 推荐阅读顺序

1. `docs/handover/refactored/00-overview/handover.md`
2. `docs/handover/refactored/00-overview/setup.md`
3. `docs/handover/refactored/01-architecture/architecture.md`
4. `docs/handover/refactored/02-api/api.md`
5. `docs/handover/refactored/03-modules/modules_compat.md`
6. `docs/handover/refactored/00-overview/cleanup_and_improvements.md`

## 当前最重要的文档

| 文档 | 用途 |
|---|---|
| `docs/handover/refactored/00-overview/handover.md` | 当前代码状态、三阶段收口结果、验证证据、后续建议 |
| `docs/handover/refactored/00-overview/cleanup_and_improvements.md` | **详细的清理清单** —— 删除了什么文件、创建了什么文件、改进了什么地方、验证证据 |
| `docs/handover/refactored/00-overview/setup.md` | 环境、依赖、数据库初始化与部署步骤 |
| `docs/handover/refactored/02-api/api.md` | 当前 `src/` 与 `pgsql_db/` 的接口参考 |
| `docs/handover/refactored/01-architecture/architecture.md` | 系统架构说明 |

## 结构与架构文档

| 文档 | 用途 |
|---|---|
| `docs/handover/refactored/01-architecture/architecture.md` | 系统架构与模块关系说明 |
| `docs/handover/refactored/01-architecture/core.md` | Core 层说明 |
| `docs/handover/refactored/01-architecture/module_sequence_diagrams.md` | 模块间调用时序图 |
| `docs/handover/refactored/02-api/function_interface_spec.md` | 函数接口与数据格式参考 |
| `docs/handover/refactored/02-api/function_dependency_matrix.md` | 函数依赖关系矩阵 |

## 交接材料

### 核心交接文档

| 文档 | 用途 |
|---|---|
| `docs/handover/refactored/00-overview/handover.md` | 总体交接说明 |
| `docs/handover/refactored/00-overview/cleanup_and_improvements.md` | **清理与改进详细记录** |
| `docs/handover/refactored/00-overview/setup.md` | 环境与运行交接 |
| `docs/handover/refactored/02-api/api.md` | API 与调用边界交接 |
| `docs/handover/refactored/03-modules/modules_compat.md` | 旧导入路径迁移说明，重点说明"哪些旧文件已经删除" |

### 技术文档

| 文档 | 用途 |
|---|---|
| `docs/handover/refactored/03-modules/modules.md` | 业务模块总览 |
| `docs/handover/refactored/04-services-utils/services.md` | Services 层说明 |
| `docs/handover/refactored/04-services-utils/utils.md` | Utils 层说明（含 defaults.py、normalization.py、resource_config.py） |
| `docs/handover/refactored/03-modules/modules_demand_planning.md` | 模块 1（需求规划）说明 |
| `docs/handover/refactored/03-modules/modules_mrp_planning.md` | 模块 3（MRP 规划）说明 |
| `docs/handover/refactored/03-modules/modules_production_planning.md` | 模块 4（生产计划）说明 |
| `docs/handover/refactored/03-modules/modules_deployment_planning.md` | 模块 5（部署规划）说明 |
| `docs/handover/refactored/03-modules/modules_logistics_execution.md` | 模块 6（物流执行）说明 |

### 算法优化报告

| 文档 | 用途 |
|---|---|
| `docs/handover/refactored/05-testing/oc_algorithm_test_report.md` | OC 算法优化测试报告 |
| `docs/handover/refactored/05-testing/bc_algorithm_test_report.md` | BC 算法优化测试报告 |

## 历史与归档

| 目录 | 用途 |
|---|---|
| `docs/_archive/` | 历史过程材料（SOW、WIP COV Generator 等） |
| `archive/handover_docs/dev/` | Dev 基线源码说明 |
| `archive/handover_docs/chainsight_refactored_handover.docx` | 交接文档 docx 归档版 |
| `archive/ChainSight_Dev/` | Dev 基线完整源码归档 |

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
**更新时间**：2026-04-19  
**版本**：Phase-3 Complete + Refactor Cleanup
