# ChainSight 文档索引

更新时间：2026-04-08

本文档用于给当前仓库提供一份可靠的阅读顺序。以 2026-04-08 的代码状态为准，`src/modules/module1.py` 到 `module6.py`、`src/core/main_integration.py`、`src/core/orchestrator.py`、`src/core/parallel_executor.py` 等旧单体兼容文件已经从代码树中移除，当前应以包目录和子包入口为准。

## 推荐阅读顺序

1. `docs/用户使用入口说明.md`
2. `docs/QUICK_REFERENCE.md`
3. `docs/交接文档/Refactored/项目交接文档.md`
4. `docs/交接文档/Refactored/setup.md`
5. `docs/交接文档/Refactored/API.md`
6. `docs/交接文档/Refactored/modules_compat.md`

## 当前最重要的文档

| 文档 | 用途 |
|---|---|
| `docs/用户使用入口说明.md` | 当前最实用的运行、排查和交付入口说明 |
| `docs/QUICK_REFERENCE.md` | 常用命令、目录、导入路径、验证命令速查 |
| `docs/交接文档/Refactored/项目交接文档.md` | 当前代码状态、三阶段收口结果、验证证据、后续建议 |
| `docs/交接文档/Refactored/setup.md` | 环境、依赖、数据库初始化与部署步骤 |
| `docs/交接文档/Refactored/API.md` | 当前 `src/` 与 `pgsql_db/` 的接口参考 |
| `docs/OPTIMIZED_CODE_STRUCTURE_DIAGRAM.md` | 优化后代码结构图与旧版到新版目录映射 |

## 结构与架构文档

| 文档 | 用途 |
|---|---|
| `docs/OPTIMIZED_CODE_STRUCTURE_DIAGRAM.md` | 以当前目录结构为准的静态结构图说明 |
| `docs/ARCHITECTURE_DIAGRAM.md` | 历史架构总览，可作背景参考，不再作为“当前文件树”权威来源 |
| `docs/MIGRATION.md` | 历史迁移说明 |
| `docs/README_REFACTORING_MAP.md` | 重构映射背景材料 |
| `docs/MODULE3_DESIGN.md` | M3 设计说明 |
| `docs/MODULE5_DESIGN.md` | M5 设计说明 |

## 交接材料

| 文档 | 用途 |
|---|---|
| `docs/交接文档/Refactored/项目交接文档.md` | 总体交接说明 |
| `docs/交接文档/Refactored/setup.md` | 环境与运行交接 |
| `docs/交接文档/Refactored/API.md` | API 与调用边界交接 |
| `docs/交接文档/Refactored/core.md` | Core 层说明 |
| `docs/交接文档/Refactored/module.md` | 业务模块总览 |
| `docs/交接文档/Refactored/modules_compat.md` | 旧导入路径迁移说明，重点说明“哪些旧文件已经删除” |
| `docs/交接文档/Refactored/services.md` | Services 层说明 |
| `docs/交接文档/Refactored/utils.md` | Utils 层说明 |

## 验证与专项报告

| 文档 | 用途 |
|---|---|
| `docs/PERFORMANCE_OPTIMIZATION_REPORT.md` | 性能优化总览 |
| `docs/CYTHON_OPTIMIZATION_REPORT.md` | Cython 专项说明 |
| `docs/DUCKDB_OPTIMIZATION_GUIDE.md` | DuckDB 优化说明 |
| `docs/ORI_DEPLOYMENT_UID_ANALYSIS.md` | UID 兼容性分析 |
| `docs/20260127变更版本与修复.md` | 历史修复记录 |

## 历史与归档

| 目录 | 用途 |
|---|---|
| `docs/_archive/` | 历史过程材料 |
| `docs/交接文档/dev/` | Dev 基线说明 |

## 当前代码树的权威入口

- CLI 入口：`run.py`
- 运行分发：`src/core/run.py`、`src/core/run/*`
- 主集成流程：`src/core/main_integration/*`
- 共享状态：`src/core/orchestrator/*`
- 业务模块：`src/modules/demand_planning/*`、`src/modules/mrp_planning/*`、`src/modules/production_planning/*`、`src/modules/deployment_planning/*`、`src/modules/logistics_execution/*`
- 数据库边界：`pgsql_db/*`

## 重要提醒

- 若文档仍提到 `src/modules/module1.py` 之类的文件名，应视为历史资料，除非文档明确写明“历史兼容层”。
- 数据库模式下，`--config` 推荐直接传配置名，不带引号，除非参数值本身含空格。
- 当前已验证通过的两天数据库回归命令为：

```powershell
.\.venv\Scripts\python run.py --config OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2025-12-16 --use-db
```
