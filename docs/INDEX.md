# ChainSight 文档索引

## 说明

本文档用于整理当前仓库 `docs/` 目录中的有效入口文档，帮助接手人员、使用人员与维护人员快速找到对应材料。

本次清理后，`docs/` 的阅读顺序建议为：

1. `docs/用户使用入口说明.md`
2. `docs/交接文档/Refactored/项目交接文档.md`
3. `docs/交接文档/Refactored/setup.md`
4. `docs/交接文档/Refactored/API.md`

## 一、对外/对接入口

| 文档 | 用途 |
|---|---|
| `docs/用户使用入口说明.md` | 面向使用者与接手人员的最小操作入口说明 |
| `docs/QUICK_REFERENCE.md` | 常用命令、目录与快速跳转参考 |
| `docs/交接文档/Refactored/项目交接文档.md` | 正式交接总文档，建议优先阅读 |
| `docs/交接文档/Refactored/setup.md` | 环境部署、运行模式、数据库初始化与常见问题 |
| `docs/交接文档/Refactored/API.md` | `src/` 与 `pgsql_db/` 当前 API 参考 |

## 二、架构与模块设计

| 文档 | 用途 |
|---|---|
| `docs/ARCHITECTURE_DIAGRAM.md` | 顶层架构可视化说明 |
| `docs/MIGRATION.md` | 迁移与演进说明 |
| `docs/MODULE3_DESIGN.md` | Module3 设计说明 |
| `docs/MODULE5_DESIGN.md` | Module5 设计说明 |
| `docs/README_REFACTORING_MAP.md` | 重构路径、模块映射与结构对照 |
| `docs/交接文档/Refactored/ARCHITECTURE.md` | 交接视角下的架构说明 |
| `docs/交接文档/Refactored/core.md` | Core 层补充说明 |
| `docs/交接文档/Refactored/module.md` | 模块总览补充说明 |
| `docs/交接文档/Refactored/services.md` | Services 层补充说明 |
| `docs/交接文档/Refactored/utils.md` | Utils 层补充说明 |
| `docs/交接文档/Refactored/modules_demand_planning.md` | M1 拆分说明 |
| `docs/交接文档/Refactored/modules_mrp_planning.md` | M3 拆分说明 |
| `docs/交接文档/Refactored/modules_production_planning.md` | M4 拆分说明 |
| `docs/交接文档/Refactored/modules_deployment_planning.md` | M5 拆分说明 |
| `docs/交接文档/Refactored/modules_logistics_execution.md` | M6 拆分说明 |
| `docs/交接文档/Refactored/modules_compat.md` | 兼容层说明 |

## 三、验证、分析与专项报告

| 文档 | 用途 |
|---|---|
| `docs/交接文档/Refactored/BC算法优化测试报告.md` | BC 场景优化结果与一致性验证 |
| `docs/交接文档/Refactored/OC算法优化测试报告.md` | OC 场景优化结果与一致性验证 |
| `docs/PERFORMANCE_OPTIMIZATION_REPORT.md` | 性能优化总体结果说明 |
| `docs/CYTHON_OPTIMIZATION_REPORT.md` | Cython 优化专项说明 |
| `docs/ORI_DEPLOYMENT_UID_ANALYSIS.md` | UID 兼容性问题分析 |
| `docs/20260127变更版本与修复.md` | 一次具体重构验证与修复记录 |

## 四、交接配套资料

| 文档 | 用途 |
|---|---|
| `docs/交接文档/dev/ChainSight_Dev源码说明汇总.md` | Dev 基线版本的源码说明汇总 |
| `docs/交接文档/Refactored/函数级接口与文件格式总表.md` | 函数输入输出与文件格式对照 |
| `docs/交接文档/Refactored/函数上下游依赖矩阵.md` | 上下游依赖矩阵 |
| `docs/交接文档/Refactored/模块级时序图文档.md` | 模块执行时序说明 |
| `docs/交接文档/Refactored/中文注释规范与维护约定.md` | 中文注释维护规范 |
| `docs/交接文档/Refactored/纯注释文档改动可提交清单.md` | 纯注释/文档改动的提交边界参考 |

## 五、保留的二进制/外部文档

| 文档 | 状态 | 说明 |
|---|---|---|
| `docs/ChainSight 性能优化.docx` | 保留 | 作为优化背景与对外材料保留，未纳入本次归档 |
| `docs/ChainSight Design Document.docx` | 保留 | 作为设计背景文档保留 |

## 六、已归档文档

历史过程性、阶段性、开发中间态材料已统一转移至：

- `docs/_archive/`

如需追溯本次清理前的优化过程、重构阶段说明或过程性职责文档，请从该目录查阅。
