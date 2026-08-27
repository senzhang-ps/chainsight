# Plan: 新基线回归定位

目标是验证：当前 `input` schema 是否由纯 `8ef13bb1` 的 pandas + DQ 逻辑能够复现。当前 `legacy` schema 由 `8ef13bb1` 基础上的未提交内容产生，仅作为辅助参考，不作为纯提交基线。

## 步骤

1. **保存当前成果**
	- 在当前 Decoupling 工作区提交所有未提交修复、测试、审计工具与文档，创建不可变 checkpoint。
	- 提交前检查工作区状态，记录提交 SHA、变更文件清单及当前最终 pandas/Polars 审计报告。

2. **创建纯提交对比环境**
	- 从 `8ef13bb1ef58513d1bd1f288db2162bf00999cdd` 创建独立 detached worktree。
	- 不在该 worktree 引入当前未提交或 checkpoint 后的修复。
	- 确认该版本仅支持固定 `test` schema，不修改历史运行入口来支持自定义 schema。

3. **纯提交 pandas + DQ 回放**
	- 在纯 `8ef13bb1` worktree 使用同一个 Excel：
	  `input/OC/OC_Paste_S1_20251224_extension/OC_Paste_S1_20251224_repare.xlsx`。
	- 范围固定为 `2025-12-15` 至 `2025-12-25`。
	- 使用 pandas 引擎，启用 DQ，不传跳过 DQ 参数。
	- 将运行输出写入该版本固定的 `test` schema，并记录实际 `run_id`、DQ 结果、配置名、运行日志和性能报告。

4. **`input` 对纯 `8ef13bb1` pandas 的审计**
	- 使用当前审计脚本或纯提交环境可用的等价只读比较工具，对以下 run 做逐表审计：
	  - 左侧：`input/db_OC_Paste_S1_20251224_repare_20260820_163016`
	  - 右侧：纯 `8ef13bb1` 生成的 `test/<run_id>`。
	- 首先确认 M1 的五张输出：OrderLog、ShipmentLog、CutLog、SupplyDemandLog、Summary。
	- 采用已验证的业务规则：排除运行元数据、技术顺序列和零值占位行；对重复 DeploymentPlan 使用全业务字段去重。

5. **按模块定位**
	- 若 M1 已一致：标记“新 input 可由纯 `8ef13bb1` M1 复现”，继续 M4。
	- 若 M1 不一致：锁定为 `85c43cc2 → 8ef13bb1` 之间引入的变更或 legacy 未提交工作区差异；对 `build_cov()`、日度订单拆分、随机采样和订单持久化逐步骤对比。
	- 仅在上游模块一致时按 **M1 → M4 → M5 → M6 → M3** 推进，防止修复下游症状。

6. **提交区间归因**
	- 如果纯 `8ef13bb1` 与 `input` 一致，则比较纯 `8ef13bb1` 与当前 checkpoint 的提交差异，定位今天或后续修复中引入的新回归。
	- 如果纯 `8ef13bb1` 与 `input` 仍不一致，则比较 `input` 的实际产生环境与该 worktree，包括未提交文件、依赖版本、DQ 处理结果、运行参数、断点恢复和随机种子状态。
	- 每个模块仅在根因确认后提交修复方案，等待确认再实施。

7. **恢复当前主线并复测**
	- 在当前 checkpoint 主线实施获确认的最小修复。
	- 先执行 pandas + DQ 完整回放，对比 `input`。
	- pandas 审计通过后，再运行 Polars + DQ 全链路并对比 pandas/`input`。
	- 最终生成新版业务审计与回归报告。

## 相关文件

- `run.py` — 当前 CLI 入口。
- `src/core/run/run_main.py` — 参数解析与 DQ/数据库模式分发。
- `src/core/run/db_runner.py` — 数据库配置加载和固定 schema 运行行为。
- `tests/_compare_legacy_refactor_schema_runs.py` — 全量数据库业务审计。
- `tests/_compare_config_schemas.py` — `cfg_*` schema 只读比较。
- `tests/regression/test_m1_two_way_compare.py` — M1 历史输出与 pandas/Polars 受控对比。
- `src/modules/demand_planning/backends.py` — M1 的订单生成、随机误差和日度拆分。

## 验证

1. 当前主线 checkpoint 提交成功且工作区干净。
2. 纯 `8ef13bb1` worktree 的 pandas + DQ 11 天运行完成，`test` schema 有唯一运行。
3. `input` 与纯提交 pandas 的审计报告明确给出 M1 首差或 M1 一致结论。
4. 按模块逐步提供可复现证据，未确认前不修改业务逻辑。
5. 最终 pandas 和 Polars 均以同一新 `input` run 审计通过。

## 决策

- 纯 `8ef13bb1` worktree 是本轮 pandas 历史对照，不使用当前工作区的修复。
- `legacy` schema 不是纯 `8ef13bb1` 对照，因为它来自该提交上的未提交工作区。
- DQ 必须启用，以复刻本次用户要求的运行条件。
