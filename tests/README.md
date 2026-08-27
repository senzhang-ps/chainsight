# 测试目录

- `unit/`：不依赖完整业务链路的快速单元测试。
- `contracts/`：模块 API、状态和持久化契约测试。
- `integration/`：跨模块、数据库和恢复流程集成测试。
- `regression/`：历史数据回放、性能和一致性回归测试。
- `diagnostics/`：由环境变量显式启用的输入、配置和差异诊断测试。
- `helpers/`：共享比对函数、运行器和不被 pytest 收集的辅助脚本。

根目录只保留 pytest 共享配置 `conftest.py` 与包初始化文件。
