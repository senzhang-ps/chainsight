# ChainSight 代码功能测试完整指导流程

## 概述
本指南提供从Dev版本运行、重构后版本运行、输出对比到测试报告生成的完整测试流程。

---

## 测试环境准备

### 1. 确认Python环境
```powershell
# 检查Python版本（建议3.9+）
python --version

# 激活虚拟环境（如果有）
# .\venv\Scripts\Activate.ps1
```

### 2. 安装依赖
```powershell
# 安装项目依赖
pip install -r requirements.txt
```

### 3. 准备测试配置文件
确保你有一个测试配置文件，例如：
- `test_files/BC_S5.xlsx`
- `test_files/BC_S9.xlsx`

---

## 步骤1: 运行Dev版本（ChainSight_Dev）

### 1.1 进入Dev目录
```powershell
cd ChainSight_Dev
```

### 1.2 执行Dev版本仿真
```powershell
# 基本运行命令
python run.py --config ..\test_files\BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10

# 或使用PowerShell脚本
.\run.ps1
```

**命令参数说明**：
- `--config`: 配置文件路径
- `--start-date`: 仿真开始日期（格式：YYYY-MM-DD）
- `--end-date`: 仿真结束日期（格式：YYYY-MM-DD）

### 1.3 记录Dev版本运行信息
```powershell
# 运行完成后，记录输出目录和运行时间
# 输出目录示例：test_files/BC_S5/run_20260121_HHMMSS
# 运行时间将在日志中显示
```

**Dev版本输出目录结构**：
```
test_files/BC_S5/
├── simulation_start.txt         # 仿真开始日期
└── run_20260121_HHMMSS/         # 运行结果目录
    ├── orchestrator/             # 协调器状态
    ├── module1/                  # Module1输出
    ├── module3/                  # Module3输出
    ├── module4/                  # Module4输出
    ├── module5/                  # Module5输出
    ├── module6/                  # Module6输出
    └── summary/                  # 汇总报告
```

---

## 步骤2: 运行重构版本（src）

### 2.1 返回项目根目录
```powershell
cd ..
```

### 2.2 执行重构版本仿真
```powershell
# 使用项目根目录的run.py（会自动调用src/core/run.py）
python run.py --config test_files\BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10

# 或使用PowerShell脚本
.\run.ps1
```

### 2.3 记录重构版本运行信息
```powershell
# 运行完成后，记录输出目录和运行时间
# 输出目录示例：outputs/BC_S5/run_20260121_HHMMSS
```

**重构版本输出目录结构**：
```
outputs/BC_S5/
├── simulation_start.txt         # 仿真开始日期
└── run_20260121_HHMMSS/         # 运行结果目录
    ├── orchestrator/             # 协调器状态
    ├── module1/                  # Module1输出
    ├── module3/                  # Module3输出
    ├── module4/                  # Module4输出
    ├── module5/                  # Module5输出
    ├── module6/                  # Module6输出
    └── summary/                  # 汇总报告
```

---

## 步骤3: 对比两个版本的输出结果

### 3.1 使用自动化对比工具

```powershell
# 进入测试文件目录
cd test_files

# 运行对比报告生成器
python generate_optimization_report.py BC_S5\run_20260121_HHMMSS ..\outputs\BC_S5\run_20260121_HHMMSS
```

**命令参数说明**：
- 第一个参数：Dev版本的运行输出目录
- 第二个参数：重构版本的运行输出目录
- `--output`（可选）：指定输出报告文件路径

**完整命令示例**：
```powershell
python generate_optimization_report.py `
    BC_S5\run_20260121_100000 `
    ..\outputs\BC_S5\run_20260121_110000 `
    --output COMPARISON_REPORT_20260121.md
```

### 3.2 手动对比关键输出文件

如果需要手动验证特定模块的输出：

```powershell
# 对比Module5的最后一天输出
$dev_m5 = Import-Csv "BC_S5\run_20260121_HHMMSS\module5\2025-10-10.csv"
$src_m5 = Import-Csv "..\outputs\BC_S5\run_20260121_HHMMSS\module5\2025-10-10.csv"

# 统计行数
Write-Host "Dev版本记录数: $($dev_m5.Count)"
Write-Host "重构版本记录数: $($src_m5.Count)"

# 对比总量
$dev_total = ($dev_m5 | Measure-Object -Property deployed_qty -Sum).Sum
$src_total = ($src_m5 | Measure-Object -Property deployed_qty -Sum).Sum
Write-Host "Dev版本总部署量: $dev_total"
Write-Host "重构版本总部署量: $src_total"
Write-Host "差异: $($src_total - $dev_total)"
```

---

## 步骤4: 生成测试报告

### 4.1 自动生成的对比报告

`generate_optimization_report.py` 会自动生成包含以下内容的Markdown报告：

1. **执行摘要**
   - 测试通过/失败状态
   - 关键发现
   - 性能提升概要

2. **测试环境**
   - 系统信息
   - Python版本
   - 依赖包版本

3. **测试配置**
   - 配置文件信息
   - 仿真日期范围
   - 输出目录

4. **性能对比**
   - 运行时间对比
   - 性能提升百分比
   - 内存使用（如可用）

5. **数据一致性验证**
   - 文件级对比结果
   - 数据差异统计
   - 差异详情

6. **模块级详细分析**
   - Orchestrator状态对比
   - Module1-6各模块输出对比
   - Summary汇总对比

7. **结论与建议**

### 4.2 报告输出位置

```powershell
# 默认输出位置
test_files\OPTIMIZATION_TEST_REPORT_YYYYMMDD_HHMMSS.md

# 自定义输出位置
python generate_optimization_report.py <dev_dir> <src_dir> --output <custom_path>
```

---

## 完整测试流程示例

以下是一个完整的测试执行示例：

```powershell
# ==================== 阶段1: 运行Dev版本 ====================
cd C:\Users\25936\Desktop\Code\chainsight\ChainSight_Dev

# 记录开始时间
$dev_start = Get-Date

# 运行Dev版本
python run.py --config ..\test_files\BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10

# 记录结束时间
$dev_end = Get-Date
$dev_duration = ($dev_end - $dev_start).TotalSeconds
Write-Host "Dev版本运行时间: $dev_duration 秒"

# 记录输出目录（查看最新的run_*目录）
$dev_output = Get-ChildItem ..\test_files\BC_S5\run_* | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host "Dev版本输出: $($dev_output.FullName)"


# ==================== 阶段2: 运行重构版本 ====================
cd ..

# 记录开始时间
$src_start = Get-Date

# 运行重构版本
python run.py --config test_files\BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10

# 记录结束时间
$src_end = Get-Date
$src_duration = ($src_end - $src_start).TotalSeconds
Write-Host "重构版本运行时间: $src_duration 秒"

# 记录输出目录
$src_output = Get-ChildItem outputs\BC_S5\run_* | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host "重构版本输出: $($src_output.FullName)"


# ==================== 阶段3: 性能对比 ====================
$speedup = $dev_duration / $src_duration
Write-Host ""
Write-Host "========== 性能对比 =========="
Write-Host "Dev版本: $dev_duration 秒"
Write-Host "重构版本: $src_duration 秒"
Write-Host "性能提升: $([math]::Round($speedup, 2))x"


# ==================== 阶段4: 生成对比报告 ====================
cd test_files

# 获取相对路径
$dev_rel = Split-Path $dev_output.FullName -Leaf
$src_rel = "..\outputs\BC_S5\" + (Split-Path $src_output.FullName -Leaf)

# 生成报告
python generate_optimization_report.py "BC_S5\$dev_rel" $src_rel

Write-Host ""
Write-Host "========== 测试完成 =========="
Write-Host "报告已生成，请查看最新的 OPTIMIZATION_TEST_REPORT_*.md 文件"
```

---

## 高级测试选项

### 恢复测试（Resume）

如果测试中断，可以从上次中断处继续：

```powershell
# Dev版本恢复
cd ChainSight_Dev
python run.py --config ..\test_files\BC_S5.xlsx --resume

# 重构版本恢复
cd ..
python run.py --config test_files\BC_S5.xlsx --resume
```

### 列出现有运行

```powershell
# 查看所有运行历史
python run.py --config test_files\BC_S5.xlsx --list-runs

# 查看运行状态和是否可恢复
python run.py --config test_files\BC_S5.xlsx --list-runs --start-date 2025-10-06 --end-date 2025-10-10
```

### 强制新运行

```powershell
# 即使存在未完成的运行，也创建新运行
python run.py --config test_files\BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10 --force-new
```

---

## 端到端集成测试

除了生产环境测试，项目还提供端到端集成测试：

```powershell
# 运行Dev版本的E2E测试
cd ChainSight_Dev
python e2e_integration_test.py

# 运行重构版本的E2E测试
cd ..\tests
python e2e_integration_test.py
```

E2E测试会创建简化的测试配置并验证：
- 模块执行顺序
- 数据流正确性
- 库存守恒
- 业务逻辑正确性

---

## 常见问题排查

### 1. 导入错误

```powershell
# 确保Python路径正确
$env:PYTHONPATH = "C:\Users\25936\Desktop\Code\chainsight"
```

### 2. 配置文件找不到

```powershell
# 使用绝对路径
python run.py --config "C:\Users\25936\Desktop\Code\chainsight\test_files\BC_S5.xlsx" --start-date 2025-10-06 --end-date 2025-10-10
```

### 3. 输出目录权限问题

```powershell
# 确保输出目录有写权限
Test-Path outputs\BC_S5\ -PathType Container
# 如果不存在，会自动创建
```

### 4. 查看详细日志

```powershell
# 日志文件位置
# ChainSight_Dev版本: test_files\BC_S5\run_*\logs\
# 重构版本: outputs\BC_S5\run_*\logs\
```

---

## 测试结果解读

### 性能指标
- **运行时间**: 完整仿真的总耗时
- **性能提升**: Dev版本时间 / 重构版本时间
- **目标**: 重构版本应显著快于Dev版本（通常3-5倍）

### 数据一致性指标
- **记录数差异**: 应 < 1%
- **数值差异**: 应 < 0.5%
- **可接受差异来源**: 
  - 浮点数精度
  - 随机数种子差异
  - 并行执行顺序差异

### 功能正确性指标
- ✅ 所有模块成功执行
- ✅ 所有日期都有输出
- ✅ 关键业务逻辑一致（库存守恒、优先级分配等）
- ✅ 没有异常或错误日志

---

## 测试检查清单

使用此清单确保完整的测试流程：

- [ ] 1. Python环境已准备
- [ ] 2. 依赖已安装
- [ ] 3. 测试配置文件已准备
- [ ] 4. Dev版本运行成功
- [ ] 5. Dev版本输出目录已记录
- [ ] 6. Dev版本运行时间已记录
- [ ] 7. 重构版本运行成功
- [ ] 8. 重构版本输出目录已记录
- [ ] 9. 重构版本运行时间已记录
- [ ] 10. 性能对比已完成
- [ ] 11. 数据对比报告已生成
- [ ] 12. 报告已审查
- [ ] 13. 差异已分析（如有）
- [ ] 14. 测试结论已记录

---

## 快速参考

### 常用命令速查

```powershell
# 运行Dev版本
cd ChainSight_Dev
python run.py --config ..\test_files\BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10

# 运行重构版本
cd ..
python run.py --config test_files\BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10

# 生成对比报告
cd test_files
python generate_optimization_report.py BC_S5\run_DEV ..\outputs\BC_S5\run_SRC

# 查看运行历史
python ..\run.py --config BC_S5.xlsx --list-runs

# 恢复中断的运行
python ..\run.py --config BC_S5.xlsx --resume
```

---

## 联系与支持

如有问题，请查看：
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - 系统架构文档
- [TEST_REPORT.md](TEST_REPORT.md) - 历史测试报告
- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - 重构总结

---

**最后更新**: 2026年1月21日
