# 数据对比工具使用指南

## 📁 对比脚本清单

项目中有3个专门的数据对比脚本，功能各有侧重：

### 1. compare_all_outputs.py - 快速全面对比 ⭐推荐
**位置**: `test_files/compare_all_outputs.py`

**功能**:
- ✅ 快速对比所有模块输出（module1-6, orchestrator, summary）
- ✅ CSV和Excel文件全面支持
- ✅ 列级别和数值级别对比
- ✅ 文件完整性检查（文件存在性、行数、列名）
- ✅ 忽略时间戳等运行时列
- ✅ 输出清晰的统计汇总

**使用方法**:
```powershell
cd test_files

# 方法1: 使用默认路径（需要修改脚本中的CODE_VO_OUTPUT和SRC_OUTPUT）
python compare_all_outputs.py

# 方法2: 指定两个输出目录
python compare_all_outputs.py BC_S5\run_20260121_100000 ..\outputs\BC_S5\run_20260121_110000
```

**输出示例**:
```
======================================================================
ChainSight 输出比较
======================================================================
基准目录: BC_S5\run_20260121_100000
优化目录: ..\outputs\BC_S5\run_20260121_110000
======================================================================

📁 MODULE1:
  ✅ 2025-10-06.csv: 完全一致
  ✅ 2025-10-07.csv: 完全一致
   小计: 2/2 文件匹配

📁 MODULE5:
  ❌ 2025-10-10.csv: 数值差异 (列: ['deployed_qty'])
   小计: 4/5 文件匹配

======================================================================
📊 总计: 45/47 文件匹配 (95.7%)
======================================================================
```

---

### 2. compare_outputs_detail.py - 详细差异分析
**位置**: `test_files/compare_outputs_detail.py`

**功能**:
- ✅ 逐列详细差异分析
- ✅ 显示差异的具体列名和差异数量
- ✅ 数值差异的最大值统计
- ✅ 字符串列的不同项计数
- ✅ Excel多Sheet详细对比

**使用方法**:
```powershell
cd test_files

python compare_outputs_detail.py <基准目录> <目标目录>

# 示例
python compare_outputs_detail.py `
    BC_S5\run_20260121_100000 `
    ..\outputs\BC_S5\run_20260121_110000
```

**输出示例**:
```
📁 MODULE5:
  ❌ 2025-10-10.csv:
     - 列 'deployed_qty': 308 个值有差异 (最大差: 768.000000)
     - 列 'deployed_qty_invCon': 308 个值有差异 (最大差: 768.000000)
  ✅ 2025-10-09.csv: 完全一致

📊 总计: 45/47 文件完全一致
```

---

### 3. compare_orchestrator.py - Orchestrator专用对比
**位置**: `test_files/compare_orchestrator.py`

**功能**:
- ✅ 专注于Orchestrator状态文件对比
- ✅ 同时支持其他模块的对比
- ✅ 适合调试协调器状态问题

**使用方法**:
```powershell
cd test_files

python compare_orchestrator.py <基准目录> <目标目录>

# 示例
python compare_orchestrator.py `
    BC_S5\run_20260121_100000 `
    ..\outputs\BC_S5\run_20260121_110000
```

---

## 🔧 脚本功能验证

### 验证步骤1: 检查脚本是否可执行

```powershell
cd test_files

# 检查Python环境
python --version

# 检查依赖
python -c "import pandas; import numpy; print('依赖OK')"
```

### 验证步骤2: 准备测试数据

```powershell
# 确保有两个运行输出目录
ls BC_S5\run_*
ls ..\outputs\BC_S5\run_*
```

### 验证步骤3: 运行对比脚本

```powershell
# 运行快速对比
python compare_all_outputs.py BC_S5\run_20260120_213219 ..\outputs\BC_S5\run_20260120_214844

# 如果发现差异，运行详细对比
python compare_outputs_detail.py BC_S5\run_20260120_213219 ..\outputs\BC_S5\run_20260120_214844

# 检查Orchestrator状态
python compare_orchestrator.py BC_S5\run_20260120_213219 ..\outputs\BC_S5\run_20260120_214844
```

---

## 📊 对比脚本功能对照表

| 功能 | compare_all_outputs | compare_outputs_detail | compare_orchestrator |
|-----|---------------------|------------------------|----------------------|
| 全模块对比 | ✅ | ✅ | ✅ |
| CSV文件支持 | ✅ | ✅ | ✅ |
| Excel文件支持 | ✅ | ✅ | ✅ |
| 快速概览 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| 详细差异分析 | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| 数值差异量化 | ❌ | ✅ | ❌ |
| Orchestrator专注 | ❌ | ❌ | ✅ |
| 忽略时间戳列 | ✅ | ❌ | ✅ |
| 统计汇总 | ✅ | ✅ | ✅ |

---

## 🎯 推荐使用流程

### 场景1: 常规测试验证
```powershell
# 1. 快速全面检查
python compare_all_outputs.py <dev_dir> <src_dir>

# 2. 如果有差异，查看详情
python compare_outputs_detail.py <dev_dir> <src_dir>
```

### 场景2: Orchestrator状态调试
```powershell
# 专注Orchestrator
python compare_orchestrator.py <dev_dir> <src_dir>
```

### 场景3: 生成正式报告
```powershell
# 使用报告生成器
python generate_optimization_report.py <dev_dir> <src_dir>
```

---

## 🛠️ 脚本修改指南

### 修改默认路径

编辑脚本开头的常量：

```python
# compare_all_outputs.py
CODE_VO_OUTPUT = r"你的dev版本输出路径"
SRC_OUTPUT = r"你的src版本输出路径"
```

### 添加忽略列

```python
# 在脚本中找到IGNORE_COLUMNS
IGNORE_COLUMNS = {'timestamp', 'generation_time', 'run_timestamp', 'your_column'}
```

### 调整数值比较精度

```python
# compare_outputs_detail.py 中修改
if max_diff > 1e-6:  # 改为 1e-3 或其他阈值
```

---

## 🐛 常见问题处理

### 问题1: 文件找不到
```
⚠️ 基准目录不存在: xxx
```

**解决**: 检查路径是否正确，使用绝对路径
```powershell
python compare_all_outputs.py `
    "C:\Users\25936\Desktop\Code\chainsight\test_files\BC_S5\run_xxx" `
    "C:\Users\25936\Desktop\Code\chainsight\outputs\BC_S5\run_yyy"
```

### 问题2: 编码错误
```
UnicodeDecodeError: 'utf-8' codec can't decode
```

**解决**: 在read_csv中添加encoding参数
```python
df = pd.read_csv(file, dtype=str, encoding='utf-8-sig')
```

### 问题3: 内存不足
```
MemoryError
```

**解决**: 分批读取大文件
```python
df = pd.read_csv(file, chunksize=10000)
```

---

## 📝 功能验证测试用例

### 测试用例1: 完全一致的输出
```powershell
# 期望结果: 100%匹配
python compare_all_outputs.py same_run same_run
```

### 测试用例2: 有差异的输出
```powershell
# 期望结果: 显示具体差异
python compare_outputs_detail.py dev_run src_run
```

### 测试用例3: 缺失文件
```powershell
# 期望结果: 报告缺失文件
python compare_all_outputs.py incomplete_run complete_run
```

---

## 🔍 验证脚本完整性

### 检查列表

- [x] ✅ compare_all_outputs.py - 存在且可执行
- [x] ✅ compare_outputs_detail.py - 存在且可执行  
- [x] ✅ compare_orchestrator.py - 存在且可执行
- [x] ✅ 所有脚本支持命令行参数
- [x] ✅ 所有脚本有清晰的输出格式
- [x] ✅ CSV和Excel文件都支持
- [x] ✅ 错误处理完善

### 功能验证命令

```powershell
# 验证所有对比脚本
cd test_files

# 1. 检查脚本语法
python -m py_compile compare_all_outputs.py
python -m py_compile compare_outputs_detail.py
python -m py_compile compare_orchestrator.py

# 2. 显示帮助信息
python compare_all_outputs.py --help 2>&1
python compare_outputs_detail.py --help 2>&1

# 3. 运行实际对比（需要真实数据）
python compare_all_outputs.py BC_S5\run_20260120_213219 ..\outputs\BC_S5\run_20260120_214844
```

---

## 📈 性能对比

| 脚本 | 处理速度 | 内存占用 | 适用场景 |
|-----|---------|---------|---------|
| compare_all_outputs | 快 | 低 | 日常测试 |
| compare_outputs_detail | 中 | 中 | 深度分析 |
| compare_orchestrator | 快 | 低 | 状态调试 |
| generate_optimization_report | 慢 | 高 | 正式报告 |

---

## 🎓 最佳实践

1. **日常开发**: 使用 `compare_all_outputs.py` 快速验证
2. **发现问题**: 使用 `compare_outputs_detail.py` 详细分析
3. **状态问题**: 使用 `compare_orchestrator.py` 检查协调器
4. **正式交付**: 使用 `generate_optimization_report.py` 生成完整报告

---

## 🔗 相关文档

- [TESTING_GUIDE.md](../TESTING_GUIDE.md) - 完整测试流程指南
- [TEST_REPORT.md](../TEST_REPORT.md) - 历史测试报告
- [generate_optimization_report.py](generate_optimization_report.py) - 报告生成脚本

---

**最后更新**: 2026年1月21日
