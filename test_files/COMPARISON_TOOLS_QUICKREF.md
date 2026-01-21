# 🎯 数据对比脚本快速参考

## 三大对比工具

### 1️⃣ compare_all_outputs.py - 快速全面对比 ⭐推荐

**用途**: 日常测试的首选工具，快速检查所有模块输出

```powershell
cd test_files
python compare_all_outputs.py <dev_dir> <src_dir>
```

**输出**: ✅/❌ 标记 + 匹配百分比

---

### 2️⃣ compare_outputs_detail.py - 详细差异分析

**用途**: 发现差异时，查看具体哪些列、哪些值不同

```powershell
cd test_files
python compare_outputs_detail.py <dev_dir> <src_dir>
```

**输出**: 差异列名 + 差异数量 + 最大差值

---

### 3️⃣ compare_orchestrator.py - Orchestrator专用

**用途**: 调试协调器状态问题

```powershell
cd test_files
python compare_orchestrator.py <dev_dir> <src_dir>
```

**输出**: Orchestrator + 各模块的匹配状态

---

## 🔧 功能验证状态

| 项目 | 状态 |
|------|------|
| 脚本语法 | ✅ 通过 |
| Python依赖 | ✅ 正常 |
| 命令行参数 | ✅ 支持 |
| CSV文件对比 | ✅ 支持 |
| Excel文件对比 | ✅ 支持 |
| 错误处理 | ✅ 完善 |

---

## 📋 典型工作流

```powershell
# Step 1: 快速检查
python compare_all_outputs.py BC_S5\run_DEV ..\outputs\BC_S5\run_SRC

# Step 2: 如果有差异，详细分析
python compare_outputs_detail.py BC_S5\run_DEV ..\outputs\BC_S5\run_SRC

# Step 3: 生成正式报告
python generate_optimization_report.py BC_S5\run_DEV ..\outputs\BC_S5\run_SRC
```

---

## 🚀 快速测试命令

```powershell
# 进入测试目录
cd C:\Users\25936\Desktop\Code\chainsight\test_files

# 查看可用的运行数据
ls BC_S5\run_*
ls ..\outputs\BC_S5\run_*

# 选择最新的两个运行进行对比
$dev = (ls BC_S5\run_* | Sort-Object LastWriteTime -Desc)[0].Name
$src = (ls ..\outputs\BC_S5\run_* | Sort-Object LastWriteTime -Desc)[0].Name
python compare_all_outputs.py "BC_S5\$dev" "..\outputs\BC_S5\$src"
```

---

## 📖 详细文档

- [DATA_COMPARISON_TOOLS_GUIDE.md](DATA_COMPARISON_TOOLS_GUIDE.md) - 完整工具指南
- [TESTING_GUIDE.md](../TESTING_GUIDE.md) - 测试流程指南

---

**验证日期**: 2026年1月21日  
**验证结果**: ✅ 所有脚本功能正常
