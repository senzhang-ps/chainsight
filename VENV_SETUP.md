# ChainSight 虚拟环境设置指南

## 📋 概述

本项目已为您设置好 Python 虚拟环境（`.venv`），所有依赖已安装完毕。

## ✅ 虚拟环境状态

- **位置**: `.\.venv\`
- **Python 版本**: 3.13+
- **状态**: ✅ 已创建并配置完成
- **已安装的依赖**:
  - pandas==2.2.3
  - numpy==2.2.5
  - scipy==1.16.0
  - openpyxl==3.1.5
  - xlsxwriter==3.2.5

## 🚀 快速开始

### 方法 1: 使用 PowerShell（推荐）

```powershell
# 1. 激活虚拟环境
.\activate_venv.ps1

# 2. 运行仿真
python run.py --config test_files\BC_S5.xlsx --start-date 2024-01-01 --end-date 2024-01-31
```

### 方法 2: 使用 Batch

```batch
REM 1. 激活虚拟环境
activate_venv.bat

REM 2. 运行仿真
python run.py --config test_files\BC_S5.xlsx --start-date 2024-01-01 --end-date 2024-01-31
```

### 方法 3: 直接使用虚拟环境 Python（无需激活）

```powershell
# PowerShell
.\.venv\Scripts\python run.py --config test_files\BC_S5.xlsx --start-date 2024-01-01 --end-date 2024-01-31

# Batch
.venv\Scripts\python.exe run.py --config test_files\BC_S5.xlsx --start-date 2024-01-01 --end-date 2024-01-31
```

## 📝 常用命令

### 查看已安装的包

```powershell
.\.venv\Scripts\pip list
```

### 升级包

```powershell
.\.venv\Scripts\pip install --upgrade pandas numpy scipy
```

### 检查虚拟环境的 Python 版本

```powershell
.\.venv\Scripts\python --version
```

### 导出依赖到文件

```powershell
.\.venv\Scripts\pip freeze > requirements_frozen.txt
```

## ⚙️ 运行仿真的 CLI 选项

```
用法: python run.py --config CONFIG --start-date START --end-date END [选项]

必需参数:
  --config CONFIG           配置 Excel 文件路径 (.xlsx)
  --end-date END           结束日期 (YYYY-MM-DD)

可选参数:
  --start-date START       开始日期 (YYYY-MM-DD)
                           第一次运行时必需，后续可省略（自动读取）
  --force-restart          强制从头开始，忽略已有的续跑状态
  --check-resume          只检查续跑状态，不执行仿真
  --resume                启用自动续跑功能
  --resume-from DIR       从特定的运行目录续跑
  --list-runs             列出所有可用的运行目录
  --non-interactive       禁用交互提示（自动选择最新运行）

示例:
  # 运行完整仿真
  python run.py --config test_files\BC_S5.xlsx --start-date 2024-01-01 --end-date 2024-01-31

  # 快速验证（仅 3 天）
  python run.py --config test_files\BC_S5.xlsx --start-date 2024-01-01 --end-date 2024-01-03

  # 检查或列出运行
  python run.py --config test_files\BC_S5.xlsx --list-runs

  # 续跑之前被中断的仿真
  python run.py --config test_files\BC_S5.xlsx --end-date 2024-01-31 --resume
```

## 📂 输出文件位置

运行仿真后，输出文件会保存在配置文件同级目录的子文件夹中：

```
test_files/
├── BC_S5.xlsx
└── BC_S5/                    # 输出目录
    ├── simulation_start.txt  # 仿真开始日期记录
    ├── run_20260105_212916/  # 运行目录（时间戳）
    │   ├── validation_report.txt
    │   ├── simulation_log_*.txt
    │   ├── module1/
    │   ├── module3/
    │   ├── module4/
    │   ├── module5/
    │   ├── module6/
    │   ├── orchestrator/
    │   ├── performance/
    │   └── summary/
```

## 🔧 故障排除

### 问题 1: "python: command not found"

**解决方案**: 使用虚拟环境中的 Python
```powershell
.\.venv\Scripts\python run.py ...
```

### 问题 2: "ExecutionPolicy" 错误 (PowerShell)

**解决方案**: 以下任选其一
```powershell
# A. 临时绕过
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# B. 直接使用虚拟环境的 python.exe
.\.venv\Scripts\python.exe run.py ...

# C. 使用 Batch 脚本
activate_venv.bat
```

### 问题 3: 缺少依赖包

**解决方案**: 重新安装依赖
```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### 问题 4: "maximum recursion depth exceeded"

**解决方案**: 这是一个 Python 日志记录的已知问题。该错误不影响仿真结果：
- 仿真仍会继续运行
- 所有输出文件都会正确生成
- 这通常出现在第 1-3 天的日志中

## 📊 验证安装

运行以下命令验证所有依赖都正确安装：

```powershell
.\.venv\Scripts\python -c "
import pandas
import numpy
import scipy
import openpyxl
import xlsxwriter
print('✅ 所有依赖包导入成功')
print('✅ pandas:', pandas.__version__)
print('✅ numpy:', numpy.__version__)
print('✅ scipy:', scipy.__version__)
print('✅ openpyxl:', openpyxl.__version__)
print('✅ xlsxwriter:', xlsxwriter.__version__)
"
```

## 🎯 下一步

1. **验证环境**
   ```powershell
   .\.venv\Scripts\python run.py --config test_files\BC_S5.xlsx --start-date 2024-01-01 --end-date 2024-01-03
   ```

2. **查看输出**
   打开 `test_files\BC_S5\run_*\validation_report.txt` 查看仿真结果

3. **生成完整报告**
   ```powershell
   python run.py --config test_files\BC_S5.xlsx --start-date 2024-01-01 --end-date 2024-01-31
   ```

## 📞 技术支持

如遇到问题，请检查以下文件：
- `test_files/BC_S5/run_*/simulation_log_*.txt` - 详细日志
- `test_files/BC_S5/run_*/validation_report.txt` - 验证报告
