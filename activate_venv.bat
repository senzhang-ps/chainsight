@echo off
REM ChainSight 虚拟环境激活脚本 (Batch)
REM 使用方法: activate_venv.bat

setlocal enabledelayedexpansion

set "PROJECT_ROOT=%~dp0"
set "VENV_PATH=%PROJECT_ROOT%.venv"

echo.
echo 🔍 检查虚拟环境...
if not exist "%VENV_PATH%" (
    echo ❌ 虚拟环境不存在: %VENV_PATH%
    echo 📝 请先运行: python -m venv .venv
    pause
    exit /b 1
)

echo ✅ 虚拟环境已找到
echo 🚀 激活虚拟环境...

call "%VENV_PATH%\Scripts\activate.bat"

if errorlevel 1 (
    echo ❌ 激活失败
    pause
    exit /b 1
)

echo ✅ 虚拟环境已激活
echo.
echo 📝 运行命令示例:
echo    python run.py --config test_files\BC_S5.xlsx --start-date 2024-01-01 --end-date 2024-01-31
echo.
pause
