# ChainSight 虚拟环境激活脚本 (PowerShell)
# 使用方法: .\activate_venv.ps1

$ErrorActionPreference = "Stop"

# 获取项目根目录
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $projectRoot ".venv"

# 检查虚拟环境是否存在
if (-not (Test-Path $venvPath)) {
    Write-Host "❌ 虚拟环境不存在: $venvPath" -ForegroundColor Red
    Write-Host "   请先运行: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# 激活虚拟环境
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

if (-not (Test-Path $activateScript)) {
    Write-Host "❌ 找不到激活脚本: $activateScript" -ForegroundColor Red
    exit 1
}

# 尝试绕过执行策略
try {
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
    & $activateScript
    Write-Host "✅ 虚拟环境已激活" -ForegroundColor Green
    Write-Host "📝 运行命令示例:" -ForegroundColor Cyan
    Write-Host "   python run.py --config test_files\BC_S5.xlsx --start-date 2024-01-01 --end-date 2024-01-31" -ForegroundColor Gray
} catch {
    Write-Host "⚠️  无法激活虚拟环境: $_" -ForegroundColor Yellow
    Write-Host "   尝试直接使用虚拟环境的 Python:" -ForegroundColor Gray
    Write-Host "   $venvPath\Scripts\python.exe run.py ..." -ForegroundColor Gray
}
