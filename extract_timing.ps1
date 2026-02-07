$content = Get-Content 'D:\PG\Code\chainsight\config\OC_Paste_S1_20251224\run_20260127_142402\simulation_log_20260127_142402.txt' -Encoding UTF8
$dayLines = $content | Select-String '第 \d+/76 天'
Write-Host "First day:" $dayLines[0].Line
Write-Host "Last day:" $dayLines[-1].Line
Write-Host "Total day lines:" $dayLines.Count

$completions = $content | Select-String '第 \d+ 天处理完成'
Write-Host "First complete:" $completions[0].Line
Write-Host "Last complete:" $completions[-1].Line
Write-Host "Total completions:" $completions.Count
