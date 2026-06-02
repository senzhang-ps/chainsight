param(
    [string]$TargetPath = "."
)

$ErrorActionPreference = "Stop"

function Write-Info {
    param([string]$Message)
    Write-Host "[opencode-init] $Message"
}

$projectRoot = [System.IO.Path]::GetFullPath($TargetPath)

if (-not (Test-Path -LiteralPath $projectRoot)) {
    throw "Target path does not exist: $projectRoot"
}

$opencodeRoot = Join-Path $projectRoot ".opencode"
$memoryDir = Join-Path $opencodeRoot "memory"
$progressDir = Join-Path $opencodeRoot "progress"
$handoffDir = Join-Path $opencodeRoot "handoff"
$gitignorePath = Join-Path $projectRoot ".gitignore"
$gitDir = Join-Path $projectRoot ".git"

$directories = @(
    $opencodeRoot,
    $memoryDir,
    $progressDir,
    $handoffDir
)

foreach ($directory in $directories) {
    if (-not (Test-Path -LiteralPath $directory)) {
        New-Item -ItemType Directory -Path $directory | Out-Null
        Write-Info "Created directory: $directory"
    }
    else {
        Write-Info "Directory already exists: $directory"
    }
}

$files = @{
    (Join-Path $opencodeRoot "README.md") = @'
# .opencode

This directory stores project-local OpenCode working context that should stay on the local machine.

## Purpose

- Keep durable project memory close to the repository
- Track current work, blockers, and handoff notes between sessions
- Provide a project-level OpenCode config entrypoint for future local customization

## Structure

- `opencode.json` - project-level OpenCode config
- `memory/` - stable project knowledge
- `progress/` - active work tracking
- `handoff/` - latest cross-session handoff summary

## Git policy

This directory is intentionally ignored by git. It is for local workflow state, not shared repository source.
'@;
    (Join-Path $opencodeRoot "opencode.json") = @'
{
  "$schema": "https://opencode.ai/config.json"
}
'@;
    (Join-Path $memoryDir "project-overview.md") = @'
# Project Overview

## Project

- Name:
- Purpose:
- Primary users:
- Current stage:

## Tech Stack

- Languages:
- Frameworks:
- Storage:
- Tooling:
'@;
    (Join-Path $memoryDir "architecture.md") = @'
# Architecture

## High-Level Shape

- Entry points:
- Core layers:
- Main runtime boundaries:
'@;
    (Join-Path $memoryDir "conventions.md") = @'
# Conventions

## Code Style

- Formatting:
- Naming:
- File organization:
'@;
    (Join-Path $memoryDir "decisions.md") = @'
# Decisions

## YYYY-MM-DD - Decision title

### Context

-
'@;
    (Join-Path $progressDir "backlog.md") = @'
# Backlog

| Priority | Status | Item | Notes |
| --- | --- | --- | --- |
| high | pending |  |  |
'@;
    (Join-Path $progressDir "current-focus.md") = @'
# Current Focus

## Active Goal

-
'@;
    (Join-Path $progressDir "work-log.md") = @'
# Work Log

## YYYY-MM-DD

### Done

-
'@;
    (Join-Path $progressDir "blockers.md") = @'
# Blockers

## Open Blockers

### Blocker

- Description:
'@;
    (Join-Path $handoffDir "latest.md") = @'
# Latest Handoff

## Current State

-
'@
}

foreach ($entry in $files.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $entry.Key)) {
        Set-Content -LiteralPath $entry.Key -Value $entry.Value -Encoding UTF8
        Write-Info "Created file: $($entry.Key)"
    }
    else {
        Write-Info "File already exists, skipped: $($entry.Key)"
    }
}

if (-not (Test-Path -LiteralPath $gitignorePath)) {
    Set-Content -LiteralPath $gitignorePath -Value ".opencode/`r`n" -Encoding UTF8
    Write-Info "Created .gitignore with .opencode/ rule"
}
else {
    $gitignoreContent = Get-Content -LiteralPath $gitignorePath -Raw
    $gitignoreLines = @($gitignoreContent -split "`r?`n" | ForEach-Object { $_.Trim() })
    if ($gitignoreLines -notcontains ".opencode/") {
        $normalized = $gitignoreContent.TrimEnd("`r", "`n")
        $appendValue = if ([string]::IsNullOrWhiteSpace($normalized)) { ".opencode/" } else { "$normalized`r`n`r`n.opencode/" }
        Set-Content -LiteralPath $gitignorePath -Value $appendValue -Encoding UTF8
        Write-Info "Added .opencode/ rule to .gitignore"
    }
    else {
        Write-Info ".gitignore already ignores .opencode/"
    }
}

if (Test-Path -LiteralPath $gitDir) {
    Write-Info "Git repository detected at $projectRoot"
}
else {
    Write-Info "No .git directory detected; initialized local OpenCode workspace anyway"
}

Write-Info "Initialization complete"
