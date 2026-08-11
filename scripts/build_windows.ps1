# build_windows.ps1 — one-click portable build for Windows
<#
.SYNOPSIS
    Build the portable WordAlign distribution for Windows.
.DESCRIPTION
    Installs PyInstaller if needed, runs the spec build, bundles extras,
    and creates dist/WordAlign-portable.zip.
.EXAMPLE
    .\scripts\build_windows.ps1
    .\scripts\build_windows.ps1 -SkipZip
#>
param(
    [switch]$SkipZip
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "=== WordAlign portable build ===" -ForegroundColor Cyan

# 1. Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[error] Python not found on PATH." -ForegroundColor Red
    exit 1
}

# 2. Ensure PyInstaller
python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[1/4] Installing PyInstaller ..."
    python -m pip install pyinstaller
}

# 3. Build
Write-Host "[2/4] Building with PyInstaller ..."
python build.py @(if ($SkipZip) { @() } else { @("--zip") })

if ($LASTEXITCODE -ne 0) {
    Write-Host "[error] Build failed." -ForegroundColor Red
    exit 1
}

# 4. Smoke test
$exe = Join-Path $Root "dist\WordAlign\WordAlign.exe"
if (Test-Path $exe) {
    Write-Host "[3/4] Smoke test: $exe --version"
    & $exe --version
    Write-Host "[4/4] Build complete!" -ForegroundColor Green
    Write-Host "    Portable app: $Root\dist\WordAlign"
    if (-not $SkipZip) {
        Write-Host "    Distributable: $Root\dist\WordAlign-portable.zip"
    }
} else {
    Write-Host "[error] WordAlign.exe not found at $exe" -ForegroundColor Red
    exit 1
}
