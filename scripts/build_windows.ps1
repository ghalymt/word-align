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
    [switch]$SkipZip,
    [switch]$CopyModels,
    [switch]$BuildBackends
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

Write-Host "[2/4] Checking main-bundle optional dependencies ..."
python -c "import tensorflow, tensorflow_hub, textgrid" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  installing TensorFlow, TensorFlow Hub, and TextGrid ..."
    python -m pip install tensorflow-cpu==2.21.0 tensorflow-hub==0.16.1 textgrid==1.6.1
    if ($LASTEXITCODE -ne 0) { throw "Optional dependency installation failed" }
}

# 3. Build
Write-Host "[2/4] Building with PyInstaller ..."
$buildArgs = @()
if (-not $SkipZip -and -not $BuildBackends) { $buildArgs += "--zip" }
if ($CopyModels) { $buildArgs += "--copy-models" }
python build.py @buildArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "[error] Build failed." -ForegroundColor Red
    exit 1
}

if ($BuildBackends) {
    Write-Host "[3/4] Building isolated backend environments ..."
    powershell -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\build_backends.ps1")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[error] Backend environment build failed." -ForegroundColor Red
        exit 1
    }
}

if (-not $SkipZip -and $BuildBackends) {
    Write-Host "[3/4] Creating ZIP from the completed bundle ..."
    python build.py --zip-only
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[error] ZIP creation failed." -ForegroundColor Red
        exit 1
    }
}

# 4. Smoke test
$exe = Join-Path $Root "dist\WordAlign\WordAlign.exe"
if (Test-Path $exe) {
    $launcher = Join-Path $Root "dist\WordAlign\Launch WordAlign.vbs"
    if (-not (Test-Path $launcher)) {
        Write-Host "[error] Launch WordAlign.vbs was not produced." -ForegroundColor Red
        exit 1
    }
    Write-Host "[3/4] Smoke test: $exe --version"
    & $exe --version
    foreach ($backend in @("qwen", "whisperx")) {
        $backendPython = Join-Path $Root "dist\WordAlign\venvs\$backend\Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $backendPython)) {
            Write-Host "[warn] Backend venv missing: $backend (run with -BuildBackends)" -ForegroundColor Yellow
        }
    }
    Write-Host "[4/4] Build complete!" -ForegroundColor Green
    Write-Host "    Portable app: $Root\dist\WordAlign"
    if (-not $SkipZip) {
        Write-Host "    Distributable: $Root\dist\WordAlign-portable.zip"
    }
} else {
    Write-Host "[error] WordAlign.exe not found at $exe" -ForegroundColor Red
    exit 1
}
