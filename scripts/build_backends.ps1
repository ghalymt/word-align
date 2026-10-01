# build_backends.ps1 — create per-backend venvs next to WordAlign.exe
<#
.SYNOPSIS
    Build isolated venvs for the heavy backends that conflict with each
    other (Qwen3-ASR, WhisperX) and stage MFA's binary into the bundle.

.DESCRIPTION
    Each backend has its own torch/transformers/CT2 pins. Stuffing them all
    into one environment causes "works in dev, broken when frozen" failures.
    Instead we give each backend its own venv under
    ``WordAlign/venvs/<backend>/`` and the launcher auto-points
    WORDALIGN_*_PYTHON at them.

    Layout created:
        WordAlign/venvs/qwen/python.exe        + qwen_asr + pinned transformers
        WordAlign/venvs/whisperx/python.exe    + whisperx + faster-whisper + pinned transformers
        WordAlign/venvs/mfa/                 + MFA, KALPY, and Kaldi runtime

    This script is idempotent: re-running it will refresh the venvs.
#>
param(
    [string]$Bundle = "",
    [string]$Py = "",
    [string]$Conda = "",
    [string]$TorchIndex = "",
    [switch]$RefreshMfa,
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Bundle)) {
    $Bundle = Join-Path $root "dist\WordAlign"
}
if ([string]::IsNullOrWhiteSpace($Py)) {
    $command = Get-Command python -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "Python not found on PATH"
    }
    $Py = $command.Source
}

if (-not (Test-Path -LiteralPath $Py)) {
    throw "Python not found at $Py"
}
$venvRoot = Join-Path $Bundle "venvs"
New-Item -ItemType Directory -Path $venvRoot -Force | Out-Null
if (-not $CpuOnly -and [string]::IsNullOrWhiteSpace($TorchIndex)) {
    $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidia) {
        & $nvidia.Source --query-gpu=name --format=csv,noheader *> $null
        if ($LASTEXITCODE -eq 0) {
            $TorchIndex = "https://download.pytorch.org/whl/cu128"
        }
    }
}

function Write-Section($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Test-Cmd($cmd, $args) {
    $p = Start-Process -FilePath $cmd -ArgumentList $args -PassThru -Wait `
                        -RedirectStandardOutput "$env:TEMP\cmd_out.log" `
                        -RedirectStandardError  "$env:TEMP\cmd_err.log"
    if ($p.ExitCode -ne 0) {
        Write-Host "  [FAIL] exit=$($p.ExitCode)"
        Write-Host "  --- stdout ---"
        Get-Content "$env:TEMP\cmd_out.log" -Tail 30
        Write-Host "  --- stderr ---"
        Get-Content "$env:TEMP\cmd_err.log" -Tail 30
        throw "Command failed: $cmd $($args -join ' ')"
    }
}

function New-Venv($name, $reqs, $extraEnv = @{}, [switch]$NoDepsFirst) {
    Write-Section "Building venv: $name"
    $dir = Join-Path $venvRoot $name
    if (Test-Path -LiteralPath $dir) {
        Write-Host "  removing old venv at $dir ..."
        Remove-Item -LiteralPath $dir -Recurse -Force
    }
    Write-Host "  creating venv ..."
    & $Py -m venv --copies $dir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed for $name" }
    $venvPy = Join-Path $dir "Scripts\python.exe"
    Write-Host "  upgrading pip ..."
    & $venvPy -m pip install --upgrade pip wheel setuptools | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed for $name" }

    $envs = $extraEnv.Clone()
    foreach ($k in $envs.Keys) { Set-Item -Path "Env:$k" -Value $envs[$k] }
    try {
        if ($NoDepsFirst -and $reqs.Count -gt 0) {
            # Install the first requirement with --no-deps (skips its
            # declared dependency tree), then install the rest normally.
            Write-Host "  installing (no-deps): $($reqs[0]) ..."
            & $venvPy -m pip install --no-deps $reqs[0]
            if ($LASTEXITCODE -ne 0) { throw "pip --no-deps install failed for $name" }
            if ($reqs.Count -gt 1) {
                $rest = $reqs[1..($reqs.Count - 1)]
                Write-Host "  installing: $($rest -join ', ') ..."
                & $venvPy -m pip install $rest
                if ($LASTEXITCODE -ne 0) { throw "pip install failed for $name" }
            }
        } else {
            Write-Host "  installing: $($reqs -join ', ') ..."
            & $venvPy -m pip install $reqs
            if ($LASTEXITCODE -ne 0) { throw "pip install failed for $name" }
        }
    } finally {
        foreach ($k in $envs.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
    }
    Write-Host "  [OK] $name ready at $dir"
    return $venvPy
}

function Install-CudaTorch($venvPython, [switch]$WithVision) {
    if ([string]::IsNullOrWhiteSpace($TorchIndex)) { return }
    Write-Host "  installing CUDA Torch runtime from $TorchIndex ..."
    $torchArgs = @("-m", "pip", "install", "--force-reinstall", "--no-deps",
        "torch==2.8.0+cu128", "torchaudio==2.8.0+cu128")
    if ($WithVision) {
        $torchArgs += "torchvision==0.23.0+cu128"
    }
    $torchArgs += @("--index-url", $TorchIndex)
    & $venvPython @torchArgs
    if ($LASTEXITCODE -ne 0) { throw "CUDA Torch installation failed" }
}

# --- Qwen3-ASR venv ----------------------------------------------------------
# qwen-asr 0.0.6 declares an EXACT pin on transformers==4.57.6 (their
# modeling code uses the new @check_model_inputs(func) decorator shape).
# Its forced-aligner module imports nagisa + DyNet at module load time,
# even when the worker only does English transcription -- so we have to
# install nagisa (which pulls in DyNet38, a C++ ext) despite never
# hitting the Japanese code paths.
#
# Flask + gradio are only used for qwen-asr's web demo, which the worker
# never invokes, so we deliberately skip those two.
New-Venv "qwen" @(
    "qwen-asr==0.0.6",
    "qwen-omni-utils",
    "transformers==4.57.6",
    "torch==2.8.0",
    "torchaudio==2.8.0",
    "accelerate==1.12.0",
    "librosa",
    "soundfile",
    "nagisa==0.2.11",
    "pytz",
    "sox",
    "soynlp==0.0.493"
) -NoDepsFirst
Install-CudaTorch (Join-Path $venvRoot "qwen\Scripts\python.exe")

# --- WhisperX venv -----------------------------------------------------------
# WhisperX 3.8.6 declares transformers>=4.48 and torch~=2.8. Pin to a
# known-good build (4.57.x) so wav2vec2 ctc_loss + ctranslate2 +
# faster-whisper all import. We deliberately skip pyannote-audio here --
# diarization is handled by the main bundle via the dedicated
# diarization adapter, and pulling pyannote into this venv doubles its
# size without adding alignment value.
New-Venv "whisperx" @(
    "whisperx==3.8.6",
    "transformers==4.57.6",
    "torch==2.8.0",
    "torchaudio==2.8.0"
)
Install-CudaTorch (Join-Path $venvRoot "whisperx\Scripts\python.exe") -WithVision

$mfaDir = Join-Path $venvRoot "mfa"
$mfaExe = Join-Path $mfaDir "Scripts\mfa.exe"
$condaPath = $Conda
if ([string]::IsNullOrWhiteSpace($condaPath)) {
    $condaCommand = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCommand) {
        $condaPath = $condaCommand.Source
    } else {
        $condaPath = "C:\ProgramData\miniconda3\Scripts\conda.exe"
    }
}
if (-not (Test-Path -LiteralPath $condaPath)) {
    throw "Conda is required for MFA. Install Miniconda or pass -Conda <conda.exe>."
}

if ($RefreshMfa -or -not (Test-Path -LiteralPath $mfaExe)) {
    Write-Section "Building MFA conda environment"
    if (Test-Path -LiteralPath $mfaDir) {
        Write-Host "  removing old MFA environment at $mfaDir ..."
        Remove-Item -LiteralPath $mfaDir -Recurse -Force
    }
    & $condaPath create --yes --prefix $mfaDir --override-channels `
        --channel conda-forge "python=3.11" "montreal-forced-aligner=3.4.2"
    if ($LASTEXITCODE -ne 0) { throw "MFA conda environment creation failed" }
}

Write-Section "Validating MFA"
& $condaPath run --prefix $mfaDir python -c "import _kalpy, montreal_forced_aligner; print('MFA runtime ready')"
if ($LASTEXITCODE -ne 0) { throw "MFA runtime validation failed" }
if (-not (Test-Path -LiteralPath $mfaExe)) { throw "MFA executable missing: $mfaExe" }

Write-Section "Done"
Write-Host "Bundle venv root: $venvRoot"
Get-ChildItem $venvRoot | Select-Object Mode, Name | Format-Table -AutoSize
