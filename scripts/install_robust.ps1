# install_robust.ps1 — optional backends for word-align's bad-audio stages
<#
.SYNOPSIS
    Install the extra venvs and models used by the robust (bad-audio) flow.

.DESCRIPTION
    Each component is optional; word-align detects what is installed and
    skips the rest with a message. Components:

        vad       Silero VAD (ONNX, runs in the main app)          ~2 MB
        sed       PretrainedSED BEATs sound-event model             ~0.4 GB  (uses the whisperx venv)
        diarize   pyannote Community-1 speaker diarization          ~0.1 GB  (uses the whisperx venv; gated)
        separate  RoFormer vocal + crowd separation (audio-separator) venv + ~1 GB
        denoise   MossFormerGAN speech enhancement (ClearerVoice)   venv + ~0.2 GB
        cohere    Cohere Transcribe 03-2026 voter                   venv + ~4 GB  (gated)
        parakeet  parakeet-ultra voter via Moondream Photon         venv + ~2.5 GB
        sedicow   SE-DiCoW overlapped-speech ASR                    ~4 GB    (uses the whisperx venv)

    Venvs go next to the existing qwen/whisperx venvs (``<Bundle>\venvs``),
    models under ``<Models>`` in the layout word-align reads. The script is
    idempotent: an existing venv is reused unless -Rebuild is given.

    Gated models need a Hugging Face login with the terms accepted:
    https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 and
    https://huggingface.co/pyannote/speaker-diarization-community-1

.EXAMPLE
    .\scripts\install_robust.ps1 -Bundle E:\Wordalign\dist\WordAlign -Models E:\Wordalign\models
.EXAMPLE
    .\scripts\install_robust.ps1 -Components vad,sed,diarize
#>
param(
    [string]$Bundle = "",
    [string]$VenvRoot = "",
    [string]$Models = "",
    [string]$Py = "",
    [string]$TorchIndex = "",
    [string[]]$Components = @("vad", "sed", "diarize", "separate", "denoise",
                              "cohere", "parakeet", "sedicow"),
    [switch]$Rebuild,
    [switch]$SkipModels,
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Bundle)) { $Bundle = Join-Path $root "dist\WordAlign" }
if ([string]::IsNullOrWhiteSpace($VenvRoot)) { $VenvRoot = Join-Path $Bundle "venvs" }
if ([string]::IsNullOrWhiteSpace($Models)) { $Models = Join-Path $root "models" }
if ([string]::IsNullOrWhiteSpace($Py)) {
    $command = Get-Command python -ErrorAction SilentlyContinue
    if (-not $command) { throw "Python not found on PATH; pass -Py <python.exe>" }
    $Py = $command.Source
}
# Components may arrive as one comma-separated string from cmd.exe.
$Components = @($Components | ForEach-Object { $_ -split "," } |
                ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })
$known = @("vad", "sed", "diarize", "separate", "denoise", "cohere", "parakeet", "sedicow")
foreach ($c in $Components) {
    if ($known -notcontains $c) { throw "Unknown component '$c'. Choose from: $($known -join ', ')" }
}

if (-not $CpuOnly -and [string]::IsNullOrWhiteSpace($TorchIndex)) {
    $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidia) { $TorchIndex = "https://download.pytorch.org/whl/cu128" }
}
$TorchVersion = "2.8.0"
$TorchSuffix = if ($TorchIndex -match "cu128") { "+cu128" } else { "" }

New-Item -ItemType Directory -Path $VenvRoot -Force | Out-Null
New-Item -ItemType Directory -Path $Models -Force | Out-Null
$fetch = Join-Path $PSScriptRoot "fetch_models.py"
$whisperxPy = Join-Path $VenvRoot "whisperx\Scripts\python.exe"

function Write-Section($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments, [string]$What)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)" }
}

function Get-VenvPython($name) { return (Join-Path $VenvRoot "$name\Scripts\python.exe") }

function New-BackendVenv {
    param([string]$Name, [string[]]$Packages, [switch]$Torch, [switch]$Vision)
    $dir = Join-Path $VenvRoot $Name
    $venvPy = Get-VenvPython $Name
    if ($Rebuild -and (Test-Path -LiteralPath $dir)) {
        Write-Host "  removing old venv at $dir ..."
        Remove-Item -LiteralPath $dir -Recurse -Force
    }
    if (-not (Test-Path -LiteralPath $venvPy)) {
        Write-Host "  creating venv $dir ..."
        Invoke-Checked $Py @("-m", "venv", "--copies", $dir) "venv creation ($Name)"
        Invoke-Checked $venvPy @("-m", "pip", "install", "-q", "--upgrade", "pip", "wheel", "setuptools") "pip bootstrap ($Name)"
    }
    $constraints = Join-Path $dir "wordalign-constraints.txt"
    if ($Torch) {
        $torchPkgs = @("torch==$TorchVersion$TorchSuffix", "torchaudio==$TorchVersion$TorchSuffix")
        if ($Vision) { $torchPkgs += "torchvision==0.23.0$TorchSuffix" }
        $pipArgs = @("-m", "pip", "install", "-q") + $torchPkgs
        if ($TorchIndex) { $pipArgs += @("--index-url", $TorchIndex) }
        Write-Host "  installing $($torchPkgs -join ', ') ..."
        Invoke-Checked $venvPy $pipArgs "torch install ($Name)"
        # Keep later installs from swapping the CUDA build for a CPU one.
        Set-Content -LiteralPath $constraints -Value $torchPkgs -Encoding ascii
    }
    if ($Packages.Count -gt 0) {
        $pipArgs = @("-m", "pip", "install", "-q") + $Packages
        if (Test-Path -LiteralPath $constraints) { $pipArgs += @("-c", $constraints) }
        if ($TorchIndex) { $pipArgs += @("--extra-index-url", $TorchIndex) }
        Write-Host "  installing $($Packages -join ', ') ..."
        Invoke-Checked $venvPy $pipArgs "package install ($Name)"
    }
    return $venvPy
}

function Fetch-Models([string]$PythonExe, [string[]]$Which) {
    if ($SkipModels) { return }
    Invoke-Checked $PythonExe (@($fetch, "--models-dir", $Models) + $Which) "model download ($($Which -join ','))"
}

# A python with huggingface_hub for downloads that need no venv of their own.
$hubPy = if (Test-Path -LiteralPath $whisperxPy) { $whisperxPy } else { $Py }

if ($Components -contains "vad") {
    Write-Section "Silero VAD"
    Fetch-Models $hubPy @("vad")
}

foreach ($c in @("sed", "diarize", "sedicow")) {
    if ($Components -contains $c -and -not (Test-Path -LiteralPath $whisperxPy)) {
        throw "Component '$c' runs in the whisperx venv; build it first with scripts\build_backends.ps1"
    }
}

if ($Components -contains "sed") {
    Write-Section "Sound-event detection (PretrainedSED BEATs, whisperx venv)"
    Fetch-Models $whisperxPy @("sed")
}

if ($Components -contains "diarize") {
    Write-Section "Speaker diarization (pyannote Community-1, whisperx venv)"
    Invoke-Checked $whisperxPy @("-c", "import pyannote.audio, sys; v=pyannote.audio.__version__; print('pyannote.audio', v); sys.exit(0 if int(v.split('.')[0]) >= 4 else 1)") "pyannote.audio >= 4 check"
    Fetch-Models $whisperxPy @("diarize")
}

if ($Components -contains "sedicow") {
    Write-Section "SE-DiCoW overlapped-speech ASR (whisperx venv)"
    Fetch-Models $whisperxPy @("sedicow")
}

if ($Components -contains "separate") {
    Write-Section "RoFormer separation venv (audio-separator)"
    $sepPy = New-BackendVenv -Name "separate" -Torch -Packages @("audio-separator[gpu]==0.47.0")
    if (-not $SkipModels) {
        $sepDir = Join-Path $Models "separation"
        New-Item -ItemType Directory -Path $sepDir -Force | Out-Null
        $code = "from audio_separator.separator import Separator; s = Separator(model_file_dir=r'$sepDir', output_dir=r'$sepDir'); " +
                "[s.load_model(model_filename=m) for m in ('model_bs_roformer_ep_317_sdr_12.9755.ckpt', 'mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt')]; print('separation models ready')"
        Invoke-Checked $sepPy @("-c", $code) "separation model download"
    }
}

if ($Components -contains "denoise") {
    Write-Section "MossFormerGAN enhancement venv (ClearerVoice)"
    $dnPy = New-BackendVenv -Name "denoise" -Torch -Vision -Packages @("clearvoice==0.1.2")
    Fetch-Models $dnPy @("denoise")
}

if ($Components -contains "cohere") {
    Write-Section "Cohere Transcribe venv (transformers 5)"
    $cohPy = New-BackendVenv -Name "cohere" -Torch -Packages @(
        "transformers>=5.4,<6", "accelerate", "sentencepiece", "protobuf",
        "soundfile", "librosa", "huggingface_hub")
    Fetch-Models $cohPy @("cohere")
}

if ($Components -contains "parakeet") {
    Write-Section "parakeet-ultra venv (Moondream Photon)"
    # kestrel (Photon) needs torch >= 2.8; install the CUDA build first.
    $pkPy = New-BackendVenv -Name "parakeet" -Torch -Packages @("moondream==2.6.1", "huggingface_hub", "numpy")
    Fetch-Models $pkPy @("parakeet")
}

Write-Section "Done"
Write-Host "Venvs : $VenvRoot"
Write-Host "Models: $Models"
Write-Host "Check what word-align can use with:  wordalign --doctor"
