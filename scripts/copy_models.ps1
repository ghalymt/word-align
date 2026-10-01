param(
    [string]$ProjectRoot = "",
    [string]$VoskSource = "D:\Subtitle edit\Vosk",
    [string]$WhisperSource = "D:\Subtitle edit\Whisper\Models",
    [string]$LlmSource = "F:\LM_studio\Models\unsloth\gemma-4-12B-it-qat-GGUF",
    [string]$MfaSource = "$env:USERPROFILE\Documents\MFA\pretrained_models"
)

$ErrorActionPreference = 'Continue'
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$root = Join-Path $ProjectRoot "models"
$logPath = Join-Path $root "_copy_log.txt"
New-Item -ItemType Directory -Path $root -Force | Out-Null
$drive = (Split-Path -Qualifier (Resolve-Path -LiteralPath $root)).TrimEnd('\')
"" | Set-Content -Path $logPath
$startAll = Get-Date

function Log($msg) {
  $stamp = (Get-Date).ToString("HH:mm:ss")
  $line = "[$stamp] $msg"
  Write-Host $line
  Add-Content -Path $logPath -Value $line
}

function Run-Robo([string]$src, [string]$dst, [string[]]$extraArgs = @()) {
  # Use cmd /c so paths with spaces are preserved correctly.
  # Wrap each path in double quotes.
  $srcQ = '"' + $src + '"'
  $dstQ = '"' + $dst + '"'
  $extra = ($extraArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '
  $cmdLine = "robocopy $srcQ $dstQ /MIR /R:2 /W:5 /NP /NFL /NDL /BYTES /MT:8 $extra"
  cmd /c $cmdLine 2>&1 | Out-Null
  return $LASTEXITCODE
}

function Copy-Tree([string]$src, [string]$dst, [string]$label,
                   [string[]]$excludeFiles = @(), [string[]]$excludeDirs = @()) {
  if (-not (Test-Path $src)) {
    Log "SKIP (missing source): $src"
    return
  }
  Log "START: $label"
  Log "  src: $src"
  Log "  dst: $dst"
  $t0 = Get-Date
  $extra = @()
  if ($excludeFiles.Count -gt 0) { $extra += "/XF"; $extra += $excludeFiles }
  if ($excludeDirs.Count -gt 0)  { $extra += "/XD"; $extra += $excludeDirs }
  $rc = Run-Robo $src $dst $extra
  $t1 = Get-Date
  $mins = [math]::Round(($t1 - $t0).TotalMinutes, 1)
  if ($rc -lt 8) {
    Log "  OK  robocopy exit=$rc elapsed=${mins}m"
  } else {
    Log "  FAIL robocopy exit=$rc elapsed=${mins}m"
  }
  $freeGB = [math]::Round((Get-PSDrive -Name $drive).Free/1GB, 1)
  Log "  ${drive}\ free now: ${freeGB} GB"
}

Log "=== WordAlign model copy (round 3) ==="
Log "${drive}\ free: $([math]::Round((Get-PSDrive -Name $drive).Free/1GB,1)) GB"

# Vosk (all 14 langs) -- ~24 GB
Copy-Tree $VoskSource "$root\vosk" "Vosk (all 14 langs)" `
  -excludeFiles @("*.zip")

# Whisper\Models -- ~28 GB
Copy-Tree $WhisperSource "$root\whisper" "Whisper\Models"

# gemma-4 GGUF -- 7.6 GB (re-run to finish remaining files)
Copy-Tree $LlmSource "$root\llm" "gemma-4-12B GGUF"

# MFA pretrained models -- 480 MB
Copy-Tree $MfaSource "$root\mfa\pretrained_models" "MFA pretrained_models"

# Qwen3 already done above; skip
# llama.cpp already done above; skip

$total = [math]::Round(((Get-Date) - $startAll).TotalMinutes, 1)
Log "=== ALL DONE in ${total} min ==="
Log "${drive}\ free: $([math]::Round((Get-PSDrive -Name $drive).Free/1GB,1)) GB"
