# Creates a new venv next to this repo and installs ONLY requirements.txt (no PyTorch).
# Usage (from repo root voice_engine):
#   powershell -ExecutionPolicy Bypass -File .\scripts\recreate_core_venv.ps1
# Optional: -VenvName myvenv -Python py -3.12

param(
    [string]$VenvName = "venv_core",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
# This file lives in <repo>/scripts → repo root is one level up.
$RepoRoot = Split-Path $PSScriptRoot -Parent
$Req = Join-Path $RepoRoot "requirements.txt"
if (-not (Test-Path $Req)) {
    Write-Error "requirements.txt not found at $Req"
}
$VenvPath = Join-Path $RepoRoot $VenvName
Write-Host "Creating venv: $VenvPath"
& $Python -m venv $VenvPath
$Py = Join-Path $VenvPath "Scripts\python.exe"
& $Py -m pip install -U pip wheel
& $Py -m pip install --no-cache-dir -r $Req
Write-Host "Done. Activate:  $($VenvPath)\Scripts\Activate.ps1"
Write-Host "Run slicer:     `"$Py`" `"$(Join-Path $RepoRoot 'slicer.py')`""
