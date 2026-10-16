$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$env:LIKENESSFORGE_PROJECT_ROOT = $PSScriptRoot
$env:HF_HOME = Join-Path $PSScriptRoot 'models/hf-home'
$env:HF_HUB_CACHE = Join-Path $PSScriptRoot 'models/huggingface'
$env:HF_XET_CACHE = Join-Path $PSScriptRoot 'models/xet'
$env:TORCH_HOME = Join-Path $PSScriptRoot 'models/torch'
$env:TORCH_EXTENSIONS_DIR = Join-Path $PSScriptRoot 'models/torch-extensions'
$env:TRITON_CACHE_DIR = Join-Path $PSScriptRoot 'models/triton'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $PSScriptRoot 'models/torchinductor'

function Find-Uv {
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Path
    }

    $candidates = @(
        (Join-Path $HOME ".local\bin\uv.exe"),
        (Join-Path $HOME ".cargo\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "uv\bin\uv.exe")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }
    return $null
}

$uv = Find-Uv
if (-not $uv) {
    Write-Host "LikenessForge: uv not found; installing Astral uv..."
    Invoke-Expression (Invoke-RestMethod https://astral.sh/uv/install.ps1)
    $uv = Find-Uv
}

if (-not $uv) {
    throw "uv installation completed but uv.exe could not be located. Open a new terminal and retry."
}

Write-Host "LikenessForge: using $(& $uv --version)"
Write-Host "LikenessForge: ensuring the project Python from .python-version is installed..."
& $uv python install
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "LikenessForge: creating/updating the managed environment and GUI/model dependencies..."
& $uv sync --extra all
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "LikenessForge: starting GUI..."
& $uv run --no-sync python scripts/prepare_windows_runtime.py --apply
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $uv run --no-sync likenessforge-gui @args
exit $LASTEXITCODE
