# Run from the owning project. Keep source data and weights out of public Git.
$projectRoot = (Get-Location).Path
$env:LIKENESSFORGE_PROJECT_ROOT = $projectRoot
$env:HF_HOME = Join-Path $projectRoot 'models/hf-home'
$env:HF_HUB_CACHE = Join-Path $projectRoot 'models/huggingface'
$env:HF_XET_CACHE = Join-Path $projectRoot 'models/xet'
$env:TORCH_HOME = Join-Path $projectRoot 'models/torch'
$env:TRITON_CACHE_DIR = Join-Path $projectRoot 'models/triton'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $projectRoot 'models/torchinductor'
# Add models/ to this project's .gitignore before downloading or preparing data.
# Then use likenessforge models download <selected-plugin> or a chosen workflow.
