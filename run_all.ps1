# run_all.ps1
# Convenience PowerShell script to setup a venv, install deps, and run loader + trainer.
# Run this from the repo root: .\run_all.ps1

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvPath = Join-Path $repoRoot '.venv'
$innerRequirements = Join-Path $repoRoot 'california_housing_regression\requirements.txt'
$loader = Join-Path $repoRoot 'california_housing_regression\scripts\load_and_save_data.py'
$trainer = Join-Path $repoRoot 'california_housing_regression\scripts\train_linear_regression.py'

Write-Host "Repository root: $repoRoot"

# Create venv if missing
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath..."
    python -m venv $venvPath
} else {
    Write-Host "Using existing virtual environment at $venvPath"
}

# Determine python executable inside venv (fallback to system python)
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} else {
    $pythonExe = 'python'
}

Write-Host "Using python: $pythonExe"

# Install requirements if file exists
if (Test-Path $innerRequirements) {
    Write-Host "Installing requirements from $innerRequirements..."
    & $pythonExe -m pip install -r $innerRequirements
} else {
    Write-Warning "Requirements file not found at $innerRequirements. Skipping pip install."
}

# Run loader
if (Test-Path $loader) {
    Write-Host "Running loader: $loader"
    & $pythonExe $loader
} else {
    Write-Error "Loader script not found at $loader"
    exit 1
}

# Run trainer
if (Test-Path $trainer) {
    Write-Host "Running trainer: $trainer"
    & $pythonExe $trainer
} else {
    Write-Error "Trainer script not found at $trainer"
    exit 1
}

Write-Host "Done. Check results in california_housing_regression\results"
