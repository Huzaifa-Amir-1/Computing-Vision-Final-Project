$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment not found. Running setup first..."
    & "$PSScriptRoot\setup.ps1"
}

& $venvPython -m streamlit run app.py
