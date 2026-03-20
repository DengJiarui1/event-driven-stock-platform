$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot\..
.\.venv\Scripts\Activate.ps1
python run_pipeline.py
