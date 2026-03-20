$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot\..
if (Test-Path .venv) {
    Write-Host '发现旧 .venv，正在删除...'
    Remove-Item -Recurse -Force .venv
}
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-core.txt -i https://pypi.org/simple
python -m pip install -r requirements-lstm.txt -i https://pypi.org/simple
Write-Host '环境重建完成。'
