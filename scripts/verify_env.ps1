$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot\..
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -c "import pandas, numpy, sklearn, akshare, torch; print('core ok'); print(torch.__version__)"
