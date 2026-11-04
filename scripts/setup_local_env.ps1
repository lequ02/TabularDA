param(
    [string]$WorkspaceRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$BasePython = 'D:\python\python.exe',
    [string]$Wheelhouse = ''
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $WorkspaceRoot).ProviderPath
$lock = Join-Path $root 'requirements-local.lock.txt'
if (-not (Test-Path -LiteralPath (Join-Path $root 'workspace.json')) -or -not (Test-Path -LiteralPath $lock)) { throw 'Not a prepared research workspace' }
$venv = [IO.Path]::GetFullPath((Join-Path $root '.venv'))
if (-not $venv.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Environment path escaped the workspace' }
if (Test-Path -LiteralPath $venv) { throw 'Existing .venv preserved; inspect it before explicitly replacing it' }
if (-not (Test-Path -LiteralPath $BasePython)) { throw 'Base Python is unavailable' }
$temporary = Join-Path $root ('.cache\venv_setup_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporary -ErrorAction Stop | Out-Null
$priorTemp = $env:TEMP
$priorTmp = $env:TMP
$priorPipConfig = $env:PIP_CONFIG_FILE
try {
    $env:TEMP = $temporary
    $env:TMP = $temporary
    $env:PIP_CONFIG_FILE = 'NUL'
    & $BasePython -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Venv creation failed; inspect the command error above' }
    $python = Join-Path $venv 'Scripts\python.exe'
    @"
[global]
index-url = https://pypi.org/simple
extra-index-url =
no-cache-dir = true
disable-pip-version-check = true
retries = 0
"@ | Set-Content -LiteralPath (Join-Path $venv 'pip.ini') -Encoding ascii
    if ($Wheelhouse) {
        $wheels = (Resolve-Path -LiteralPath $Wheelhouse).ProviderPath
        & $python -m pip install --no-index --no-cache-dir --find-links $wheels -r $lock
    } else {
        & $python -m pip install --index-url https://pypi.org/simple --retries 0 --no-cache-dir -r $lock
    }
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; package versions were not substituted' }
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Local dependencies are inconsistent' }
    & $python -c 'import sys; from pathlib import Path; import numpy,pandas,matplotlib,scipy,statsmodels,openpyxl,pymupdf; assert Path(sys.prefix).resolve()==Path(sys.argv[1]).resolve(); print(sys.executable)' $venv
    if ($LASTEXITCODE -ne 0) { throw 'Local interpreter/import check failed' }
} finally {
    $env:TEMP = $priorTemp
    $env:TMP = $priorTmp
    $env:PIP_CONFIG_FILE = $priorPipConfig
}
# The helper never deletes an existing venv. Only its own empty setup directory is removed.
if ($null -eq (Get-ChildItem -LiteralPath $temporary -Force | Select-Object -First 1)) { [IO.Directory]::Delete($temporary, $false) }
'Local reporting venv is ready. Production experiments use the established server environment.'
