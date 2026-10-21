param([ValidateSet('auto','openvino','tensorrt','onnx')][string]$Backend = 'auto',
      [ValidateSet('CPU','GPU')][string]$Device = 'GPU', [switch]$Cpu)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $project
try {
    if ($Cpu) { $Backend = 'onnx' }
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Run deployment/setup.ps1 first.' }
    if ($Backend -eq 'tensorrt' -and -not (Test-Path -LiteralPath 'models/registry.local.json')) {
        throw 'Run deployment/setup.ps1 -Backend tensorrt on this PC first, or use Auto.'
    }
    & .\.venv\Scripts\python.exe -m app.app --backend $Backend --device $Device
    if ($LASTEXITCODE -ne 0) { throw 'Application stopped with an error; read the message above.' }
} finally {
    Pop-Location
}
