param([ValidateSet('auto','openvino','tensorrt','onnx')][string]$Backend = 'auto', [switch]$Cpu)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $project
try {
    if ($Cpu) { $Backend = 'onnx' }
    & py -3.10 -c "import struct,sys; assert sys.platform == 'win32' and struct.calcsize('P') == 8, 'Use Windows x64 and Python 3.10 x64'"
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.10 x64 with the Python launcher, then retry.' }
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & py -3.10 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
    }
    $requirements = switch ($Backend) {
        'tensorrt' { 'deployment/requirements-gpu.txt' }
        'onnx' { 'requirements-cpu.txt' }
        default { 'requirements-auto.txt' }
    }
    & .\.venv\Scripts\python.exe -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check Internet access and the error above.' }
    if ($Backend -eq 'tensorrt') {
        & .\.venv\Scripts\python.exe tools/prepare_deployment.py
    } else {
        & .\.venv\Scripts\python.exe tools/prepare_deployment.py --backend $Backend
    }
    if ($LASTEXITCODE -ne 0) { throw 'Model setup failed. Existing verified engines were preserved.' }
    Write-Host 'Setup passed. Run deployment/start.ps1. Use -Backend openvino, tensorrt, or onnx to force a backend.'
} finally {
    Pop-Location
}
