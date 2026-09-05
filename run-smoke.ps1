$ErrorActionPreference = 'Stop'

$runner = Join-Path $PSScriptRoot 'metal-smoke.exe'
$spirvVal = 'C:\msys64\mingw64\bin\spirv-val.exe'
foreach ($path in @($runner, $spirvVal)) {
    if (-not (Test-Path $path)) {
        throw "Required Metal API smoke input is missing: $path"
    }
}

$env:METAL2VULKAN_SPIRV_VAL = $spirvVal
foreach ($executor in @('standalone', 'reims')) {
    & $runner --executor $executor
    if ($LASTEXITCODE -ne 0) {
        throw "metal-smoke $executor failed with exit code $LASTEXITCODE"
    }
}
