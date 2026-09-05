$ErrorActionPreference = 'Stop'

$runner = Join-Path $PSScriptRoot 'metal-smoke.exe'
$spirvVal = 'C:\msys64\mingw64\bin\spirv-val.exe'
$llvmAs = 'C:\msys64\mingw64\bin\llvm-as.exe'
$llvmDis = 'C:\msys64\mingw64\bin\llvm-dis.exe'
foreach ($path in @($runner, $spirvVal, $llvmAs, $llvmDis)) {
    if (-not (Test-Path $path)) {
        throw "Required Metal API smoke input is missing: $path"
    }
}

$env:METAL2VULKAN_SPIRV_VAL = $spirvVal
$env:METAL2VULKAN_LLVM_DIS = $llvmDis
$env:METAL_API_LLVM_AS = $llvmAs
foreach ($executor in @('standalone', 'reims')) {
    & $runner --executor $executor
    if ($LASTEXITCODE -ne 0) {
        throw "metal-smoke $executor failed with exit code $LASTEXITCODE"
    }
}
