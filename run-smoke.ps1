param(
    [string]$Runner = '',
    [string]$ReimsRunner = '',
    [switch]$ProviderSmoke,
    [switch]$CaptureMatrix
)
$ErrorActionPreference = 'Stop'

if (-not $Runner) {
    $candidates = @(
        (Join-Path $PSScriptRoot 'target\x86_64-pc-windows-gnu\release\metal-smoke.exe'),
        (Join-Path $PSScriptRoot 'target\release\metal-smoke.exe'),
        (Join-Path $PSScriptRoot 'metal-smoke.exe')
    )
    $Runner = $candidates | Where-Object { Test-Path $_ -PathType Leaf } | Select-Object -First 1
}
if (-not $Runner -or -not (Test-Path $Runner -PathType Leaf)) {
    throw 'Build metal-smoke or supply -Runner with its executable path.'
}
if ($ReimsRunner -and -not (Test-Path $ReimsRunner -PathType Leaf)) {
    throw "Reims runner is missing: $ReimsRunner"
}

function Resolve-Tool([string]$Variable, [string]$Name) {
    $configured = [Environment]::GetEnvironmentVariable($Variable)
    if ($configured) {
        $command = Get-Command $configured -CommandType Application -ErrorAction SilentlyContinue
        if ($command) { return $command.Source }
        throw "Invalid tool path in ${Variable}: $configured"
    }
    $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $fallback = Join-Path 'C:\msys64\mingw64\bin' ($Name + '.exe')
    if (Test-Path $fallback -PathType Leaf) { return $fallback }
    throw "Install $Name on PATH or set $Variable."
}

$env:METAL2VULKAN_SPIRV_VAL = Resolve-Tool 'METAL2VULKAN_SPIRV_VAL' 'spirv-val'
$env:METAL2VULKAN_LLVM_DIS = Resolve-Tool 'METAL2VULKAN_LLVM_DIS' 'llvm-dis'
$env:METAL_API_LLVM_AS = Resolve-Tool 'METAL_API_LLVM_AS' 'llvm-as'
& $Runner --executor standalone
if ($LASTEXITCODE -ne 0) { throw "Standalone smoke failed: $LASTEXITCODE" }
if ($ProviderSmoke) {
    $providerRunner = Join-Path (Split-Path $Runner -Parent) 'provider-smoke.exe'
    if (-not (Test-Path $providerRunner -PathType Leaf)) {
        throw "provider-smoke.exe is missing next to the runner: $providerRunner"
    }
    & $providerRunner
    if ($LASTEXITCODE -ne 0) { throw "Provider smoke failed: $LASTEXITCODE" }
}
if ($CaptureMatrix) {
    $captureRunner = Join-Path (Split-Path $Runner -Parent) 'provider-capture.exe'
    if (-not (Test-Path $captureRunner -PathType Leaf)) {
        throw "provider-capture.exe is missing next to the runner: $captureRunner"
    }
    $suiteDir = Join-Path $PSScriptRoot 'conformance'
    $outputDir = Join-Path $PSScriptRoot 'target\windows-captures'
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    foreach ($version in 1..8) {
        $suite = if ($version -eq 1) { 'suite.json' } else { "suite-v$version.json" }
        foreach ($rail in @('direct', 'objects', 'objects-async')) {
            $output = Join-Path $outputDir "vulkan-$rail-v$version.json"
            $captureArgs = @('--suite', (Join-Path $suiteDir $suite), '--output', $output)
            if ($rail -ne 'direct') { $captureArgs += @('--api', 'objects') }
            if ($rail -eq 'objects-async') { $captureArgs += '--async' }
            & $captureRunner @captureArgs
            if ($LASTEXITCODE -ne 0) { throw "Capture failed: $rail v$version" }
        }
    }
    Write-Host "Captures written to $outputDir"
}
if ($ReimsRunner) {
    & $ReimsRunner
    if ($LASTEXITCODE -ne 0) { throw "Reims smoke failed: $LASTEXITCODE" }
}
