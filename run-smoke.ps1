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
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $outputDir = Join-Path $PSScriptRoot "target\windows-captures\$stamp"
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    # Discover every committed suite the way tools/lavapipe-smoke.sh does:
    # suite.json is v1, suite-vN.json is vN, and the list grows with the repo
    # instead of stopping at a hand-maintained version.
    $suites = @((Join-Path $suiteDir 'suite.json'))
    $version = 2
    while (Test-Path (Join-Path $suiteDir "suite-v$version.json")) {
        $suites += (Join-Path $suiteDir "suite-v$version.json")
        $version++
    }
    foreach ($suitePath in $suites) {
        $suite = Split-Path $suitePath -Leaf
        $label = if ($suite -eq 'suite.json') { 'v1' } else { $suite -replace '^suite-', '' }
        foreach ($rail in @('direct', 'objects', 'objects-async')) {
            $output = Join-Path $outputDir "vulkan-$rail-$label.json"
            $captureArgs = @('--suite', $suitePath, '--output', $output)
            if ($rail -ne 'direct') { $captureArgs += @('--api', 'objects') }
            if ($rail -eq 'objects-async') { $captureArgs += '--async' }
            & $captureRunner @captureArgs
            if ($LASTEXITCODE -ne 0) { throw "Capture failed: $rail $label" }
        }
    }
    Write-Host "Captures written to $outputDir"
}
if ($ReimsRunner) {
    & $ReimsRunner
    if ($LASTEXITCODE -ne 0) { throw "Reims smoke failed: $LASTEXITCODE" }
}
