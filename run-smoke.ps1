param(
    [string]$Runner = '',
    [string]$ReimsRunner = ''
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
if ($ReimsRunner) {
    & $ReimsRunner
    if ($LASTEXITCODE -ne 0) { throw "Reims smoke failed: $LASTEXITCODE" }
}
