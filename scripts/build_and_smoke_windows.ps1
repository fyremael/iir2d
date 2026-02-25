param(
    [switch]$SkipGpuSmoke
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$BuildDir = "build_win"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $false)]
        [string[]]$Args = @()
    )
    & $Command @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Command $($Args -join ' ')"
    }
}

function Invoke-CmdChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CmdLine
    )
    cmd.exe /c $CmdLine
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): cmd /c $CmdLine"
    }
}

function Resolve-NvccPath {
    $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $cudaRoot) {
        $dirs = Get-ChildItem -Path $cudaRoot -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^v\d+\.\d+$' } |
            Sort-Object {
                $parts = $_.Name.TrimStart('v').Split('.')
                [int]$parts[0] * 100 + [int]$parts[1]
            } -Descending
        foreach ($d in $dirs) {
            $p = Join-Path $d.FullName "bin\nvcc.exe"
            if (Test-Path $p) { return $p }
        }
    }
    $cmd = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    return $null
}

function Get-NvccVersion {
    param(
        [Parameter(Mandatory = $true)]
        [string]$NvccPath
    )
    $out = & $NvccPath --version 2>&1
    $text = ($out | Out-String)
    $m = [regex]::Match($text, "release\s+(\d+)\.(\d+)")
    if (-not $m.Success) { return $null }
    return [pscustomobject]@{
        Major = [int]$m.Groups[1].Value
        Minor = [int]$m.Groups[2].Value
        Text = "$($m.Groups[1].Value).$($m.Groups[2].Value)"
    }
}

function Resolve-VsDevCmdPath {
    $candidates = @(
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Resolve-NinjaPath {
    $cmd = Get-Command ninja -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($py) {
        $scriptDir = Join-Path (Split-Path -Parent $py.Source) "Scripts"
        $cand = Join-Path $scriptDir "ninja.exe"
        if (Test-Path $cand) { return $cand }
    }
    $appDataCand = "C:\Users\$env:USERNAME\AppData\Roaming\Python\Python312\Scripts\ninja.exe"
    if (Test-Path $appDataCand) { return $appDataCand }
    return $null
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "cmake not found in PATH"
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python not found in PATH"
}
 $nvcc = Resolve-NvccPath
if (-not $nvcc) {
    throw "nvcc not found in PATH. Install CUDA toolkit and ensure nvcc is on PATH."
}
$nvccVersion = Get-NvccVersion -NvccPath $nvcc
if (-not $nvccVersion) {
    throw "Could not determine nvcc version from: $nvcc"
}
# Windows MSVC + CUDA compatibility floor for this project setup.
if ($nvccVersion.Major -lt 11 -or ($nvccVersion.Major -eq 11 -and $nvccVersion.Minor -lt 8)) {
    throw "Unsupported CUDA toolkit version $($nvccVersion.Text). Require CUDA >= 11.8 on Windows."
}
$vsDevCmd = Resolve-VsDevCmdPath
if (-not $vsDevCmd) {
    throw "VsDevCmd.bat not found. Install Visual Studio Build Tools (C++ workload)."
}
$nvccCmake = $nvcc -replace "\\", "/"
$ninja = Resolve-NinjaPath
if ($ninja) {
    $ninjaCmake = $ninja -replace "\\", "/"
}
if (Test-Path "$Root\$BuildDir") {
    Remove-Item -Recurse -Force "$Root\$BuildDir"
}

if ($ninja) {
    $BuildDir = "build_win_ninja"
    if (Test-Path "$Root\$BuildDir") {
        Remove-Item -Recurse -Force "$Root\$BuildDir"
    }
    Invoke-CmdChecked "call `"$vsDevCmd`" -arch=x64 && cmake -S . -B $BuildDir -G Ninja -DCMAKE_MAKE_PROGRAM=`"$ninjaCmake`" -DCMAKE_CUDA_COMPILER=`"$nvccCmake`""
    Invoke-CmdChecked "call `"$vsDevCmd`" -arch=x64 && cmake --build $BuildDir"
} else {
    $BuildDir = "build_win_vs2019"
    if (Test-Path "$Root\$BuildDir") {
        Remove-Item -Recurse -Force "$Root\$BuildDir"
    }
    Invoke-CmdChecked "call `"$vsDevCmd`" -arch=x64 && cmake -S . -B $BuildDir -G `"Visual Studio 16 2019`" -A x64 -DCMAKE_CUDA_COMPILER=`"$nvccCmake`""
    Invoke-CmdChecked "call `"$vsDevCmd`" -arch=x64 && cmake --build $BuildDir --config Release"
}

$dll = Get-ChildItem -Path "$Root\$BuildDir" -Recurse -Filter "iir2d_jax.dll" |
    Select-Object -First 1
if (-not $dll) {
    throw "Could not find iir2d_jax.dll under $Root\$BuildDir"
}

Copy-Item -Force $dll.FullName "$Root\python\iir2d_jax\iir2d_jax.dll"

$env:PYTHONPATH = "$Root\python"
Invoke-Checked -Command "python" -Args @("$Root\scripts\smoke_core_status.py")
if (-not $SkipGpuSmoke) {
    Invoke-Checked -Command "python" -Args @("$Root\smoke_jax.py")
}
