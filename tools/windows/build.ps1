# Build av1gpu on Windows x64:
#   - MSVC from Visual Studio 2022 or its Build Tools ("Desktop development with C++");
#   - CUDA from -CudaRoot, or the parts av1gpu needs fetched by get-cuda.ps1;
#   - libjpeg-turbo, libpng, zlib and libwebp built static by vcpkg (pinned release);
#   - CMake 3.24 or newer, and Ninja (downloaded if not on PATH).
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1              # all GPUs, Turing .. Blackwell
#   powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Archs native -Package
#
# The executable is build\av1gpu.exe; it needs nvjpeg64_*.dll from the toolkit beside it (-Package puts both in
# av1gpu-<version>-windows-x86_64.zip, with a .sha256).
param(
    [string]$Archs = "",                 # CMAKE_CUDA_ARCHITECTURES; empty: the CMakeLists default
    [string]$CudaRoot = "",              # a CUDA 12.x / 13.x toolkit; empty: get-cuda.ps1 -Version $CudaVersion
    [string]$CudaVersion = "12.9.1",
    [string]$VcpkgRoot = "",             # an existing vcpkg checkout; empty: build-deps\vcpkg-<tag>
    [string]$BuildDir = "build",
    [int]$Jobs = 0,                      # parallel compile jobs (0: Ninja's default)
    [switch]$Package,
    [string]$PackageVersion = ""         # zip name: av1gpu-<this>-windows-x86_64 (default: the CMake project version)
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$deps = Join-Path $repo "build-deps"
$vcpkgTag = "2026.10.16"
$triplet = "x64-windows-static"
New-Item -ItemType Directory -Force -Path $deps | Out-Null
function Run($exe) {
    & $exe @args
    if ($LASTEXITCODE -ne 0) { throw "$exe failed with exit code $LASTEXITCODE" }
}

# MSVC environment: Visual Studio 2022 (CUDA 12.9 accepts its compilers, 19.3x / 19.4x)
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vs = & $vswhere -version "[17.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -latest -property installationPath
    if (-not $vs) { throw "needs Visual Studio 2022 or its Build Tools with the C++ workload" }
    cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
    }
}
if (-not (Get-Command ninja.exe -ErrorAction SilentlyContinue)) {
    $nd = Join-Path $deps "ninja"
    if (-not (Test-Path "$nd\ninja.exe")) {
        New-Item -ItemType Directory -Force -Path $nd | Out-Null
        Invoke-WebRequest "https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-win.zip" -OutFile "$nd\ninja.zip"
        Expand-Archive "$nd\ninja.zip" $nd -Force
    }
    $env:PATH = "$nd;$env:PATH"
}

# CUDA
if (-not $CudaRoot) {
    $CudaRoot = Join-Path $deps "cuda-$CudaVersion"
    & "$PSScriptRoot\get-cuda.ps1" -Version $CudaVersion -Dest $CudaRoot
    if (-not $?) { throw "get-cuda.ps1 failed" }
}
$CudaRoot = (Resolve-Path $CudaRoot).Path
$env:CUDA_PATH = $CudaRoot

# image libraries
if (-not $VcpkgRoot) {
    $VcpkgRoot = Join-Path $deps "vcpkg-$vcpkgTag"
    if (-not (Test-Path "$VcpkgRoot\vcpkg.exe")) {
        Invoke-WebRequest "https://github.com/microsoft/vcpkg/archive/refs/tags/$vcpkgTag.zip" -OutFile "$deps\vcpkg.zip"
        Expand-Archive "$deps\vcpkg.zip" $deps -Force
        Remove-Item "$deps\vcpkg.zip"
        Run "$VcpkgRoot\bootstrap-vcpkg.bat" -disableMetrics
    }
}
if ($env:VCPKG_DEFAULT_BINARY_CACHE) { New-Item -ItemType Directory -Force -Path $env:VCPKG_DEFAULT_BINARY_CACHE | Out-Null }
Run "$VcpkgRoot\vcpkg.exe" install libjpeg-turbo libpng libwebp zlib --triplet $triplet

# configure + build
$cfg = @("-S", $repo, "-B", $BuildDir, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_TOOLCHAIN_FILE=$VcpkgRoot\scripts\buildsystems\vcpkg.cmake", "-DVCPKG_TARGET_TRIPLET=$triplet",
    "-DCUDAToolkit_ROOT=$CudaRoot", "-DCMAKE_CUDA_COMPILER=$CudaRoot\bin\nvcc.exe")
if ($Archs) { $cfg += "-DCMAKE_CUDA_ARCHITECTURES=$Archs" }
Run cmake @cfg
$b = @("--build", $BuildDir)
if ($Jobs -gt 0) { $b += @("-j", "$Jobs") }
Run cmake @b

if ($Package) {
    if (-not $PackageVersion) {
        $PackageVersion = (Select-String -Path "$repo\CMakeLists.txt" -Pattern 'project\(av1gpu VERSION ([0-9.]+)').Matches[0].Groups[1].Value
    }
    $name = "av1gpu-$PackageVersion-windows-x86_64"
    $pkg = Join-Path $BuildDir $name
    if (Test-Path $pkg) { Remove-Item -Recurse -Force $pkg }
    New-Item -ItemType Directory -Force -Path $pkg | Out-Null
    Copy-Item "$BuildDir\av1gpu.exe", "$repo\LICENSE", "$repo\THIRD_PARTY_NOTICES.md" $pkg
    $nvjpeg = Get-ChildItem "$CudaRoot\bin" -Recurse -Filter "nvjpeg64_*.dll" | Select-Object -First 1
    if (-not $nvjpeg) { throw "nvjpeg64_*.dll not found under $CudaRoot\bin" }
    Copy-Item $nvjpeg.FullName $pkg
    # only Windows system DLLs and nvJPEG: no Visual C++ runtime, no CUDA runtime DLL
    $imports = (dumpbin /nologo /dependents "$pkg\av1gpu.exe" | Select-String -Pattern '^\s+(\S+\.dll)$').Matches |
        ForEach-Object { $_.Groups[1].Value }
    Write-Host "av1gpu.exe imports: $($imports -join ', ')"
    $bad = $imports | Where-Object { $_ -match '^(msvcp|vcruntime|ucrtbase|api-ms-win-crt|cudart|jpeg|turbojpeg|libpng|zlib|libwebp)' }
    if ($bad) { throw "not linked in statically: $($bad -join ', ')" }
    cmd /c "`"$pkg\av1gpu.exe`" --help >nul 2>&1"
    if ($LASTEXITCODE -ne 0) { throw "av1gpu.exe --help failed ($LASTEXITCODE)" }
    $zip = Join-Path $BuildDir "$name.zip"
    if (Test-Path $zip) { Remove-Item $zip }
    Compress-Archive -Path $pkg -DestinationPath $zip
    $sha = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
    Set-Content -Path "$zip.sha256" -Value "$sha  $name.zip" -Encoding ascii -NoNewline
    Write-Host "$zip ($sha)"
}
