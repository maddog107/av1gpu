# Fetch the parts of the CUDA toolkit av1gpu needs (nvcc, cudart, CCCL, nvJPEG) from NVIDIA's redistributable
# archives into one directory, checking each archive's SHA-256 against NVIDIA's manifest.  About 140 MB instead of
# the full installer; no administrator rights or Visual Studio integration needed (av1gpu builds with Ninja).
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\get-cuda.ps1 -Version 12.9.1 -Dest C:\cuda-12.9.1
#
# Then pass -DCUDAToolkit_ROOT=<Dest> -DCMAKE_CUDA_COMPILER=<Dest>\bin\nvcc.exe to CMake (build.ps1 does).
param(
    [string]$Version = "12.9.1",
    [string]$Dest = "$PSScriptRoot\..\..\build-deps\cuda-$Version"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is many times slower with the progress bar
$base = "https://developer.download.nvidia.com/compute/cuda/redist"
$components = @("cuda_nvcc", "cuda_cudart", "cuda_cccl", "libnvjpeg", "cuda_profiler_api")

if (Test-Path "$Dest\bin\nvcc.exe") { Write-Host "CUDA $Version already in $Dest"; exit 0 }
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "av1gpu-cuda-$Version"
New-Item -ItemType Directory -Force -Path $tmp, $Dest | Out-Null
$manifest = Invoke-RestMethod "$base/redistrib_$Version.json"
foreach ($c in $components) {
    $entry = $manifest.$c.'windows-x86_64'
    if (-not $entry) { throw "$c has no windows-x86_64 archive in CUDA $Version" }
    $zip = Join-Path $tmp (Split-Path $entry.relative_path -Leaf)
    Write-Host "$c $($manifest.$c.version)"
    Invoke-WebRequest "$base/$($entry.relative_path)" -OutFile $zip
    $sha = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
    if ($sha -ne $entry.sha256) { throw "SHA-256 mismatch for $zip" }
    $x = Join-Path $tmp $c
    if (Test-Path $x) { Remove-Item -Recurse -Force $x }
    Expand-Archive -Path $zip -DestinationPath $x
    # each archive holds one top-level folder (bin, include, lib, ...): merge its contents into Dest
    $top = (Get-ChildItem $x | Select-Object -First 1).FullName
    foreach ($f in Get-ChildItem $top -Recurse -File) {
        $to = Join-Path $Dest $f.FullName.Substring($top.Length + 1)
        New-Item -ItemType Directory -Force -Path (Split-Path $to) | Out-Null
        Copy-Item -LiteralPath $f.FullName -Destination $to -Force
    }
}
Remove-Item -Recurse -Force $tmp
Write-Host "CUDA $Version in $Dest"
