# run.ps1 -Bin C:\av1gpu\av1gpu.exe -Tag win: every case of cases.txt (in C:\data) -> C:\res\<Tag>\{md5.txt,exit.txt,logs\}
param([string]$Bin, [string]$Tag)
$ErrorActionPreference = "Continue"
Set-Location C:\data
$R = "C:\res\$Tag"
Remove-Item -Recurse -Force $R, C:\data\out -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path "$R\logs", C:\data\out | Out-Null
$exit = @()
foreach ($line in [System.IO.File]::ReadAllLines("C:\data\cases.txt")) {
    if (-not $line) { continue }
    $name, $envs, $argline = $line.Split("`t")
    if ($name -like "s_*") { New-Item -ItemType Directory -Force -Path "out\$name" | Out-Null }
    $set = ""
    foreach ($e in ($envs -split " ")) { if ($e -and $e -ne "-") { $set += "set $e&& " } }
    $a = $argline -replace "/", "\"     # Windows-style paths on the command line (lists keep forward slashes)
    cmd /c "$set`"$Bin`" $a > `"$R\logs\$name.out`" 2> `"$R\logs\$name.err`""
    $exit += "$name $LASTEXITCODE"
}
[System.IO.File]::WriteAllLines("$R\exit.txt", $exit)
$root = (Resolve-Path C:\data\out).Path
$md5 = Get-ChildItem C:\data\out -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring($root.Length + 1) -replace "\\", "/"
    "$rel $((Get-FileHash -Algorithm MD5 -LiteralPath $_.FullName).Hash.ToLower())"
} | Sort-Object -CaseSensitive
[System.IO.File]::WriteAllLines("$R\md5.txt", [string[]]$md5, (New-Object System.Text.UTF8Encoding $false))
"$($md5.Count) files"
