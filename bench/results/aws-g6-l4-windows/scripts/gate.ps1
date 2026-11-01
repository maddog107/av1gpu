# gate.ps1 -Bin ...: replay gates.txt's encodes (as decoder_gate.py runs them: --batch DIR --outdir <run>\q<Q> --qindex Q
# --debug-outputs OPTS) in C:\data, outputs in C:\gate\<run>\, hashes in C:\res\gate_md5.txt
param([string]$Bin)
Set-Location C:\data
New-Item -ItemType Directory -Force -Path C:\data\gate_img | Out-Null
Copy-Item C:\av1gpu\tests\gate_images\*.png C:\data\gate_img\
Remove-Item -Recurse -Force C:\gate -ErrorAction SilentlyContinue
foreach ($line in [System.IO.File]::ReadAllLines("C:\test\gates.txt")) {
    if (-not $line) { continue }
    $run, $imgs, $q, $extra, $opts = $line.Split("`t")
    if ($opts -eq "-") { $opts = "" }
    $qs = if ($q -eq "lossless") { @("lossless") } else { $q.Split(",") }
    foreach ($v in $qs) {
        $sub = if ($v -eq "lossless") { "lossless" } else { "q$v" }
        $qa = if ($v -eq "lossless") { "--lossless" } else { "--qindex $v" }
        $out = "C:\gate\$run\$sub"
        New-Item -ItemType Directory -Force -Path $out | Out-Null
        cmd /c "`"$Bin`" --batch $imgs --outdir $out $qa --debug-outputs $opts > nul 2> `"C:\gate\$run\$sub.err`""
        "$run $sub $LASTEXITCODE"
    }
}
$root = "C:\gate"
$md5 = Get-ChildItem C:\gate -Recurse -File | Where-Object { $_.Extension -ne ".err" } | ForEach-Object {
    $rel = $_.FullName.Substring($root.Length + 1) -replace "\\", "/"
    "$rel $((Get-FileHash -Algorithm MD5 -LiteralPath $_.FullName).Hash.ToLower())"
} | Sort-Object -CaseSensitive
New-Item -ItemType Directory -Force -Path C:\res | Out-Null
[System.IO.File]::WriteAllLines("C:\res\gate_md5.txt", [string[]]$md5, (New-Object System.Text.UTF8Encoding $false))
"$($md5.Count) files"
