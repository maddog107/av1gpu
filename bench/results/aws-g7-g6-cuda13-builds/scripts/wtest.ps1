# wtest.ps1 -Builds c13,c12 [-Parts ident,startup,speed]: the release zips (unpacked in C:\av1gpu\<build>) on this
# Windows L4.  ident: outputs of the size range (defaults, --yuv 444 --depth 10) and Kodak -> C:\res\<build>_<set>.sha256;
# startup: one JPEG, compute cache empty / warm / disabled, 2 rounds; speed: C:\data\sr3 (1,433 MP), a warm-up and 3
# timed runs per build, 2 rounds with the order alternating, MP/s from --stats.  Everything in C:\res.
param([string[]]$Builds = @("c13", "c12"), [string[]]$Parts = @("ident", "startup", "speed"))
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path C:\res, C:\out | Out-Null
function Exe($b) { (Get-ChildItem "C:\av1gpu\$b" -Recurse -Filter av1gpu.exe | Select-Object -First 1).FullName }
function Run($b, $envs, $a) {   # cmd line with environment settings; returns wall seconds
    $set = ""; foreach ($e in $envs) { $set += "set $e&& " }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    cmd /c "$set`"$(Exe $b)`" $a > C:\res\last.out 2> C:\res\last.err"
    $sw.Stop(); $script:rc = $LASTEXITCODE
    return $sw.Elapsed.TotalSeconds
}
"driver: $(& 'C:\Windows\System32\nvidia-smi.exe' --query-gpu=name,driver_version,driver_model.current --format=csv,noheader)" | Tee-Object -Append C:\res\info.txt
foreach ($b in $Builds) { "$b $(Exe $b) $((Get-FileHash (Exe $b)).Hash.ToLower())" | Tee-Object -Append C:\res\info.txt }
if ($Parts -contains "ident") {
    foreach ($b in $Builds) {
        foreach ($s in @(@("sr", "C:\data\sr", ""), @("sr444", "C:\data\sr", "--yuv 444 --depth 10"), @("kodak", "C:\data\kodak", ""))) {
            $o = "C:\out\$b\$($s[0])"; Remove-Item -Recurse -Force $o -ErrorAction SilentlyContinue
            $t = Run $b @() "--batch $($s[1]) --outdir $o $($s[2])"
            "ident $b $($s[0]): rc=$rc $([math]::Round($t, 2)) s, $((Get-ChildItem $o).Count) files" | Tee-Object -Append C:\res\ident.txt
            Get-FileHash "$o\*" | ForEach-Object { "$((Split-Path $_.Path -Leaf)) $($_.Hash.ToLower())" } | Sort-Object |
                Set-Content "C:\res\$($b)_$($s[0]).sha256"
        }
    }
}
if ($Parts -contains "startup") {
    Set-Content C:\data\one\list.txt "C:\data\one\a.jpg"
    foreach ($r in 1..2) {
        foreach ($b in $Builds) {
            $c = "C:\cc\$b"; Remove-Item -Recurse -Force $c, C:\out\one -ErrorAction SilentlyContinue
            $e = Run $b @("CUDA_CACHE_PATH=$c") "--batch C:\data\one\list.txt --outdir C:\out\one"
            $kb = [math]::Round(((Get-ChildItem $c -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum) / 1MB, 1)
            $w = Run $b @("CUDA_CACHE_PATH=$c") "--batch C:\data\one\list.txt --outdir C:\out\one"
            $d = Run $b @("CUDA_CACHE_DISABLE=1") "--batch C:\data\one\list.txt --outdir C:\out\one"
            "startup $b round $r`: empty $([math]::Round($e, 2)) s (cache $kb MB), warm $([math]::Round($w, 2)) s, disabled $([math]::Round($d, 2)) s, rc=$rc" |
                Tee-Object -Append C:\res\startup.txt
        }
    }
}
if ($Parts -contains "speed") {
    $v = @{}; foreach ($b in $Builds) { $v[$b] = @() }
    foreach ($r in 1..2) {
        $order = if ($r % 2) { $Builds } else { $Builds[($Builds.Count - 1)..0] }
        foreach ($b in $order) {
            for ($k = 0; $k -le 3; $k++) {
                Remove-Item -Recurse -Force C:\out\spd -ErrorAction SilentlyContinue
                $t = Run $b @() "--batch C:\data\sr3 --outdir C:\out\spd --stats"
                $j = Get-Content C:\res\last.out | Where-Object { $_ -like "{*" } | Select-Object -Last 1 | ConvertFrom-Json
                if ($k -gt 0) {
                    $v[$b] += $j.mp_s
                    "speed $b round $r run $k`: $($j.mp_s) MP/s, wall $([math]::Round($t, 2)) s, cpu $($j.cpu_cores), rc=$rc" | Tee-Object -Append C:\res\speed.txt
                }
            }
        }
    }
    foreach ($b in $Builds) {
        $s = $v[$b] | Sort-Object; $m = ($s[2] + $s[3]) / 2
        "median $b`: $m MP/s ($($v[$b] -join ' '))" | Tee-Object -Append C:\res\speed.txt
    }
}
'WTEST_DONE'
