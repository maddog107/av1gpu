# speed.ps1 -Bin ... -Tag ...: warm-up + 3 runs per set (no outputs written) -> C:\res\<Tag>\speed.txt
param([string]$Bin, [string]$Tag)
Set-Location C:\data
$R = "C:\res\$Tag"; New-Item -ItemType Directory -Force -Path $R | Out-Null
$out = @()
foreach ($s in @("jpg3.txt", "png40x2.txt", "mix3.txt", "mix3.txt --yuv 444 --depth 10")) {
    for ($k = 0; $k -le 3; $k++) {
        $line = cmd /c "`"$Bin`" --batch $s --stats 2>nul" | Where-Object { $_ -like "{*" }
        $d = $line | ConvertFrom-Json
        if ($k -gt 0) { $out += "$($s -replace ' ', '_') $k $($d.mp_s) $($d.cpu_cores)" }
    }
}
[System.IO.File]::WriteAllLines("$R\speed.txt", [string[]]$out)
$out
