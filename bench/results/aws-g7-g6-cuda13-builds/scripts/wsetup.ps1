# wsetup.ps1: AWS's GRID driver (R580, 582.78; WDDM), hardware-accelerated GPU scheduling on, then the test data:
# C:\data\sr (the 100 size-range JPEGs of openimages_v4_gamut.csv, SHA-256 checked), C:\data\kodak (24 PNGs),
# C:\data\sr3 (the size range three times, for speed), C:\data\one\a.jpg.  Reboots at the end.
$ProgressPreference = 'SilentlyContinue'
$H = 'C:\Users\Administrator'
$p = Start-Process -FilePath "$H\grid58278.exe" -ArgumentList '-s', '-noreboot' -Wait -PassThru
"DRIVER_EXIT=$($p.ExitCode)"
New-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers -Name HwSchMode -Value 2 -PropertyType DWord -Force | Out-Null
New-Item -ItemType Directory -Force -Path C:\data\sr, C:\data\kodak, C:\data\sr3, C:\data\one | Out-Null
Copy-Item "$H\a.jpg" C:\data\one\a.jpg
$bad = 0
foreach ($r in (Import-Csv "$H\openimages_v4_gamut.csv")) {
    $f = "C:\data\sr\$($r.file)"
    for ($t = 0; $t -lt 3; $t++) {
        try { Invoke-WebRequest $r.url -OutFile $f -UseBasicParsing } catch { Start-Sleep 2; continue }
        if ((Get-FileHash $f).Hash.ToLower() -eq $r.sha256) { break }
    }
    if ((Get-FileHash $f).Hash.ToLower() -ne $r.sha256) { "BAD $($r.file)"; $bad++ }
}
"sr: $((Get-ChildItem C:\data\sr).Count) files, $bad bad"
foreach ($i in 1..24) {
    $n = 'kodim{0:D2}.png' -f $i
    Invoke-WebRequest "https://r0k.us/graphics/kodak/kodak/$n" -OutFile "C:\data\kodak\$n" -UseBasicParsing
}
"kodak: $((Get-ChildItem C:\data\kodak).Count) files"
foreach ($k in 0..2) { foreach ($f in Get-ChildItem C:\data\sr) { Copy-Item $f.FullName "C:\data\sr3\r$($k)_$($f.Name)" } }
"sr3: $((Get-ChildItem C:\data\sr3).Count) files"
Get-FileHash C:\data\kodak\*.png | ForEach-Object { "$((Split-Path $_.Path -Leaf)) $($_.Hash.ToLower())" } | Set-Content C:\data\kodak.sha256
'WSETUP_DONE'
Restart-Computer -Force
