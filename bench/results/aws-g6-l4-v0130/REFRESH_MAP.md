# REFRESH_MAP: README.md / docs/RESULTS.md numbers that depend on av1gpu's speed or file size (v0.13.0 refresh)

Line numbers are those of the docs at 7e79fbe. "repro" = I recomputed the formula from the stored data and got the printed value; "NO" = it did not.
Shorthand: `SP(x)` = `V13.speed['x/new']`, `SPo(x)` = `V13.speed['x/old']`, `CH(x)` = `V13.speed_change['x']`, `BD(k)` = `V13.bd_rate[k].median` (`.mean` where a mean is printed), `BDS(k,cls)` = `V13.bd_rate_by_size[k][cls].median`, `Q60(img)` = the `V13.quality.av1gpu` row with that image and q 60.
Fixed baselines: s6 = 28.752 / s9 = 187.561 (aws-g6-l4-gamut-enc2 speed, libaom 6/9); sv[p], svb[p] = aws-c6a-gamut-svt speed / bd_rate svt_p{p}_vs_aom_s6; nv = 15.66 (aws-g6-l4-nvenc best, 8 workers); aom_fmt[f] = aws-g6-l4-formats aom_s6 mp_s; tun[label] = aws-c6a-cpu-tuning primary.timings; JPEG total of the size range = 160,393,126 B (sum of bench/openimages_v4_gamut.csv bytes).
Abbreviations: D = `SP('jpeg/size_range/default_enc4').mp_s`, NS = `SP('jpeg/size_range/nosmall_enc4').mp_s`, T3 = `SP('jpeg/size_range/turbo3ns_enc4').mp_s`.

Notes on the keys as scripts/analyze.py writes them (they take precedence over guesses below):
- Ready-made ratios in `V13.derived`: `default_vs_aom_s6` (D / s6), `default_vs_nvenc`, `turbo3ns_vs_svt_p9`, `turbo3ns_vs_aom_s9`,
  `vs_svt_at_same_size.{default,nosmall,turbo3ns}` (= speed / `svt_interp_at`, the README's "4.4–8.6×"), `cvvdp_vs_svt`,
  `cvvdp_vs_aom`, plus `svt_interp_at`, `svt46_interp`, `q60_99`, `cvvdp_svt_speed`, `cvvdp_aom_speed`.
- Standard set: `V13.bd_rate_std.{av1gpu_vs_aom_s6, av1gpu_nosmall_vs_aom_s6, av1gpu_turbo3_nosmall_vs_aom_s6, av1gpu_vs_webp}`
  (against aws-g6-l4's stored libaom speed 6 / WebP rows), rows in `V13.quality_std`.
- Start-up: `V13.single['photo.jpg' | 'big.jpg' | 'thumb512.jpg' | 't320.jpg' | 'pair batch (photo.jpg + thumb512.jpg)'].{new, old, old_noise}.median` (ms).
- v0.12.0: `V13.v0120_on_l4.{bd_rate, totals}`; per config `V13.new_vs_old[cfg].{all, lt1, 1to2, ge2, bytes_change, bd_rate_new_vs_old}`.
- Not refreshed by this run: the luma-only CVVDP (`bd_rate_cvvdp.av1gpu_vs_aom_s6_luma_only`) — keep the tiles4x4 value with a note.
- RGBA: the lost `data.tar.gz` was rebuilt on the Mac from the 2026-10-31 transcript; all 39 set PNGs match the v0.9.0 run's
  input hashes byte for byte. If it is uploaded, the alpha rows (speed `alpha/*`, `V13.alpha`) ARE refreshed and the
  "measured with v0.9.0" notes for them are not needed; if not, `V13.not_measured` lists `alpha`.

## 1. README.md

### Hero, TL;DR, Overview, key-figures table

| line | current | source / formula | notes |
|---|---|---|---|
| 19 | alt "…av1gpu's 421 KB AVIF" | `Q60('b1aa6e8182e3f22f_huge.jpg').bytes` /1000 | repro (421,035 B). 6,031 KB = JPEG, fixed. Image itself: see (c) |
| 20 | caption "av1gpu's AVIF at the default quality, 421 KB" | same | repro |
| 25 | "9× faster than libaom on 16 CPU threads" | D / s6 | repro (9.07) |
| 25-26 | "with the same file sizes" | BD('av1gpu_vs_aom_s6') ≈ 0 | wording if it moves to +0.4–0.6 %, see (d) |
| 26 | "a third of one CPU core" | `SP('jpeg/size_range/default_enc4').cpu_cores` | repro (0.276) |
| 26 | "86 % smaller than the JPEGs" | 1 − `V13.totals['60'].bytes` / 160,393,126 | repro (85.9 %) |
| 26-27 | "31 % smaller than WebP" | −BD('av1gpu_vs_webp') | repro (−30.6 %) |
| 27 | "17× faster than NVENC" | D / nv | repro (16.6) |
| 34 | "17× slower" | D / nv | repro |
| 34-35 | "about 26 % bigger files" | 1 / (1 + BD('av1gpu_vs_nvenc_even_height')) − 1 | repro (26.0 %); uses even-height (all-96 gives 29.4 %) |
| 39-40 | "about 7 days of GPU time" (50 trillion px, "one 70 W card") | none | source unknown: 50e6 MP / 7 d = 83 MP/s, matches neither L4 (261 → 2.2 d) nor RTX PRO 2000 110.8 MP/s (5.2 d). See (a) |
| 41-42 | "40 % smaller than the WebP it serves today, 1.5 TB less" | Bellazon production | (a): not re-measured |
| 42 | "(at equal quality, av1gpu's files are 31 % smaller than WebP's)" | −BD('av1gpu_vs_webp') | repro |
| 46 | "about 7 GPU-days on one card" | as line 39 | source unknown |
| 47 | "**261 MP/s** … (with `GP_ENCODERS=4`)" | D | repro (260.66) |
| 48 | "**0.36 s** for one photo … (2.5 MP JPEG, L4)" | `V13.single['photo.jpg'].new.median` / 1000 | repro (v0.9.0 358.7 ms) |
| 49 | "**+0.1 %** … +0.7 % against libaom with its tiling off" | BD('av1gpu_vs_aom_s6'); BD('av1gpu_vs_aom_s6_notiles') | repro (+0.08, +0.72) |
| 50 | "**−31 %** file size against WebP" | BD('av1gpu_vs_webp') | repro |
| 51 | "**0.3 cores** of CPU while encoding JPEGs" | `SP('jpeg/size_range/default_enc4').cpu_cores` | repro |
| 52 | "bit-exact … in dav1d and libaom" | `V13.decoder_gate` | wording check only |

### See it run (demo)

| line | current | source / formula | notes |
|---|---|---|---|
| 56 | "99 camera photos (478 MP, 154 MB of JPEG)" | input set, fixed | |
| 59 | alt "…in 3.4 seconds of wall time at 192.8 MP/s using 0.32 CPU cores; 154 MB … becomes 22 MB" | aws-g6-l4-speed-v06 results.json "demo" (v0.6, 2026-11-01) | (c): not re-recorded; add "(v0.6)" note or re-record |
| 62 | "on longer batches the speed settles at the figures below" | wording | true either way |

### Results intro, trade-off chart and paragraph

| line | current | source / formula | notes |
|---|---|---|---|
| 66-70 | method (EPYC 7R13, avifenc 1.4…) | fixed | optionally name "av1gpu v0.13.0" |
| 74 | alt "…arrow marking av1gpu 9.1× faster…" | D / s6 | repro |
| 74 | alt "default 261 MP/s at +0.1 percent" | D; BD('av1gpu_vs_aom_s6') | repro |
| 74 | alt "no small preset 299 at +2.2" | NS; BD('av1gpu_nosmall_vs_aom_s6') | repro (298.9, +2.23) |
| 74 | alt "turbo3 420 at +5.7" | T3; BD('av1gpu_turbo3_nosmall_vs_aom_s6') | repro; bytes should be unchanged (check `V13.new_vs_old`) |
| 74 | alt SVT / libaom / NVENC values | fixed | |
| 77 | "From libaom-speed-6 sizes up to +6 %" | BD turbo3 nosmall rounded | repro (5.74) |
| 78 | "4.4–8.6× faster than the best CPU encoder" | min/max over k∈{default,nosmall,turbo3ns} of SP(k).mp_s / `V13.derived.svt_interp_at[k]` | repro (8.60, 5.88, 4.36). The brief's "2.4–4×" is not in the current text |
| 79-80 | "SVT-AV1's slow presets go further (preset 2: −5.1 %, at 4.6 MP/s); av1gpu's own `--preset slow` has not been timed on the L4 yet" | CPU fixed | wording: still true unless V13 times `slow` |
| 81 | "makes it 1–15 % faster" | CPU only | skip |
| 82 | "at libaom-speed-6 sizes av1gpu is then 7.0–8.4× faster" | D / `V13.derived.svt46_interp.pgo_untiled` (7.04) and D / tun['tuned_aom_s6_j2'] (31.09 → 8.39) | repro |

### Smallest files (match-small)

| line | current | source / formula | notes |
|---|---|---|---|
| 92 | alt "av1gpu default 261 MP/s" | D | repro; rest CPU/NVENC fixed |
| 95 | "**9.1× faster than libaom running on all 16 CPU threads**" | D / s6 | repro |
| 96 | "(0.3 of a core)" | `SP('jpeg/size_range/default_enc4').cpu_cores` | repro |
| 96-97 | "SVT-AV1 brackets the same size with preset 6 … and preset 4" | needs svb[4] < BD default < svb[6] | wording check (holds for +0.6 %) |

### Fastest (match-fast)

| line | current | source / formula | notes |
|---|---|---|---|
| 103 | alt "turbo3 … 420 MP/s"; "av1gpu turbo3 +5.7 percent" | T3; BD('av1gpu_turbo3_nosmall_vs_aom_s6') | repro |
| 106 | "**2.2–2.3× as fast as both**" | T3 / s9 (2.24), T3 / sv[9] (2.29) | repro |
| 106 | "+5.7 % against libaom speed 6" | BD turbo3 nosmall | repro |
| 108 | "av1gpu uses 0.4 of a CPU core" | `SP('jpeg/size_range/turbo3ns_enc4').cpu_cores` | repro (0.444) |

### Same GPU: vs NVENC

| line | current | source / formula | notes |
|---|---|---|---|
| 115 | alt "av1gpu default 261 MP/s"; "av1gpu +0.1 percent" | D; BD('av1gpu_vs_aom_s6') | repro |
| 118 | "**17× faster …, with files 21 % smaller.**" | D / nv; −BD('av1gpu_vs_nvenc_even_height') | repro (16.6, 20.6 %) |
| 119-124 | NVENC 16 MP/s, 4 cores, +27.9 / +29.5 % | NVENC only | skip |

### FPGA (Pulsar)

| line | current | source / formula | notes |
|---|---|---|---|
| 131 | "**645 images/s** … with `--preset turbo3 --small-preset none`" | `SP('kodak/turbo3_nosmall').img_s` | repro (645.4) |
| 131-132 | "(522 counting CUDA start-up)" | `SP('kodak/turbo3_nosmall').img_s_wall` | repro |
| 132 | "365 with `--small-preset none`" | `SP('kodak/nosmall').img_s` | repro |
| 132 | "152 with the default" | `SP('kodak/default').img_s` | repro; Kodak is 0.39 MP, so base preset: expect the +43 % here |
| 130-131 | "$0.80/hour" | fixed; no cost-per-image in the text | if one is added: 0.80 / 3600 / img_s |

### Quality and file size (quality, by-size charts), main table

| line | current | source / formula | notes |
|---|---|---|---|
| 140 | alt "The av1gpu and libaom speed 6 curves coincide" | `V13.quality.av1gpu` vs gamut aom_s6 | wording check after V13 |
| 145 | alt "av1gpu: +0.1, +1.9, +0.2 and −2.1 percent" | BDS('av1gpu_vs_aom_s6', '< 1 MP' / '1-5 MP' / '5-10 MP' / '>= 10 MP') | repro. by_size_chart recomputes these from V13.quality rows; the alt must equal the chart |
| 150 | "**261**", "**+0.1 %**", "**−30.6 %**", "0.3" | D; BD('av1gpu_vs_aom_s6'); BD('av1gpu_vs_webp'); default_enc4 cpu_cores | repro |
| 151 | "299", "+2.2 %", "0.3" | NS; BD('av1gpu_nosmall_vs_aom_s6'); nosmall_enc4 cpu_cores (0.32) | repro |
| 152 | "420¹", "+5.7 %", "0.4" | T3; BD turbo3 nosmall; turbo3ns_enc4 cpu_cores | repro |
| 163 | "¹ With the small-image preset on, `turbo3` runs at 331 MP/s, +1.2 %" | `SP('jpeg/size_range/turbo3_enc4').mp_s`; BD('av1gpu_turbo3_vs_aom_s6') | repro (331.0, +1.25). Bytes CHANGE: images < 1 MP go to `base`, which v0.13 changes |
| 164 | "² … av1gpu's default (v0.3.0 then) ran at 116 MP/s alongside it" | aws-g6-l4-nvenc av1gpu_same_session | (a): keep |
| 172-174 | "against untiled libaom speed 6 av1gpu's default is +0.7 %" | BD('av1gpu_vs_aom_s6_notiles') | repro; SVT −2.9 / +2.6 / +9.8 are CPU-only |

### 4:4:4 and 10-bit

| line | current | source / formula | notes |
|---|---|---|---|
| 182 | alt "8-bit 4:2:0: av1gpu 261 MP/s … 8-bit 4:4:4: 198 … 10-bit 4:2:0: 253 … 10-bit 4:4:4: 194" | D; `SP('formats/444_8')`, `SP('formats/420_10')`, `SP('formats/444_10')` .mp_s | repro; libaom 29/24/22/19 fixed |
| 182 | alt "within 0.4 percent" | max over f of abs(BD('av1gpu_{f}_vs_aom_s6_{f}')), rounded up (make_charts `within`) | repro (0.397 → 0.4) |
| 186 | "within 0.4 % … at 8.1–11.5× the speed" | same; SP(fmt)/aom_fmt[f] min/max | repro (9.11, 8.14, 11.51, 10.27) |
| 187-188 | "about 2 % smaller … (av1gpu −2.2 %, libaom −2.0 %)" | BD('av1gpu_420_10_vs_av1gpu_420_8'); libaom fixed | repro (−2.18) |
| 188 | "with no measurable speed cost for av1gpu" | SP('formats/420_10')/D − 1 | does not reproduce: −2.8 % (RESULTS l.507 says "2–3 %"); wording (d) |
| 189 | "4:4:4 makes little difference here" | BD('av1gpu_444_8_vs_av1gpu_420_8') | +0.40 |

### Lossless (output unchanged in v0.13; speeds only)

| line | current | source / formula | notes |
|---|---|---|---|
| 195-196 | "47 PNGs … 42.7 MP" | set of the v0.6 table | V13 lossless/* run the 51-PNG / 49.1 MP set (mk_ll_sets.py today) |
| 200 | ycgco row MP/s "45" | v0.6, 47 PNGs ×4, `--palette off` (RESULTS l.560: 44.5) | cannot map 1:1: `SP('lossless/cmp_4x_ycgco').mp_s` is 51 PNGs with palette auto (v0.9.0: 44.3). Either keep "(v0.6)" or relabel to V13 set |
| 201 | GBR row MP/s "44" | v0.6 43.6 | same; V13 `SP('lossless/cmp_4x').mp_s` (v0.9.0: 45.3) |
| 200-201 | sizes 90.6 %, 11.99 / 4.82 MB, 108.5 %, 13.37 / 6.74 MB | unchanged bytes | confirm via `V13.new_vs_old` (lossless identical) |
| 206 | "about 5× faster" | av1gpu MP/s / 8.8 and / 9.3 | repro (5.1, 4.7) |
| 207-208 | "a single pass of 42.7 MP, CUDA start-up included, ran at 19 MP/s" | v0.6 one-pass (18.9 / 19.4) | V13 equivalent: `SP('lossless/cmp_1pass').mp_s_wall` (51-PNG set; v0.9.0 21.8) |
| 209 | "palette mode … takes 10 % off the graphics" | palette study | (a) |

### Screen content (palette)

| line | current | source / formula | notes |
|---|---|---|---|
| 215-218 | "about 24 % smaller", "9.5 % smaller than libaom", "3.3 % bigger", "22 % smaller", "0.37× … (57 against 153 MP/s on the L4)" | aws-g6-l4-palette-study | (a): not re-measured; v0.13 base/turbo changes may move these too |

### Transparency (alpha: refresh only if V13 has `alpha/*`, else note "measured with v0.9.0" / "files of v0.4.0")

| line | current | source / formula | notes |
|---|---|---|---|
| 231 | default row 699 KB / 91 KB / 50.2 dB / 50 | sizes: aws-g6-l4-tiles4x4 `alpha['new/alpha_default']` (`--palette off`, v0.4.0 files); speed: v0.9.0 `--palette off` (v090-check NOTES.md l.70) | V13 schema has only total bytes, no colour/alpha split and no `--palette off` run → (a). Bytes WILL change (all 39 PNGs < 1 MP → base) |
| 232 | q100 row 1,088 KB / 480 KB / exact / 44 | same (`alpha_q100`) | colour part changes; alpha part (lossless) unchanged |
| 235 | lossless row 7.72 MB / 486 KB / exact / 30 | same (`alpha_lossless`) | bytes unchanged; speed (a) |
| 239 | "Exact alpha comes out about 9 % smaller than libaom's" | 480,059 vs 527,573 | lossless path, unchanged |
| 241 | "(685 KB)" | `SP('alpha/default').bytes` / 20 | repro (13,708,280 / 20 = 685,414) |
| 241 | "(32, 30 and 29 MP/s)" | `SP('alpha/default'|'alpha/q100'|'alpha/lossless').mp_s_wall` | repro (31.46, 30.07, 29.33) |

### CVVDP

| line | current | source / formula | notes |
|---|---|---|---|
| 248 | alt "av1gpu +0.1 and +4.2 percent" | BD('av1gpu_vs_aom_s6'); `V13.bd_rate_cvvdp.av1gpu_vs_aom_s6.median` | repro; others fixed |
| 253-254 | "**+4.2 %** … instead of +0.1 %" | same | repro |
| 254 | "The difference is in luma and grows with image size" | luma_only (not refreshed) + `V13.bd_rate_cvvdp_by_size.av1gpu_vs_aom_s6` | wording check |
| 255 | "about 3.6× faster than SVT-AV1 and 5.3× faster than libaom" | D / `V13.derived.cvvdp_svt_speed`; D / `V13.derived.cvvdp_aom_speed` | repro (72.6 → 3.59, 49.2 → 5.30) |
| 256 | "av1gpu at 0.46 bits per pixel" | `V13.cvvdp_target.median_bpp` | repro (0.4616) |
| 256 | "on all 99 photos" | `V13.cvvdp_target.reached` / `.images` | repro |

### What the files look like

| line | current | source / formula | notes |
|---|---|---|---|
| 262-263 | "160.4 MB, into 22.5 MB of AVIF, at a mean SSIMULACRA2 of 72" | 160.4 fixed; `V13.totals['60'].bytes`, `.ssimulacra2` | repro (22,538,643; 71.99) |
| 263-264 | "`--quality 70` gave 33.4 MB at 78" | `V13.totals['70']` | repro (33,430,493; 78.4) |
| 268 | "90 KB → 27 KB … −70 %" | `Q60('66ce30dc11941998_tiny.jpg').bytes`; 1 − bytes/90,278 | repro (27,245; 69.8 %) |
| 269 | "177 KB → 60 KB … −66 %" | `Q60('4d1e8704fa4bc4e5_small.jpg')` vs 177,070 | repro (59,903) |
| 270 | "2.06 MB → 177 KB … −91 %" | `Q60('8c67f6e6c2ca8a6d_typical.jpg')` vs 2,060,084 | repro (177,359) |
| 271 | "1.88 MB → 213 KB … −89 %" | `Q60('ca12d94f3a1dca2c_large.jpg')` vs 1,881,462 | repro (213,338) |
| 272 | "6.08 MB → 578 KB … −90 %" | `Q60('df28dcbd0a84d456_huge.jpg')` vs 6,082,968 | repro (578,346) |
| 276 | "SSIMULACRA2 against the source: 67.9, 77.3, 73.6, 71.1 and 69.0" | `.ssimulacra2` of the same five rows | repro |
| 286-287 | "With v0.3.0's 2 × 2 tiles the RTX PRO 2000 and the L4 produced byte-identical files … these are the L4's files with the 4 × 4 default" | wording (d) | |

### Same file size, side by side

| line | current | source / formula | notes |
|---|---|---|---|
| 296 | alt "av1gpu (421.0 KB, SSIMULACRA2 72), libaom speed 6 (412.1 KB, 71), libaom speed 9 (395.9 KB, 66), WebP (417.6 KB, 62) and JPEG (420.8 KB, 36)" | av1gpu: `Q60('b1aa6e8182e3f22f_huge.jpg')` | 421.0 repro; "SSIMULACRA2 72" NO: row has 71.4. The other encoders were "pushed to av1gpu's size": source unknown (no script/results dir found). See (c) |
| 300 | summary "(578 KB each)" | `Q60('df28dcbd0a84d456_huge.jpg').bytes` | repro |
| 302 | alt "av1gpu (578.3 KB, SSIMULACRA2 69), libaom speed 6 (577.8 KB, 68) … JPEG (558.7 KB, 49)" | av1gpu row as above | av1gpu repro (578,346 / 69.0); other encoders source unknown |
| 308 | "re-encoded with av1gpu (4 × 4 default, on the L4)" | wording (d) | |

### Status, Download, Building, Windows (not re-measured: see (a))

| line | current | notes |
|---|---|---|
| 322-323 | RTX PRO 4500 "same bytes as the L4 for all 5,679 files of the v0.9.0 release check" | (a) |
| 323-324 | RTX 5090 byte-identical | (a) |
| 352 | "+14–16 % on an RTX PRO 4500" | (a) |
| 353 | "0.34 s there (v0.10.0, driver 595)" | (a) |
| 407-411 | Clang: "1:47 instead of 4:54", "8–17 % slower (v0.3.0 code …: default 70 vs 83 MP/s, turbo3 103 vs 125, slow 14.4 vs 15.7)" | (a) |
| 430-434 | Windows "1,226 files", "2,125 files", "67–83 % of the Linux speed … (JPEG 86 against 128 MP/s)", "224 files each" | (a) |

### Usage, v0.13 paragraph, tiles paragraph, image ops

| line | current | source / formula | notes |
|---|---|---|---|
| 479 | "0.36 s for a 2.5 MP JPEG" | `V13.single['photo.jpg'].new.median` | repro (v0.9.0 358.7) |
| 480 | "0.47 s for 10 MP" | `V13.single['big.jpg'].new.median` | repro (467.1) |
| 480 | "0.35 s for a 512 × 384 thumbnail" | `V13.single['thumb512.jpg'].new.median` | repro (349.5) |
| 480 | "a `--batch` of a photo and its thumbnail 0.58 s" | `V13.single['pair'].new.median` | repro (575.3) |
| 480-481 | "(0.34, 0.42, 0.34 and 0.67 s on an RTX PRO 4500 with v0.10.0" | aws-g6-g7-recon-lazy | (a) |
| 517-518 | "about 10 % faster on large photos" | `CH('jpeg/size_range/nosmall_enc4')` (turbo on every image) or `CH('jpeg/size_range/default_enc4')` | must be confirmed; "large photos" has no exact key (no ≥ 1 MP subset in the schema) |
| 518 | "40 % faster on images below 1 MP" | `CH('jpeg/standard/default_enc3')`, `CH('thumbs/default')`, `CH('kodak/default')` | must be confirmed |
| 518-519 | "files about 0.3–0.5 % larger at the same quality" | `V13.new_vs_old['default'].bd_rate` (and nosmall/std configs) | must be confirmed |
| 519 | "`--topdown 0 …` gives v0.12's decisions (and bytes)" | needs an identity run | not in schema; wording (d) |
| 528-529 | "(58 SMs; size-range set, default settings, median of 6, v0.9.0)" | label → v0.13.0 | wording |
| 529 | "4 × 4 runs at 252.4 MP/s against 230.0 for 2 × 2 … +10 %" | `SP('jpeg/size_range/default_enc3').mp_s`, `SP('jpeg/size_range/tiles211_enc3').mp_s`, ratio − 1 | repro (+9.7 %) |
| 530 | "8 × 8 (`2,3,3`) at 269.0 (+17 %)" | `SP('jpeg/size_range/tiles233_enc3').mp_s` / tiles211 − 1 | repro (+16.9 %) |
| 530-531 | "4 × 4 cost +0.58 % BD-rate against 2 × 2 on the photos of 2 MP and more (SSIMULACRA2; CVVDP +0.56 %)" | aws-g6-l4-tiles4x4 (v0.3.0 vs 4×4 files, 2026-10-31) | not in V13 (no 2 × 2 quality rows): keep with "measured at v0.4.0 (2026-10-31)" |
| 531-532 | "moved the default from −0.1 % to +0.1 % against libaom speed 6; images below 2 MP are byte-identical" | tiles4x4 v030_on_l4 vs bd_rate | wording (d): historical step, not today's default |
| 532-533 | RTX 5090 "4 × 4 was 39 % faster than 2 × 2 for 0.2 % more bytes" | pickae v0.2.0 | (a) |
| 556-557 | "169 MP/s of output … against 195" | aws-g6-l4-crop-resize (v0.6 port, 2026-11-02) | (a) |

## 2. docs/RESULTS.md

### Intro and RTX PRO 2000 (development)

| line | current | source / formula | notes |
|---|---|---|---|
| 17-19 | "the default now uses 4 × 4, which cost 0.2 points on the L4's size-range photos" | tiles4x4: +0.08 − (−0.12) | wording (d): v0.13 adds +0.3–0.5 on top |
| 44-79 | held-out BD-rates, 110.8 MP/s, 0.55 J/MP, preset table | dev machine, pre-repository code | (a) |
| 97-100 | production "236 MB … 63 MB, 73 % smaller" | Bellazon snapshot | (a) |

### NVIDIA L4: intro, setup, method

| line | current | source / formula | notes |
|---|---|---|---|
| 108-110 | "When the default changed to 4 × 4 tiles (2026-10-31), av1gpu's side of every compression result below was re-encoded … aws-g6-l4-tiles4x4/" | → V13 (aws-g6-l4-v0130) | wording (d) |
| 110-114 | "av1gpu speeds below are v0.9.0's … (main c553ffb …), re-run on 2026-11-03 … driver 595.112.07) with v0.8.0 alternating … aws-g6-g7-v090-check/" | `V13.environment`, `V13.cpu`, date, v0.12.0 as old | wording (d) |
| 118-120 | driver 595.91.07, CUDA 13.2 (first run) | `V13.environment` for the av1gpu rows | wording |
| 130-131 | "av1gpu: two such rounds, alternating with the previous version, and the median of the 6" | confirm V13 method | |
| 132-133 | "single pass over the standard set … 28.1 MP/s instead of 39.2 MP/s" | first run (aws-g6-l4, v0.1-era, 2026-10-25) | (a) |

### Speed

| line | current | source / formula | notes |
|---|---|---|---|
| 137 | "av1gpu uses about 0.2–0.4 CPU cores" | min/max `cpu_cores` of SP(jpeg/*) | repro (0.15–0.44) |
| 141 | default enc3: 81.2 / 252.4 / 0.2–0.3 | `SP('jpeg/standard/default_enc3')`, `SP('jpeg/size_range/default_enc3')` .mp_s; cpu_cores range of default_enc2–4 | repro |
| 142 | enc2: 77.1 / 225.6 | `…/default_enc2` | repro |
| 143 | enc4: 80.9 / **260.7** | `…/default_enc4` | repro |
| 144 | nosmall enc4: **191.0** / **298.9** / 0.3 | `…/nosmall_enc4` | repro |
| 145 | turbo3 enc4: 331.0 / 0.4 | `SP('jpeg/size_range/turbo3_enc4')` | mp_s repro; cores NO: 0.3495 → 0.3, printed 0.4 |
| 146 | turbo3ns: **419.5** / 0.4 | `SP('jpeg/size_range/turbo3ns_enc4')` | repro |
| 147-148 | libaom 21.1 / 28.8 / 166.1 / 187.6 | fixed | |
| 151 | "96–100 % SM utilisation at 68–71 W" | W: min/max `watts` of SP(jpeg/size_range/*); SM %: not in schema | W repro (67.8–71.0); SM % from v0.9.0 dmon logs: source not in V13 |
| 151 | "about 0.27 J/MP at 261 MP/s (v0.8.0: 0.29)" | `SP('…/default_enc4').j_per_mp`, D; `SPo(…).j_per_mp` | repro (0.271) |
| 152 | "72 W cap with either version, at 1,600–1,820 MHz" | not in schema | source: v0.9.0 dmon logs; drop or re-measure |
| 152-153 | "Four … 3 % faster than three on the size range and the same on the standard set; two were 11 % and 5 % slower" | SP enc4/enc3 − 1 (both sets); enc2/enc3 − 1 | repro (+3.3, −0.5, −10.6, −5.2) |
| 155-157 | "v0.8.0 … 231.5 / 213.8 / 236.4 MP/s with 3 / 2 / 4 encoders, 276.9 with `--small-preset none`, 294.6 with `turbo3` and 376.9 … standard set at 76.2 and 179.0" | `SPo(…/default_enc3|enc2|enc4|nosmall_enc4|turbo3_enc4|turbo3ns_enc4)`, `SPo(jpeg/standard/default_enc3|nosmall_enc4)` (76.2, 179.0 checked) | relabel v0.12.0 |
| 157-160 | "9.0 % … 10.3 % … 5.5 %"; "7.9 % … 12.4 % … 11.3 %"; "6.6 % … 6.7 %" | `CH(jpeg/size_range/default_enc3|enc4|enc2)`, `CH(…/nosmall_enc4|turbo3_enc4|turbo3ns_enc4)`, `CH(jpeg/standard/default_enc3|nosmall_enc4)` | relabel v0.13.0 vs v0.12.0 |
| 162-163 | "v0.8.0 against a second copy of itself … 0.2–0.7 %" | noise binary: not in schema | add if V13 has one, else drop |
| 163 | "Every output is byte-identical to v0.8.0's" | `V13.new_vs_old` | wording (d): false for default/base/turbo3 in v0.13 |
| 163-166 | "The gains come from #56–#58 …" | | wording: v0.13's PRs |
| 166-167 | "the size range runs at 217 MP/s against 201" | `SP('…/default_enc3').mp_s_wall` / `SPo(…).mp_s_wall` | repro (216.7 / 200.6) |
| 170-172 | thumbnails "280 … 67.5 MP/s (445 images/s) … 161.0 (1,062 images/s) … (v0.8.0: 62.8 and 148.3)" | `SP('thumbs/default')`, `SP('thumbs/nosmall')` mp_s / img_s; SPo; 280 = `V13.sets` | repro |
| 173-174 | Kodak "59.6, 143.4 and 253.8 MP/s … 152, 365 and 645 images per second" | `SP('kodak/default'|'kodak/nosmall'|'kodak/turbo3_nosmall')` mp_s, img_s | repro |
| 177-178 | "`--small-preset none` is 2.4× faster there, for files 3 % larger" | `SP('jpeg/standard/nosmall_enc4').mp_s` / `SP('jpeg/standard/default_enc4').mp_s`; `V13.bd_rate_std` nosmall − default | repro (2.36; 3.43 − 0.34 = 3.1). Will shrink a lot (base +43 %). If bd_rate_std skipped: (a) |
| 180-181 | "about 420 MP/s, 1.6× the default and 2.2× libaom speed 9 (188)" | T3; T3 / D; T3 / s9 | repro (1.61, 2.24) |

### Single-image start-up

| line | current | source / formula | notes |
|---|---|---|---|
| 186-190 | "v0.9.0 cut it (#56) … median of 12 runs … aws-g6-g7-v090-check" | | keep as history; add a V13 table |
| 192-198 | v0.9.0 / v0.8.0 table, L4 and RTX PRO 4500 | L4 cols = v090 single B13/A13 | keep (history, labelled); new table: `V13.single[photo.jpg|big.jpg|thumb512.jpg|t320.jpg|pair].new|old.median` (L4 only; RTX PRO 4500 columns: (a)) |
| 200-211 | "2.6–3.0× faster", "708 ms instead of 2,119", "543 ms", "629", "359 and 506 ms", "506 against 359", "200 ms … 48 ms", "144 ms", "38 MB … 7–8 MB" | v0.9.0 history | (a). If an upload figure is wanted for v0.13: single['photo.jpg'].new + single['thumb512.jpg'].new |
| 213-229 | v0.10.0 table and bullets | aws-g6-g7-recon-lazy (v0.10.0) | (a) |

### Compression (size range = V13.bd_rate; standard = V13.bd_rate_std, may be skipped)

| line | current | source / formula | notes |
|---|---|---|---|
| 239 | default: standard "**+0.3 %** (−0.1 %)" | `V13.bd_rate_std` default (median, mean) | if skipped: (a) "v0.12.0 files" (aws-g6-l4, bytes unchanged v0.1–v0.12 below 2 MP). Standard photos are ~1 MP → base: this WILL change |
| 239 | default: size range "**+0.1 %** (−0.1 %)" | BD('av1gpu_vs_aom_s6') median, mean | repro (+0.08, −0.13) |
| 240 | nosmall: "+3.4 % (+2.9 %)" / "+2.2 % (+1.4 %)" | `V13.bd_rate_std` nosmall; BD('av1gpu_nosmall_vs_aom_s6') | repro |
| 241 | turbo3 nosmall "+5.7 % (+5.6 %)¹" | BD('av1gpu_turbo3_nosmall_vs_aom_s6') | repro; std column empty (or `V13.bd_rate_std` turbo3 nosmall if run) |
| 247 | "¹ `turbo3` with the small-image preset left on: +1.2 % (+2.0 %)" | BD('av1gpu_turbo3_vs_aom_s6') | repro (+1.25, +1.96); changes (base) |
| 261-262 | "the default is 29.7 % (standard) and 30.6 % (size range) smaller" | `V13.bd_rate_std` av1gpu vs WebP (needs webp-icc-standard rows); BD('av1gpu_vs_webp') | size range repro; standard needs a vs-WebP key in bd_rate_std (not named in the schema) |
| 262 | "(30.0 % and 31.2 % before the WebP ICC correction)" | aws-g6-l4 / aws-g6-l4-gamut (v0.3.0) | history: drop or keep labelled |
| 266 | default by size "+0.1 % (40) +1.9 % (20) +0.2 % (23) −2.1 % (17)" | BDS('av1gpu_vs_aom_s6', cls) and `.n` | repro |
| 267 | nosmall by size "+3.9 % (40) …" | BDS('av1gpu_nosmall_vs_aom_s6', cls) | repro |
| 274 | "(At 1 MP and above both av1gpu rows are the same `turbo` encodes.)" | | still true |
| 275 | "av1gpu compares best on large images" | trend of BDS | wording check |
| 277-282 | Tiling: "−0.1 % (−0.4 %) … +2.1 % (+1.1 %) … +5.6 % (+5.4 %) … +1.1 % … +1.1, −0.6 and −2.4 % … grew 0.26 % … about +12 % on this set" | tiles4x4 `v030_on_l4` (v0.3.0) | wording (d). "+12 %" NO: README tiles paragraph gives +10 % (252.4 / 230.0 = +9.7 %); source of 12 % unknown |

### SVT-AV1 on the same CPU

| line | current | source / formula | notes |
|---|---|---|---|
| 312-313 | "default (261 MP/s, +0.1 %) is 8.6× faster than SVT-AV1 interpolated to the same size (30 MP/s)" | D; BD default; D / `V13.derived.svt_interp_at.default` | repro (30.3, 8.60) |
| 314 | "at +2.2 %, … `--small-preset none` (299 MP/s) is 6.1× faster than preset 6 (49 MP/s, +2.1 %)" | NS / sv[6] | NO: 298.85 / 49.40 = 6.05 (6.1 only from rounded 299/49). With nosmall at +2.5–2.7 % use NS / `svt_interp_at.nosmall` (now 5.9) and reword |
| 315 | "`turbo3` (420 MP/s, +5.7 %) is 2.3× as fast as preset 9 (183 MP/s, +9.8 %)" | T3 / sv[9] | repro (2.29) |
| 316-317 | "SVT-AV1's slow presets make smaller files than any av1gpu setting measured on the L4 … `--preset slow` … −4.8 % on the development set" | | wording; dev number (a) |

### CPU builds, tiling (untiled anchor)

| line | current | source / formula | notes |
|---|---|---|---|
| 374 | av1gpu default "+0.1 % / +0.7 %" | BD('av1gpu_vs_aom_s6'); BD('av1gpu_vs_aom_s6_notiles') | repro |
| 375 | nosmall "+2.2 % / +2.5 %" | BD('av1gpu_nosmall_vs_aom_s6'); BD('av1gpu_nosmall_vs_aom_s6_notiles') | repro (+2.23, +2.54) |
| 383 | "(+0.7 % against untiled libaom speed 6)" | BD notiles | repro |
| 384 | "261 MP/s (v0.9.0) is 9.1× stock libaom speed 6, 8.8× the tuned build and 8.4× the tuned build at 8 × `-j 2` (31.1 MP/s, tiled)" | D / tun['stock_aom_s6_j1'] (9.08), / tun['tuned_aom_s6_j1'] (8.78), / tun['tuned_aom_s6_j2'] (8.39) | repro; relabel v0.13.0 |
| 385-387 | "33.3 MP/s … (7.8×) and 37.0 MP/s with PGO (7.0×) … 34.5 MP/s (7.6×). The fastest CPU configuration … about 7.0× slower" | `V13.derived.svt46_interp.stock_untiled`, `.pgo_untiled`, `.pgo_tiled_8x2`; D / each; min | repro (33.25, 37.04, 34.45; 7.84, 7.04, 7.57) |

### NVENC AV1 on the same L4

| line | current | source / formula | notes |
|---|---|---|---|
| 414-415 | "av1gpu's default (v0.3.0 then) … 115.6 and 117.1 MP/s on 0.12 cores, 7.4× faster" | same-session | (a): keep |
| 415-416 | "v0.9.0 runs at 261 MP/s on 0.3 cores, **16.6× NVENC's best**" | D; default_enc4 cpu_cores; D / nv | repro (16.6); relabel |
| 422-423 | "av1gpu's default files are 22.7 % smaller than NVENC's (96 photos)" | BD('av1gpu_vs_nvenc') and `.n` | repro |
| 425 | "av1gpu's files are 20.6 % smaller than NVENC's" (even height) | BD('av1gpu_vs_nvenc_even_height') | repro; JSON n = 75, the text's "76" is NVENC's count |
| 438-439 | "av1gpu's default quality gives 22.5 MB at 72.0 on the same photos" | `V13.derived.q60_99.bytes`, `.ssimulacra2` | repro (99 photos); libaom fixed |
| 457 | "against av1gpu's 104 MP/s there" (RTX PRO 2000) | dev | (a) |

### FPGA: Bluedot Pulsar-AVIF

| line | current | source / formula | notes |
|---|---|---|---|
| 470-471 | "av1gpu v0.9.0 … repeated to 1,000 MP (2,544 images) … (aws-g6-g7-v090-check/; v0.8.0 in the same session: 141, 344 and 585 images/s)" | `V13.sets` (kodak repeats); `SPo('kodak/*').img_s` | relabel v0.13.0 / v0.12.0 |
| 475 | default: 152 / 144 / 3.4 / 912,368 | `SP('kodak/default')` img_s / img_s_wall / cpu_cores / bytes × 24 / images per run (bytes / 106 for 2,544) | repro |
| 476 | nosmall: 365 / 322 / 7.0 / 917,825 | `SP('kodak/nosmall')` | repro |
| 477 | turbo3 nosmall: **645** / **522** / 11.9 / 946,869 | `SP('kodak/turbo3_nosmall')` | repro; bytes should be unchanged |
| 486 | "$0.80/hour on spot" | fixed | no per-image cost printed |

### 4:4:4 and 10-bit

| line | current | source / formula | notes |
|---|---|---|---|
| 495-496 | "The BD-rates below are the 4 × 4 default's (aws-g6-l4-tiles4x4/; with v0.3.0's 2 × 2 tiles: −0.1, +0.3, +0.1 and +0.0 %)" | | wording (d) |
| 496-497 | "av1gpu's speeds are v0.9.0's, `GP_ENCODERS=4` (…; v0.8.0 in the same session: 236.4, 174.7, 231.4 and 172.3 MP/s)" | `SPo('jpeg/size_range/default_enc4')`, `SPo('formats/*')` | relabel v0.12.0 |
| 501-504 | av1gpu MP/s 260.7 / 197.9 / 253.3 / 193.9 | D; `SP('formats/444_8'|'420_10'|'444_10').mp_s` | repro |
| 501-504 | BD-rate col +0.1 / +0.4 / +0.3 / +0.4 % | BD('av1gpu_{f}_vs_aom_s6_{f}') | repro (+0.08, +0.40, +0.27, +0.35) |
| 506-507 | "about 24 % slower; 10-bit costs it 2–3 %. libaom loses 23 % at 10 bits, so av1gpu's lead grows from 9.1× to 11.5×" | SP(444_8)/D − 1, SP(444_10)/SP(420_10) − 1; SP(420_10)/D − 1, SP(444_10)/SP(444_8) − 1; D / aom_fmt['420_8'], SP(420_10) / aom_fmt['420_10'] | repro (−24.1, −23.4; −2.8, −2.0; 9.1, 11.5) |
| 508-509 | "−2.2 % for av1gpu … in 4:2:0, −2.0 % … in 4:4:4" | BD('av1gpu_420_10_vs_av1gpu_420_8'), BD('av1gpu_444_10_vs_av1gpu_444_8') | repro |
| 510 | "+0.4 % for av1gpu" (4:4:4 vs 4:2:0) | BD('av1gpu_444_8_vs_av1gpu_420_8') | repro |

### Lossless (bytes unchanged in v0.13)

| line | current | source / formula | notes |
|---|---|---|---|
| 527 | "(44.8 MP/s on the size-range photos)" (JPEG lossless, no longer accepted) | history | (a) |
| 538-540 | "800/800 files byte-identical … speed is within noise (115.7 vs 114.9 MP/s, 18 runs each)" | lossless merge, 2026-10-31 | (a) |
| 541-543 | "lossless test files … byte-identical to v0.3.0's" | | wording: add "and v0.13.0's" if `V13.new_vs_old` confirms |
| 547-549 | "(v0.6; v0.5.0 in the same session: 17.4 / 39.3 and 17.8 / 39.9). Lossless is 9–11 % faster" | v0.6 history | (a) |
| 550-553 | "v0.9.0 … 51 PNGs and 49.1 MP … at 21.8 / 45.3 MP/s, and 22.1 / 44.3 with `--ycgco-re` … v0.8.0 … 20.3 / 43.7 and 20.6 / 42.7" | `SP('lossless/cmp_1pass').mp_s_wall` / `SP('lossless/cmp_4x').mp_s`; `SP('lossless/cmp_1pass_ycgco').mp_s_wall` / `SP('lossless/cmp_4x_ycgco').mp_s`; SPo same | repro; relabel |
| 553-554 | "Without palette mode … 49.8 and 49.1 MP/s (v0.8.0: 46.8 and 47.2)" | v090-check NOTES/summary (`--palette off`) | not in schema → (a) "measured with v0.9.0" |
| 559-560 | MP/s "18.9 / 43.6", "19.4 / 44.5" | v0.6, 47-PNG set, `--palette off` | (a) keep labelled v0.6 (the V13 set differs); bytes unchanged |
| 569 | "about 126 ms for a Kodak image" | | source unknown |

### Palette mode

| line | current | notes |
|---|---|---|
| 585-594 | BD-rates −23.6 / −19.5 / −9.5 / −22.3 / −12.7 / +0.7 / −6.8 / −5.9 %, lossless −10.1 / −9.9 / −8.7 / −3.7 % | (a) aws-g6-l4-palette-study (pre-v0.8.0 main); v0.13 may change the lossy rows |
| 601-613 | speed table 229.3 / 226.3 … 152.7 / 57.0 … 0.37× / 0.25×; "0.09 ms of a 64 ms photo encode" | (a) same study (v0.6-method speed check) |

### Transparency (refresh only if V13 has `alpha/*`)

| line | current | source / formula | notes |
|---|---|---|---|
| 651-652 | "Speed on the size-range set is unchanged within noise (114.9 vs 114.4 MP/s pooled over 36 runs each)" | alpha merge, 2026-10-31 | (a) |
| 656 | default row 698,892 / 592,901 / 90,523 / 50.2 dB / 50.4 | sizes: tiles4x4 `alpha['new/alpha_default']`; speed: v0.9.0 `--palette off` | (a) unless the run adds a `--palette off` alpha pass with per-item bytes. Colour bytes WILL change (base) |
| 657 | q100 row 1,088,428 / 592,901 / 480,059 / 43.9 | tiles4x4 `alpha_q100` | colour part changes |
| 658 | lossless row 7,723,961 / 7,222,941 / 485,582 / 30.4 | tiles4x4 `alpha_lossless` | bytes unchanged |
| 666-668 | "re-measured with the 4 × 4 default; only `ui_hud_1080p.png` … grew by 235 and 211 bytes" | tiles4x4 new vs old alpha | wording (d) |
| 669 | "av1gpu's are about 9 % smaller than libaom's" (exact alpha) | 480,059 vs 527,573 | unchanged |
| 670-671 | "v0.9.0's speeds; … without alpha (`--no-alpha`) at 57.3 MP/s (v0.8.0 …: 44.4, 39.3, 28.1 and 49.8)" | v0.9.0 `--palette off` | (a) |
| 672-673 | "685,414 bytes for the default row, and v0.9.0 encodes the set at 31.5, 30.1, 29.3 and 36.8 MP/s" | `SP('alpha/default').bytes` / 20; `SP('alpha/default'|'q100'|'lossless'|'no_alpha').mp_s_wall` | repro; relabel |

### CPU threads, crop/resize

| line | current | source / formula | notes |
|---|---|---|---|
| 683 | "a JPEG batch keeps only 0.1–0.4 CPU cores busy" | min/max cpu_cores of SP(jpeg/*) | repro (0.15–0.44) |
| 704-742 | MP/s table and bullets (126.0 → 127.4 …, +13 % … +45 %, 123.4 …) | aws-g6-l4-threads (2026-11-01, v0.5/v0.6-era main, g6.8xlarge) | (a) |
| 744-745 | "128.0 MP/s … with v0.4, and run at 252.4 MP/s with v0.9.0" | `SP('jpeg/size_range/default_enc3').mp_s` | repro; relabel; 128.0 (v0.4) stays |
| 756-805 | crop/resize: 1,748 files, 124 MP/s, 235.8 / 233.4 / 230.9 / 231.4 / 232.2, 169.0 / 195.2, pickae 66–73 / 72–79 | aws-g6-l4-crop-resize (v0.6 port, 2026-11-02) | (a) |

### CVVDP

| line | current | source / formula | notes |
|---|---|---|---|
| 812-815 | "For the 4 × 4 default, av1gpu's files were encoded and scored again … its unchanged files below 2 MP got the same scores to the last digit" | | wording (d): in v0.13 every size changes |
| 820 | "**+4.2 %**", "+0.1 %", "0.462", "99/99" | `V13.bd_rate_cvvdp.av1gpu_vs_aom_s6.median`; BD('av1gpu_vs_aom_s6'); `V13.cvvdp_target.median_bpp`; `.reached`/`.images` | repro |
| 831-832 | "worse than SSIMULACRA2 does on 84 of the 99 photos" | `V13.av1gpu_per_image_agreement.jod_worse_than_ssimulacra2` / `.images` | repro |
| 832 | "(+2.1 % below 1 MP, +6.1 % at 10 MP and over)" | `V13.bd_rate_cvvdp_by_size.av1gpu_vs_aom_s6['< 1 MP'|'>= 10 MP'].median` | repro (+2.06, +6.12) |
| 833 | "Scored without colour, CVVDP puts it at +6.2 %" | `V13.bd_rate_cvvdp.av1gpu_vs_aom_s6_luma_only` only if present | else keep with note "v0.4.0 files (tiles4x4)" |
| 835-837 | "about 73 MP/s … about 49 MP/s, so av1gpu (v0.9.0, 261 MP/s) is about 3.6× and 5.3× faster. On SSIMULACRA2 the same comparison gives about 8.6×" | `V13.derived.cvvdp_svt_speed`, `.cvvdp_aom_speed`; D / each; D / `svt_interp_at.default` | repro; relabel |
| 838-839 | "With v0.3.0's 2 × 2 tiles av1gpu was +3.9 % on CVVDP (0.459 …; +5.9 % at 10 MP and over), so the 4 × 4 default costs about a quarter of a point" | tiles4x4 v030_on_l4 | wording (d); could add v0.12.0 → v0.13.0 from `V13.v0120_on_l4.bd_rate_cvvdp` |

### Correctness, compared with RTX PRO 2000

| line | current | source / formula | notes |
|---|---|---|---|
| 845-870 | gate and identity rows of v0.6 / v0.9.0 / v0.10.0 | history | add a v0.13.0 row from `V13.decoder_gate` and `V13.new_vs_old` (identical files for slow/fast/faster/turbo2/turbo3ns, lossless) |
| 905-907 | "104–116 MP/s with the default settings of v0.3.0 … v0.9.0 runs at 226–261 MP/s on the L4" | v0.3.0 (a); min/max `SP('jpeg/size_range/default_enc2|enc3|enc4').mp_s` | repro (225.6–260.7); relabel |
| 908-909 | "+0.1 % against libaom speed 6 here (−0.1 % with v0.3.0's 2 × 2 tiles…), −0.6 % on the held-out set" | BD('av1gpu_vs_aom_s6'); rest history/dev | wording |
| 911-912 | "av1gpu (v0.9.0) is 9.1× libaom speed 6 at the same file size, and 1.4× libaom speed 9, whose files are 15 % larger" | D / s6; D / s9 | repro (9.07, 1.39); relabel |
| 913-914 | "1 h 39 min … roughly $1.40" | first run (2026-10-25) | (a) |

### Windows, release builds, RTX 5090, image ops (all (a))

| line | current | notes |
|---|---|---|
| 918-968 | Windows table 127.5 / 76.8 / 85.6 …, 15.3 s / 0.4 s, 6.6 / 6.3 s, 1.1–1.3 s, 133.9 / 131.5 MP/s | (a) v0.5.0 (2026-11-01) and v0.7.1 zips (2026-11-02) |
| 976-1020 | 9.3–9.7 s, 1.2–1.3 s, 298 / 349 MP/s, table 351.0 / 406.9 …, 434.4 / 406.4 (+6.9 %), 831 / 776 images/s, 0.51 / 1.03 s, 4,394 outputs, build times | (a) v0.6.0–v0.7.1 (2026-11-02), v0.9.0 RTX PRO 4500 (2026-11-03) |
| 1047-1098 | RTX 5090 v0.2.0 tables, incl. "L4 … same version" columns (39.2, 103.8, 116.0, 100.9, 126.8, 167.1, 209.8–213.9, 113.4, 94.5, 109.8, 92.8), 0.83 / 0.44 / 0.5 J/MP, +39 % / 0.21 % | (a) pickae v0.2.0; the L4 columns are matched-version: do NOT refresh |
| 1114-1150 | RTX 5090 v0.9.0: 214.3; "L4, v0.9.0" column 81.2 / 21.1 / 166.1; 2.6×, 1.77×, 6.5× / 0.81×, "(the L4 …: 3.9× and 0.49×)"; BD table "L4 +0.3 % (−0.1 %)", "−29.7 % (−30.5 %)"; 71.43 MB; 700 MP/s, 249 s, 60× / 25× | (a) pickae v0.9.0 (2026-11-03). Keep the L4 column at v0.9.0 (= v0.12.0 bytes): the av1gpu-vs-WebP row "matches to the digit" only with the same files |
| 1152-1215 | trim/resize/tone (RTX 5090): 816 outputs, 7.42 / 7.27 / 3.0 s, +4 % / +2 %, 54.4 / 53.9 / 48.3 / 46.0 s, 0.05 % | (a) pickae, main 4721191 (2026-11-06) |
| 1217-1340 | L4 image-ops review: 22,231 outputs, within 0.4 %, −3.7 to +2.4 ms, 0 to 1.4 %, +3 to +6 %, 20 %, 16 to 44 % | (a) v0.11.0 (b08c856, 2026-11-07, aws-g6-l4-v0110-imageops) |

## (a) Not re-measured in this run: add "measured with vX.Y" instead of refreshing

| doc:line | number(s) | measured with / source |
|---|---|---|
| README 39-41, 46 | 7 GPU-days, 40 % smaller than WebP, 1.5 TB | source unknown (Bellazon production / estimate) |
| README 59 (demo) | 3.4 s, 192.8 MP/s, 0.32 cores, 22 MB | v0.6, 2026-11-01, aws-g6-l4-speed-v06 `demo` (logs/rec.json; scripts in aws-g6-l4-speed-v04) |
| README 164 ²; RESULTS 414-415 | 116 / 115.6 / 117.1 MP/s, 0.12 cores, 7.4× | v0.3.0 with NVENC, 2026-10-29, aws-g6-l4-nvenc `av1gpu_same_session` |
| README 200-201, 207-208; RESULTS 559-560 | lossless 45 / 44 / 19 MP/s, 18.9 / 43.6, 19.4 / 44.5 | v0.6 (2026-11-01), 47 PNGs, `--palette off` |
| RESULTS 553-554 | 49.8 / 49.1 MP/s (v0.8.0 46.8 / 47.2) | v0.9.0 `--palette off` (v090-check summary.md l.478-479) |
| README 215-218; RESULTS 585-613 | palette BD-rates and speeds | aws-g6-l4-palette-study (pre-v0.8.0 main) |
| README 231-235; RESULTS 656-671 | alpha sizes (`--palette off`), MP/s 50.4 / 43.9 / 30.4 / 57.3 | sizes: tiles4x4 `alpha` (v0.4.0 4×4 files, 2026-10-31); speeds: v0.9.0 `--palette off` (2026-11-03). Refresh only if V13 has alpha with per-item bytes; lossless/exact-alpha parts unchanged |
| README 322-324, 352-353, 480-481 | RTX PRO 4500 5,679 files, +14–16 %, 0.34 s, 0.34 / 0.42 / 0.34 / 0.67 s | v0.7.1 (aws-g7-g6-cuda13-builds, 2026-11-02), v0.9.0 (2026-11-03), v0.10.0 (aws-g6-g7-recon-lazy) |
| README 407-411 | Clang 70 vs 83, 103 vs 125, 14.4 vs 15.7, 1:47 vs 4:54 | v0.3.0 code (no results dir found: source unknown beyond the text) |
| README 430-434; RESULTS 918-968 | Windows | v0.5.0 (2026-11-01), v0.7.1 zips (2026-11-02), aws-g6-l4-windows |
| README 532-533; RESULTS 1022-1215 | RTX 5090 | pickae: v0.2.0 (cbea9b8), v0.9.0 (7c2bf33, 2026-11-03), main 4721191 (2026-11-06) |
| README 530-531 | +0.58 % / +0.56 % (4×4 vs 2×2) | tiles4x4 (v0.3.0 vs 4×4 branch, 2026-10-31) |
| README 556-557; RESULTS 748-805 | crop/resize 169 / 195 MP/s etc. | v0.6 port (2026-11-02), aws-g6-l4-crop-resize |
| RESULTS 1217-1340 | image-ops review | v0.11.0 (2026-11-07), aws-g6-l4-v0110-imageops |
| RESULTS 704-742 | CPU-threads table | 2026-11-01 (v0.5/v0.6-era main, g6.8xlarge), aws-g6-l4-threads |
| RESULTS 321-387 (CPU sides) | CPU-tuning study | aws-c6a-cpu-tuning (2026-10-31): CPU numbers fixed; only the av1gpu ratios refresh |
| RESULTS 42-100, 457, 905, 909 | RTX PRO 2000 dev numbers, 104 MP/s, production 73 % | development machine (pre-repo code), Bellazon snapshot |
| RESULTS 132-133, 913-914 | 28.1 / 39.2 MP/s; 1 h 39 min, $1.40 | first L4 run (2026-10-25), aws-g6-l4 |
| RESULTS 151-152 | SM utilisation 96–100 %, 1,600–1,820 MHz | v0.9.0 dmon logs (not in V13 schema) |
| RESULTS 162-163 | noise 0.2–0.7 % | v0.9.0 second-copy binary (only if V13 has a noise run) |
| RESULTS 192-229 | single-image v0.9.0 / v0.8.0 and v0.10.0 tables, RTX PRO 4500 columns | v090-check, recon-lazy (keep as history; add a V13 L4 table) |
| RESULTS 527, 538-540, 547-549, 651-652 | 44.8 MP/s; 115.7 / 114.9; 17.4 / 39.3 …; 114.9 / 114.4 | lossless/alpha merges (2026-10-31), v0.6 |
| RESULTS 833 | CVVDP luma only +6.2 % | tiles4x4 (v0.4.0 files) unless V13 has `av1gpu_vs_aom_s6_luma_only` |
| RESULTS 239-240, 261, 177-178 (standard set) | +0.3 % (−0.1 %), +3.4 % (+2.9 %), 29.7 %, "3 % larger" | if `V13.bd_rate_std` is skipped: "v0.12.0 files" (aws-g6-l4, aws-g6-l4-turbo-nosmall, webp-icc-standard). These rows change most (base) |

## (b) Charts (make_charts.py) and their alt/caption numbers

make_charts.py must first be pointed at V13: `l4_speed()` loads `aws-g6-g7-v090-check` (l.117-120, docstring "v0.9.0") → `aws-g6-l4-v0130`; `tiles4x4()` (l.43-46, docstring "Below 2 MP they are the same files as before", now false) → `aws-g6-l4-v0130`, which needs `bd_rate`, `bd_rate_cvvdp` and `quality['av1gpu']` (rows image/q/bytes/ssimulacra2) in the tiles4x4 format.

| chart (README line) | V13 keys feeding it | numbers in SVG text | alt/caption numbers (README line) |
|---|---|---|---|
| banner (l.3-4) | D (`jpeg/size_range/default_enc4`), s6 | "libaom-speed-6 file sizes, 9.1× faster than libaom on all 16 CPU threads at once" (l.526) | alt none; "libaom-speed-6 file sizes" is wording (d) |
| tradeoff (l.73-74) | D, NS, T3 (`l4_speed` default_enc4 / nosmall_enc4 / turbo3ns_enc4); BD av1gpu_vs_aom_s6 / av1gpu_nosmall_vs_aom_s6 / av1gpu_turbo3_nosmall_vs_aom_s6 | "9.1×" arrow (D/s6), point positions | l.74: 9.1×, 261 / +0.1, 299 / +2.2, 420 / +5.7 |
| match-small (l.91-92) | D, default_enc4 cpu_cores; (BD default passed, not drawn: sizes=False) | subtitle "about 0.3 CPU cores", "9.1× the speed", "SVT-AV1 reaches those sizes between its presets 4 and 6" | l.92: 261 MP/s |
| match-fast (l.102-103) | T3; BD av1gpu_turbo3_nosmall_vs_aom_s6 | subtitle "2.2–2.3× as fast", "files grow 6 %"; bar labels 420, +5.7 % | l.103: 420 MP/s, +5.7 % |
| match-nvenc (l.114-115) | D; BD av1gpu_vs_aom_s6 | subtitle "17× faster"; bars 261, +0.1 % | l.115: 261, +0.1 % |
| quality (l.139-140) | `V13.quality.av1gpu` (6 q points × 100 images: bytes, ssimulacra2) | subtitle "av1gpu lies on top of libaom speed 6" (wording) | l.140: no numbers, "curves coincide" (wording) |
| by-size (l.144-145) | recomputed from `V13.quality.av1gpu` per image vs gamut aom_s6 (should equal `V13.bd_rate_by_size.av1gpu_vs_aom_s6`) | bar labels; hard-coded subtitle "av1gpu matches speed 6 and does better as images get larger; NVENC is 22–36 % bigger" (l.431) | l.145: +0.1, +1.9, +0.2, −2.1 |
| formats (l.181-182) | D, `formats/444_8|420_10|444_10` mp_s; BD av1gpu_{f}_vs_aom_s6_{f} | "files ±x % vs libaom" per row; "within 0.4 %" | l.182: 261/198/253/194, "within 0.4 percent" |
| metrics (l.247-248) | BD av1gpu_vs_aom_s6, `V13.bd_rate_cvvdp.av1gpu_vs_aom_s6` (replace the t4 entries) | "+0.1 / +4.2 %" | l.248: +0.1 and +4.2 |
| pipeline (l.660-661) | none in code | HARD-CODED "CPU · about 0.3 cores" (make_charts.py l.481): check against default_enc4 cpu_cores | alt none |

## (c) docs/img assets

| file | shows | numbers / caption | action |
|---|---|---|---|
| demo.webp | terminal recording, 99 photos | README l.59 alt: 3.4 s, 192.8 MP/s, 0.32 cores, 154 → 22 MB; source aws-g6-l4-speed-v06 results.json `demo` (v0.6, 2026-11-01; made with aws-g6-l4-speed-v04/scripts rec.py + termgif3.py) | leave with "(v0.6)" in alt/caption, or re-record with v0.13.0 |
| hero-car.jpg | right half = av1gpu AVIF of b1aa6e8182e3f22f_huge.jpg (421 KB, v0.4.0 4×4 file) | README l.19-20: 421 KB | caption → `Q60('b1aa6e8182e3f22f_huge.jpg').bytes`; image can stay (visually the same at this scale) if the caption names the version; generator script not in the repo |
| compare-car.png | 2× crops, av1gpu + 4 encoders pushed to av1gpu's size | baked into the image and alt l.296 (421.0 KB / 72; 412.1 / 71; 395.9 / 66; 417.6 / 62; 420.8 / 36) | av1gpu's size changes, so the "same size" pairing is off: regenerate all, or leave with a note "(av1gpu v0.4.0 file)". Script/results: source unknown |
| compare-dog.png | same, df28dcbd0a84d456_huge.jpg | alt l.302 (578.3 / 69; 577.8 / 68; 565.0 / 66; 576.7 / 60; 558.7 / 49), summary l.300 "578 KB each" | same as compare-car |
| thumb-tiny / small / typical / large / huge.jpg | crops of av1gpu output | captions l.268-272 sizes, −% and l.276 SSIMULACRA2 | update captions from `Q60(...)`; images can stay (crops at thumbnail scale) |

## (d) Wording that v0.13.0's new defaults make wrong or stale

| doc:line | text | why |
|---|---|---|
| README 25-26, banner SVG | "with the same file sizes" / "libaom-speed-6 file sizes" | default moves +0.3–0.5 %; still fine if BD default stays below about +1 %, check |
| README 188 | "no measurable speed cost for av1gpu" (10-bit) | already −2.8 % (RESULTS says 2–3 %) |
| README 286-287 | "With v0.3.0's 2 × 2 tiles the RTX PRO 2000 and the L4 produced byte-identical files for all five; these are the L4's files with the 4 × 4 default" | the files are now v0.13.0's |
| README 308 | "re-encoded with av1gpu (4 × 4 default, on the L4)" | name v0.13.0, or say the crops are v0.4.0 files |
| README 516-519 | v0.13 paragraph: "about 10 % … 40 % … 0.3–0.5 %", "gives v0.12's decisions (and bytes)" | confirm against V13; "(and bytes)" needs an identity run |
| README 528-533 | tiles paragraph labelled v0.9.0; "moved the default from −0.1 % to +0.1 %"; "images below 2 MP are byte-identical" | relabel speeds; BD step is the v0.3.0→4×4 history |
| RESULTS 17-19 | "the default now uses 4 × 4, which cost 0.2 points" | the default also changed in v0.13 |
| RESULTS 44 | dev table "av1gpu default (`turbo`, `base` below 1 MP)" | pre-v0.13 presets; label "(development build)" |
| RESULTS 108-114 | "re-encoded … 4 × 4 tiles (2026-10-31) … aws-g6-l4-tiles4x4", "speeds below are v0.9.0's … v0.8.0 alternating" | now v0.13.0 / v0.12.0, aws-g6-l4-v0130 |
| RESULTS 155-167 | "Against v0.8.0, v0.9.0 is …", "Every output is byte-identical to v0.8.0's", "gains come from #56–#58" | v0.13 is not byte-identical for default / base / turbo3-with-small-preset |
| RESULTS 277-282 | Tiling paragraph: "These av1gpu rows use the 4 × 4-tile default", "Files below 2 MP are byte-identical either way (the standard set entirely)", "+12 %" | v0.13 files differ from v0.3.0's below 2 MP too; +12 % doesn't reproduce (+9.7 %) |
| RESULTS 314 | "at +2.2 % … nosmall … 6.1× faster than preset 6 (49 MP/s, +2.1 %)" | the pairing relies on nosmall ≈ preset 6's size |
| RESULTS 495-497 | "BD-rates below are the 4 × 4 default's (aws-g6-l4-tiles4x4/; with v0.3.0's 2 × 2 tiles …)", "speeds are v0.9.0's" | v0.13 files / v0.12.0 as old |
| RESULTS 541-543 | lossless "byte-identical to v0.3.0's" | add v0.13.0 if confirmed |
| RESULTS 666-668 | alpha "re-measured with the 4 × 4 default …" | colour bytes change in v0.13 |
| RESULTS 812-815 | CVVDP "its unchanged files below 2 MP got the same scores to the last digit" | no unchanged files in v0.13 default |
| RESULTS 838-839 | "With v0.3.0's 2 × 2 tiles … +3.9 % … 4 × 4 default costs about a quarter of a point" | history; add v0.12.0 → v0.13.0 |
| RESULTS 858-860 | "4 × 4 default: files below 2 MP … byte-identical to v0.3.0" | history row, fine; add a v0.13.0 row |
| RESULTS 905-912 | "v0.9.0 runs at 226–261", "(v0.9.0) is 9.1× …" | relabel |
| RESULTS 1103-1105, 1123, 1134 | RTX 5090 bytes "the same totals as in aws-g6-l4/…", "71.43 MB … same total as on the L4", "matches to the digit" | true only against v0.9.0 = v0.12.0 L4 files; keep the L4 column at v0.12.0 |
| make_charts.py 44-45, 117 | docstrings "Below 2 MP they are the same files as before", "with v0.9.0" | update with the code |

## Source unknown / does not reproduce

- README 39-40, 46 "about 7 days of GPU time" / "7 GPU-days": no stored source; 50e12 px / 7 d = 83 MP/s matches neither the L4 nor the RTX PRO 2000 figures.
- README 41-42 "40 % smaller than the WebP it serves today, 1.5 TB less": production data, no results dir.
- README 296 / 302 compare-car and compare-dog: the libaom 6 / 9, WebP and JPEG sizes and scores (412.1 / 71, 395.9 / 66 …): no script or results dir found. av1gpu "SSIMULACRA2 72" for the car does NOT reproduce (stored q60 row 71.4).
- README 407-411 Clang timings: no results dir found.
- RESULTS 282 "about +12 % on this set" (tiles): does NOT reproduce (+9.7 % from v0.9.0 data; README says +10 %).
- RESULTS 314 "6.1×": does NOT reproduce exactly (298.85 / 49.40 = 6.05).
- RESULTS 145 turbo3 CPU cores "0.4": does NOT reproduce (0.3495 → 0.3).
- RESULTS 151-152 SM utilisation 96–100 % and 1,600–1,820 MHz: only in v0.9.0 logs, no V13 key.
- RESULTS 569 "about 126 ms for a Kodak image" (lossless, one image at a time): version and source unknown.
- RESULTS 261 standard-set "29.7 %" vs WebP: no named V13 key (bd_rate_std has av1gpu vs libaom only in the brief); needs webp-icc-standard rows.
- README 517-518 "about 10 % faster on large photos": no exact V13 key for "large photos" (closest `speed_change['jpeg/size_range/nosmall_enc4']`).
