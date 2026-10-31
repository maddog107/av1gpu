# CPU baselines: thread splits and tuned builds (AWS c6a.4xlarge)

- 2026-10-31, AWS us-east-1, c6a.4xlarge spot (AMD EPYC 7R13, 8 cores / 16 vCPUs, the CPU of the published runs), Ubuntu 24.04.5, kernel 7.0.0-1014-aws, GCC 13.3.0, CMake 3.28.3; avifenc 1.4.0, libaom 3.13.1, SVT-AV1 v4.2.0.
- Images: the 99 size-range photos of the published c6a run (477.7 MP). Each timed run encodes them 3 times (297 files, 1,433 MP) at q60, `-y 420`, libaom `tune=iq`, SVT-AV1 `tune=3`, with `bench.py`'s own speed code (`bench.cpu_speed`, one timed pass).
- Builds (exact flags in results.json `builds`): **stock** = the published build (CMake Release, `-O3`; SVT-AV1's `SVT_AV1_LTO` defaults to ON with GCC, so the published SVT-AV1 already had LTO). **tuned** = `-march=native` + LTO for all three: libaom via `AOM_EXTRA_C(XX)_FLAGS=-march=native` + CMake IPO (it has no option of its own), SVT-AV1 via its own `NATIVE=ON` and `SVT_AV1_LTO=ON`, libavif via `CMAKE_C(XX)_FLAGS` + IPO. **tuned + PGO** = tuned plus SVT-AV1's own `SVT_AV1_PGO`, trained with avifenc on the separate 100-image ~1 MP set (`bench/images`), never on the timing set.
- Two instances. The **primary** host matched the published speeds (control below) and gives every number here unless marked. The first instance launched (**replica**) was 7-8 % slower at every encoder with the same bytes (no CPU steal or background load), so the same job ran on both; the replica also did the quality scoring, which is host-independent (it reproduced the published q60 bytes and scores exactly).

## Findings

1. **Tuned builds (question a).** The published builds were CMake Release (`-O3`), and SVT-AV1 already had LTO. `-march=native` + LTO made libaom 3.3-3.5 % faster at speed 6 (0.8-2.2 % at speed 9) and SVT-AV1 1.4-4.5 % faster, depending on the preset. SVT-AV1's own PGO, trained on a different image set, gained the most: 11-13 % at presets 2-6 and 9-11 % at presets 8-9 over stock (two rounds on the primary host; 11-18 % on the replica). **Every tuned and PGO output is byte-identical to the stock output**, so the published BD-rates hold unchanged.
2. **Thread splits (question b).** Of the five encoders split (libaom speed 6 and 9, SVT-AV1 presets 4, 6 and 9), 16 × `-j 1` is the fastest or within noise of it for libaom speed 9 and SVT-AV1 presets 6 and 9. 8 × `-j 2` is faster for libaom speed 6 (+2.4 to +4.3 % over three comparisons) and SVT-AV1 preset 4 (+3.6 to +4.9 %), and about the same for preset 6 (+0.9 to +1.1 %). More threads per encode are slower everywhere: at 1 × `-j 16` libaom speed 6 runs 35 % slower and SVT-AV1 preset 9 84 % slower.
3. **Bytes across splits.** SVT-AV1 writes the same files at every split. libaom writes different files at any `-j` > 1 (the same files for 2, 4, 8 and 16 threads; all 99 images change at speed 6, 30 of 99 at speed 9). The compression is effectively the same: at q60, 0.11 % fewer bytes and 0.014 lower mean SSIMULACRA2 at speed 6; BD-rate +0.10 % at speed 6 and +0.00 % at speed 9 against the `-j 1` files.
4. **Best combinations measured** (primary host, against stock 16 × `-j 1`): SVT-AV1 preset 4 with PGO at 8 × `-j 2`: 17.9 MP/s vs 15.6 (+15 %). Preset 6 with PGO: 55.1-55.3 MP/s at 16 × `-j 1` and 55.5 at 8 × `-j 2`, vs 49.4 (+12 %). libaom speed 6, tuned, at 8 × `-j 2`: 31.1 vs 28.7 MP/s (+8 %).
5. **Tiling.** avifenc 1.4 tiles automatically when no tile option is given, so the published baselines (and every run here) used up to 8 tiles per image; `--autotiling` changes nothing. One tile per image runs at the same speed at 16 × `-j 1` (-0.7 to +1.6 %) and is 0.4-0.8 % smaller at equal SSIMULACRA2 (BD-rate median: libaom s6 -0.52 %, s9 -0.81 %; SVT-AV1 p4 -0.52 %, p6 -0.52 %, p9 -0.37 %). Comparisons between the CPU encoders move by at most ~0.3 point (both gain about the same); against an untiled libaom speed 6 anchor, a codec that does not tile would score about 0.5 point worse than against the published (tiled) anchor.
6. **Host variance.** One of the two c6a.4xlarge instances ran every encoder 7-8 % slower, with the same bytes; per-session controls are needed before comparing absolute speeds across sessions.

## Control: stock build, 16 × `-j 1` (MP/s)

| Encoder | published | primary host, start | primary, end | replica host |
|---|---:|---:|---:|---:|
| libaom speed 6 | 28.7 | 28.7 (-0.1 %) | 28.6 (-0.4 %) | 26.7 (-7.2 %) |
| libaom speed 9 | 186.4 | 186.6 (+0.1 %) |  | 174.1 (-6.6 %) |
| SVT-AV1 preset 2 | 4.6 | 4.6 (-0.6 %) |  | 4.2 (-8.4 %) |
| SVT-AV1 preset 4 | 15.5 | 15.6 (+0.1 %) |  | 14.3 (-7.9 %) |
| SVT-AV1 preset 6 | 49.4 | 49.4 (+0.0 %) | 49.8 (+0.9 %) | 45.9 (-7.0 %) |
| SVT-AV1 preset 8 | 136.0 | 136.0 (+0.0 %) |  | 125.8 (-7.5 %) |
| SVT-AV1 preset 9 | 182.8 | 182.6 (-0.1 %) |  | 170.3 (-6.9 %) |

## A: thread splits, stock build (MP/s; change vs 16 × `-j 1` in the same block of runs)

avifenc 1.4 tiles automatically unless given tile options (as in the published runs): libavif uses up to 8 tiles of at least 512 × 512 pixels whatever `-j` is, which here is 8 tiles for 56 of the 99 images (452 of the 478 MP), 4 for 4, 2 for 33 and 1 for 6. So `-j N` gives libaom or SVT-AV1 N threads over those tiles. libavif passes `-j N` to libaom as `g_threads` and to SVT-AV1 4.x as `level_of_parallelism` = N, which SVT-AV1 4.2 caps at 6 (`-j 8` and `-j 16` both run level 6).

| Encoder (primary) | 16 × `-j 1` | 8 × `-j 2` | 4 × `-j 4` | 2 × `-j 8` | 1 × `-j 16` | output vs 16 × `-j 1` |
|---|---:|---:|---:|---:|---:|---|
| libaom speed 6 | **28.5** | 29.6 (+3.7 %) | 28.6 (+0.2 %) | 25.4 (-10.9 %) | 18.5 (-35.2 %) | -0.106 % bytes (the same files at every `-j` > 1) |
| libaom speed 9 | **188.9** | 170.9 (-9.5 %) | 139.4 (-26.2 %) | 95.0 (-49.7 %) | 52.6 (-72.2 %) | +0.014 % bytes (the same files at every `-j` > 1) |
| SVT-AV1 preset 4 | **15.7** | 16.3 (+3.6 %) | 15.7 (+0.0 %) | 12.7 (-19.6 %) | 8.1 (-48.5 %) | identical |
| SVT-AV1 preset 6 | **49.4** | 49.9 (+1.1 %) | 43.3 (-12.3 %) | 29.8 (-39.7 %) | 17.1 (-65.4 %) | identical |
| SVT-AV1 preset 9 | **184.5** | 152.4 (-17.4 %) | 100.5 (-45.5 %) | 55.7 (-69.8 %) | 28.9 (-84.3 %) | identical |

| Encoder (replica) | 16 × `-j 1` | 8 × `-j 2` | 4 × `-j 4` | 2 × `-j 8` | 1 × `-j 16` | output vs 16 × `-j 1` |
|---|---:|---:|---:|---:|---:|---|
| libaom speed 6 | **26.3** | 27.4 (+4.4 %) | 26.0 (-1.0 %) | 23.0 (-12.6 %) | 16.8 (-36.2 %) | -0.106 % bytes (the same files at every `-j` > 1) |
| libaom speed 9 | **172.2** | 158.6 (-7.9 %) | 129.0 (-25.1 %) | 88.0 (-48.9 %) | 50.0 (-70.9 %) | +0.014 % bytes (the same files at every `-j` > 1) |
| SVT-AV1 preset 4 | **14.7** | 15.6 (+6.0 %) | 15.2 (+3.3 %) | 12.2 (-17.3 %) | 7.7 (-47.6 %) | identical |
| SVT-AV1 preset 6 | **46.2** | 46.4 (+0.5 %) | 40.9 (-11.6 %) | 28.0 (-39.5 %) | 15.7 (-66.0 %) | identical |
| SVT-AV1 preset 9 | **171.2** | 140.8 (-17.7 %) | 94.4 (-44.9 %) | 51.3 (-70.0 %) | 26.4 (-84.6 %) | identical |

Second round on the primary host (`R2`, 16 × `-j 1` and 8 × `-j 2` back to back): libaom speed 6 28.7 → 29.4 (+2.4 %); SVT-AV1 preset 4 15.6 → 16.3 (+4.8 %).

## B: tuned builds, 16 × `-j 1` (MP/s; change vs stock)

Round 1: stock from the control block, tuned and PGO from the B block (~20–40 min later). Round 2 (primary only): stock, tuned and PGO back to back per encoder.

| Encoder | stock r1 / r2 | tuned r1 / r2 | tuned vs stock | tuned + PGO r1 / r2 | PGO vs stock | replica: tuned, PGO vs stock |
|---|---:|---:|---:|---:|---:|---:|
| libaom speed 6 | 28.7 / 28.7 | 29.7 / 29.6 | +3.5 % / +3.3 % | — | — | +0.5 % |
| libaom speed 9 | 186.6 / 186.8 | 190.6 / 188.3 | +2.2 % / +0.8 % | — | — | -0.5 % |
| SVT-AV1 preset 2 | 4.6 | 4.6 | +1.5 % | 5.1 | +12.8 % | +1.0 %, +17.9 % |
| SVT-AV1 preset 4 | 15.6 / 15.6 | 15.8 / 15.8 | +1.6 % / +1.4 % | 17.2 / 17.3 | +10.8 % / +11.0 % | +0.1 %, +17.8 % |
| SVT-AV1 preset 6 | 49.4 / 49.8 | 50.5 / 51.0 | +2.3 % / +2.5 % | 55.1 / 55.3 | +11.6 % / +11.1 % | +1.8 %, +15.6 % |
| SVT-AV1 preset 8 | 136.0 / 134.0 | 138.5 / 140.1 | +1.9 % / +4.5 % | 149.2 / 149.4 | +9.7 % / +11.4 % | +6.0 %, +13.4 % |
| SVT-AV1 preset 9 | 182.6 / 181.9 | 190.5 / 189.8 | +4.3 % / +4.3 % | 198.6 / 199.4 | +8.7 % / +9.6 % | +7.6 %, +10.6 % |

Every tuned and PGO output at 16 × `-j 1` is byte-identical to the stock output: yes.

Best split from A (primary), re-timed with each build:

| Encoder | split | build | MP/s | vs stock 16 × `-j 1` (control block) | output |
|---|---|---|---:|---:|---|
| libaom speed 6 | 8 × `-j 2` | stock | 29.9 | +4.3 % | as stock at this split |
| libaom speed 6 | 8 × `-j 2` | tuned | 31.1 | +8.3 % | as stock at this split |
| SVT-AV1 preset 4 | 8 × `-j 2` | stock | 16.3 | +4.9 % | identical |
| SVT-AV1 preset 4 | 8 × `-j 2` | tuned | 16.6 | +6.5 % | identical |
| SVT-AV1 preset 4 | 8 × `-j 2` | pgo | 17.9 | +15.3 % | identical |
| SVT-AV1 preset 6 | 8 × `-j 2` | stock | 49.9 | +0.9 % | identical |
| SVT-AV1 preset 6 | 8 × `-j 2` | tuned | 51.2 | +3.6 % | identical |
| SVT-AV1 preset 6 | 8 × `-j 2` | pgo | 55.5 | +12.3 % | identical |

## Tiling

Passing `--autotiling` explicitly (primary, stock: libaom speed 6 at 1 × `-j 16`, libaom speed 6 at 4 × `-j 4`, libaom speed 9 at 1 × `-j 16`, libaom speed 9 at 4 × `-j 4`) gave the same files as the runs without it, as expected from avifenc's default. The tiling question is therefore the reverse one: what avifenc's default tiling costs or gains against one tile per image (`--tilerowslog2 0 --tilecolslog2 0`, bench.py `--no-tiling`).

Speed, stock build, primary host, default tiling and one tile back to back (MP/s):

| Encoder | 16 × `-j 1`, default tiling | 16 × `-j 1`, one tile | change | 8 × `-j 2`, one tile | change |
|---|---:|---:|---:|---:|---:|
| libaom speed 6 | 28.8 | 28.6 | -0.7 % | 28.9 | +0.3 % |
| libaom speed 9 | 185.7 | 188.7 | +1.6 % |  |  |
| SVT-AV1 preset 4 | 15.5 | 15.6 | +0.6 % | 16.4 | +5.6 % |
| SVT-AV1 preset 6 | 49.5 | 49.7 | +0.4 % |  |  |
| SVT-AV1 preset 9 | 182.4 | 183.1 | +0.4 % |  |  |

## Checks against the published runs (replica host)

- libaom speed 6 stock q60: 99/99 files the same size as in aws-g6-l4-gamut; largest SSIMULACRA2 difference 0.000000
- libaom speed 9 stock q60: 99/99 files the same size as in aws-c6a-gamut-svt; largest SSIMULACRA2 difference 0.000000
- SVT-AV1 preset 2 stock q60: 99/99 files the same size as in aws-c6a-gamut-svt; largest SSIMULACRA2 difference 0.000000
- SVT-AV1 preset 4 stock q60: 99/99 files the same size as in aws-c6a-gamut-svt; largest SSIMULACRA2 difference 0.000000
- SVT-AV1 preset 6 stock q60: 99/99 files the same size as in aws-c6a-gamut-svt; largest SSIMULACRA2 difference 0.000000
- SVT-AV1 preset 8 stock q60: 99/99 files the same size as in aws-c6a-gamut-svt; largest SSIMULACRA2 difference 0.000000
- SVT-AV1 preset 9 stock q60: 99/99 files the same size as in aws-c6a-gamut-svt; largest SSIMULACRA2 difference 0.000000

## libaom at `-j` > 1: q60 SSIMULACRA2 of the changed files (99 images)

| Run | total bytes | vs `-j 1` | mean SSIMULACRA2 | vs `-j 1` | images better / worse / same |
|---|---:|---:|---:|---:|---:|
| A_stock_aom_s6_j2 | 28815308 | -0.106 % | 75.874 | -0.014 | 34 / 65 / 0 |
| A_stock_aom_s9_j2 | 30864446 | +0.014 % | 74.702 | -0.001 | 14 / 16 / 69 |

## Quality sweeps (q 30–80, 99 images): BD-rate on SSIMULACRA2 (median / mean over images)

`_j2` / `_j4`: 8 × `-j 2` / 4 × `-j 4`; `_tiles`: `--autotiling` passed (avifenc's default anyway); `_notiles`: one tile per image (`--tilerowslog2 0 --tilecolslog2 0`). Scored on the replica host.

| Run | vs libaom s6 (anchor) | vs the published stock -j 1 rows of the same encoder |
|---|---:|---:|
| stock_aom_s6_j2 | +0.10 % / +0.08 % | +0.10 % / +0.08 % |
| stock_aom_s9_j2 | +14.96 % / +17.16 % | +0.00 % / +0.02 % |
| stock_aom_s6_j4_tiles | +0.10 % / +0.08 % | +0.10 % / +0.08 % |
| stock_svt_p6_j4_tiles | +2.07 % / +3.22 % | +0.00 % / +0.00 % |
| stock_aom_s6_notiles | -0.52 % / -0.61 % | -0.52 % / -0.61 % |
| stock_svt_p6_notiles | +1.81 % / +2.65 % | -0.52 % / -0.55 % |
| stock_svt_p4_notiles | -3.23 % / -3.51 % | -0.52 % / -0.61 % |
| stock_aom_s9_notiles | +14.32 % / +16.09 % | -0.81 % / -0.92 % |
| stock_svt_p9_notiles | +9.37 % / +11.45 % | -0.37 % / -0.41 % |

- avifenc 1.4 tiles automatically by default (up to 8 tiles of at least 512 × 512 pixels, whatever `-j` is), so the published CPU baselines were tiled. One tile per image (`--tilerowslog2 0 --tilecolslog2 0`) runs at the same speed at 16 × `-j 1` and is about 0.5 % smaller at equal SSIMULACRA2 for both libaom and SVT-AV1 (sweeps above).
- SVT-AV1 adds `-mno-avx` after `NATIVE`'s `-march=native`, so `NATIVE` does not let GCC auto-vectorise its generic C code with AVX2/FMA; its hand-written AVX2 kernels are used by every build (runtime dispatch). The CPU has no AVX-512.
- libaom and libavif have no `-march=native`, LTO or PGO options of their own; no PGO build was made for libaom.
- Not rebuilt in any build: libjpeg-turbo, libyuv, libpng, zlib (Ubuntu 24.04 packages).
- Every run gave identical bytes in all three repeats of each image: all builds and splits are deterministic.
- The replica host also drifted ~3 % slower during its session (libaom s6 26.7 at the start, 25.9 at the end), so its differences of a few per cent are less certain than the primary's (28.70 at the start, 28.61 at the end).
