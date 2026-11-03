# av1gpu analysis kernels: perf/analysis against main cde1f74 (NVIDIA L4, g6.8xlarge, 2026-11-03)

Agent an08 of the v0.9 speed round: the open-loop analysis (`k_analysis*`, `k_an_*`) and the partition DP. Two
byte-identical changes. One g6.8xlarge on-demand instance (no g6.8xlarge spot capacity at launch): NVIDIA L4, driver
595.112.07, CUDA 13.2, 32 vCPUs (`logs/env.txt`). main cde1f74 (`~/old`) and the branch were built on the instance for
`sm_89` with the same toolkit: `~/new` = change 1, `~/new2` = changes 1 + 2 = the branch head. The method and most
scripts are `../aws-g6-l4-speed-v06/scripts/` as stored (`identity.sh`, `cmpset.sh`, `mk_sets.py`, `mk_ll_sets.py`,
`mk_mono.sh`, `sanitizer.sh`, `race_ab.sh`); the palette sets come from
`../aws-g6-l4-palette-study/scripts/prep_sets.py`; `scripts/` here has the rest (`spd_an.py` speed, `identity_pal.sh`,
`checks2.sh`, `verify2.sh`, `after2.sh`, the profiling scripts). `mk_ll_sets.py` reads `~/br` (charts, encoder.cpp):
here `~/br` was main, so the graphics PNGs differ a little from the stored ones; both binaries ran on the same files.

## Where the time goes (main, photos)

ncu `gpu__time_duration.sum`, kernels serialised, one pass of each set, default settings (`logs/prof/base_*_ksum.txt`):

| set | kernel ms | k_recon | k_analysis (warp per block) | k_analysis_small (thread per block) | k_entropy_win | k_dp |
|---|---:|---:|---:|---:|---:|---:|
| size range (100 JPEGs, 478 MP; mostly `turbo`) | 3,653 | 2,348 (64 %) | 356 (9.7 %) | 138 (3.8 %) | 558 (15 %) | 7.0 (0.2 %) |
| standard (100 JPEGs, 79 MP; mostly the `base` small preset) | 1,538 | 842 (55 %) | 300 (19.5 %) | 177 (11.5 %) | 141 (9 %) | 1.7 (0.1 %) |
| thumbnails (280 JPEGs, 42 MP; `base` small preset) | 731 | 314 (43 %) | 184 (25 %) | 115 (16 %) | 18 (2.5 %) | 0.0 |

- The analysis is spread over 19 shape kernels; none is above 3.5 % of the GPU time. On the thumbnails it is as large
  as the reconstruction. The "dp" time of the batch stage line is mostly waiting: the kernel itself is under 0.2 %.
- SASS sampling (ncu source counters, `logs/prof/*_sass_summary.txt`): in `k_analysis<16,8>` (standard set) 14.7 % of
  the stall samples were in single-lane code, mostly the intra edge filter of each RD candidate (`av1g_edge_prepare`
  on lane 0 in `warp_residual`); 6.4 % in `k_analysis<16,16>`, 1.8 % in `k_analysis<64,64>` (size range).
- Occupancy (ncu): `k_analysis<64,64>` 8.3 % theoretical, `k_analysis<32,32>` 16.7 %, both limited by shared memory
  (38.9 / 36.9 KB static per CTA: 2 CTAs per SM); the other warp kernels are register-limited at 33 % (128 registers
  from their launch bounds).
- ptxas (`logs/prof/ptxv_*`): every `k_analysis` kernel had a 192-byte stack frame (the serial edge filter's local
  copy); `k_analysis_small<8,8>` spills 432 bytes at 128 registers.

## Changes

1. **Warp-parallel intra edge filter in the analysis RD stage** (`k_analysis.cuh`). `warp_residual` ran
   `av1g_edge_prepare` on lane 0 for every directional RD candidate. It now runs the warp version that k_recon
   already uses (`k_recon.cuh: warp_edge_prepare`), copied under its own names (`an_edge_*`), so the two files stay
   independent: every lane reads its outputs' inputs, `__syncwarp`, then writes; the same integer arithmetic. The
   stack frame of every `k_analysis` kernel drops from 192 to 24 bytes. ncu kernel time, one pass: k_analysis
   183.9 -> 169.8 ms on the thumbnails (-7.7 %), 299.7 -> 268.9 ms on the standard set (`logs/prof/ef1_*`).
2. **Less shared memory per warp for the DCT-only shapes, one warp per CTA for 64x64 / 64x32** (`k_analysis.cuh`,
   `k_an_d.cu`). Shapes with a 32- or 64-sample side have only DCT_DCT for intra luma, so every RD evaluation follows a
   fresh `warp_residual`: the quantized levels now reuse the residual's storage (`ASmem::qInRes`). The column pass has
   read the whole residual (and `__syncwarp`ed) before `warp_quant_rate_rt` writes a level, and a `__syncwarp` before
   the next residual's writes covers the last level reads (racecheck found it missing in a first build; see the
   checks). 64x64 and 64x32 run one warp per CTA instead of two, so the per-SM limit counts warps, not pairs. Each warp
   still analyses one block in the same order. ncu (`logs/prof/n2_*`, `o_3232`): `k_analysis<32,32>` 3 CTAs per SM
   (25 % theoretical occupancy, was 16.7 %), `<64,64>` 5 warps per SM (10.4 %, was 8.3 %), `<64,32>` 7 (14.6 %).

Tried and dropped: an inlined compile-time-size copy of `coeff_bits` and a register nonzero mask for the eob search in
the thread-per-block small kernels: no change in ncu (k_analysis_small 113.8 -> 113.5 ms on the thumbnails).

**Blackwell (sm_120):** change 1 removes work and is not occupancy-tuned. Change 2 is occupancy-sensitive: it lowers
the shared memory per CTA, which can only raise the CTAs per SM; the counts above are for the L4's 100 KB per SM. The
CUDA guide lists 100 KB of shared memory per SM for compute capability 12.0 as well, so the same counts should hold
there, but this was not measured on Blackwell. Neither change alters registers (all `k_analysis` kernels stay at 128).

## Byte identity (`logs/identity_new2.txt`, `logs/identity_pal_new2.txt`)

The branch head (changes 1 + 2) against main, `cmp` of every output: speed-v06's full identity list (`identity.sh`:
size range and standard at default, presets, qualities, 4:4:4, 10-bit, 4:4:4 10-bit; thumbnails; gray / mono; odd sizes;
lossless; RGBA / alpha; WebP; mixed batch list; single-file mode) plus the palette sets (screen, flat_png, flat_jpg)
at default, `--palette on`, `--yuv 444 --depth 10` and `--lossless`. Every run had rc 0 for both binaries.

| set | options | inputs | main files | branch files | differ | skipped (both) |
|---|---|---:|---:|---:|---:|---:|
| sr_default | default | 100 | 100 | 100 | 0 | 0 |
| sr_turbo3_nosmall | `--preset turbo3 --small-preset none` | 100 | 100 | 100 | 0 | 0 |
| sr_nosmall | `--small-preset none` | 100 | 100 | 100 | 0 | 0 |
| sr_444 | `--yuv 444` | 100 | 100 | 100 | 0 | 0 |
| sr_10 | `--depth 10` | 100 | 100 | 100 | 0 | 0 |
| sr_444_10 | `--yuv 444 --depth 10` | 100 | 100 | 100 | 0 | 0 |
| sr_base_nosmall | `--preset base --small-preset none` | 100 | 100 | 100 | 0 | 0 |
| sr_slow_nosmall | `--preset slow --small-preset none` | 100 | 100 | 100 | 0 | 0 |
| sr_fast_nosmall | `--preset fast --small-preset none` | 100 | 100 | 100 | 0 | 0 |
| sr_q30 | `--quality 30` | 100 | 100 | 100 | 0 | 0 |
| sr_q90 | `--quality 90` | 100 | 100 | 100 | 0 | 0 |
| sr_q95 | `--quality 95` | 100 | 100 | 100 | 0 | 0 |
| sr_tune_none | `--tune none` | 100 | 100 | 100 | 0 | 0 |
| sr_tiles_2_1_1 | `--tiles 2,1,1` | 100 | 100 | 100 | 0 | 0 |
| sr_default_enc1 | default, `GP_ENCODERS=1` | 100 | 100 | 100 | 0 | 0 |
| sr_default_enc6 | default, `GP_ENCODERS=6` | 100 | 100 | 100 | 0 | 0 |
| sr_slow | `--preset slow` | 100 | 100 | 100 | 0 | 0 |
| sr_444_q90 | `--yuv 444 --quality 90` | 100 | 100 | 100 | 0 | 0 |
| sr_444_q95 | `--yuv 444 --quality 95` | 100 | 100 | 100 | 0 | 0 |
| sr_444_slow | `--yuv 444 --preset slow` | 100 | 100 | 100 | 0 | 0 |
| sr_444_10_q95 | `--yuv 444 --depth 10 --quality 95` | 100 | 100 | 100 | 0 | 0 |
| sr_10_q95 | `--depth 10 --quality 95` | 100 | 100 | 100 | 0 | 0 |
| mono_default | default | 47 | 47 | 47 | 0 | 0 |
| mono_q90 | `--quality 90` | 47 | 47 | 47 | 0 | 0 |
| mono_q95 | `--quality 95` | 47 | 47 | 47 | 0 | 0 |
| mono_slow | `--preset slow` | 47 | 47 | 47 | 0 | 0 |
| mono_nosmall | `--small-preset none` | 47 | 47 | 47 | 0 | 0 |
| mono_10 | `--depth 10` | 47 | 47 | 47 | 0 | 0 |
| mono_tune_none | `--tune none` | 47 | 47 | 47 | 0 | 0 |
| std_q95 | `--quality 95` | 100 | 100 | 100 | 0 | 0 |
| std_slow | `--preset slow` | 100 | 100 | 100 | 0 | 0 |
| thumbs_q95 | `--quality 95` | 280 | 280 | 280 | 0 | 0 |
| thumbs_slow | `--preset slow` | 280 | 280 | 280 | 0 | 0 |
| kodak_q95 | `--quality 95` | 24 | 24 | 24 | 0 | 0 |
| kodak_444_q90 | `--yuv 444 --quality 90` | 24 | 24 | 24 | 0 | 0 |
| rgba_q95 | `--quality 95` | 39 | 39 | 39 | 0 | 0 |
| odd_q95 | `--quality 95` | 30 | 30 | 30 | 0 | 0 |
| odd_slow | `--preset slow` | 30 | 30 | 30 | 0 | 0 |
| std_default | default | 100 | 100 | 100 | 0 | 0 |
| std_nosmall | `--small-preset none` | 100 | 100 | 100 | 0 | 0 |
| thumbs_default | default | 280 | 280 | 280 | 0 | 0 |
| thumbs_nosmall | `--small-preset none` | 280 | 280 | 280 | 0 | 0 |
| kodak_default | default | 24 | 24 | 24 | 0 | 0 |
| kodak_lossless | `--lossless` | 24 | 24 | 24 | 0 | 0 |
| kodak_lossless_ycgco | `--lossless --ycgco-re` | 24 | 24 | 24 | 0 | 0 |
| rgba_default | default | 39 | 39 | 39 | 0 | 0 |
| rgba_aq100 | `--alpha-quality 100` | 39 | 39 | 39 | 0 | 0 |
| rgba_lossless | `--lossless` | 39 | 39 | 39 | 0 | 0 |
| rgba_no_alpha | `--no-alpha` | 39 | 39 | 39 | 0 | 0 |
| gray_default | default | 9 | 9 | 9 | 0 | 0 |
| gray_lossless | `--lossless` | 9 | 8 | 8 | 0 | 1 |
| odd_default | default | 30 | 30 | 30 | 0 | 0 |
| odd_444_10 | `--yuv 444 --depth 10` | 30 | 30 | 30 | 0 | 0 |
| odd_lossless | `--lossless` | 30 | 10 | 10 | 0 | 20 |
| misc_default | default | 13 | 13 | 13 | 0 | 0 |
| png16_10_default | `--depth 10` | 8 | 8 | 8 | 0 | 0 |
| png16_10_lossless | `--lossless` | 8 | 8 | 8 | 0 | 0 |
| llgraphics_lossless | `--lossless --ycgco-re` | 27 | 27 | 27 | 0 | 0 |
| webp_default | default | 7 | 7 | 7 | 0 | 0 |
| webp_lossless | `--lossless` | 7 | 7 | 7 | 0 | 0 |
| mixed_list | default | 33 | 33 | 33 | 0 | 0 |
| mixed_list_444_10 | `--yuv 444 --depth 10` | 33 | 33 | 33 | 0 | 0 |
| lljpeg_set | default | 25 | 25 | 25 | 0 | 0 |
| single_file | default | 10 | 10 | 10 | 0 |  |
| single_file | `--yuv 444 --depth 10` | 10 | 10 | 10 | 0 |  |
| single_file | `--preset slow` | 10 | 10 | 10 | 0 |  |
| pal_screen_default | default | 40 | 40 | 40 | 0 | 0 |
| pal_screen_palette_on | `--palette on` | 40 | 40 | 40 | 0 | 0 |
| pal_screen_444_10 | `--yuv 444 --depth 10` | 40 | 40 | 40 | 0 | 0 |
| pal_screen_lossless | `--lossless` | 40 | 40 | 40 | 0 | 0 |
| pal_flat_png_default | default | 24 | 24 | 24 | 0 | 0 |
| pal_flat_png_palette_on | `--palette on` | 24 | 24 | 24 | 0 | 0 |
| pal_flat_png_444_10 | `--yuv 444 --depth 10` | 24 | 24 | 24 | 0 | 0 |
| pal_flat_png_lossless | `--lossless` | 24 | 24 | 24 | 0 | 0 |
| pal_flat_jpg_default | default | 24 | 24 | 24 | 0 | 0 |
| pal_flat_jpg_palette_on | `--palette on` | 24 | 24 | 24 | 0 | 0 |
| pal_flat_jpg_444_10 | `--yuv 444 --depth 10` | 24 | 24 | 24 | 0 | 0 |
| pal_flat_jpg_lossless | `--lossless` | 24 | 0 | 0 | 0 | 24 |
| **all (78 runs)** | | | **5,030** | **5,030** | **0** | |

Change 1 alone (`~/new`) was also compared on 8 sets before the final run started (`logs/v1/identity_new_partial.txt`:
size range default, turbo3, nosmall, 4:4:4, 10-bit, 4:4:4 10-bit, base, slow): all identical. A first run of the
final check, with the sanitizers sharing the GPU, ran out of GPU memory on one set (both binaries rc 1); it was
stopped when racecheck flagged the missing `__syncwarp` (`logs/v2pre/`), and the whole list was run again on the
fixed build with only racecheck alongside.

## Speed (the stored method)

`scripts/spd_an.py` (speed-v06's method): each set repeated to >= 1,000 MP (`bench.speed_set`), a warm-up and 3 timed runs (`--stats` MP/s), 2 rounds with the binary order alternating (new2, new, old / old, new, new2), median of the 6 timed runs; the default 3 encoders. `new` = change 1, `new2` = changes 1 + 2 (the branch), `old` = main. Raw runs: `logs/runs/ab3.jsonl`; the noise run (main against main, same method): `logs/runs/noise.jsonl`. Wall MP/s counts the whole process (CUDA start-up included).

| set | main MP/s | change 1 MP/s | branch MP/s | change 1 vs main | change 2 vs change 1 | branch vs main | branch img/s | main img/s | rounds main | rounds branch | branch wall MP/s | main wall MP/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| size range (100 JPEGs), default | 225.66 | 226.48 | 228.25 | +0.4 % | +0.8 % | **+1.1 %** | 47.8 | 47.2 | 225.0 / 226.4 | 231.0 / 228.2 | 197.9 | 195.3 |
| standard (100 JPEGs), default | 72.93 | 74.24 | 75.98 | +1.8 % | +2.3 % | **+4.2 %** | 96.5 | 92.6 | 73.3 / 72.4 | 76.2 / 75.3 | 70.8 | 68.2 |
| thumbnails (280 JPEGs), default | 60.08 | 60.89 | 61.91 | +1.4 % | +1.7 % | **+3.1 %** | 408.2 | 396.1 | 59.8 / 60.3 | 62.5 / 60.6 | 58.5 | 56.8 |

Output bytes were the same in every timed run of every binary: True.

**Noise run** (main against main, the same binary under two labels, same alternation):

| set | main (label old2) MP/s | main (label old) MP/s | difference | rounds old2 | rounds old |
|---|---:|---:|---:|---|---|
| size range (100 JPEGs), default | 225.84 | 225.53 | +0.1 % | 225.7 / 228.2 | 225.8 / 225.3 |
| standard (100 JPEGs), default | 72.31 | 72.53 | -0.3 % | 73.1 / 71.5 | 72.9 / 72.3 |
| thumbnails (280 JPEGs), default | 59.93 | 60.13 | -0.3 % | 60.6 / 59.4 | 60.7 / 59.7 |

**Single-JPEG cold process** (`av1gpu in.jpg out.avif`, no `--batch`; wall time per process including CUDA start-up; a warm-up, then 12 runs per binary, order alternating; median; `logs/runs/single.jsonl`):

| file | MP | main ms | change 1 ms | branch ms | branch vs main |
|---|---:|---:|---:|---:|---:|
| 6057c6daef8fc78a_typical.jpg | 2.16 | 1103.4 | 1100.9 | 1101.0 | -0.2 % |
| a239a24eddebdbba_huge.jpg | 12.00 | 1309.9 | 1307.9 | 1303.6 | -0.5 % |

**Reading the numbers.** The noise run's medians agree within 0.3 %, but the rounds of one binary differ by up to
1.3 % (size range) and 2.2 % (standard set), so:

- **Standard set +4.2 %: outside the noise**; both branch rounds (76.2, 75.3) are above both main rounds (73.3, 72.4).
  Change 1 +1.8 % and change 2 +2.3 % each, at the edge of the round spread on their own.
- **Thumbnails +3.1 %** on the median of 6; round 1 +4.5 %, round 2 +0.6 %. The median is above the noise run's
  0.3 %, the second round alone is not; the ncu kernel times (k_analysis -7.7 % from change 1) support a real gain.
- **Size range +1.1 %: inside the noise.** The size range is mostly `turbo` (K 1, no tx search), where the analysis is
  10-13 % of the GPU time; the gain there is real at kernel level but too small to separate from run-to-run noise.
- **Single JPEG cold process -0.2 % / -0.5 %: inside the noise.** About 1.1-1.3 s per process, mostly CUDA start-up.

Speed sets (files, MP, repeats): size_range: 300 files, 1434 MP, x3; standard: 1300 files, 1023 MP, x13; thumbs: 6720 files, 1019 MP, x24.

## Decoder gate, tests, sanitizers (branch head)

- **Decoder gate** (`tests/decoder_gate.py`, dav1d 1.4.1 and aomdec, `logs/gate_new2.txt`): 12 runs, **497/497
  encodes and 3/3 alpha items identical** in both decoders, all PASS: 30 size-range photos (3 of every 10) at qindex
  60 / 120 / 200, 4:4:4 10-bit, `--preset slow`; 28 thumbnails at 60 / 120; the 30 odd sizes in 4:2:0 and 4:4:4
  10-bit; the gray and gray + alpha PNGs; the 47 monochrome inputs at 26 / 120; the palette sets screen (also 4:4:4
  10-bit), flat_png and flat_jpg.
- **ctest** 9/9 passed; `gpu_fuzz_test` seeds 1-400 and 77-276 (more 4:4:4, 10-bit, lossless): 0 mismatches;
  `gpu_dsp_test` passed (`logs/ctest_new2.txt`).
- **compute-sanitizer racecheck** (analysis mode, `race_ab.sh`: 3 size-range photos, default and `--yuv 444 --depth
  10`; `logs/race/`): 8 hazards per run, all in `k_entropy_win` / `vload` / `vstore` (gpu_entropy.cu, the documented
  producer / consumer hand-off of docs/RESULTS.md); nothing in the analysis kernels. A first build of change 2 without
  the `__syncwarp` before `warp_residual`'s writes showed reports between `coeff_bits` reads of the levels and the next
  residual's writes (`logs/v2pre/race_new2_0_partial_head.txt`); the fix went in before the final runs.
- **memcheck / synccheck** (`sanitizer.sh`, alone on the GPU, `logs/sanitizer_new2b.txt`):
  - memcheck [default] rc 99, 29 s, outputs differing from the plain run: 0; ========= ERROR SUMMARY: 5 errors
  - memcheck [--yuv 444 --depth 10] rc 99, 31 s, outputs differing from the plain run: 0; ========= ERROR SUMMARY: 6 errors
  - synccheck [default] rc 0, 12 s, outputs differing from the plain run: 0; ========= ERROR SUMMARY: 0 errors
  - synccheck [--yuv 444 --depth 10] rc 0, 12 s, outputs differing from the plain run: 0; ========= ERROR SUMMARY: 0 errors
  0 access errors: memcheck's 5 / 6 "errors" are its `--leak-check full` reports, 5 / 6 leaks of 256 KB, the
  encoders' variance-boost tables of v0.6's summary (pipeline.cpp, unchanged from main). Every instrumented output
  equals the plain run's.
