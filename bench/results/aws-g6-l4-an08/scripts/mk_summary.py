#!/usr/bin/env python3
"""mk_summary.py OUT.md IDTABLE.md SPEED.md CHECKS.md: assemble bench/results/aws-g6-l4-an08/summary.md from the fixed
text below and the generated parts (identity table, speed tables, checks)."""
import sys

out, idt, spd, chk = sys.argv[1:5]
HEAD = r"""# av1gpu analysis kernels: perf/analysis against main cde1f74 (NVIDIA L4, g6.8xlarge, 2026-11-03)

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

"""
TAIL = r"""
Change 1 alone (`~/new`) was also compared on 8 sets before the final run started (`logs/v1/identity_new_partial.txt`:
size range default, turbo3, nosmall, 4:4:4, 10-bit, 4:4:4 10-bit, base, slow): all identical. A first run of the
final check, with the sanitizers sharing the GPU, ran out of GPU memory on one set (both binaries rc 1); it was
stopped when racecheck flagged the missing `__syncwarp` (`logs/v2pre/`), and the whole list was run again on the
fixed build with only racecheck alongside.
"""
with open(out, "w") as f:
    f.write(HEAD)
    f.write(open(idt).read())
    f.write(TAIL)
    f.write("\n")
    f.write(open(spd).read())
    f.write("\n")
    f.write(open(chk).read())
