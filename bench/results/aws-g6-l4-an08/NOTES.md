# an08 working notes (perf/analysis)

Instance: av1gpu-an08 (i-0fb91781bf847cce2), g6.8xlarge on-demand (no g6.8xlarge spot capacity in any subnet at
05:26 UTC; the template tries the larger type first), L4, driver 595.112.07, CUDA 13.2. Dead-man `shutdown -h +180`
from 05:26 UTC. ~/old = main cde1f74 (tests OFF); ~/new = change 1 (tests ON); ~/new2 = changes 1 + 2 (the branch
head; tests ON after verify2.sh's rebuild). All `-DCMAKE_CUDA_ARCHITECTURES=89`.

Set-up differences from speed-v06: `mk_ll_sets.py` reads `~/br` (charts, encoder.cpp): here `~/br -> ~/old` (main),
so the graphics PNGs differ a little from the stored ones; both binaries run on the same files. Palette sets from
`../aws-g6-l4-palette-study/scripts/prep_sets.py` (~/pal/sets).

## Profile of main (ncu gpu__time_duration.sum, kernels serialised, one pass, default settings)

| set | total kernel ms | k_recon | k_analysis (warp per block) | k_analysis_small | entropy win |
|---|---:|---:|---:|---:|---:|
| size range (100, 478 MP; mostly turbo) | 3653 | 2348 (64 %) | 356 (9.7 %) | 138 (3.8 %) | 558 (15 %) |
| standard (100, 79 MP; mostly base small preset) | 1538 | 842 (55 %) | 300 (19.5 %) | 177 (11.5 %) | 141 (9 %) |
| thumbnails (280, 42 MP; base small preset) | 731 | 314 (43 %) | 184 (25 %) | 115 (16 %) | 18 (2.5 %) |

k_dp: < 0.2 % everywhere (the "dp" stage time in the batch stage line is mostly waiting on other encoders' work).

SASS sampling (ncu SourceCounters), share of warp-stall samples in single-lane code (lane-0 serial parts):
k_analysis<16,8> on the standard set 14.7 %, k_analysis<16,16> on the size range 6.4 %, k_analysis<64,64> 1.8 %.
The biggest serial part is the intra edge filter / upsampling of each RD candidate (`av1g_edge_prepare` on lane 0
in `warp_residual`). k_recon already has a warp-parallel copy (`warp_edge_prepare`, k_recon.cuh).

Occupancy (ncu): k_analysis<64,64> 8.3 % theoretical (2 CTAs of 2 warps per SM, limited by 38.9 KB static shared
memory per CTA), k_analysis<32,32> 16.7 % (2 CTAs of 4 warps, 36.9 KB). The other warp kernels are register-limited
at 33 % (128 registers, launch bounds).

ptxas: k_analysis (warp kernels) 128 regs (launch bounds), 192-byte stack (the serial edge filter's local copy),
no or tiny spills; k_analysis_small<8,8> 128 regs with 432 B spill stores; 16x4 / 4x16 small kernels 168 regs.

## Changes

1. warp-parallel edge preparation in the analysis `warp_residual` (k_analysis.cuh: `an_edge_*`, the k_recon.cuh
   algorithm under its own names). ncu: k_analysis 183.9 -> 169.8 ms (thumbnails), 299.7 -> 268.9 ms (standard).
   Stack frame of every k_analysis kernel 192 -> 24 bytes. Identity, change 1 alone (verify.sh, stopped after 8
   sets to start the final run): 8/8 sets identical (`logs/v1/identity_new_partial.txt`); gate 12 runs PASS, ctest.
2. DCT-only shapes (a 32- or 64-sample side) keep the quantized levels in the residual's shared memory; 64x64 /
   64x32 one warp per CTA. ncu: k_analysis<32,32> 3 CTAs per SM (25 % theoretical occupancy, was 16.7 %),
   <64,64> 5 one-warp CTAs (10.4 %, was 8.3 %), <64,32> 7 (14.6 %).

Tried and dropped: an inlined compile-time-size copy of coeff_bits plus a register nonzero mask for the eob search
in the thread-per-block small kernels (k_analysis_small): no measurable change in ncu (113.8 -> 113.5 ms thumbnails).

## Status / log
- 05:26 launched; 05:31 builds done; 05:33 profiles; 05:47 change 1 built, quick identity OK; 05:52 verify.sh (change 1)
- 05:57 change 2 built (~/new2), quick identity OK on std; 05:59 verify.sh stopped (its ctest was cut off by the stop,
  so logs/v1 has only the 8-set identity, the gate and the first sanitizer pass), verify2.sh (1 + 2) started
- 06:02 racecheck on that build flagged coeff_bits level reads in res against the next warp_residual's writes (no
  __syncwarp between them): fixed (`__syncwarp` before the residual writes), runs stopped (logs/v2pre), verify2.sh
  restarted on the fixed build with only racecheck alongside (the first pass had run out of GPU memory on one set
  while memcheck / synccheck shared the GPU)
- 06:21 identity 66/66 identical; 06:23 palette 12/12; 06:24-06:51 speed (3-way A/B, noise, single); 06:52 memcheck /
  synccheck alone: clean. All results in summary.md.
