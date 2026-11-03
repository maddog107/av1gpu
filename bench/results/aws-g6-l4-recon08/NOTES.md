# recon08 working notes

Agent recon08, branch `perf/recon` from main cde1f74 (v0.8.0). Goal: faster k_recon on JPEG photos, byte-identical.
Instance `av1gpu-recon08`: g6.8xlarge on-demand (no g6.8xlarge spot capacity in any subnet), us-east-1, dead-man
shutdown +180 min.

## Baseline profile (main, `--prof`, one run each; `logs/prof1/`, `logs/prof2/`)

- Size range default: stage ms analysis 638, recon 2343, postfilter 119, gpu total 3175: recon is ~74 % of the GPU
  stages. Recon CTA-cycles = 16 % of 58 SMs x 4 CTAs x clock x recon time: the SB wavefront leaves the SMs mostly
  idle, per-SB latency and the work of the other encoders' kernels overlap.
- Phases (size range): luma setup 7.1, pass1 0.5, pass2 13.8, luma final 30.5 (with the extra ticks of
  `scripts/prof_ticks.diff`: 8.2 levels / CTA-wide final quantization of 32x32 and 64x64 blocks, 23.2 prediction +
  inverse transform + record), chroma edges+CfL 5.7, SATD 9.4, chroma RD 19.3, chroma final 11.6, maps 2.0 %.
- Standard set: recon CTA-cycles 1.5 % of the capacity (one batch; the rest goes to the small-image encoder).
- Thumbnails run on the small-image encoder (base preset), whose phases the profile does not aggregate: stage
  analysis 323, recon 321 of 733 ms.

## What was tried

1. **Kept: `rc_inv.cuh` in 32-bit arithmetic, dequantization by shift.** The inverse 1-D transforms run on one
   thread per row / column (H or W threads), so their instruction chain is on the critical path of every coded tx
   block. Quick A/B, size range: +4.8 %, +4.5 %, +5.0 % (three runs, each with main in the same run); standard set
   +3.0 %.
2. **Dropped: fewer barriers.** (a) `cta_inverse_add` range flags as barrier reductions (`__syncthreads_or`,
   `bar.red.or`) instead of a reset barrier + shared atomics; (b) luma final levels in place (no copy into S.qfin);
   (c) ~12 fewer CTA barriers per block (pass-1 size known per thread, pass-2 list and edges behind one barrier,
   no trailing barriers after the luma record / chroma pick / U-V counters, CfL average and alpha per thread,
   resets before the first chroma barrier, chroma txb_ctx hoisted). All byte-identical on the quick sets, but no
   gain: size range c2 = c1 -0.8 %, c3 = c1 -1.0 %; standard c3 = c1 -0.6 % (inside the noise, never better). The
   barriers are not where the time goes: the phases are bound by single-thread / single-warp chains (global loads,
   serial loops), and removing a barrier only moves the wait to the next one. Kept as `scripts/dropped_barriers.diff`
   (with a one-pass eob + culLevel in the record) for reference.
3. **Kept: forward transform loops with a compile-time trip count** (`dev_tx.cuh` `fwd_col_pass` /
   `fwd_row_pass`, used by warp_eval_rt in k_recon and k_pal: below 32 the template value is the length, so the k
   loop unrolls fully, same fmaf order per output). Quick A/B, size range: c4 (1 + 3) 249.0 MP/s against c1 245.6
   and main 236.2 (+1.4 % over c1, both rounds).

Final verification of abf26f5 (c1 + fwd loops; the branch's 1d432f5 has the same encoder sources):
`scripts/final.sh`, 06:08-06:26 UTC: identity 78 runs, 5,030 / 5,030 files, 0 differ; decoder gate 16/16 PASS;
ctest 9/9 and the GPU tests 0 mismatches; memcheck / synccheck 0 errors; racecheck only the known k_entropy_win
pairs. Speed A/B 06:26-06:44 (summary.md): size range +4.1 %, standard +3.4 %, thumbnails +3.2 % img/s
(noise -1.6 / -1.1 / -0.4 %), single-file -0.6 / -0.6 / 0.0 %. AWS torn down at 06:45.

## Ideas not done (for a later round)

- Chroma RD / luma pass 2 with 4x4 tx blocks on half-warps (two tasks per warp): exact for n <= 16 (each lane
  holds at most one element, the xor-16 shuffle step adds zeros), would halve the rounds of these phases.
- Thread 0's serial loops: chroma RD top-K and the delta screening (19 candidates, a cost-table load each), the
  pick loop; rc_edges' serial DC sum (w + h samples) and integer division.
- base_bits on lane 0 (chain of cost-table loads) in every warp evaluation.

## Scripts (`scripts/`)

setup8.sh (instance), prof1.sh / pf_build.sh (profiles), bnew.sh (build + quick identity), q*.sh (quick A/B),
spd8.py (A/B speed, speed-v04 method), identity8.sh (speed-v06 identity.sh + the palette sets), checks8.sh (decoder
gate, ctest, GPU tests), resusage.sh (ptxas resource usage), final.sh (the final verification, in order).
