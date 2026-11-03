# ent08: GPU entropy and loop filters (NVIDIA L4, g6.4xlarge, 2026-11-03)

Branch `perf/entropy-filters` against main **cde1f74** (v0.8.0), both built on the instance with the same toolkit
(CUDA 13.2, driver 595.112.07) for `sm_89`. One g6.4xlarge on-demand instance (spot had no capacity in any of the 5
subnets), EPYC 7R13, 16 vCPUs (`logs/env.txt`). Scripts in `scripts/`, logs in `logs/`, working notes in `NOTES.md`.

## Where the time goes (main)

### Entropy: how much is on the critical path

In batch mode the entropy threads code each finished batch while the encoders work on the next ones, so most of the
entropy GPU time overlaps encoding. nsys, kernels classified by stream (`scripts/nsys_an.py`, `logs/prof/nsys_*.txt`):

| run (main) | GPU span | entropy-stream kernels busy | of that, while encoder streams were busy | entropy alone = tail after the last encoder kernel |
|---|---:|---:|---:|---:|
| size range, 100 JPEGs (478 MP, 9 batches) | 2,254 ms | 771 ms | 683 ms | **88 ms (3.9 %)** |
| standard 100 x 3 (236 MP) | 3,243 ms | 475 ms | 437 ms | 41 ms (1.3 %) |
| thumbnails 280 x 3 (127 MP) | 2,094 ms | 98 ms | 70 ms | 28 ms (1.3 %) |

So the entropy wall time printed by `--stats` (7.3 s in the brief's 1,132 MP batch) is summed over the three entropy
threads and almost all hidden. On the critical path are the last batch's entropy (the tail) and the SM slots the
entropy CTAs hold while `k_recon` runs. The tail is one `k_entropy_win` launch, as long as its longest tile's serial
chain: a 1-2 MP photo is one tile (turbo tiles from 2 MP), a large photo's 4 x 4 tiles are 0.5-1.5 MP each.
`k_entropy_win` was 744 of the 901 ms of entropy kernels in the size-range run (11 launches, 25-100 ms each).

Speed of light (`scripts/sol.sh` with `scripts/exp_patch.py`; not byte-identical, only to bound the possible gain;
`logs/prof/sol.txt`, 2 runs each, >= 1,000 MP): with `k_entropy_win` not coding, the size range ran +2-4 % (252.3 /
252.6 MP/s; main 247.6 / 237.6, the unmodified experiment build 242.0 / 239.1); the standard set and the thumbnails
did not move (78.4 / 77.7 against 78.6 / 77.4 MP/s; 419 / 416 against 418 / 413 img/s). Batch throughput is set by
analysis and reconstruction, not by the entropy coder.

### The loop filters and the CDEF strength search (lfsearch)

Kernel time per run (nsys; the default `--tune iq`: deblocking level from q, libaom's LVL4 CDEF search):

| run (main) | k_select_small | k_cdef_sse_t | k_cdef (apply) | k_deblock | k_mark | rest | share of all kernel time |
|---|---:|---:|---:|---:|---:|---:|---:|
| size range (100) | 64.0 ms (0.64 ms / image) | 60.5 | 37.7 | 13.5 | 5.3 | < 3 | 3.4 % |
| standard x 3 (300) | 97.7 ms (0.33 ms / image) | 36.4 | 22.3 | 9.2 | 6.0 | < 5 | 2.9 % |
| thumbnails x 3 (840) | 174.9 ms (0.21 ms / image) | 47.6 | 35.8 | 14.5 | 8.8 | < 10 | 10.8 % |

`k_select_small` (the joint preset search) is one 1024-thread CTA per image running 75 serial steps with four
barriers each: latency on one SM, on an encoder stream whose latency the other encoders hide. Replacing it with
memsets (speed of light) changed nothing outside the noise on any set (sr 245.9 / 245.6, std 77.6 / 76.6 MP/s,
thumbnails 417 / 415 img/s), so it was not rewritten this round. The CDEF apply is left alone (pipe08 takes it out of
the photo path). `k_cdef_sse_t` is GPU work, 1.1 % of the size range's kernel time.

### Single-file mode (one 8 MP JPEG, cold process)

Process wall 1.23 s, of which the kernels span 310 ms (nsys, `logs/prof/nsys_single_8mp_kernels.txt`). Entropy is on
the critical path here: ~8.5 ms of host copies of the decision record in `run_batch`, 2.6 ms upload, ~6 ms plan and
module loading, 22 ms of kernels (`k_entropy_win` 18.4 ms: 16 tiles of 0.5 MP). The loop filters take ~1.8 ms. The
rest of the second is CUDA start-up, decoding and allocations outside these files.

### Inside k_entropy_win

Per-symbol cycles per tile (`AV1G_ENTROPY_PROFILE` build with a reader in `run_batch`, `scripts/prof_patch.py`; L4 at
2,040 MHz; `logs/prof/entropy_prof_*_8mp.txt`): the coder thread (phase B) took 112-115 cycles per symbol, the
bottleneck of every large tile (producer, phase A: 88-129). ncu's warp-stall sampling of that thread
(`logs/prof/ncu_k_entropy_win_main_consumer.txt`) shows its serial chain: range -> `FLO` (short scoreboard) -> shift,
with the flush test of `low` and its branch on the same chain.

## Changes

1. **`k_entropy_win`: the coder thread split in two, and the producer's next symbol prefetched** (637f113). B1 (warp 1
   lane 0) runs only the range (r, u, v, d) from symbol to symbol and hands `(r - u) | d << 16` to B2 (warp 2 lane 0)
   through a second shared ring of 8 windows; B2 does `low`, the flush, carries and bytes. Same arithmetic as
   `rc_code`, split where `low` only takes `r - u` and `d`. The waits back off with `__nanosleep(32)`. In the
   producer, `win_group` fetches the next group member's lane and word one step ahead. CTA 96 threads (was 64),
   14.7 KB shared memory (was 13.7 KB), 40 registers per thread and no local memory as before. Per symbol on the 8 MP
   photo: B1 63 cycles, B2 8-17, the producer (now the limit) 83-118; tile time 119-139 -> 93-128 cycles per symbol,
   the longest tile 118.8 -> 96.6 (-19 %).
2. **Single-file `run_batch`: one frame uploaded straight from its record** (a8d2969), without the host copy into
   new vectors (~7 ms for 8 MP). Batch mode never calls `run_batch` (it codes device records with
   `run_batch_device`), so this commit changes single-file mode only.

Tried and dropped: a binary-tree select for the 16-entry CDF groups (the producer got slower: 88 -> 100 cycles per
symbol on coefficient tiles); an unrolled single coder loop with the range update before the flush branch (no
change: 113 -> 112).

## Speed

### k_entropy_win alone (nsys, single-file mode, the GPU otherwise idle; `scripts/kern_single.sh`, `logs/prof/kern_single.txt`)

| input | main | commit 1 | both commits |
|---|---:|---:|---:|
| 8 MP photo (16 tiles of 0.5 MP) | 18.43 ms | 15.37 ms (-16.6 %) | 15.37 ms |
| 1.8 MP photo (one tile) | 26.47 ms | 24.11 ms (-8.9 %) | 24.11 ms |

(The same in each of 3 runs.) The one-tile photo gains less: its tile is producer-bound (block-level symbols, 16-entry
CDF groups), and the producer only got the prefetch.

### A/B (the stored method: each set repeated to >= 1,000 MP, a warm-up + 3 timed runs per binary, 2 rounds with the order alternating, median of 6; `scripts/spd8.py`, `scripts/analyze8.py`, `logs/spd_*.jsonl`, `logs/speed_analysis.txt`)

A = main cde1f74, B = the PR head (both commits); noise = main against a copy of main, run right after.

| set | main | PR head | B/A | per round (A first, B first) | noise B/A | noise per round |
|---|---:|---:|---:|---|---:|---|
| size range, default (MP/s) | 228.92 | 225.84 | 0.987 | 0.993, 1.001 | 1.002 | 0.995, 1.018 |
| standard 100, default (MP/s) | 73.89 | 74.40 | 1.007 | 1.011, 1.006 | 0.996 | 1.001, 0.985 |
| thumbnails 280, default (img/s) | 407.74 | 407.71 | 1.000 | 0.989, 1.002 | 0.997 | 1.012, 0.992 |
| single 8 MP JPEG, cold process (wall, median of 12) | 1,247.1 ms | **1,231.3 ms** | **0.987** | A 1,239-1,279, B 1,227-1,246 ms | 1.003 | |
| single 8 MP JPEG, commit 1 alone | 1,242.0 ms | 1,246.6 ms | 1.004 | | | |

- **Batch mode: no change outside the noise** on any set. The per-round ratios straddle 1 like the noise run's
  (main against main moved 0.985-1.018 per round), and the GPU drifted by ~1.5 % over the session (the size range's
  main ran 233.7 at the start and 225.2 at the end). nsys of one size-range pass agrees: `k_entropy_win` 825.4 ->
  824.7 ms in total, the entropy tail 87.9 -> 83.8 ms (`logs/prof/nsys_sr_*_post.txt`). In batch mode the entropy
  CTAs share their SMs with k_recon / k_analysis, so their per-symbol chain is not what sets their speed, and the
  speed-of-light bound for the whole entropy coder there was 2-4 % on the size range and 0 elsewhere.
- **Single-file mode: -15.8 ms (-1.3 %) per 8 MP JPEG**, outside the noise (main against main +0.3 %; the ranges of
  the 12 runs barely overlap). Commit 1 alone measured +0.4 %, inside the noise: its 3.1 ms on this photo is below
  what the cold-process wall time resolves; the rest is commit 2 (the host copy of the record) and run-to-run noise.

## Byte identity

`cmp` of every output against main cde1f74 built on the same instance (`scripts/identity8.sh`: speed-v06's
identity sets and options plus the palette sets at the default, `--palette on`, `--yuv 444 --depth 10` and
`--lossless`; `logs/identity*.txt`). Commit 1 (`~/br`) on every set; the PR head (`~/br2`, both commits) on the
single-file sets, the only mode that runs commit 2's code. The set-up had made only `ll/kodak` of the lossless sets
(mk_ll_sets.py reads `~/br`, which did not exist yet), so the first pass of the gray, monochrome, misc, png16_10,
graphics, WebP and ll/jpeg sets ran with empty or short inputs; `scripts/fix_llsets.sh` made them and
`scripts/rerun8.sh` ran those sets again (`logs/identity_rerun.txt`; the table uses the rerun). The mixed `--batch`
list was made before the fix: 29 entries (speed-v06: 33). `pal_flat_jpg_lossless` and the JPEGs of `odd_lossless`
are refused by `--lossless` in both binaries (skipped).

| set | options | inputs | main files | branch files | differ | skipped (both) | rc | from |
|---|---|---:|---:|---:|---:|---:|---|---|
| sr_default | `default` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_turbo3_nosmall | `--preset turbo3 --small-preset none` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_nosmall | `--small-preset none` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_444 | `--yuv 444` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_10 | `--depth 10` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_444_10 | `--yuv 444 --depth 10` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_base_nosmall | `--preset base --small-preset none` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_slow_nosmall | `--preset slow --small-preset none` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_fast_nosmall | `--preset fast --small-preset none` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_q30 | `--quality 30` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_q90 | `--quality 90` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_q95 | `--quality 95` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_tune_none | `--tune none` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_tiles_2_1_1 | `--tiles 2,1,1` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_default_enc1 | `default` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_default_enc6 | `default` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_slow | `--preset slow` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_444_q90 | `--yuv 444 --quality 90` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_444_q95 | `--yuv 444 --quality 95` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_444_slow | `--yuv 444 --preset slow` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_444_10_q95 | `--yuv 444 --depth 10 --quality 95` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| sr_10_q95 | `--depth 10 --quality 95` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| mono_default | `default` | 47 | 47 | 47 | 0 | 0 | 0/0 | identity_rerun.txt |
| mono_q90 | `--quality 90` | 47 | 47 | 47 | 0 | 0 | 0/0 | identity_rerun.txt |
| mono_q95 | `--quality 95` | 47 | 47 | 47 | 0 | 0 | 0/0 | identity_rerun.txt |
| mono_slow | `--preset slow` | 47 | 47 | 47 | 0 | 0 | 0/0 | identity_rerun.txt |
| mono_nosmall | `--small-preset none` | 47 | 47 | 47 | 0 | 0 | 0/0 | identity_rerun.txt |
| mono_10 | `--depth 10` | 47 | 47 | 47 | 0 | 0 | 0/0 | identity_rerun.txt |
| mono_tune_none | `--tune none` | 47 | 47 | 47 | 0 | 0 | 0/0 | identity_rerun.txt |
| std_q95 | `--quality 95` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| std_slow | `--preset slow` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| thumbs_q95 | `--quality 95` | 280 | 280 | 280 | 0 | 0 | 0/0 | identity.txt |
| thumbs_slow | `--preset slow` | 280 | 280 | 280 | 0 | 0 | 0/0 | identity.txt |
| kodak_q95 | `--quality 95` | 24 | 24 | 24 | 0 | 0 | 0/0 | identity.txt |
| kodak_444_q90 | `--yuv 444 --quality 90` | 24 | 24 | 24 | 0 | 0 | 0/0 | identity.txt |
| rgba_q95 | `--quality 95` | 39 | 39 | 39 | 0 | 0 | 0/0 | identity.txt |
| odd_q95 | `--quality 95` | 30 | 30 | 30 | 0 | 0 | 0/0 | identity.txt |
| odd_slow | `--preset slow` | 30 | 30 | 30 | 0 | 0 | 0/0 | identity.txt |
| std_default | `default` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| std_nosmall | `--small-preset none` | 100 | 100 | 100 | 0 | 0 | 0/0 | identity.txt |
| thumbs_default | `default` | 280 | 280 | 280 | 0 | 0 | 0/0 | identity.txt |
| thumbs_nosmall | `--small-preset none` | 280 | 280 | 280 | 0 | 0 | 0/0 | identity.txt |
| kodak_default | `default` | 24 | 24 | 24 | 0 | 0 | 0/0 | identity.txt |
| kodak_lossless | `--lossless` | 24 | 24 | 24 | 0 | 0 | 0/0 | identity.txt |
| kodak_lossless_ycgco | `--lossless --ycgco-re` | 24 | 24 | 24 | 0 | 0 | 0/0 | identity.txt |
| rgba_default | `default` | 39 | 39 | 39 | 0 | 0 | 0/0 | identity.txt |
| rgba_aq100 | `--alpha-quality 100` | 39 | 39 | 39 | 0 | 0 | 0/0 | identity.txt |
| rgba_lossless | `--lossless` | 39 | 39 | 39 | 0 | 0 | 0/0 | identity.txt |
| rgba_no_alpha | `--no-alpha` | 39 | 39 | 39 | 0 | 0 | 0/0 | identity.txt |
| gray_default | `default` | 9 | 9 | 9 | 0 | 0 | 0/0 | identity_rerun.txt |
| gray_lossless | `--lossless` | 9 | 8 | 8 | 0 | 1 | 0/0 | identity_rerun.txt |
| odd_default | `default` | 30 | 30 | 30 | 0 | 0 | 0/0 | identity.txt |
| odd_444_10 | `--yuv 444 --depth 10` | 30 | 30 | 30 | 0 | 0 | 0/0 | identity.txt |
| odd_lossless | `--lossless` | 30 | 10 | 10 | 0 | 20 | 0/0 | identity.txt |
| misc_default | `default` | 13 | 13 | 13 | 0 | 0 | 0/0 | identity_rerun.txt |
| png16_10_default | `--depth 10` | 8 | 8 | 8 | 0 | 0 | 0/0 | identity_rerun.txt |
| png16_10_lossless | `--lossless` | 8 | 8 | 8 | 0 | 0 | 0/0 | identity_rerun.txt |
| llgraphics_lossless | `--lossless --ycgco-re` | 27 | 27 | 27 | 0 | 0 | 0/0 | identity_rerun.txt |
| webp_default | `default` | 7 | 7 | 7 | 0 | 0 | 0/0 | identity_rerun.txt |
| webp_lossless | `--lossless` | 7 | 7 | 7 | 0 | 0 | 0/0 | identity_rerun.txt |
| mixed_list | `default` | 29 | 29 | 29 | 0 | 0 | 0/0 | identity.txt |
| mixed_list_444_10 | `--yuv 444 --depth 10` | 29 | 29 | 29 | 0 | 0 | 0/0 | identity.txt |
| lljpeg_set | `default` | 25 | 25 | 25 | 0 | 0 | 0/0 | identity_rerun.txt |
| pal_screen_default | `default` | 40 | 40 | 40 | 0 | 0 | 0/0 | identity.txt |
| pal_screen_on | `--palette on` | 40 | 40 | 40 | 0 | 0 | 0/0 | identity.txt |
| pal_screen_444_10 | `--yuv 444 --depth 10` | 40 | 40 | 40 | 0 | 0 | 0/0 | identity.txt |
| pal_screen_lossless | `--lossless` | 40 | 40 | 40 | 0 | 0 | 0/0 | identity.txt |
| pal_flat_png_default | `default` | 24 | 24 | 24 | 0 | 0 | 0/0 | identity.txt |
| pal_flat_png_on | `--palette on` | 24 | 24 | 24 | 0 | 0 | 0/0 | identity.txt |
| pal_flat_png_444_10 | `--yuv 444 --depth 10` | 24 | 24 | 24 | 0 | 0 | 0/0 | identity.txt |
| pal_flat_png_lossless | `--lossless` | 24 | 24 | 24 | 0 | 0 | 0/0 | identity.txt |
| pal_flat_jpg_default | `default` | 24 | 24 | 24 | 0 | 0 | 0/0 | identity.txt |
| pal_flat_jpg_on | `--palette on` | 24 | 24 | 24 | 0 | 0 | 0/0 | identity.txt |
| pal_flat_jpg_444_10 | `--yuv 444 --depth 10` | 24 | 24 | 24 | 0 | 0 | 0/0 | identity.txt |
| pal_flat_jpg_lossless | `--lossless` | 24 | 0 | 0 | 0 | 24 | 0/0 | identity.txt |
| single-file mode (br1) | `default` | 10 | 10 | 10 | 0 | | 0 failed runs | identity_single_br1.txt |
| single-file mode (br1) | `--yuv 444 --depth 10` | 10 | 10 | 10 | 0 | | 0 failed runs | identity_single_br1.txt |
| single-file mode (br1) | `--preset slow` | 10 | 10 | 10 | 0 | | 0 failed runs | identity_single_br1.txt |
| single-file mode (br2) | `default` | 10 | 10 | 10 | 0 | | 0 failed runs | identity_single_br2.txt |
| single-file mode (br2) | `--yuv 444 --depth 10` | 10 | 10 | 10 | 0 | | 0 failed runs | identity_single_br2.txt |
| single-file mode (br2) | `--preset slow` | 10 | 10 | 10 | 0 | | 0 failed runs | identity_single_br2.txt |
| **all (75 batch runs, 6 single-file runs)** | | | **5052** | | **0** | | | |

## Decoder gate, tests, sanitizers (commit 1 unless noted)

- **Decoder gate** (`tests/decoder_gate.py`, dav1d 1.4.1 and aomdec 3.8.2; `logs/gate.txt`): all N/N identical in both
  decoders: 30 size-range photos at qindex 60 / 120 / 200 / 13 and 4:4:4 10-bit at 120; 28 thumbnails at 60 / 120;
  30 odd sizes; 4 gray + alpha PNGs (3 alpha items); 39 RGBA PNGs (35 alpha items, `--expect-alpha`); Kodak
  `--lossless`; the palette sets screen (40), flat_png (24), flat_jpg (24) at the default, and screen and flat_png
  `--lossless` (flat_jpg `--lossless`: JPEGs refused, nothing to gate).
  After the set fix, the PR head on gray (9 PNGs, 3 alpha items), monochrome (47) and ll/misc (13, 2 alpha items) at
  qindex 120: all N/N (`logs/gate_post.txt`).
- **ctest** 9/9 passed; **gpu_fuzz_test** seeds 1-400 and 77-276 (4:4:4 / 10-bit / lossless mix): 0 mismatches
  (`logs/ctest.txt`); the PR head again (`logs/fuzz_br2.txt`), plus `--check-entropy` (GPU entropy == CPU writer)
  on the 10 single-file inputs.
- **compute-sanitizer** 2026.1 (`logs/sanitizer.txt`, `logs/racecheck_summary.txt`): memcheck and synccheck on 2
  size-range photos (2.1 MP turbo with 4 x 4 tiles, 0.75 MP base preset): 0 errors, outputs equal to the plain run.
  racecheck (analysis mode) on 2 thumbnails, main and branch: every report is inside `k_entropy_win`, the hand-off
  of the shared rings through the volatile window counters (documented in docs/RESULTS.md). Main has the producer
  ring writes against the coder's reads and the counter pair; the branch has the same for the first ring
  (producer -> B1), the same again for the second ring (B1 -> B2: lines 1344 / 1347 against 1369 / 1373 / 1380) and
  the counters (1261 / 1262). No other kernel reports anything.

## Blackwell (sm_120)

Not measured on Blackwell. The change does not depend on register or occupancy tuning (no minimum-blocks launch
bound; 40 registers and no local memory as before). The entropy CTA grows from 64 to 96 threads and from 13.7 to
14.7 KB of shared memory; next to `k_recon` (16 K registers, 23.5 KB per CTA) an SM that hosts one entropy CTA still
fits three `k_recon` CTAs on the L4, as before. The gain relies on the producer, B1 and B2 being separate warps
(separate schedulers) and on the latency of the integer chain (`IMAD`, `FLO`, shifts), which sm_120 shares with sm_89.
