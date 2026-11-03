# pipe08 notes (working log; summary.md has the results)

Agent pipe08, branch `perf/pipeline`, area: single-image path and start-up, thumbnails, host pipeline.
Instance `av1gpu-pipe08`: g6.4xlarge on-demand (spot had no capacity), us-east-1, NVIDIA L4, driver 595.112.07,
CUDA 13.2. main = cde1f74 built as `~/old`, the branch as `~/new` (k_recon objects linked from `~/old` with
`AV1GPU_RECON_OBJECTS`: the branch does not touch them). States after each commit: `~/c1` .. `~/c6`.

## Where a cold `av1gpu in.jpg out.avif` spends its time (L4, main, 320 x 213 JPEG)

Stage timestamps (a temporary instrumented copy, `scripts/instrument.py`, never committed; median of 5):

| phase | 32 queues (main) | 1 queue |
|---|---:|---:|
| exec to main() | 3.5 ms | 3.4 |
| cudaSetDevice (context creation) | 373.5 | 97.7 |
| GpEncoder::init: gp_init_tables (table uploads = lazy loads of 12 kernel modules) | 91.9 | 91.1 |
| GpEncoder::init: variance-boost LUT (256 x 1024 entries on the host) | 24.1 | 24.1 |
| ingest_create: nvJPEG hardware decoder (+ 6000 x 6000 pre-allocation) | 76.0 | 75.6 |
| ingest_create: 256 MiB pinned arena (memset + cudaHostRegister) | 24.2 | 24.1 |
| decode | 9.8 | 9.6 |
| second encoder (small-image preset): tables 2.4 + LUT 23.8 | 26.2 | 26.1 |
| encode (k_recon wavefront: 11 steps x 4.9 ms = 53 ms of it) | 76.7 | 77.3 |
| ingest_destroy (35 ms of it: freeing nvJPEG's pre-allocation) | 76.3 | 75.9 |
| GPU entropy + write | 10.3 | 10.3 |
| atexit to process end (CUDA teardown, kernel) | 300.8 | 69.6 |
| **total** | **1094** | **586** |

- `CUDA_DEVICE_MAX_CONNECTIONS=32` (main sets it for every run) is the biggest item: each hardware queue costs
  ~12 ms at context creation and ~7 ms at exit on this driver (GSP RPCs). A trivial CUDA program (one empty kernel):
  645 ms with 32, 252 with 8 (CUDA's default), 153 with 1. The release v0.8.0 binary: 1116 ms -> 598 ms with
  `GP_MAX_CONNECTIONS=1`, same output.
- Short lists (`--batch`, warm cache, main, wall incl. start-up, median of 5): pair (2.4 MP + 512 x 384 thumbnail)
  1345 / 937 / 874 / 850 ms with 32 / 8 / 4 / 1 queues; 5 size-range photos 1572 / 1171 / 1104 / 1086; 10 thumbnails
  1279 / 884 / 816 / 777. 50 photos (contended run, rough): 2780 / 2441 / 2951 (4 queues serialise real work).
- The binary's size (540 MB release, 100 MB sm_89 build) did not matter on the L4 (page cache warm): both ~1.10 s.
- nsys (`-t cuda,osrt`): the thumbnail's GPU time is 66 ms, 53 ms of it 11 k_recon launches (the wavefront of the
  base small-image preset, ~4.9 ms per step): recon08's area.
- perf: 10 % of all samples in `av1f_vb_build_luts_bd` (two encoders in single mode, 24 ms each).

## Changes (one commit each)

1. Hardware queues: 1 in single-file mode and for `--batch` lists of at most 8 encodes, 32 otherwise
   (`GP_MAX_CONNECTIONS` still overrides).
2. `wantRecon` off in single-file mode unless `--check` / `--debug-outputs`; the CDEF apply (output only feeds the
   reconstruction download) is skipped without `wantRecon` (also in batch mode, where it ran for nothing).
3. Variance-boost LUT: only the row of the encoder's qindex (k_sb_stats reads lut[base_q_idx][var] with base_q_idx =
   the encoder's qindex). Checked: all 3.67 M entries (q 0..255, strengths 0..300, 8 and 10 bit) equal
   av1f_vb_build_luts_bd's.
4. Lazy pinned arena (`ingest_config.lazy_arena`) in single-file mode: baseline JPEGs never use it.
5. Single-file mode decodes the JPEG in a thread while GpEncoder::init runs.
6. Single-file mode on Linux ends with `_exit` after the outputs are closed; no ingest_destroy / CUDA teardown.

## Not done (measured, left for later)

- gp_init_tables uploads the tables into all 12 kernel modules on every run (each upload is a lazy module load):
  in single-file mode the 4:4:4 / 10-bit k_recon variants, lossless and palette modules are not needed for an 8-bit
  4:2:0 photo. A per-module upload on first use could save ~30-40 ms, but touches k_misc.cu and every launcher.
- nvJPEG hardware-decoder setup (~72 ms) and its 6000 x 6000 pre-allocation (freeing it: 35 ms): skipping the
  pre-allocation would also change which images take the CPU route (> 6000 px), so not done.

## Progress (UTC)

- 05:30 instance up; 05:50 main / branch built; 05:53 identity (commits 1-4) + gate + ctest + memcheck started.
- 06:04 identity.sh done (all differ 0; but see below), gate / ctest / memcheck done (all pass).
- 06:06 ~/new rebuilt with commits 1-6; ident_fix.sh (sets remade, rerun) and ident_single.sh: all identical.
- 06:27-06:52 speed: spd8.py single (main, c1..c6, main copy), batch (size range, standard, thumbnails; new / main /
  main copy), split (c1, c2, c3 on the size range). summary.md written; PR opened; instance torn down.

## Problems on the way

- `mk_ll_sets.py` needs `~/br` (a source tree); without it `~/ll/*` (except kodak) and the sets built from them
  were empty on the first identity pass. `scripts/ident_fix.sh` makes them (`~/br -> ~/old`) and reruns those sets.
