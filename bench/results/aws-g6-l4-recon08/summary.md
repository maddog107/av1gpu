# k_recon speed round (recon08) on an NVIDIA L4

AWS g6.8xlarge, on demand, because no subnet had g6.8xlarge spot capacity. NVIDIA L4, driver 595.112.07, CUDA 13.2,
EPYC 7R13 with 32 vCPUs, 2026-11-03 (`logs/env.txt`). Both builds are made on the instance with
`-DCMAKE_CUDA_ARCHITECTURES=89`: perf/recon (the code of 1d432f5, tested as abf26f5, the same encoder sources) and
main cde1f74 (v0.8.0). Only byte-identical changes count: every output of every set below is `cmp`-identical to
main's.

## Where the reconstruction time goes

These are main's numbers with `--prof`, one run per set (`logs/prof1/`). With the extra profile ticks of
`scripts/prof_ticks.diff` the luma final splits in two (`logs/prof2/`).

- **Size range, default.** The stage times are analysis 638 ms, recon 2343 ms, postfilter 119 ms, GPU total
  3175 ms, so recon is about 74 % of the GPU stage time. Its CTA-cycles fill only 16 % of the 58 SMs x 4 recon CTAs
  over the recon time. The SB wavefront is latency-bound, and the SMs are mostly idle or busy with the other
  encoders' kernels.
- **Phases.** The time splits as follows:

  | phase | share |
  |---|---:|
  | luma setup | 7.1 % |
  | luma pass 1 | 0.5 % |
  | luma pass 2 (tx-type search) | 13.8 % |
  | luma final quantization (CTA-wide, 32x32 / 64x64) | 8.2 % |
  | luma prediction + inverse transform + record | 23.2 % |
  | chroma edges + CfL | 5.7 % |
  | chroma SATD screening | 9.4 % |
  | chroma RD | 19.3 % |
  | chroma final | 11.6 % |
  | maps + block record | 2.0 % |
- **Standard set.** Recon CTA-cycles are 1.5 % of the capacity: the set is mostly small images on the small-image
  encoder.
- **Thumbnails.** They run on the small-image encoder (base preset), whose phases `--prof` does not aggregate.
  Stage analysis is 323 ms and recon 321 ms of 733 ms.

## Changes

1. **`rc_inv.cuh`: the inverse transforms in 32-bit arithmetic, the dequantization by a shift** (f123194).
   - `cta_inverse_add` runs each row and each column as one thread's serial chain (H or W threads of the CTA), which
     puts it on the critical path of every coded tx block. The chain used av1core's int64 butterflies.
   - With in-range inputs nothing needs more than 31 bits. The values are within r bits (r = BitDepth + 8 <= 18 for
     the rows, 16 for the columns), so products are below 2^29 and sums below 2^30, and iadst4's sums are checked at
     r + 12 bits.
   - Until the first failed range check, the values are therefore exactly the int64 ones, and the first failed check
     fails here too. A transform with a failed check is discarded and redone with halved levels, so the residual and
     the flags are the same.
   - The arithmetic wraps through `uint32_t`, so the values after a failed check are defined (and unused).
   - `av1g_dequant_coef`'s division by 1, 2 or 4 is of a non-negative value, so it is a shift.
2. **`dev_tx.cuh`: the forward transform loops with a compile-time trip count** (1d432f5).
   - `fwd_col_pass` / `fwd_row_pass` are the runtime-size transform of `warp_eval_rt`. k_recon uses it in the luma
     tx-type search, chroma RD, the tx-size trial and the final RD quantization, and k_pal uses it too.
   - Both passes are templated on the outputs per lane. Below 32 that is also the transform length, so the inner
     loop now unrolls fully, and the coefficient loads of every k go out together.
   - The fmaf order of every output is the same.

ptxas on sm_89 (`logs/resusage.txt`, `scripts/resusage.sh`): every k_recon instantiation still uses 128 registers
(`__launch_bounds__(128, 4)`, unchanged), and shared memory is unchanged (23,472 B for 4:2:0 8-bit). The spill
stores / loads (bytes) of k_recon<1,8>'s non-inlined functions, main against this branch:

| function | main | perf/recon |
|---|---:|---:|
| `cta_inverse_add` | 1212 / 1360 | 196 / 392 |
| `rc_chroma` | 1528 / 3484 | 768 / 2496 |
| `rc_luma` | 4712 / 11328 | 5200 / 12292 |

`rc_final_quant_rd` is a separate function now (240 / 452). Before, it was inlined. The kernel's stack frame grows
from 3120 to 3392 bytes.

## Byte identity

Every output is compared with `cmp` (`scripts/identity8.sh`, the v0.6 release check's `identity.sh` plus the palette
sets of `bench/results/aws-g6-l4-palette-study/scripts/prep_sets.py`; `logs/identity.txt` has a digest per set).
The sets are speed-v06's: the size range and standard JPEGs at default, the presets, qualities, 4:4:4, 10-bit,
4:4:4 10-bit, tiles and encoder counts; thumbnails; monochrome; Kodak; RGBA; odd sizes 1 x 1 ... 4097 x 17; the
lossless sets; WebP; the mixed `--batch` list; and single-file mode. On top of them come the palette sets (screen,
flat_png, flat_jpg) at default, `--palette on`, `--yuv 444 --depth 10` and `--lossless`. `--lossless` refuses JPEGs,
so the flat_jpg lossless run skips all 24 files in both builds. During the work, each step also passed a quick
identity check on the size range, the thumbnails and the standard set (`logs/spd8/`, `scripts/bnew.sh`).

| set | options | inputs | main files | perf/recon files | differ | skipped (both) |
|---|---|---:|---:|---:|---:|---:|
| size range (100 JPEGs) | default | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--preset turbo3 --small-preset none` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--small-preset none` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--yuv 444` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--depth 10` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--yuv 444 --depth 10` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--preset base --small-preset none` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--preset slow --small-preset none` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--preset fast --small-preset none` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--quality 30` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--quality 90` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--quality 95` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--tune none` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--tiles 2,1,1` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | default, `GP_ENCODERS=1` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | default, `GP_ENCODERS=6` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--preset slow` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--yuv 444 --quality 90` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--yuv 444 --quality 95` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--yuv 444 --preset slow` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--yuv 444 --depth 10 --quality 95` | 100 | 100 | 100 | 0 | 0 |
| size range (100 JPEGs) | `--depth 10 --quality 95` | 100 | 100 | 100 | 0 | 0 |
| monochrome (47: 40 gray JPEGs of the size range, gray PNGs and JPEGs) | default | 47 | 47 | 47 | 0 | 0 |
| monochrome (47: 40 gray JPEGs of the size range, gray PNGs and JPEGs) | `--quality 90` | 47 | 47 | 47 | 0 | 0 |
| monochrome (47: 40 gray JPEGs of the size range, gray PNGs and JPEGs) | `--quality 95` | 47 | 47 | 47 | 0 | 0 |
| monochrome (47: 40 gray JPEGs of the size range, gray PNGs and JPEGs) | `--preset slow` | 47 | 47 | 47 | 0 | 0 |
| monochrome (47: 40 gray JPEGs of the size range, gray PNGs and JPEGs) | `--small-preset none` | 47 | 47 | 47 | 0 | 0 |
| monochrome (47: 40 gray JPEGs of the size range, gray PNGs and JPEGs) | `--depth 10` | 47 | 47 | 47 | 0 | 0 |
| monochrome (47: 40 gray JPEGs of the size range, gray PNGs and JPEGs) | `--tune none` | 47 | 47 | 47 | 0 | 0 |
| standard (100 JPEGs) | `--quality 95` | 100 | 100 | 100 | 0 | 0 |
| standard (100 JPEGs) | `--preset slow` | 100 | 100 | 100 | 0 | 0 |
| thumbnails (280 JPEGs) | `--quality 95` | 280 | 280 | 280 | 0 | 0 |
| thumbnails (280 JPEGs) | `--preset slow` | 280 | 280 | 280 | 0 | 0 |
| Kodak (24 PNGs) | `--quality 95` | 24 | 24 | 24 | 0 | 0 |
| Kodak (24 PNGs) | `--yuv 444 --quality 90` | 24 | 24 | 24 | 0 | 0 |
| RGBA (39 PNGs) | `--quality 95` | 39 | 39 | 39 | 0 | 0 |
| odd sizes (30: PNG, 4:2:0 and 4:4:4 JPEG) | `--quality 95` | 30 | 30 | 30 | 0 | 0 |
| odd sizes (30: PNG, 4:2:0 and 4:4:4 JPEG) | `--preset slow` | 30 | 30 | 30 | 0 | 0 |
| standard (100 JPEGs) | default | 100 | 100 | 100 | 0 | 0 |
| standard (100 JPEGs) | `--small-preset none` | 100 | 100 | 100 | 0 | 0 |
| thumbnails (280 JPEGs) | default | 280 | 280 | 280 | 0 | 0 |
| thumbnails (280 JPEGs) | `--small-preset none` | 280 | 280 | 280 | 0 | 0 |
| Kodak (24 PNGs) | default | 24 | 24 | 24 | 0 | 0 |
| Kodak (24 PNGs) | `--lossless` | 24 | 24 | 24 | 0 | 0 |
| Kodak (24 PNGs) | `--lossless --ycgco-re` | 24 | 24 | 24 | 0 | 0 |
| RGBA (39 PNGs) | default | 39 | 39 | 39 | 0 | 0 |
| RGBA (39 PNGs) | `--alpha-quality 100` | 39 | 39 | 39 | 0 | 0 |
| RGBA (39 PNGs) | `--lossless` | 39 | 39 | 39 | 0 | 0 |
| RGBA (39 PNGs) | `--no-alpha` | 39 | 39 | 39 | 0 | 0 |
| gray and gray + alpha PNGs (9) | default | 9 | 9 | 9 | 0 | 0 |
| gray and gray + alpha PNGs (9) | `--lossless` | 9 | 8 | 8 | 0 | 1 |
| odd sizes (30: PNG, 4:2:0 and 4:4:4 JPEG) | default | 30 | 30 | 30 | 0 | 0 |
| odd sizes (30: PNG, 4:2:0 and 4:4:4 JPEG) | `--yuv 444 --depth 10` | 30 | 30 | 30 | 0 | 0 |
| odd sizes (30: PNG, 4:2:0 and 4:4:4 JPEG) | `--lossless` | 30 | 10 | 10 | 0 | 20 |
| ll/misc (13: palette, RGBA, odd, 4100 × 40, 6144 × 4096) | default | 13 | 13 | 13 | 0 | 0 |
| 16-bit PNGs with 10 bits (8) | `--depth 10` | 8 | 8 | 8 | 0 | 0 |
| 16-bit PNGs with 10 bits (8) | `--lossless` | 8 | 8 | 8 | 0 | 0 |
| graphics PNGs (23) | `--lossless --ycgco-re` | 27 | 27 | 27 | 0 | 0 |
| WebP (7: lossless, lossy, with alpha) | default | 7 | 7 | 7 | 0 | 0 |
| WebP (7: lossless, lossy, with alpha) | `--lossless` | 7 | 7 | 7 | 0 | 0 |
| mixed `--batch` LIST (33) | default | 33 | 33 | 33 | 0 | 0 |
| mixed `--batch` LIST (33) | `--yuv 444 --depth 10` | 33 | 33 | 33 | 0 | 0 |
| ll/jpeg (25: 4:2:0, 4:4:4, gray, 4:2:2, odd) | default | 25 | 25 | 25 | 0 | 0 |
| palette: screen (40) | default | 40 | 40 | 40 | 0 | 0 |
| palette: screen (40) | `--palette on` | 40 | 40 | 40 | 0 | 0 |
| palette: screen (40) | `--yuv 444 --depth 10` | 40 | 40 | 40 | 0 | 0 |
| palette: screen (40) | `--lossless` | 40 | 40 | 40 | 0 | 0 |
| palette: flat_png (24) | default | 24 | 24 | 24 | 0 | 0 |
| palette: flat_png (24) | `--palette on` | 24 | 24 | 24 | 0 | 0 |
| palette: flat_png (24) | `--yuv 444 --depth 10` | 24 | 24 | 24 | 0 | 0 |
| palette: flat_png (24) | `--lossless` | 24 | 24 | 24 | 0 | 0 |
| palette: flat_jpg (24) | default | 24 | 24 | 24 | 0 | 0 |
| palette: flat_jpg (24) | `--palette on` | 24 | 24 | 24 | 0 | 0 |
| palette: flat_jpg (24) | `--yuv 444 --depth 10` | 24 | 24 | 24 | 0 | 0 |
| palette: flat_jpg (24) | `--lossless` | 24 | 0 | 0 | 0 | 24 |
| single-file mode, no `--batch` (10 files) | default | 10 | 10 | 10 | 0 |  |
| single-file mode, no `--batch` (10 files) | `--yuv 444 --depth 10` | 10 | 10 | 10 | 0 |  |
| single-file mode, no `--batch` (10 files) | `--preset slow` | 10 | 10 | 10 | 0 |  |
| **all (78 runs: 75 batch, 3 single-file)** | | | **5,030** | **5,030** | **0** | |

## Decoder gate, tests, sanitizers

- **Decoder gate** (`tests/decoder_gate.py`, dav1d 1.4.1 and aomdec 3.8.2, `scripts/checks8.sh`, `logs/gate.txt`):
  16 runs, all PASS, every encode identical in both decoders:
  - 30 size-range photos at qindex 60 / 120 / 200, 13 / 26, 4:4:4 10-bit at 120, and `--preset slow`;
  - 28 thumbnails at 60 / 120;
  - the odd sizes in 4:2:0 and in 4:4:4 10-bit;
  - gray and gray + alpha;
  - the 47 monochrome inputs at 26 / 120;
  - the 39 RGBA PNGs (alpha items identical);
  - the palette sets screen / flat_png / flat_jpg at default and `--palette on`.
- **ctest:** 9/9 passed. `gpu_fuzz_test` (seeds 1-400, and 77-276 with more 4:4:4 / 10-bit / lossless),
  `gpu_postfilter_test`, `gpu_dsp_test` and `varboost_test` show 0 mismatches (`logs/ctest.txt`). None of these runs
  k_recon itself (the fuzz tests drive the entropy coder, and gpu_dsp_test drives av1core's dsp), so for these
  changes the identity runs and the gate are the real tests.
- **compute-sanitizer 2026.1** on `av1gpu --batch` with the 3 photos of the v0.6 release check, at default and at
  `--yuv 444 --depth 10`:
  - memcheck: 0 errors (`logs/memcheck.txt`);
  - synccheck: 0 errors (`logs/synccheck.txt`);
  - racecheck: only the known `k_entropy_win` / `vload` / `vstore` pairs in gpu_entropy.cu (docs/RESULTS.md,
    Correctness), and no `k_recon` report (`logs/race_new.txt`). Neither change touches synchronisation.

## Speed

`scripts/spd8.py` uses the method of `../aws-g6-l4-speed-v04/`:
- Each set is repeated to at least 1,000 MP. Each binary gets a warm-up and 3 timed runs, done twice with the binary
  order alternating, and the result is the median of the 6 runs (av1gpu's `--stats`).
- The noise run is main again: the same binary from a second path, in the same rounds.
- Single-file mode times one cold process per file (`av1gpu in.jpg out.avif`, CUDA start-up included): a warm-up,
  then 12 runs per binary alternating, median wall time.

Raw runs are in `logs/spd8/final.jsonl` and the summary in `logs/speed_final.txt`. Nothing else ran on the GPU
(`logs/final.log`).

| set | repeated to | main | perf/recon | change | main again (noise) | noise |
|---|---|---:|---:|---:|---:|---:|
| size range (100 JPEGs), default | 300 files, 1434 MP (3x) | 237.69 MP/s | 247.32 | +4.1 % | 233.84 | -1.6 % |
| standard (100 JPEGs), default | 1300 files, 1023 MP (13x) | 75.50 MP/s | 78.07 | +3.4 % | 74.64 | -1.1 % |
| thumbnails (280 JPEGs), default | 6720 files, 1019 MP (24x) | 411.35 img/s | 424.59 | +3.2 % | 409.75 | -0.4 % |

| file | main | perf/recon | change | main again (noise) | noise |
|---|---:|---:|---:|---:|---:|
| single JPEG, cold process: 3879c92c11164a6a_large.jpg (3264, 2448) | 1256 ms | 1248 ms | -0.6 % | 1252 ms | -0.3 % |
| single JPEG, cold process: 1068f4f5ac7e8d01_typical.jpg (1280, 1920) | 1081 ms | 1074 ms | -0.6 % | 1082 ms | +0.1 % |
| single thumbnail, cold process: d512_92a594bfb7613b85_huge.jpg (512, 384) | 1107 ms | 1107 ms | +0.0 % | 1112 ms | +0.4 % |

- **Batch sets.** These are outside the noise:
  - size range: +4.1 %, with the noise run at -1.6 % (against the second main run it is +5.8 %);
  - standard set: +3.4 %, noise -1.1 %;
  - thumbnails: +3.2 % img/s, noise -0.4 %.

  perf/recon is ahead of both main runs in every round. The two rounds of one binary differ by up to 2 %
  (thumbnails).
- **Single-file mode.** The 8 MP and 2.5 MP JPEGs take about 7 ms less (-0.6 %), against a noise of 0.3 % and 0.1 %.
  The cold process is about 1.1 s of CUDA start-up and decode, so the recon gain hardly shows there. The thumbnail
  is unchanged.

### Per commit (quick A/B, size range, same method, main in every run)

The quick runs measured the size range at 1,434 MP, and the standard set at 315 MP with `MINMP=250`
(`logs/spd8/q_*.jsonl`).

| build | size range | standard |
|---|---|---|
| commit 1 (c1) | +4.8 %, +4.5 %, +5.0 %, +4.0 % (four runs) | +3.0 %, +2.8 % |
| commit 1 + 2 (c4 = this branch) | +5.4 % (c1 in the same run: +4.0 %) | +3.3 % (c1: +2.8 %) |

## Tried and dropped

These changes were byte-identical on the quick sets, but no faster (`scripts/dropped_barriers.diff`, NOTES.md):

- **(c2)** The range flags of `cta_inverse_add` as barrier reductions (`__syncthreads_or`, `bar.red.or`), and the
  luma levels used in place. Against c1 on the size range: -0.8 %.
- **(c3)** About 12 fewer CTA barriers per block:
  - the pass-1 size known per thread, and the pass-2 list and edges behind one barrier;
  - no trailing barriers after the luma record, the chroma pick and the U / V counters;
  - the CfL average and alpha computed per thread, and the resets before the first chroma barrier;
  - txb_ctx hoisted.

  Against c1: size range -1.0 %, standard -0.6 %.

The phases are bound by single-thread / single-warp chains (global loads, serial loops), not by the barriers between
them. The inverse transform was such a chain, and commit 1 shortened it.

## Blackwell (sm_120)

Not measured here. Neither change tunes registers or occupancy, and the launch bounds and shared memory are
unchanged. Commit 1 lowers register pressure. Commit 2 unrolls loops, and on sm_89 `rc_luma`'s spills grow by about
10 %, which the measured gain already includes. ptxas may allocate differently for sm_120, so the final g7 check
should confirm commit 2.
