# av1gpu start-up and the single-image path (NVIDIA L4, g6.4xlarge, 2026-11-03)

Agent pipe08, branch `perf/pipeline` against main **cde1f74** (v0.8.0). Production converts every forum upload (a JPEG
and its thumbnail) with one av1gpu process, so the fixed cost of a process matters more than batch throughput. This
round measures where a cold `av1gpu in.jpg out.avif` spends its time and removes what it can without changing a
byte of output.

**Result:** a cold single JPEG (`av1gpu in.jpg out.avif`) takes **389–453 ms instead of 1,104–1,186 ms** (2.6–2.8×, 320 × 213 to 10 MP); a `--batch` of a photo and its thumbnail **718 instead of 1,357 ms**. Long batches: the size range **+5.1 %** (241.8 against 230.1 MP/s; noise 0.6 %), the standard set and the thumbnails +1.6 % and +1.1 %, inside the noise. Every output file is identical to main's (5,654 / 5,654).

## Setup

One g6.4xlarge on-demand instance (spot had no capacity): NVIDIA L4 (sm_89, 23 GB, 72 W), driver 595.112.07,
CUDA 13.2 (V13.2.51), AMD EPYC 7R13 with 16 vCPUs, Ubuntu 24.04, kernel 7.0.0-1014-aws. main and the branch built on
the instance with `-DCMAKE_CUDA_ARCHITECTURES=89`; the branch links main's k_recon objects (`AV1GPU_RECON_OBJECTS`;
it does not touch them). `~/c1` .. `~/c6` are the trees after each commit, for the per-commit timings. The published
v0.8.0 Linux binary (CUDA 13.4) was measured too. Scripts in `scripts/` (`setup.sh`, `ident_all.sh`, `ident_fix.sh`,
`ident_single.sh`, `checks2.sh`, `spd8.py`, `wall.py`), logs in `logs/`.

## Where the time goes (main)

`logs/startup_phases.txt`: a temporary instrumented copy (`scripts/instrument.py`, not committed), strace, perf and
nsys. A 320 × 213 JPEG, cold process, warm page cache, median of 5:

| phase | main | main with 1 hardware queue |
|---|---:|---:|
| exec to `main()` | 3.5 ms | 3.4 |
| `cudaSetDevice` (context) | 373.5 | 97.7 |
| `gp_init_tables` (12 table uploads = 12 lazy kernel-module loads) | 91.9 | 91.1 |
| variance-boost LUT, all 256 × 1,024 entries on the host | 24.1 | 24.1 |
| nvJPEG hardware decoder + 6000 × 6000 pre-allocation | 76.0 | 75.6 |
| 256 MiB pinned staging arena | 24.2 | 24.1 |
| decode | 9.8 | 9.6 |
| second encoder (small-image preset): tables 2.4 + LUT 23.8 | 26.2 | 26.1 |
| encode | 76.7 | 77.3 |
| `ingest_destroy` (35 ms of it: nvJPEG freeing its pre-allocation) | 76.3 | 75.9 |
| GPU entropy, write | 10.3 | 10.3 |
| exit (CUDA teardown, kernel) | 300.8 | 69.6 |
| **total** | **1,094** | **586** |

- **Hardware queues.** main sets `CUDA_DEVICE_MAX_CONNECTIONS=32` for every run. The driver creates every queue
  with the context and destroys it at exit: about 12 ms and 7 ms each on this driver. A trivial CUDA program (one
  empty kernel) takes 645 ms with 32 queues, 252 with 8 (CUDA's default) and 153 with 1. That was half of a cold
  single image. The v0.8.0 release binary: 1,116 ms with its 32 queues, 598 ms with `GP_MAX_CONNECTIONS=1`, same
  output. `GP_MAX_CONNECTIONS` already works in v0.8.0.
- **Binary size** did not matter on the L4: the 540 MB release binary and the 100 MB sm_89 build start equally fast
  (1,095 / 1,100 ms; warm page cache).
- **The encode of a thumbnail** is 77 ms, 53 ms of it 11 k_recon launches of ~4.9 ms each (the SB wavefront of the
  `base` small-image preset; nsys). That is recon08's area.
- **CPU:** 0.11 s user and 0.94 s system time for 1.18 s of wall time (release binary, 32 queues): driver ioctls and
  the kernel's teardown, not av1gpu's code.

## Changes (one commit each)

1. **One hardware queue for a single image and for `--batch` lists of at most 8 encodes**; longer lists keep 32.
   `GP_MAX_CONNECTIONS` still overrides. Short lists on main with 32 / 8 / 4 / 1 queues: photo + thumbnail 1,345 /
   937 / 874 / 850 ms, 5 photos 1,572 / 1,171 / 1,104 / 1,086, 10 thumbnails 1,279 / 884 / 816 / 777.
2. **No CDEF apply and no reconstruction download unless an output reads the reconstruction.** The CDEF apply
   only fed the reconstruction download (`wantRecon`); the CDEF search and `cdef_idx` come from
   `av1f_lf_search_async_pf` / `gp_launch_set_cdef`. Batch mode had `wantRecon` off and still ran the apply; single
   mode had it on (the default) and downloaded the reconstruction of every image. Now only `--check` and
   `--debug-outputs` (single) and `GP_WANT_RECON` (batch, set by `--debug-outputs`, which the decoder gate uses)
   turn it on.
3. **Variance-boost LUT: only the row of the encoder's qindex.** `k_sb_stats` reads `lut[base_q_idx][var]`, and every
   frame of an encoder has `base_q_idx` = its qindex. The row comes from the same `av1f_varboost_qindex()`; all
   3,670,016 entries (q 0–255, strengths 0–300, 8 and 10 bits) equal `av1f_vb_build_luts_bd`'s.
4. **Single image: the 256 MiB staging arena is pinned on first use** (`ingest_config.lazy_arena`). Only CPU-route
   JPEGs (progressive, arithmetic, > 6000 px) use it; batch mode keeps pinning it up front (#36).
5. **Single image: the JPEG is decoded in a thread while the encoder initialises** (the nvJPEG set-up overlaps the
   table uploads).
6. **Single image, Linux: `_exit` after the outputs are closed**, without `ingest_destroy` and CUDA's atexit
   teardown; the kernel frees it all at exit, faster. Windows keeps the orderly end.

Shared header: `ingest/src/ingest_api.h` gets one field at the end of `ingest_config` (`lazy_arena`, 0 by default;
`ingest_default_config` zeroes the struct). No other header changes.

## Byte identity

**5,654 of 5,654 output files identical** (`cmp`), 101 runs, main cde1f74 against the branch, both built on this
instance (CUDA 13.2, sm_89):

- speed-v06's whole identity list (`scripts/ident_all.sh` runs its `identity.sh`): the size range (22 settings:
  presets, qualities, 4:4:4, 10-bit, 4:4:4 10-bit, tiles, `--tune none`, 1 and 6 encoders), the standard set,
  280 thumbnails, monochrome (47), Kodak, RGBA (39), gray + alpha, odd sizes 1 × 1 … 4097 × 17, the lossless sets,
  16-bit PNGs, graphics, WebP, the mixed `--batch` list, `ll/jpeg`, and single-file mode (10 files × 3 settings);
- the palette sets (screen 40, flat_png 24, flat_jpg 24) at default, `--palette on`, `--yuv 444 --depth 10` and
  `--lossless` (JPEGs are refused with `--lossless`, by both);
- short `--batch` lists, which now get one hardware queue: the mixed list in slices of 8, default and 4:4:4 10-bit,
  and the production pair (a photo and its 512 × 384 thumbnail), default, 4:4:4 10-bit and `--preset slow`;
- single-file mode on 97 files from every set (thumbnails, size range, standard, RGBA, gray, odd, WebP, screen,
  flat_jpg, mono, 16-bit, misc, `ll/jpeg`) at default, `--palette on`, `--yuv 444 --depth 10`, `--preset slow`,
  `--quality 90`, and the 47 non-JPEGs at `--lossless` (96 / 45 output files: two inputs share the name `g0.png`);
- error exits in single-file mode (missing file, a JPEG with `--lossless`, a crop): same return codes and messages.

Two binaries were checked. The batch-mode sets ran on the branch at commit 4 (`identity.txt`, `ident_extra.txt`):
commits 5 and 6 only change `run_single` and the single-file end of `main`. Every single-file set, the short lists
and the sets of the second pass ran on the final branch (commits 1–6; `identity_fix.txt`, `ident_single.txt`).
The first pass ran without the lossless sets: `mk_ll_sets.py` reads a source tree at `~/br`, which this instance did
not have, so `~/ll/*` (except Kodak) and the gray, WebP and mixed sets built from them were empty or short.
`scripts/ident_fix.sh` made them (`~/br` → main's tree) and ran every set that uses them again; the table lists
those sets from that second pass only.

The skipped column counts inputs both binaries skipped (`--lossless` refuses JPEGs; one gray PNG is not supported
lossless).

<details><summary>Every run</summary>

| set | options | branch state | inputs | main files | branch files | differ | skipped (both) |
|---|---|---|---:|---:|---:|---:|---:|
| sr_default | `default` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_turbo3_nosmall | `--preset turbo3 --small-preset none` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_nosmall | `--small-preset none` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_444 | `--yuv 444` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_10 | `--depth 10` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_444_10 | `--yuv 444 --depth 10` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_base_nosmall | `--preset base --small-preset none` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_slow_nosmall | `--preset slow --small-preset none` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_fast_nosmall | `--preset fast --small-preset none` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_q30 | `--quality 30` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_q90 | `--quality 90` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_q95 | `--quality 95` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_tune_none | `--tune none` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_tiles_2_1_1 | `--tiles 2,1,1` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_default_enc1 | `GP_ENCODERS=1` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_default_enc6 | `GP_ENCODERS=6` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_slow | `--preset slow` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_444_q90 | `--yuv 444 --quality 90` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_444_q95 | `--yuv 444 --quality 95` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_444_slow | `--yuv 444 --preset slow` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_444_10_q95 | `--yuv 444 --depth 10 --quality 95` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| sr_10_q95 | `--depth 10 --quality 95` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| mono_default | `default` | 1-6 | 47 | 47 | 47 | 0 | 0 |
| mono_q90 | `--quality 90` | 1-6 | 47 | 47 | 47 | 0 | 0 |
| mono_q95 | `--quality 95` | 1-6 | 47 | 47 | 47 | 0 | 0 |
| mono_slow | `--preset slow` | 1-6 | 47 | 47 | 47 | 0 | 0 |
| mono_nosmall | `--small-preset none` | 1-6 | 47 | 47 | 47 | 0 | 0 |
| mono_10 | `--depth 10` | 1-6 | 47 | 47 | 47 | 0 | 0 |
| mono_tune_none | `--tune none` | 1-6 | 47 | 47 | 47 | 0 | 0 |
| std_q95 | `--quality 95` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| std_slow | `--preset slow` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| thumbs_q95 | `--quality 95` | 1-4 | 280 | 280 | 280 | 0 | 0 |
| thumbs_slow | `--preset slow` | 1-4 | 280 | 280 | 280 | 0 | 0 |
| kodak_q95 | `--quality 95` | 1-4 | 24 | 24 | 24 | 0 | 0 |
| kodak_444_q90 | `--yuv 444 --quality 90` | 1-4 | 24 | 24 | 24 | 0 | 0 |
| rgba_q95 | `--quality 95` | 1-4 | 39 | 39 | 39 | 0 | 0 |
| odd_q95 | `--quality 95` | 1-4 | 30 | 30 | 30 | 0 | 0 |
| odd_slow | `--preset slow` | 1-4 | 30 | 30 | 30 | 0 | 0 |
| std_default | `default` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| std_nosmall | `--small-preset none` | 1-4 | 100 | 100 | 100 | 0 | 0 |
| thumbs_default | `default` | 1-6 | 280 | 280 | 280 | 0 | 0 |
| thumbs_nosmall | `--small-preset none` | 1-4 | 280 | 280 | 280 | 0 | 0 |
| kodak_default | `default` | 1-6 | 24 | 24 | 24 | 0 | 0 |
| kodak_lossless | `--lossless` | 1-4 | 24 | 24 | 24 | 0 | 0 |
| kodak_lossless_ycgco | `--lossless --ycgco-re` | 1-4 | 24 | 24 | 24 | 0 | 0 |
| rgba_default | `default` | 1-6 | 39 | 39 | 39 | 0 | 0 |
| rgba_aq100 | `--alpha-quality 100` | 1-4 | 39 | 39 | 39 | 0 | 0 |
| rgba_lossless | `--lossless` | 1-4 | 39 | 39 | 39 | 0 | 0 |
| rgba_no_alpha | `--no-alpha` | 1-4 | 39 | 39 | 39 | 0 | 0 |
| gray_default | `default` | 1-6 | 9 | 9 | 9 | 0 | 0 |
| gray_lossless | `--lossless` | 1-6 | 9 | 8 | 8 | 0 | 1 |
| odd_default | `default` | 1-6 | 30 | 30 | 30 | 0 | 0 |
| odd_444_10 | `--yuv 444 --depth 10` | 1-4 | 30 | 30 | 30 | 0 | 0 |
| odd_lossless | `--lossless` | 1-4 | 30 | 10 | 10 | 0 | 20 |
| misc_default | `default` | 1-6 | 13 | 13 | 13 | 0 | 0 |
| png16_10_default | `--depth 10` | 1-6 | 8 | 8 | 8 | 0 | 0 |
| png16_10_lossless | `--lossless` | 1-6 | 8 | 8 | 8 | 0 | 0 |
| llgraphics_lossless | `--lossless --ycgco-re` | 1-6 | 27 | 27 | 27 | 0 | 0 |
| webp_default | `default` | 1-6 | 7 | 7 | 7 | 0 | 0 |
| webp_lossless | `--lossless` | 1-6 | 7 | 7 | 7 | 0 | 0 |
| mixed_list | `default` | 1-6 | 33 | 33 | 33 | 0 | 0 |
| mixed_list_444_10 | `--yuv 444 --depth 10` | 1-6 | 33 | 33 | 33 | 0 | 0 |
| lljpeg_set | `default` | 1-6 | 25 | 25 | 25 | 0 | 0 |
| pal_screen_default | `default` | 1-4 | 40 | 40 | 40 | 0 | 0 |
| pal_screen_palette_on | `--palette on` | 1-4 | 40 | 40 | 40 | 0 | 0 |
| pal_screen_444_10 | `--yuv 444 --depth 10` | 1-4 | 40 | 40 | 40 | 0 | 0 |
| pal_screen_lossless | `--lossless` | 1-4 | 40 | 40 | 40 | 0 | 0 |
| pal_flat_png_default | `default` | 1-4 | 24 | 24 | 24 | 0 | 0 |
| pal_flat_png_palette_on | `--palette on` | 1-4 | 24 | 24 | 24 | 0 | 0 |
| pal_flat_png_444_10 | `--yuv 444 --depth 10` | 1-4 | 24 | 24 | 24 | 0 | 0 |
| pal_flat_png_lossless | `--lossless` | 1-4 | 24 | 24 | 24 | 0 | 0 |
| pal_flat_jpg_default | `default` | 1-4 | 24 | 24 | 24 | 0 | 0 |
| pal_flat_jpg_palette_on | `--palette on` | 1-4 | 24 | 24 | 24 | 0 | 0 |
| pal_flat_jpg_444_10 | `--yuv 444 --depth 10` | 1-4 | 24 | 24 | 24 | 0 | 0 |
| pal_flat_jpg_lossless | `--lossless` | 1-4 | 24 | 0 | 0 | 0 | 24 |
| short_mixed_00 | `default` | 1-4 | 8 | 8 | 8 | 0 | 0 |
| short_mixed_00_444_10 | `--yuv 444 --depth 10` | 1-4 | 8 | 8 | 8 | 0 | 0 |
| short_mixed_01 | `default` | 1-4 | 8 | 8 | 8 | 0 | 0 |
| short_mixed_01_444_10 | `--yuv 444 --depth 10` | 1-4 | 8 | 8 | 8 | 0 | 0 |
| short_mixed_02 | `default` | 1-4 | 8 | 8 | 8 | 0 | 0 |
| short_mixed_02_444_10 | `--yuv 444 --depth 10` | 1-4 | 8 | 8 | 8 | 0 | 0 |
| short_mixed_03 | `default` | 1-4 | 5 | 5 | 5 | 0 | 0 |
| short_mixed_03_444_10 | `--yuv 444 --depth 10` | 1-4 | 5 | 5 | 5 | 0 | 0 |
| short_pair | `default` | 1-4 | 2 | 2 | 2 | 0 | 0 |
| short_pair_444_10 | `--yuv 444 --depth 10` | 1-4 | 2 | 2 | 2 | 0 | 0 |
| short_pair_slow | `--preset slow` | 1-4 | 2 | 2 | 2 | 0 | 0 |
| single_file | `default` | 1-6 | 10 | 10 | 10 | 0 |  |
| single_file | `--yuv 444 --depth 10` | 1-6 | 10 | 10 | 10 | 0 |  |
| single_file | `--preset slow` | 1-6 | 10 | 10 | 10 | 0 |  |
| single_default | `default` | 1-6 | 97 | 96 | 96 | 0 |  |
| single_lossless | `--lossless` | 1-6 | 47 | 45 | 45 | 0 |  |
| single_palette_on | `--palette on` | 1-6 | 97 | 96 | 96 | 0 |  |
| single_444_10 | `--yuv 444 --depth 10` | 1-6 | 97 | 96 | 96 | 0 |  |
| single_slow | `--preset slow` | 1-6 | 97 | 96 | 96 | 0 |  |
| single_q90 | `--quality 90` | 1-6 | 97 | 96 | 96 | 0 |  |
| final_short_mixed_00 | `default` | 1-6 | 8 | 8 | 8 | 0 | 0 |
| final_short_mixed_01 | `default` | 1-6 | 8 | 8 | 8 | 0 | 0 |
| final_short_mixed_02 | `default` | 1-6 | 8 | 8 | 8 | 0 | 0 |
| final_short_mixed_03 | `default` | 1-6 | 8 | 8 | 8 | 0 | 0 |
| final_short_mixed_04 | `default` | 1-6 | 1 | 1 | 1 | 0 | 0 |
| final_short_pair.txt | `default` | 1-6 | 2 | 2 | 2 | 0 | 0 |
| **all (101 runs)** | | | | **5654** | **5654** | **0** | |

</details>


## Decoder gate, tests, sanitizer

- **Decoder gate** (`tests/decoder_gate.py`, dav1d 1.4.1 and aomdec 3.8.2, the branch at commit 4; `logs/gate.txt`):
  photos (30 size-range photos at qindex 60 / 120 / 200: 90/90; 4:4:4 10-bit at 120: 30/30), thumbnails (28 at
  60 / 120: 56/56), RGBA (39/39, alpha items 35/35), odd sizes (30/30), and the palette sets: screen 40/40 (default
  and `--palette on`), flat_png 24/24, flat_jpg 24/24 (default and `--palette on`). All identical in both decoders.
  The gate runs `--batch --debug-outputs`, so it also shows that the reconstruction is still produced there.
- **ctest:** 9/9 passed; `gpu_fuzz_test` (seeds 1–400), `gpu_postfilter_test` (200 cases) and `varboost_test`:
  0 mismatches (`logs/ctest.txt`).
- **compute-sanitizer memcheck** (2026.1): 0 errors in single-file mode (2.4 MP photo, 320 × 213 thumbnail, an RGBA
  PNG, a 10 MP photo with `--yuv 444 --depth 10`; at commit 4 and again at commit 6) and in batch mode (the pair and
  3 photos with 1 queue, 12 photos with 32 queues; `logs/memcheck.txt`, `logs/ident_single.txt`). Racecheck was not
  run: no change touches device-side synchronisation.
- Single-file `--check` (reconstruction identical to av1core) and `--debug-outputs` (`.obu`, `.recon.yuv` identical
  to main's) still work (`logs/ident_single.txt`).

## Speed

### Single image, cold process

`av1gpu IN OUT` (no `--batch`), one process per run, wall time from the launcher (`scripts/wall.py`): a warm-up per
binary, then 10 runs each with all 8 binaries alternating run by run; median (`logs/spd8_single.txt`). main, the
trees after each commit (`c1` … `c6`; `c6` = this branch) and a copy of main (the noise run).

| input | main | c1 queues | c2 recon | c3 LUT row | c4 lazy arena | c5 overlap | c6 `_exit` | main copy (noise) | main → branch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 320 × 213 JPEG | 1,103.7 | 585.1 | 590.1 | 537.5 | 514.6 | 433.0 | **389.1** | 1,112.7 | −65 % (2.84×) |
| 512 × 384 JPEG | 1,146.1 | 629.0 | 628.9 | 583.9 | 552.7 | 475.3 | **433.2** | 1,149.1 | −62 % (2.65×) |
| 2.4 MP JPEG | 1,137.2 | 614.7 | 610.2 | 583.5 | 552.7 | 468.8 | **422.5** | 1,133.4 | −63 % (2.69×) |
| 10 MP JPEG | 1,185.5 | 667.2 | 635.3 | 609.8 | 581.9 | 496.8 | **453.1** | 1,193.1 | −62 % (2.62×) |
| `--batch` 2.4 MP + 512 × 384 | 1,357.1 | 839.2 | 847.5 | 709.6 | 717.3 | 715.8 | **718.4** | 1,359.3 | −47 % (1.89×) |

All times in ms. Noise: main against its copy differed by −0.3 to +0.8 % (4–9 ms). Inside the noise: commit 2 on
the three small inputs (it saves 32 ms on the 10 MP one: the CDEF apply and the reconstruction download) and
commits 4–6 on the `--batch` pair (they change single-file mode only; the 718 vs 710 ms are noise).

Per upload (a JPEG and its thumbnail) on this L4: two single-file runs went from ~2,250 ms to ~810 ms (2.4 MP and
320 × 213: 1,137 + 1,104 → 423 + 389), one 2-image `--batch` from 1,357 to 718 ms. The batch still wins by ~90 ms,
because it starts CUDA once; commits 4–6 would help it too if batch mode got the same treatment for short lists
(not done here).

### Batch mode (long lists)

speed-v06's method (`scripts/spd8.py batch`): each set repeated to at least 1,000 MP (`bench.speed_set`), a warm-up and
3 timed runs per binary, 2 rounds with the binary order alternating (new, main, main copy; then reversed), median of
the 6 timed runs. MP/s and img/s are av1gpu's `--stats`; wall MP/s counts the whole process. Default settings (3
encoders). "main copy" is the noise run: main against a copy of itself in the same rounds. `logs/spd8_batch.jsonl`.

| set | branch MP/s | main MP/s | change | main copy MP/s (noise) | rounds, branch | rounds, main | branch wall MP/s | main wall MP/s | branch img/s | main img/s |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| size range (100 JPEGs × 3, 1,433 MP) | **241.8** | 230.1 | **+5.1 %** | 231.4 (+0.6 %) | 243.9 / 240.9 | 233.5 / 226.8 | 208.6 | 199.5 | 50.6 | 48.1 |
| standard (100 JPEGs × 13, 1,023 MP) | 76.5 | 75.3 | +1.6 % | 75.8 (+0.7 %) | 76.4 / 76.6 | 74.7 / 76.2 | 71.4 | 70.3 | 97.1 | 95.7 |
| thumbnails (280 JPEGs repeated, 1,000+ MP) | 63.8 | 63.1 | +1.1 % | 61.7 (−2.2 %) | 62.2 / 64.9 | 63.1 / 62.3 | 60.3 | 59.5 | 420.9 | 415.7 |

- **Size range: +5.1 %**, outside the noise (both rounds 3–6 % up; main against its copy 0.6 %). The post-filter
  stage went from 1,552 to 1,233 ms per run (the CDEF apply that fed nothing, commit 2). Split by commit (`scripts/spd8.py split`, the trees after commits 1, 2 and 3, same method): 231.9 → 236.5 → 239.5 MP/s, so about +2 % from commit 2 and +1.3 % from commit 3 (batch mode initialises six encoders after its timer starts: 6 × 24 ms of LUT); each step is near the noise on its own (`logs/spd8_split.jsonl`).
- **Standard set and thumbnails: +1.6 % and +1.1 %, inside the noise** (the main copy moved by +0.7 % and −2.2 %,
  and the rounds of one binary differ by up to 4 %). Most of their images go through the `base` small-image preset.
- Output bytes were the same in every run of a set (`bytes` in `--stats`).



## Not done, for later

- **Table uploads into modules a run does not use.** `gp_init_tables` uploads into all 12 kernel modules (k_misc,
  the analysis modules, the four k_recon variants, lossless, palette), and with lazy loading each upload loads its
  module: ~90 ms. An 8-bit 4:2:0 photo needs neither the 4:4:4 / 10-bit k_recon variants nor the lossless module. A
  per-module upload on first use could save an estimated 30–40 ms per single image, but it touches k_misc.cu and
  every launcher, and needs the full identity run again.
- **nvJPEG pre-allocation.** Without it (`INGEST_HW_MAX_SIDE=0`) a single image is 43 ms faster (freeing it is 35 ms
  of `ingest_destroy`; commit 6 removes most of that anyway), but the same limit routes JPEGs above 6000 px to the
  CPU decoder, so turning it off would change which decoder those images get.
- **Thumbnail encodes** are latency-bound in k_recon's wavefront (4.9 ms per step for the `base` preset).

## Blackwell

Not measured here. The queue cost is a driver cost, so it should show on Blackwell too. To check on a g7 with the
v0.8.0 release binary alone: `python3 scripts/wall.py 10 ./av1gpu -- in.jpg out{bin}.avif`, with and without
`GP_MAX_CONNECTIONS=1` in the environment; then the branch's release build against v0.8.0 the same way, and `cmp`
the outputs.
