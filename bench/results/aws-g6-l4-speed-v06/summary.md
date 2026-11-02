# av1gpu speed, v0.6 against v0.5.0 (NVIDIA L4, g6.4xlarge, 2026-11-01)

The v0.6.0 release check: main **6a4956e** (#32–#40: the analysis, reconstruction, loop-filter and entropy kernels and
the batch-mode start-up, all meant to be byte-identical) against **v0.5.0** (b6545c1), both built on the instance for
`sm_89` and run alternately in the same session. One g6.4xlarge spot instance (NVIDIA L4, driver 595.112.07, CUDA
13.2, EPYC 7R13 with 16 vCPUs; `logs/env.txt`). The instance was set up with main at ef2522f. When #38–#40
were merged, main was rebuilt from 6a4956e and every check was run again (`logs/pipe6.log`,
`logs/env_final.txt`). All speeds are 6a4956e's.

The method is the same as `../aws-g6-l4-speed-v04/`. Each set is repeated to at least 1,000 MP (the lossless and
RGBA sets as published), then a warm-up and 3 timed runs. That is done twice, with the binary order alternating,
and the result is the median of the 6 timed runs. MP/s is av1gpu's own `--stats` count unless marked "wall": wall
MP/s is the megapixels over the whole process's wall time, CUDA start-up included. CPU cores are process CPU time
over wall time. Power and SM clock come from `nvidia-smi dmon -s puc` (1 s samples during the timed runs). Rounds
are the medians of each round's 3 runs. Raw runs are in `runs/*.jsonl`, the summary in `results.json`
(`scripts/analyze_spd.py`), and the scripts in `scripts/`.

## Release check

### Byte identity, v0.6 against v0.5.0

Every output file was compared with `cmp` (`scripts/identity.sh`, `scripts/cmpset.sh`; `logs/identity.txt` has a
digest per set). Sets: the 100 size-range and 100 standard JPEGs; 280 thumbnails made from the size range with
ImageMagick (100 fill-and-crop downscales to 512 × 384, 100 thumbnails within 320 × 320, 80 crops of 512 × 384 from the
photos of 4 MP and more); 40 size-range photos converted to gray JPEGs, plus gray PNGs and JPEGs; the PNG and WebP sets
of the lossless and RGBA runs; odd sizes from 1 × 1 to 4097 × 17; a mixed `--batch` list; and single-file mode
(`scripts/mk_sets.py`, `scripts/mk_mono.sh`). The first pass, on main at ef2522f, before #38–#40 were merged, was
identical too (`logs/identity_ef2522f.txt`). Three of its sets (`webp_default`, `webp_lossless`, `mixed_list`) first
failed to run: GPU memory was full while sanitizer runs shared the GPU. Their reruns were identical
(`logs/identity_rerun_ef2522f.txt`; its three `single_file` lines are an artefact of that version of
`rerun_failed.sh`, which treated the single-file lines as batch sets). On 6a4956e every set ran alone and none
needed a rerun.

| set | options | inputs | v0.5.0 files | v0.6 files | differ | skipped (both) |
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
| graphics PNGs (23) | `--lossless --ycgco-re` | 23 | 23 | 23 | 0 | 0 |
| WebP (7: lossless, lossy, with alpha) | default | 7 | 7 | 7 | 0 | 0 |
| WebP (7: lossless, lossy, with alpha) | `--lossless` | 7 | 7 | 7 | 0 | 0 |
| mixed `--batch` LIST (33) | default | 33 | 33 | 33 | 0 | 0 |
| mixed `--batch` LIST (33) | `--yuv 444 --depth 10` | 33 | 33 | 33 | 0 | 0 |
| ll/jpeg (25: 4:2:0, 4:4:4, gray, 4:2:2, odd) | default | 25 | 25 | 25 | 0 | 0 |
| single-file mode, no `--batch` (10 files) | default | 10 | 10 | 10 | 0 |  |
| single-file mode, no `--batch` (10 files) | `--yuv 444 --depth 10` | 10 | 10 | 10 | 0 |  |
| single-file mode, no `--batch` (10 files) | `--preset slow` | 10 | 10 | 10 | 0 |  |
| **all (66 runs: 63 batch, 3 single-file)** | | | **4,698** | **4,698** | **0** | |

### Decoder gate, tests, sanitizers

- **Decoder gate** (`tests/decoder_gate.py`, dav1d 1.4.1 and aomdec 3.8.2, v0.6): 14 runs, **676/676
  encodes and 38/38 alpha items identical** in both decoders (`logs/gate.txt`):
  - 30 size-range photos (3 of every 10 by file name): qindex 60 / 120 / 200; 4:4:4 10-bit at 120; `--preset slow`;
    qindex 13 / 26 (quality 95 / 90: CDEF on with zero strengths, for #39) in 4:2:0 and in 4:4:4; 4:4:4 `--preset slow`;
  - the 39 RGBA PNGs (`--expect-alpha`); Kodak `--lossless`; 28 thumbnails at qindex 60 / 120; the 30 odd sizes in
    4:2:0 and 4:4:4 10-bit; the gray and gray + alpha PNGs;
  - the 47 monochrome inputs at qindex 13 / 26 / 120 and with `--preset slow`.
- **ctest:** 5/5 passed. Also `gpu_fuzz_test` (seeds 1–400, and 77–276 with more 4:4:4, 10-bit and lossless cases),
  `gpu_postfilter_test` (200 cases), `gpu_dsp_test` and `varboost_test`: 0 mismatches (`logs/ctest.txt`).
- **compute-sanitizer** (2026.1) on `av1gpu --batch` with 3 size-range photos (2.16, 1.71 and 0.78 MP: turbo with
  4 × 4 tiles, turbo, and the base small-image preset), default settings and `--yuv 444 --depth 10`
  (`scripts/sanitizer.sh`, `scripts/race_ab.sh`):
  - **memcheck:** 0 access errors. With `--leak-check full` it lists 5 / 6 leaks of 256 KB: each encoder's
    variance-boost tables (`pipeline.cpp`), freed only in the destructor of encoders that are never deleted.
    That code is the same in v0.5.0.
  - **synccheck:** 0 errors (the named barriers and half-CTA groups of #37 included).
  - **racecheck** (analysis mode), with v0.5.0 and the #36 / #37 merges run the same way (`logs/race/summary.txt`):
    - The `k_entropy_win` pairs: the producer/consumer hand-off through volatile flags, documented in
      docs/RESULTS.md. Their hazard counts are the same in v0.5.0 and v0.6; only the line numbers moved.
    - **1 report pair in `k_recon`** (`k_recon.cuh:1373` read / `:1374` write, 4:2:0 only): a compiler-widened shared
      load whose extra value is never used. It is new with #37 and has no effect on the output. The 4:2:0 concurrent
      U / V chroma finals read `S.eobUV[0..1]` after a barrier. The compiler does this with one `LDS.128 [0x5af0]`,
      which also covers `S.nTxb` at 0x5afc, and every thread executes it. Only thread 0 uses that register
      (`@!P0 IADD3 / STS [0x5afc]`), and thread 0 is the writer. The SASS is in `logs/race/sass_k_recon_1373.txt`.
      Bisect: 63683ef (#36, just before #37) has only the entropy pairs, and 980cc10 (#37) adds this pair.

```
racecheck (analysis mode) on av1gpu --batch, 3 size-range photos: hazards per pair of source locations

== v0.5.0 b6545c1, default: RACECHECK SUMMARY: 8 hazards displayed (8 errors, 0 warnings) | rc 0 825 s
    7046576  Error: Read k_entropy_win gpu_entropy.cu:1061  <->  Write k_entropy_win gpu_entropy.cu:1015
    1927868  Error: Read k_entropy_win gpu_entropy.cu:1061  <->  Write k_entropy_win gpu_entropy.cu:998
     278256  Error: Read vload gpu_entropy.cu:957  <->  Write vstore gpu_entropy.cu:958
     244888  Error: Write vstore gpu_entropy.cu:958  <->  Read vload gpu_entropy.cu:957
     228056  Error: Read k_entropy_win gpu_entropy.cu:1058  <->  Write k_entropy_win gpu_entropy.cu:1015
      61456  Error: Read k_entropy_win gpu_entropy.cu:1058  <->  Write k_entropy_win gpu_entropy.cu:998
       1584  Error: Read k_entropy_win gpu_entropy.cu:1061  <->  Write k_entropy_win gpu_entropy.cu:1033
        256  Error: Write vstore gpu_entropy.cu:958  <->  Read __nanosleep gpu_entropy.cu:957
         72  Error: Read k_entropy_win gpu_entropy.cu:1058  <->  Write k_entropy_win gpu_entropy.cu:1033
== v0.5.0 b6545c1, --yuv 444 --depth 10: RACECHECK SUMMARY: 8 hazards displayed (8 errors, 0 warnings) | rc 0 961 s
    7133332  Error: Read k_entropy_win gpu_entropy.cu:1061  <->  Write k_entropy_win gpu_entropy.cu:1015
    1968372  Error: Read k_entropy_win gpu_entropy.cu:1061  <->  Write k_entropy_win gpu_entropy.cu:998
     282392  Error: Read vload gpu_entropy.cu:957  <->  Write vstore gpu_entropy.cu:958
     249072  Error: Write vstore gpu_entropy.cu:958  <->  Read vload gpu_entropy.cu:957
     230260  Error: Read k_entropy_win gpu_entropy.cu:1058  <->  Write k_entropy_win gpu_entropy.cu:1015
      63340  Error: Read k_entropy_win gpu_entropy.cu:1058  <->  Write k_entropy_win gpu_entropy.cu:998
       1520  Error: Read k_entropy_win gpu_entropy.cu:1061  <->  Write k_entropy_win gpu_entropy.cu:1033
        280  Error: Write vstore gpu_entropy.cu:958  <->  Read __nanosleep gpu_entropy.cu:957
         88  Error: Read k_entropy_win gpu_entropy.cu:1058  <->  Write k_entropy_win gpu_entropy.cu:1033
== v0.6 main 6a4956e, default: RACECHECK SUMMARY: 103 hazards displayed (102 errors, 1 warning) | rc 0 562 s
    7046576  Error: Read k_entropy_win gpu_entropy.cu:1090  <->  Write win_group gpu_entropy.cu:992
    1927868  Error: Read k_entropy_win gpu_entropy.cu:1090  <->  Write k_entropy_win gpu_entropy.cu:1061
     279176  Error: Read vload gpu_entropy.cu:1020  <->  Write vstore gpu_entropy.cu:1021
     246872  Error: Write vstore gpu_entropy.cu:1021  <->  Read vload gpu_entropy.cu:1020
     228056  Error: Read k_entropy_win gpu_entropy.cu:1087  <->  Write win_group gpu_entropy.cu:992
      61456  Error: Read k_entropy_win gpu_entropy.cu:1087  <->  Write k_entropy_win gpu_entropy.cu:1061
      52064  Error: Read [clone k_recon] rc_chroma k_recon.cuh:1373  <->  Write [clone k_recon] rc_chroma k_recon.cuh:1374
       1584  Error: Read k_entropy_win gpu_entropy.cu:1090  <->  Write win_group gpu_entropy.cu:988
        232  Error: Write vstore gpu_entropy.cu:1021  <->  Read __nanosleep gpu_entropy.cu:1020
         72  Error: Read k_entropy_win gpu_entropy.cu:1087  <->  Write win_group gpu_entropy.cu:988
          4  Warning: Read [clone k_recon] rc_chroma k_recon.cuh:1373  <->  Write [clone k_recon] rc_chroma k_recon.cuh:1374
== v0.6 main 6a4956e, --yuv 444 --depth 10: RACECHECK SUMMARY: 8 hazards displayed (8 errors, 0 warnings) | rc 0 616 s
    7133332  Error: Read k_entropy_win gpu_entropy.cu:1090  <->  Write win_group gpu_entropy.cu:992
    1968372  Error: Read k_entropy_win gpu_entropy.cu:1090  <->  Write k_entropy_win gpu_entropy.cu:1061
     282824  Error: Read vload gpu_entropy.cu:1020  <->  Write vstore gpu_entropy.cu:1021
     251264  Error: Write vstore gpu_entropy.cu:1021  <->  Read vload gpu_entropy.cu:1020
     230260  Error: Read k_entropy_win gpu_entropy.cu:1087  <->  Write win_group gpu_entropy.cu:992
      63340  Error: Read k_entropy_win gpu_entropy.cu:1087  <->  Write k_entropy_win gpu_entropy.cu:1061
       1520  Error: Read k_entropy_win gpu_entropy.cu:1090  <->  Write win_group gpu_entropy.cu:988
        256  Error: Write vstore gpu_entropy.cu:1021  <->  Read __nanosleep gpu_entropy.cu:1020
         88  Error: Read k_entropy_win gpu_entropy.cu:1087  <->  Write win_group gpu_entropy.cu:988
== 63683ef (#36 merged, before #37), default: RACECHECK SUMMARY: 8 hazards displayed (8 errors, 0 warnings) | rc 0 590 s
    7046576  Error: Read k_entropy_win gpu_entropy.cu:1061  <->  Write k_entropy_win gpu_entropy.cu:1015
    1927868  Error: Read k_entropy_win gpu_entropy.cu:1061  <->  Write k_entropy_win gpu_entropy.cu:998
     277904  Error: Read vload gpu_entropy.cu:957  <->  Write vstore gpu_entropy.cu:958
     245368  Error: Write vstore gpu_entropy.cu:958  <->  Read vload gpu_entropy.cu:957
     228056  Error: Read k_entropy_win gpu_entropy.cu:1058  <->  Write k_entropy_win gpu_entropy.cu:1015
      61456  Error: Read k_entropy_win gpu_entropy.cu:1058  <->  Write k_entropy_win gpu_entropy.cu:998
       1584  Error: Read k_entropy_win gpu_entropy.cu:1061  <->  Write k_entropy_win gpu_entropy.cu:1033
        256  Error: Write vstore gpu_entropy.cu:958  <->  Read __nanosleep gpu_entropy.cu:957
         72  Error: Read k_entropy_win gpu_entropy.cu:1058  <->  Write k_entropy_win gpu_entropy.cu:1033
== 980cc10 (#37 merged), default: RACECHECK SUMMARY: 103 hazards displayed (100 errors, 3 warnings) | rc 0 551 s
    7046576  Error: Read k_entropy_win gpu_entropy.cu:1061  <->  Write k_entropy_win gpu_entropy.cu:1015
    1927868  Error: Read k_entropy_win gpu_entropy.cu:1061  <->  Write k_entropy_win gpu_entropy.cu:998
     277952  Error: Read vload gpu_entropy.cu:957  <->  Write vstore gpu_entropy.cu:958
     245112  Error: Write vstore gpu_entropy.cu:958  <->  Read vload gpu_entropy.cu:957
     228056  Error: Read k_entropy_win gpu_entropy.cu:1058  <->  Write k_entropy_win gpu_entropy.cu:1015
      61456  Error: Read k_entropy_win gpu_entropy.cu:1058  <->  Write k_entropy_win gpu_entropy.cu:998
      52016  Error: Read [clone k_recon] rc_chroma k_recon.cuh:1373  <->  Write [clone k_recon] rc_chroma k_recon.cuh:1374
       1584  Error: Read k_entropy_win gpu_entropy.cu:1061  <->  Write k_entropy_win gpu_entropy.cu:1033
        256  Error: Write vstore gpu_entropy.cu:958  <->  Read __nanosleep gpu_entropy.cu:957
         72  Error: Read k_entropy_win gpu_entropy.cu:1058  <->  Write k_entropy_win gpu_entropy.cu:1033
         20  Warning: Read [clone k_recon] rc_chroma k_recon.cuh:1373  <->  Write [clone k_recon] rc_chroma k_recon.cuh:1374
```

## Speed

### JPEG, size range

| setting | v0.6 MP/s | v0.5.0 MP/s | change | v0.6 rounds | v0.5.0 rounds | v0.6 wall MP/s | v0.5.0 wall MP/s | v0.6 CPU cores | v0.5.0 CPU cores | v0.6 W | v0.6 SM MHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default (3 encoders) | 224.7 | 120.9 | +85.9 % | 226.3 / 221.5 | 123.8 / 118.5 | 195.4 | 111.5 | 0.27 | 0.17 | 70.8 | 1658 |
| `GP_ENCODERS=2` | 207.1 | 118.0 | +75.6 % | 208.6 / 205.9 | 119.1 / 117.1 | 182.1 | 109.0 | 0.24 | 0.16 | 68.0 | 1770 |
| `GP_ENCODERS=4` | 229.7 | 120.1 | +91.2 % | 231.2 / 228.6 | 120.9 / 119.7 | 199.5 | 110.8 | 0.29 | 0.17 | 69.9 | 1636 |
| `--small-preset none`, 3 encoders | 261.7 | 140.1 | +86.8 % | 261.2 / 262.2 | 139.5 / 140.5 | 223.0 | 127.6 | 0.30 | 0.19 | 69.8 | 1633 |
| `--small-preset none`, 4 encoders | 271.5 | 142.0 | +91.3 % | 270.6 / 274.2 | 141.8 / 143.5 | 230.0 | 129.2 | 0.31 | 0.19 | 69.5 | 1617 |
| `--preset turbo3`, 4 encoders | 291.1 | 161.2 | +80.5 % | 291.4 / 290.7 | 159.7 / 163.2 | 244.2 | 144.9 | 0.36 | 0.23 | 68.5 | 1642 |
| `--preset turbo3 --small-preset none`, 4 encoders | 366.8 | 209.6 | +75.0 % | 365.8 / 367.8 | 211.6 / 205.3 | 295.3 | 182.7 | 0.42 | 0.27 | 70.3 | 1577 |
| `--tiles 2,1,1` (2 × 2 tiles), 3 encoders | 205.0 | 112.6 | +82.0 % | 206.2 / 202.8 | 112.8 / 112.5 | 180.3 | 104.4 | 0.25 | 0.16 | 69.3 | 1737 |
| `--tiles 2,3,3` (8 × 8 tiles), 3 encoders | 236.7 | 126.5 | +87.0 % | 236.7 / 237.9 | 124.9 / 128.1 | 204.7 | 116.2 | 0.28 | 0.18 | 71.0 | 1676 |

### JPEG, standard

| setting | v0.6 MP/s | v0.5.0 MP/s | change | v0.6 rounds | v0.5.0 rounds | v0.6 wall MP/s | v0.5.0 wall MP/s | v0.6 CPU cores | v0.5.0 CPU cores | v0.6 W | v0.6 SM MHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default (3 encoders) | 72.4 | 38.3 | +89.3 % | 72.3 / 72.7 | 38.2 / 38.8 | 67.9 | 36.9 | 0.17 | 0.14 | 71.6 | 1661 |
| `GP_ENCODERS=2` | 71.2 | 37.5 | +90.0 % | 70.9 / 72.6 | 37.2 / 38.1 | 66.8 | 36.2 | 0.18 | 0.12 | 70.5 | 1710 |
| `GP_ENCODERS=4` | 72.7 | 37.5 | +93.7 % | 72.8 / 72.1 | 37.6 / 37.4 | 68.1 | 36.2 | 0.17 | 0.15 | 71.4 | 1632 |
| `--small-preset none`, 3 encoders | 168.6 | 95.7 | +76.2 % | 167.5 / 169.7 | 95.8 / 95.4 | 145.8 | 87.5 | 0.34 | 0.26 | 71.3 | 1612 |
| `--small-preset none`, 4 encoders | 172.1 | 95.4 | +80.4 % | 172.2 / 172.0 | 94.8 / 96.6 | 148.4 | 87.3 | 0.34 | 0.25 | 70.5 | 1685 |

### Thumbnails

| setting | v0.6 MP/s | v0.5.0 MP/s | change | v0.6 rounds | v0.5.0 rounds | v0.6 wall MP/s | v0.5.0 wall MP/s | v0.6 CPU cores | v0.5.0 CPU cores | v0.6 W | v0.6 SM MHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 59.7 | 32.4 | +83.9 % | 60.0 / 59.4 | 32.4 / 32.5 | 56.5 | 31.4 | 1.27 | 1.27 | 71.1 | 1658 |
| `--small-preset none` | 143.8 | 88.4 | +62.8 % | 144.7 / 142.9 | 88.1 / 88.7 | 126.8 | 81.3 | 1.41 | 1.22 | 71.6 | 1693 |

### Kodak

| setting | v0.6 MP/s | v0.5.0 MP/s | change | v0.6 rounds | v0.5.0 rounds | v0.6 wall MP/s | v0.5.0 wall MP/s | v0.6 CPU cores | v0.5.0 CPU cores | v0.6 W | v0.6 SM MHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 54.4 | 30.7 | +76.8 % | 53.2 / 54.7 | 31.0 / 30.6 | 51.7 | 29.8 | 3.09 | 1.90 | 71.5 | 1686 |
| `--small-preset none` | 129.9 | 77.5 | +67.6 % | 130.3 / 129.4 | 77.0 / 79.1 | 115.6 | 71.9 | 6.10 | 3.60 | 71.2 | 1676 |
| `--preset turbo3 --small-preset none` | 219.1 | 129.3 | +69.4 % | 216.0 / 222.5 | 129.4 / 129.3 | 181.4 | 114.7 | 9.95 | 6.22 | 71.2 | 1737 |

### Formats, size range, 4 encoders

| setting | v0.6 MP/s | v0.5.0 MP/s | change | v0.6 rounds | v0.5.0 rounds | v0.6 wall MP/s | v0.5.0 wall MP/s | v0.6 CPU cores | v0.5.0 CPU cores | v0.6 W | v0.6 SM MHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8-bit 4:4:4 | 169.6 | 97.8 | +73.5 % | 168.3 / 170.4 | 97.3 / 98.2 | 152.3 | 91.4 | 0.22 | 0.15 | 70.8 | 1635 |
| 10-bit 4:2:0 | 229.5 | 120.2 | +90.9 % | 230.2 / 226.7 | 122.5 / 116.3 | 199.1 | 110.8 | 0.32 | 0.19 | 70.2 | 1670 |
| 10-bit 4:4:4 | 166.4 | 95.0 | +75.2 % | 168.9 / 164.3 | 95.6 / 93.4 | 149.6 | 88.9 | 0.23 | 0.15 | 70.6 | 1626 |

### Lossless (wall incl. start-up for one pass; av1gpu's MP/s for the set 4×)

| setting | v0.6 MP/s | v0.5.0 MP/s | change | v0.6 rounds | v0.5.0 rounds | v0.6 CPU cores | v0.5.0 CPU cores | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `--lossless`, one pass (wall) | 18.9 | 17.4 | +8.5 % | 18.8 / 19.0 | 17.4 / 17.5 | 1.33 | 1.30 | | |
| `--lossless --ycgco-re`, one pass (wall) | 19.4 | 17.8 | +8.7 % | 19.3 / 19.4 | 17.8 / 17.8 | 1.44 | 1.40 | | |
| `--lossless`, set 4× | 43.6 | 39.3 | +10.8 % | 43.3 / 43.9 | 39.0 / 40.2 | 1.41 | 1.34 | | |
| `--lossless --ycgco-re`, set 4× | 44.5 | 39.9 | +11.4 % | 43.7 / 44.9 | 39.9 / 39.1 | 1.38 | 1.37 | | |

### RGBA, 4 encoders, wall incl. start-up

| setting | v0.6 MP/s | v0.5.0 MP/s | change | v0.6 rounds | v0.5.0 rounds | v0.6 CPU cores | v0.5.0 CPU cores | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 43.9 | 27.0 | +62.7 % | 43.8 / 44.1 | 27.2 / 27.0 | 3.20 | 2.14 | | |
| `--alpha-quality 100` | 38.8 | 25.4 | +52.8 % | 38.5 / 39.2 | 25.4 / 25.2 | 2.88 | 1.65 | | |
| `--lossless` | 27.6 | 25.0 | +10.8 % | 27.6 / 27.8 | 24.8 / 25.0 | 1.39 | 1.23 | | |
| `--no-alpha` | 49.4 | 29.4 | +68.2 % | 49.1 / 49.9 | 29.3 / 29.4 | 3.66 | 2.07 | | |


- **Size range, default:** 224.7 MP/s with 3 encoders against 120.9 (+86 %), and 229.7 against 120.1 with 4
  (+91 %). This is above the perf agents' combined estimate of 190–200 MP/s, which was made before #38–#40
  were merged. Both rounds agree within 2 %.
- **Standard set, default:** 72.4 against 38.3 MP/s (+89 %). Its images are about 1 MP, most of them below,
  so most go through the `base` small-image preset.
- **Wall clock with start-up included:** #36 moved some driver work ahead of the point where av1gpu's batch timer
  starts. The wall-clock gain is close to the timed one, so the speed-up is not just time moved out of the timer:

  | set, default | v0.6 / v0.5.0, av1gpu's timer | v0.6 / v0.5.0, wall clock |
  |---|---:|---:|
  | size range, 3 encoders | +86 % | +75 % |
  | size range, 4 encoders | +91 % | +80 % |
  | standard set | +89 % | +84 % |
  | thumbnails | +84 % | +80 % |
  | Kodak | +77 % | +73 % |

  The time outside av1gpu's timer (process wall time minus `wall_ms`) is 0.96 s for v0.6 and 1.00 s for v0.5.0
  on every JPEG setting. It is a fixed cost per process, so the wall-clock gain is smaller only because the
  timed part got shorter.
- **Power and clocks:** on the size range (default, 3 encoders) v0.6 draws 70.8 W, with the SM clock at
  1,658 MHz and 100 % SM utilisation: 0.315 J/MP. v0.5.0 draws 68.7 W at 1,785 MHz: 0.568 J/MP. On the
  standard set it is 0.99 J/MP against 1.83. The L4 is at its 72 W cap in both versions, below its 2,040 MHz
  maximum clock, and v0.6 runs about 130 MHz slower under that cap. So the gain comes from less work per pixel,
  not from more power or higher clocks. Over all the samples at 95 % SM or more, the GPU averaged 69 W and
  76 °C (`logs/dmon.txt`). The clock-limit reasons were logged only before and after the runs
  (`logs/clocks_*.txt`).
- **v0.5.0 on this instance** ran 2–6 % below the v0.4 figures of `../aws-g6-l4-speed-v04/` (size range 120.9
  against 128.0, standard 38.3 against 39.4). v0.5.0 only added the Windows port, so this is instance-to-instance
  variance; every comparison here is within one session.
- **Bytes per run** were the same in every run of every setting (`bytes_same` in results.json).
- **Lossless set:** on this instance the 47 comparison PNGs came to 18,526,228 bytes, against 18,543,751 in
  the -v04 run. `mk_ll_sets.py` and its sources are the same, but this instance's `rsvg-convert` rendered the
  chart PNGs a little differently. Both binaries ran on the same files.

### Kodak in images per second

The 24 Kodak PNGs (768 × 512) were repeated 106 times, 2,544 files and 1,000 MP, at quality 60 with the default 3
encoders. "wall img/s" counts the whole process, CUDA start-up included. On v0.4.0 the same three settings gave
80 / 209 / 349 img/s (349 → 308 with start-up), from a different instance; v0.5.0 here gives 78 / 197 / 329.

| setting | v0.6 img/s | v0.5.0 img/s | v0.6 rounds | v0.6 wall img/s | v0.5.0 wall img/s | v0.6 MP/s | v0.5.0 MP/s | v0.6 CPU cores | v0.5.0 CPU cores |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 138.2 | 78.2 | 135.3 / 139.2 | 131.4 | 75.8 | 54.4 | 30.7 | 3.09 | 1.90 |
| `--small-preset none` | 330.3 | 197.1 | 331.4 / 329.0 | 294.0 | 182.9 | 129.9 | 77.5 | 6.10 | 3.60 |
| `--preset turbo3 --small-preset none` | 557.1 | 328.9 | 549.3 / 565.8 | 461.3 | 291.7 | 219.1 | 129.3 | 9.95 | 6.22 |

- **The FPGA comparison:** with `turbo3 --small-preset none`, v0.6 encodes 557 Kodak images per second, or 461
  counting CUDA start-up over a 1,000 MP run, against the FPGA encoder's published 409.85.
- **CPU:** the Kodak images are PNGs, decoded on the CPU. At 557 img/s that decoding keeps about 10 of the 16
  vCPUs busy, against 0.3–0.4 cores for JPEG input. With fewer CPUs, PNG decoding would limit these rates
  before the GPU does (docs/RESULTS.md, "CPU threads").

### Single-file mode

10 files (`sets/single`: JPEGs from 0.18 to 24 MP, Kodak PNGs, RGBA, gray, 63 × 65) one process each, with no
`--batch`. Wall time per process including CUDA start-up; a warm-up pass, then 3 rounds with the binary order
alternating (`scripts/single_time.py`, `runs/single.jsonl`, `logs/single.txt`). v0.6 takes 11.38 s for the 10 files
(medians of the rounds 11.35–11.39), against 13.32 s for v0.5.0 (13.29–13.34): 15 % less, and faster for every
file. So the path that #36 did not change is not slower. All outputs were identical (the identity table above).

```
new 10 files, total s per round [11.39, 11.38, 11.35] median 11.38 | per file median {'2287cba126a77fed_huge.jpg': 1.184, '25d19e868909e5e2_huge.jpg': 1.685, '2e67ed20c61dfe3e_typical.jpg': 1.13, '72496a1a66a3d206_small.jpg': 1.189, '9fba6f263f3c4f3e_tiny.jpg': 1.141, 'chart_by-size-dark.png': 0.959, 'g0.png': 0.995, 'kodim01.png': 1.029, 'kodim02.png': 1.018, 'o63x65_420.jpg': 1.047} | failures 0
old 10 files, total s per round [13.29, 13.32, 13.34] median 13.32 | per file median {'2287cba126a77fed_huge.jpg': 1.426, '25d19e868909e5e2_huge.jpg': 2.063, '2e67ed20c61dfe3e_typical.jpg': 1.376, '72496a1a66a3d206_small.jpg': 1.465, '9fba6f263f3c4f3e_tiny.jpg': 1.393, 'chart_by-size-dark.png': 1.03, 'g0.png': 1.08, 'kodim01.png': 1.128, 'kodim02.png': 1.097, 'o63x65_420.jpg': 1.264} | failures 0
```

### README demo

`rec.py` / `termgif3.py` from `../aws-g6-l4-speed-v04/scripts/`, 99 photos, default settings (`logs/rec.json`):

```
[av1gpu] 99 images (477.7 MP) in 2478.1 ms: 39.95 img/s, 192.76 MP/s; 9 batches; out 22.52 MB (0.377 bpp); CPU 796 ms = 0.32 cores; skipped (unsupported): 0; WebP/PNG via CPU decode: 0; write failures 0
real	0m3.411s
```
