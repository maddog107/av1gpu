# Measured results

The RTX PRO 2000 Blackwell the encoder was developed on (private images, below), and an NVIDIA L4 on AWS with
the public `bench/` image sets ([NVIDIA L4](#nvidia-l4-aws-g64xlarge)). Recent releases were also checked on an RTX PRO
4500 Blackwell on AWS ([Release builds](#release-builds-cuda-13-and-cuda-12)). A contributor added measurements
on a GeForce RTX 5090 ([RTX 5090](#geforce-rtx-5090-contributed-by-pickae)).

**The L4 run is the reference.** It uses public images, current baselines and a current CPU (AMD EPYC, 16
vCPUs). The development section below compares against the development server's dual Xeon Silver 4110,
which is slow per thread, so its CPU figures flatter the GPU. It is kept for its larger held-out image set.

## RTX PRO 2000 Blackwell (development)

These measurements were taken during development, before the code was extracted into this
repository. The encoder source is the same; only the CLI and build were reorganized. The test images
were private and are not distributed. `bench/` runs the same kind of comparison on public images,
and results from other GPUs are welcome. They were taken with the tiling of v0.3.0 and earlier (2 × 2 tiles for
images of 2 MP and more); the default now uses 4 × 4, which cost 0.2 points on the L4's size-range photos
([below](#compression)).

### Setup

- **GPU:** NVIDIA RTX PRO 2000 Blackwell (compute capability 12.0, 34 SMs, 16 GB, 70 W board power;
  SM clock about 2.5–2.6 GHz under load), CUDA 13.4. The GPU was shared with other jobs, so timed runs
  waited for an idle GPU and were repeated.
- **Host:** 2× Intel Xeon Silver 4110 (16 cores, 32 threads, 2017).
- **Baselines:** avifenc 1.4.0 with libaom 3.13.1, `-y 420`, `tune=iq` (avifenc's default) and
  avifenc's default auto-tiling. WebP is libwebp via `vips webpsave`.
- **Images:** 300 held-out JPEG originals from a photo community's uploads, 1,455 MP in total, from
  thumbnails to 35 MP. A separate 40-image development set was used for all tuning. All are baseline
  JPEGs with IJG quality-90 tables. Subsampling: 4:2:0 68.5 %, 4:4:4 26.1 %, grayscale 5.1 %. 41 % have
  odd dimensions and 31 % carry an ICC profile. 295 of the 300 go through the GPU path; the other 5 are
  over 35.6 MP or are not really JPEGs.
- **Metric:** SSIMULACRA2 (libjxl), measured against the JPEG as decoded by libjpeg-turbo. Each
  image's curve has 5 quality points. The BD-rate (log-size difference at equal SSIMULACRA2) is
  computed per image, then the median over images is reported. Negative means smaller files.

### Compression

Held-out set, BD-rate against libaom speed 6:

| Encoder | Median | Mean | tiny < 0.3 MP | small < 1 MP | typical < 4 MP | large < 9.4 MP | huge |
|---|---:|---:|---:|---:|---:|---:|---:|
| **av1gpu default** (`turbo`, `base` below 1 MP) | **−0.56 %** | −1.17 % | −0.38 % | −2.92 % | +0.79 % | −0.89 % | −4.15 % |
| av1gpu `--preset slow` | −4.84 % | −5.54 % | −2.2 % | −5.0 % | −4.8 % | −5.9 % | −7.4 % |
| av1gpu `--preset turbo --small-preset none` | +0.78 % | −0.01 % | +3.3 % | +1.2 % | +0.8 % | −0.9 % | −4.1 % |
| av1gpu `--preset turbo2` | +3.5 % | +3.0 % | | | | | |
| libaom speed 9 | +14.0 % | | | | | | |

Against WebP q82 at the same SSIMULACRA2 as each image's WebP, the default produces files a median of
**30.7 % smaller**, and `--preset slow` 33.4 % smaller.

### Speed

`av1gpu --batch` wall-clock throughput, with the source files in the page cache and GPU entropy coding:

| Configuration | MP/s | Notes |
|---|---:|---|
| **default** (3 encoders × 48 MP batches) | **110.8** | median of 3 paired rounds over the 295 held-out images (1,240 MP); about 58–61 W average board power, so roughly 0.55 J/MP |
| `--preset slow` | 17.3 | |

For comparison, single-threaded CPU encoders on the same host managed libaom speed 6 at 1.27 MP/s,
libaom speed 9 at 8.45 MP/s and WebP at 4.13 MP/s per thread.

The batch encoder needs about 1 GB of GPU memory, plus about 0.6 GB for the JPEG decoder. On the CPU,
the encoder process stays below one core with the hardware JPEG decoder, and the CPU JPEG route adds
about 0.2 cores.

Relative preset speeds come from an earlier build. Later optimizations, mostly in the reconstruction
kernel and in overlapping batches, made it about 1.5× faster. The earlier build's absolute numbers are
lower, but the ordering still holds (development set, BD-rate against libaom s6):

| Preset | BD-rate (dev) | MP/s (earlier build) |
|---|---:|---:|
| base | −3.8 % | 23.8 |
| fast | −3.0 % | 40.0 |
| faster | −1.5 % | 49.6 |
| turbo | −0.3 % | 71.2 |
| turbo2 | +2.7 % | 81.2 |

### Correctness

- **Decoder identity:** all 1,475 held-out encodes (295 images × 5 quality points) in batch mode were
  decoded by both dav1d and libaom's aomdec. The decoded pictures equal the encoder's reconstruction
  byte for byte. Earlier builds passed the same gate on the development set (195/195 over five
  quality points) and on the held-out set at q60, q120 and q180. Those runs included grayscale,
  tiled and very large images.
- **Fuzzing:** 14,000+ random-decision cases (all supported tools: partitions, modes, CfL, filter
  intra, edge filter, QM, delta-q, tiles, deblocking, CDEF) match dav1d and aomdec. The GPU entropy
  coder matches the CPU writer on more than 70 million symbols.
- **Sanitizers:** compute-sanitizer memcheck, racecheck and synccheck are clean on representative
  inputs, including several concurrent batches.
- **Browsers:** Chrome decodes sampled outputs at every size class and quality point.

### In production

Since October 2026 av1gpu has converted every new upload of the community site the encoder was
developed for. At one snapshot it had processed 316 uploads (632 files including thumbnails): 236 MB
of source JPEGs became 63 MB of AVIF, 73 % smaller. That figure depends on the source files and is not
a forecast.

## NVIDIA L4 (AWS g6.4xlarge)

The first run outside the development machine (2026-10-25), with this repository's code and the public
image sets in `bench/`. Raw results: `bench/results/aws-g6-l4*/` (`summary.md`, `results.json`); correctness,
sanitizer, power and version logs: `bench/results/aws-g6-l4/logs/`.

When the default changed to 4 × 4 tiles (2026-10-31), av1gpu's side of every compression result below was
re-encoded and re-scored on an L4 against the stored CPU baselines: `bench/results/aws-g6-l4-tiles4x4/`. The
av1gpu speeds below are v0.9.0's where they name no other version (main c553ffb: faster start-up,
reconstruction, analysis and entropy kernels, byte-identical to v0.8.0), re-run on 2026-11-03 on a g6.4xlarge
(on demand, driver 595.112.07) with v0.8.0 alternating in the same session: `bench/results/aws-g6-g7-v090-check/`
(v0.6 against v0.5.0: `bench/results/aws-g6-l4-speed-v06/`; v0.4 against v0.3.0: `bench/results/aws-g6-l4-speed-v04/`).
The CPU encoders' speeds are from the earlier runs.

### Setup

- **GPU:** NVIDIA L4 (Ada, compute capability 8.9, 58 SMs, 23 GB, 72 W board power), driver 595.91.07,
  CUDA 13.2 (V13.2.51), built with `-DCMAKE_CUDA_ARCHITECTURES=native`. AWS g6.4xlarge spot instance,
  Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04.5, kernel 7.0.0-1014-aws, gcc 13.3, CMake 3.28).
- **CPU:** AMD EPYC 7R13, 16 vCPUs (8 cores with SMT).
- **Baselines:** avifenc 1.4.0 with libaom 3.13.1, both built from source, `-y 420 -j 1`, `tune=iq`; one
  avifenc per vCPU, all 16 at once, with avifenc's default automatic tiling (up to 8 tiles per image; see
  [CPU builds](#cpu-builds-tuning-threads-and-tiling)). WebP is cwebp 1.3.2 `-m 4`. Decoders: dav1d 1.4.1, aomdec 3.13.1.
  Metric: libjxl v0.11.1 `ssimulacra2`.
- **Images:** the 100 ~1 MP Open Images JPEGs of `bench/openimages_v4_validation_100.csv` ("standard"),
  and 100 Open Images JPEGs of 0.18–24 MP in `bench/openimages_v4_gamut.csv` ("size range": 20 each of
  < 0.3, 0.3–1, 1–4, 4–9.4 and 9.4–35.6 MP; original Flickr uploads for the larger classes, 478 MP in all,
  mean 4.8 MP). Both are CC BY 2.0.
- **Speed runs** repeat each set up to at least 1,000 MP (`bench.py --min-mp`; one warm-up, then 3 timed runs;
  av1gpu: two such rounds, alternating with the previous version, and the median of the 6 timed runs). In the first run, a
  single pass over the standard set (79 MP, less than two GPU batches) measured 28.1 MP/s instead of
  39.2 MP/s, because start-up and pipeline fill dominate such a short run.

### Speed

MP/s at quality 60. The CPU rows use the whole machine; av1gpu uses about 0.2–0.4 CPU cores.

| Encoder | standard (~1 MP) | size range | CPU cores |
|---|---:|---:|---:|
| av1gpu default (`GP_ENCODERS=3`) | 81.2 | 252.4 | 0.2–0.3 |
| av1gpu default, `GP_ENCODERS=2` | 77.1 | 225.6 | 0.2–0.3 |
| av1gpu default, `GP_ENCODERS=4` | 80.9 | **260.7** | 0.2–0.3 |
| av1gpu `--small-preset none` (`turbo` for every image), `GP_ENCODERS=4` | **191.0** | **298.9** | 0.3 |
| av1gpu `--preset turbo3`, `GP_ENCODERS=4` | | 331.0 | 0.4 |
| av1gpu `--preset turbo3 --small-preset none`, `GP_ENCODERS=4` | | **419.5** | 0.4 |
| libaom speed 6, 16 × avifenc `-j 1` | 21.1 | 28.8 | 16 |
| libaom speed 9, 16 × avifenc `-j 1` | 166.1 | 187.6 | 16 |

libaom speed 10 gave the same files and speed as speed 9. While encoding the size range, the L4 averaged
96–100 % SM utilisation at 68–71 W (`nvidia-smi dmon`, 1 s samples), about 0.27 J/MP at 261 MP/s (v0.8.0: 0.29).
It runs at its 72 W cap with either version, at 1,600–1,820 MHz. Four concurrent encoders ran 3 % faster than
three on the size range and the same on the standard set; two were 11 % and 5 % slower.

v0.8.0 in the same session ran the size range at 231.5 / 213.8 / 236.4 MP/s with 3 / 2 / 4 encoders, 276.9
with `--small-preset none`, 294.6 with `turbo3` and 376.9 with `turbo3 --small-preset none`. It ran the
standard set at 76.2 and 179.0. Against v0.8.0, v0.9.0 is:
- 9.0 % faster on the size range with the default 3 encoders, 10.3 % with 4 and 5.5 % with 2;
- 7.9 % faster with `--small-preset none`, 12.4 % with `turbo3` and 11.3 % with `turbo3 --small-preset none`;
- 6.6 % faster on the standard set, and 6.7 % with `--small-preset none`.

v0.8.0 against a second copy of itself in the same session moved by 0.2–0.7 %, so every gain is outside the
noise. Every output is byte-identical to v0.8.0's. The gains come from #56–#58: no CDEF apply and no
reconstruction download when nothing reads them, and one variance-boost row per encoder (#56); 32-bit inverse
transforms and unrolled forward-transform loops in the reconstruction (#57); a warp-parallel edge filter and less
shared memory in the analysis (#58). Counted over the whole process's wall time, CUDA start-up included, the size
range runs at 217 MP/s against 201.

Small images:
- **Thumbnails:** 280 made from the size range (512 × 384 crops and downscales, and thumbnails within
  320 × 320) run at 67.5 MP/s (445 images/s) by default and 161.0 (1,062 images/s) with `--small-preset none`
  (v0.8.0: 62.8 and 148.3).
- **Kodak:** the 24 PNGs (0.39 MP each) run at 59.6, 143.4 and 253.8 MP/s with the default,
  `--small-preset none` and `turbo3 --small-preset none`, which is 152, 365 and 645 images per second.

The default sends images below 1 MP to the slower `base` preset (`--small-preset`), for better compression
on small images. On a set of nothing but ~1 MP images that dominates the speed: `--small-preset none` is
2.4× faster there, for files 3 % larger (below).

`turbo3 --small-preset none` runs at about 420 MP/s, 1.6× the default and 2.2× libaom
speed 9 (188).

### Single-image start-up

Bellazon converts each upload, a photo and its thumbnail, as it arrives, one av1gpu process at a time, so there the
fixed cost of a process counts more than batch throughput. v0.9.0 cut it (#56): one hardware queue instead of 32 for
a single image and for lists of up to 8 encodes, the JPEG decoded while the encoder sets up, the pinned staging
buffer only when a JPEG needs the CPU decoder, one variance-boost row per encoder, no reconstruction download, and a
plain `_exit` once the output is written (Linux). Cold `av1gpu IN OUT`, one process per run, wall time including CUDA
start-up, median of 12 runs with the binaries alternating (`bench/results/aws-g6-g7-v090-check/`):

| Input | L4, v0.9.0 | L4, v0.8.0 | RTX PRO 4500, v0.9.0 | RTX PRO 4500, v0.8.0 |
|---|---:|---:|---:|---:|
| 2.5 MP JPEG (1280 × 1920) | **359 ms** | 1,059 | **506** | 1,027 |
| 10.5 MP JPEG (4320 × 2432) | **467** | 1,214 | **578** | 1,139 |
| 512 × 384 thumbnail of the 2.5 MP photo | **350** | 1,060 | **498** | 1,036 |
| 320 × 196 thumbnail | **377** | 1,098 | **523** | 1,059 |
| the photo and its 512 × 384 thumbnail as one `--batch` | **575** | 1,235 | **677** | 1,190 |

- **2.6–3.0× faster on the L4, about 2× on the RTX PRO 4500.** A second copy of v0.8.0 in the same runs differed by
  0.6 % or less. An upload converted as two processes takes 708 ms instead of 2,119 on the L4; one `--batch` of the
  two, 575 ms.
- **One hardware queue is half of it.** v0.8.0 with `GP_MAX_CONNECTIONS=1` takes 543 ms for the photo on the L4 and
  629 on the RTX PRO 4500. The rest of #56, and #59's upload of a single image without a host copy, take it to
  359 and 506 ms.
- **The RTX PRO 4500 starts slower than the L4** with v0.9.0: 506 against 359 ms for the photo. Its kernel code is
  larger: loading all 29 kernel modules of the binary takes about 200 ms there against 48 ms on the L4 (a standalone
  loader timing each module, `scripts/modload.c`), and 144 ms of that are the four reconstruction modules, 38 MB each for sm_120 against 7–8 MB
  for sm_89 (nvcc 13.4 emits about 8× the instructions for `rc_luma` on sm_120). av1gpu loads every module at start
  (the table uploads, `bench/results/aws-g6-l4-pipe08/`), though a photo uses one of the four reconstruction
  variants; skipping the other three would save roughly 3 × 35 ms per process on Blackwell (not measured).

### Compression

BD-rate on SSIMULACRA2 against libaom speed 6, median over images (mean in brackets):

| Encoder | standard | size range |
|---|---:|---:|
| **av1gpu default** | **+0.3 %** (−0.1 %) | **+0.1 %** (−0.1 %) |
| av1gpu `--small-preset none` | +3.4 % (+2.9 %) | +2.2 % (+1.4 %) |
| av1gpu `--preset turbo3 --small-preset none` | | +5.7 % (+5.6 %)¹ |
| libaom speed 9 | +17.7 % (+20.7 %) | +15.0 % (+17.2 %) |
| SVT-AV1 preset 2 / 4 / 6 / 8 / 9 (below) | | −5.1 / −2.6 / +2.1 / +7.7 / +9.8 % |
| WebP³ | +40.4 % (+44.3 %) | +42.1 % (+44.9 %) |
| NVENC AV1 (ffmpeg `av1_nvenc` p7, below) | | +29.5 % (+46.6 %)² |

¹ `turbo3` with the small-image preset left on: +1.2 % (+2.0 %).

² 94 of the 99 photos (on the other 5 its quality range does not overlap libaom's), scored on the L4 with
the same tools as the rest of the table; see [NVENC AV1 on the same L4](#nvenc-av1-on-the-same-l4). The image
pipeline used for NVENC decodes photos with an odd height with wrong chroma; on the 76 even-height photos
NVENC is +27.9 %.

³ Scored with each .webp's own ICC profile. The first published runs decoded WebP with dwebp, which leaves
the profile out of its PNG, so the photos with Adobe RGB, Camera RGB, Generic RGB or display profiles were
scored as colour-shifted: +41.3 % (+55.3 %) on the standard set (12 photos) and +45.1 % (+61.2 %) on the size
range (13). `bench.py` now copies the profile back. The standard set was re-scored after pickae's RTX 5090 run
pointed at it ([discussion #19](https://github.com/maddog107/av1gpu/discussions/19)); its .webp files were
byte-identical to the L4 run's (`bench/results/webp-icc-standard/`).

Against WebP at each image's WebP quality, the default is 29.7 % (standard) and 30.6 % (size range)
smaller (30.0 % and 31.2 % before the WebP ICC correction). Size range by image size (median, images in brackets):

| Encoder | < 1 MP | 1–5 MP | 5–10 MP | ≥ 10 MP |
|---|---:|---:|---:|---:|
| av1gpu default | +0.1 % (40) | +1.9 % (20) | +0.2 % (23) | −2.1 % (17) |
| av1gpu `--small-preset none` | +3.9 % (40) | +1.9 % (20) | +0.2 % (23) | −2.1 % (17) |
| libaom speed 9 | +20.1 % (40) | +18.5 % (20) | +12.8 % (23) | +13.4 % (17) |
| NVENC AV1 | +35.7 % (38) | +29.2 % (18) | +28.5 % (21) | +22.2 % (17) |
| SVT-AV1 preset 4 | −2.0 % (39) | −2.2 % (20) | −3.4 % (23) | −3.9 % (17) |
| SVT-AV1 preset 6 | +5.1 % (39) | +2.9 % (20) | +0.7 % (23) | +0.2 % (17) |
| SVT-AV1 preset 9 | +14.9 % (39) | +12.5 % (20) | +6.3 % (23) | +7.2 % (17) |

(At 1 MP and above both av1gpu rows are the same `turbo` encodes.) The groups are small, but they show the
same trend as the held-out set: av1gpu compares best on large images.

**Tiling.** These av1gpu rows use the 4 × 4-tile default (since 2026-10-31). With v0.3.0's tiling (2 × 2 tiles
for images of 2 MP and more, `turbo3` 4 × 2) the same photos gave −0.1 % (−0.4 %) for the default, +2.1 %
(+1.1 %) with `--small-preset none` and +5.6 % (+5.4 %) for `turbo3 --small-preset none` (+1.1 % with the
small-image preset on), and 1–5, 5–10 and ≥ 10 MP gave +1.1, −0.6 and −2.4 %. Files below 2 MP are
byte-identical either way (the standard set entirely); at 2 MP and more the default's files grew 0.26 % at the
same quality settings. The speed this buys is in the README's tiles paragraph (about +12 % on this set).

### SVT-AV1 on the same CPU

SVT-AV1 was run later (2026-10-29) on a c6a.4xlarge, which has the same AMD EPYC 7R13 with 16 vCPUs (no
g6.4xlarge was available), with `bench.py --skip-gpu --aom-speeds 6,9 --svt-presets 2,4,6,8,9 --svt-tune 3`
on the 99 size-range photos. Raw results: `bench/results/aws-c6a-gamut-svt/`.

- **Build:** SVT-AV1 v4.2.0 and libavif 1.4.0 (with libaom 3.13.1) from source. avifenc
  `-c svt -a preset=N -a tune=3 -y 420 -j 1`, one per vCPU; libavif turns on SVT-AV1's still-picture
  mode (`avif=1`) for single images. All-intra SVT-AV1 runs presets 10–13 as preset 9.
- **Tune:** `tune=3` ("IQ") is SVT-AV1's still-image tune, the counterpart of libaom's `tune=iq`. On 20 of
  the photos at preset 8 it scored +6.0 % against libaom speed 6, against +16.5 % for the default tune
  (PSNR) and +7.6 % for `tune=4` (MS-SSIM and SSIMULACRA2), so the full run used it.
- **Control:** libaom ran in the same session at 28.7 (speed 6) and 186.4 MP/s (speed 9), against 28.8 and
  187.6 on the g6.4xlarge, and all 594 of its speed 6 encodes had the same bytes and SSIMULACRA2 scores as
  in the main run, so the BD-rates below compare directly with the rest of this section.

| SVT-AV1 | MP/s | vs libaom speed 6, median (mean) | vs WebP |
|---|---:|---:|---:|
| preset 2 | 4.6 | −5.1 % (−5.5 %) | −34.7 % |
| preset 4 | 15.5 | −2.6 % (−2.9 %) | −32.8 % |
| preset 6 | 49.4 | +2.1 % (+3.2 %) | −28.0 % |
| preset 8 | 136.0 | +7.7 % (+9.6 %) | −23.8 % |
| preset 9 | 182.8 | +9.8 % (+11.9 %) | −22.8 % |

SVT-AV1 is a better trade than libaom at every speed: at libaom speed 6's file size it runs at about the
same speed (interpolating between presets 4 and 6 gives about 29 MP/s), and at speed 9's speed its files
are 5 points smaller. Against av1gpu on one L4:

- at about libaom speed 6's file size, av1gpu's default (261 MP/s, +0.1 %) is 8.6× faster than SVT-AV1
  interpolated to the same size (30 MP/s);
- at +2.2 %, av1gpu with `--small-preset none` (299 MP/s) is 6.1× faster than preset 6 (49 MP/s, +2.1 %);
- at the fast end, `turbo3` (420 MP/s, +5.7 %) is 2.3× as fast as preset 9 (183 MP/s, +9.8 %), with smaller files;
- SVT-AV1's slow presets make smaller files than any av1gpu setting measured on the L4 (preset 2: −5.1 %,
  at 4.6 MP/s). av1gpu's `--preset slow` measured −4.8 % on the development set but has not been run here.

### CPU builds: tuning, threads and tiling

A reviewer asked whether the CPU encoders were built tuned, and whether one `avifenc -j 1` per vCPU is the
fastest way to use the CPU. We measured both on a c6a.4xlarge (the same EPYC 7R13, 16 vCPUs) on
2026-10-31, with the 99 size-range photos and `bench.py`'s own timing code. Stock libaom and SVT-AV1,
re-timed at the start and end of the session, matched the published speeds within 1 %. A second instance of
the same type ran every encoder 7–8 % slower with the same bytes; it is kept as a replication and not used
below. Raw results and every run: `bench/results/aws-c6a-cpu-tuning/`.

- **Builds.** The published builds are CMake Release (`-O3`) with GCC 13.3. SVT-AV1's build already turns
  on LTO by default. *Tuned* adds `-march=native` (Zen 3) and LTO to libaom, SVT-AV1 and libavif. *PGO* adds
  SVT-AV1's own profile-guided build option, trained with avifenc on the other image set (`bench/images`),
  never on the timing set. Every tuned and PGO file is byte-identical to the stock one, so all BD-rates
  above still hold.
- **Threads.** `bench.py --avifenc-threads N` runs avifenc `-j N`, 16 / N at once. libavif hands N to
  SVT-AV1 4.2 as its parallelism level, which it caps at 6.
- **Tiling.** avifenc 1.4 tiles automatically when no tile option is given: up to 8 tiles of at least
  512 × 512, here 8 tiles for 56 of the 99 photos. So the CPU baselines in this section and above are tiled.
  av1gpu's default preset (`turbo`, and `turbo2` and `turbo3`) uses 4 × 4 tiles for images of 2 MP and more;
  `base`, `slow` and the small-image preset use only the tiles AV1's limits need.

Speed with each build, 16 × `-j 1` (MP/s):

| Encoder | Stock | Tuned | Tuned + PGO | Fastest measured |
|---|---:|---:|---:|---|
| libaom speed 6 | 28.7 | 29.7 (+3.5 %) | | 31.1: tuned, 8 × `-j 2` |
| libaom speed 9 | 186.6 | 190.6 (+2.2 %) | | tuned, 16 × `-j 1` |
| SVT-AV1 preset 2 | 4.6 | 4.6 (+1.5 %) | 5.1 (+12.8 %) | PGO, 16 × `-j 1` |
| SVT-AV1 preset 4 | 15.6 | 15.8 (+1.6 %) | 17.2 (+10.8 %) | 18.0: PGO, 8 × `-j 2` |
| SVT-AV1 preset 6 | 49.4 | 50.5 (+2.3 %) | 55.2 (+11.6 %) | 55.5: PGO, 8 × `-j 2` |
| SVT-AV1 preset 8 | 136.0 | 138.5 (+1.9 %) | 149.2 (+9.7 %) | PGO, 16 × `-j 1` |
| SVT-AV1 preset 9 | 182.6 | 190.5 (+4.3 %) | 198.6 (+8.7 %) | PGO, 16 × `-j 1` |

Splitting the 16 vCPUs differently (stock builds, MP/s):

| Encoder | 16 × `-j 1` | 8 × `-j 2` | 4 × `-j 4` | 2 × `-j 8` | 1 × `-j 16` |
|---|---:|---:|---:|---:|---:|
| libaom speed 6 | 28.5 | 29.6 (+3.7 %) | 28.6 | 25.4 | 18.5 (−35 %) |
| libaom speed 9 | 188.9 | 170.9 (−9.5 %) | 139.4 | 95.0 | 52.6 (−72 %) |
| SVT-AV1 preset 4 | 15.7 | 16.3 (+3.6 %) | 15.7 | 12.7 | 8.1 (−49 %) |
| SVT-AV1 preset 6 | 49.4 | 49.9 (+1.1 %) | 43.3 | 29.8 | 17.1 (−65 %) |
| SVT-AV1 preset 9 | 184.5 | 152.4 (−17 %) | 100.5 | 55.7 | 28.9 (−84 %) |

8 × `-j 2` gains 2–5 % for libaom speed 6 and SVT-AV1 preset 4 (three comparisons each); everywhere else
16 × `-j 1` is fastest, and more threads per encode are always slower. SVT-AV1 writes the same files at
every split. libaom writes slightly different files at any `-j` above 1, with the same compression
(BD-rate +0.10 % at speed 6, +0.00 % at speed 9).

**Tiling.** One tile per image (`bench.py --no-tiling`) runs at the same speed at 16 × `-j 1` (−0.7 to
+1.6 %) and makes files 0.4–0.8 % smaller at equal SSIMULACRA2 (median BD-rate against the tiled files:
libaom speed 6 −0.52 %, speed 9 −0.81 %; SVT-AV1 presets 4, 6 and 9 −0.52, −0.52 and −0.37 %). Against an
untiled libaom speed 6 anchor every comparison moves by about half a point:

| Median BD-rate | vs libaom speed 6, avifenc default (tiled) | vs libaom speed 6, one tile |
|---|---:|---:|
| av1gpu default | +0.1 % | +0.7 % |
| av1gpu `--small-preset none` | +2.2 % | +2.5 % |
| libaom speed 9, one tile | +14.3 % | +14.7 % |
| SVT-AV1 preset 4, one tile | −3.2 % | −2.9 % |
| SVT-AV1 preset 6, one tile | +1.8 % | +2.6 % |
| SVT-AV1 preset 9, one tile | +9.4 % | +9.8 % |
| NVENC AV1 | +29.5 % | +30.3 % |
| WebP (with its ICC profiles) | +42.1 % | +43.5 % |

**What it means for av1gpu.** At its own file size (+0.7 % against untiled libaom speed 6), av1gpu's
261 MP/s (v0.9.0) is 9.1× stock libaom speed 6, 8.8× the tuned build and 8.4× the tuned build at 8 × `-j 2`
(31.1 MP/s, tiled). Interpolating between SVT-AV1 presets 4 and 6 at the same size gives 33.3 MP/s for the
stock build (7.8×) and 37.0 MP/s with PGO (7.0×), both untiled at 16 × `-j 1`; with tiling, PGO and
8 × `-j 2`, 34.5 MP/s (7.6×). The fastest CPU configuration we found is about 7.0× slower than one L4.

### NVENC AV1 on the same L4

The L4 also has NVIDIA's fixed-function video encoder (two NVENC engines on this card). We measured its
AV1 encoder on the same 99 size-range photos (2026-10-29) through Bellazon's earlier production pipeline,
unchanged: one ffmpeg process per image,

```
ffmpeg -i in.jpg -c:v av1_nvenc -preset p7 -tune uhq -rc vbr -b:v 0 -spatial-aq 1 -aq-strength 10 \
       -rc-lookahead 32 -lookahead_level 3 -cq <CQ> -pix_fmt yuv420p -color_range pc -frames:v 1 -f ivf out.ivf
```

then the AV1 OBUs muxed into an AVIF. p7 with `tune uhq` is NVENC's slowest, highest-quality setting.
ffmpeg n8.1.3 (BtbN static build; the 9.0 and master builds need NVENC API 13.1, a newer driver than
595.112.07). Raw results: `bench/results/aws-g6-l4-nvenc/`.

**Speed** at CQ 32 (99 photos, 477.7 MP per pass), by the number of encodes in flight; two rounds:

| Encodes at once | round 1 | round 2 | CPU cores |
|---:|---:|---:|---:|
| 1 | 5.7 | 5.7 | 0.95 |
| 2 | 11.0 | 10.9 | 1.9 |
| 4 | 15.5 | 15.5 | 3.2 |
| 8 | **15.6** | **15.7** | 3.9 |
| 16 | 14.1 | 14.1 | 4.2 |

It levels off at about 16 MP/s once both engines are busy. av1gpu's default (v0.3.0 then), timed in the same
session with `bench.gpu_speed`, ran at 115.6 and 117.1 MP/s on 0.12 cores, 7.4× faster; v0.9.0 runs at
261 MP/s on 0.3 cores, **16.6× NVENC's best**. The CPU cores are
ffmpeg decoding JPEGs and converting to 4:2:0 for the encoder; av1gpu decodes on the GPU.

**Compression:** six points, CQ 18, 25, 32, 38, 44 and 50, decoded with avifdec 1.4.0 and scored with the
same `ssimulacra2` as everything else. Against libaom speed 6 its files are **+29.5 %** (median; mean
+46.6 %) at equal quality, worse than libaom speed 9 (+15.0 %), and still 11.3 % smaller than WebP (85
photos, with WebP's ICC profiles); measured the other way, av1gpu's default files are 22.7 % smaller than
NVENC's (96 photos). Part of that is a bug in the image pipeline, not in NVENC: photos with an odd height
decode with wrong chroma (below). On the 76 even-height photos NVENC is +27.9 % against libaom speed 6, and
av1gpu's files are 20.6 % smaller than NVENC's. The 61 photos without an ICC profile (the pipeline drops ICC)
give +29.1 %, so that is not the cause. Totals
per CQ:

| CQ | AVIF bytes | Mean SSIMULACRA2 |
|---:|---:|---:|
| 18 | 85.5 MB | 79.1 |
| 25 | 54.2 MB | 75.6 |
| 32 | 31.8 MB | 68.7 |
| 38 | 18.4 MB | 59.1 |
| 44 | 10.8 MB | 46.5 |
| 50 | 6.1 MB | 28.5 |

For comparison, av1gpu's default quality gives 22.5 MB at 72.0 on the same photos, and libaom speed 6
28.9 MB at 75.9.

Other things we noticed:

- **Deterministic:** a second run on another L4 instance gave byte-identical totals at every CQ.
- **libaom rejects CQ 18:** libaom 3.13.1 (and 3.11) fails all 99 CQ 18 files with "Bitstream not
  supported by this decoder"; dav1d and libaom 3.13.3 decode them. CQ 25 and above decode everywhere. We
  scored every NVENC file with avifdec's dav1d codec (identical output to libaom where both decode).
- **Odd sizes are padded:** NVENC encodes 4:2:0 at even dimensions, so a 1,023-pixel-wide photo becomes a
  1,024-pixel-wide AVIF with a repeated edge column (cropped off before scoring).
- **Odd heights decode with wrong chroma:** found later, in the [CVVDP run](#cvvdp-a-second-metric). The 23
  photos with an odd height decode with a chroma error at every CQ (25–30 dB at CQ 18 against about 45 dB
  for even heights), while luma is fine. This is in the ffmpeg + muxer pipeline, not in NVENC, and it costs
  NVENC about 1.6 points of the +29.5 %.
- **8,192-pixel limit** per side; the pipeline sends bigger images to libaom instead (none in this set).

The same pipeline on the development machine's RTX PRO 2000 Blackwell (one NVENC engine) peaked at
about 9.4 MP/s (4 encodes at once; 9.5 MP/s with the JPEGs decoded to raw 4:2:0 in RAM beforehand, so the
engine itself is the limit), against av1gpu's 104 MP/s there, and scored about +25 % against libaom speed 6
(scored on a different machine, so not directly comparable with the L4's +29.5 %).

### FPGA: Bluedot Pulsar-AVIF

Bluedot's Pulsar-AVIF is an AVIF encoder on an AMD Alveo U250 FPGA card (about $11,000 retail in 2026; $8,995
at its 2018 launch). Its only published speed is in Google's
[Deploying AVIF for more responsive websites](https://web.dev/articles/avif-updates-2023) (last updated
2023-05-24): 409.85 frames/s on the Kodak set (24 images of 768 × 512, encoded 24 at once), 60 ms per image at
305 % CPU, on an Intel Xeon Platinum 8171 host with 10 cores. The same table has avifenc at 59.26 (libaom) and
18.11 (SVT-AV1) frames/s on a 32-core Xeon Platinum 8370C, and the article says Pulsar-AVIF has "similar
compression efficiency" without numbers or settings.

av1gpu v0.9.0 on the L4 with the same 24 PNGs at quality 60, repeated to 1,000 MP (2,544 images) in batch mode,
median of 6 runs (`bench/results/aws-g6-g7-v090-check/`; v0.8.0 in the same session: 141, 344 and 585 images/s):

| Encoder | images/s | with CUDA start-up | CPU cores | bytes, the 24 images |
|---|---:|---:|---:|---:|
| av1gpu default | 152 | 144 | 3.4 | 912,368 |
| av1gpu `--small-preset none` | 365 | 322 | 7.0 | 917,825 |
| av1gpu `--preset turbo3 --small-preset none` | **645** | **522** | 11.9 | 946,869 |
| Pulsar-AVIF (published) | 409.85 | | 3.05 | not published |

- **Speed only.** Pulsar's quality setting and file sizes aren't published, so there is no equal-quality
  comparison.
- **Small images are av1gpu's weakest case.** All 24 are below 1 MP, so the default encodes them with the slower
  `base` preset (`--small-preset`), and each fills only a little of the GPU.
- **CPU:** the PNGs are decoded on the CPU, which accounts for most of av1gpu's CPU use here. JPEGs are decoded
  by the GPU's JPEG engine.
- **Cost:** the L4 instance (g6.4xlarge) rents for about $0.80/hour on spot.

### 4:4:4 and 10-bit

`--yuv 444` (AV1 High profile) and `--depth 10` arrived on 2026-10-31. They were measured on a g6.4xlarge with
`bench/formats.py`: the size-range photos, bench.py's six quality points, SSIMULACRA2 against the source JPEG
(10-bit outputs decoded to 16-bit PNGs), and libaom speed 6 through avifenc 1.4.0 in the same format
(`-y 444`, `-d 10`; `tune=iq`, avifenc defaults, 16 × `-j 1`). The libaom 8-bit 4:2:0 encodes in the same
session reproduced the published files and scores (600/600). Raw results: `bench/results/aws-g6-l4-formats/`.
The BD-rates below are the 4 × 4 default's (`bench/results/aws-g6-l4-tiles4x4/`; with v0.3.0's 2 × 2 tiles:
−0.1, +0.3, +0.1 and +0.0 %). av1gpu's speeds are v0.9.0's, `GP_ENCODERS=4` (`bench/results/aws-g6-g7-v090-check/`;
v0.8.0 in the same session: 236.4, 174.7, 231.4 and 172.3 MP/s); libaom's are from the formats run.

| Format | av1gpu MP/s | libaom speed 6 MP/s | av1gpu vs libaom, same format (median BD-rate) |
|---|---:|---:|---:|
| 8-bit 4:2:0 | 260.7 | 28.6 | +0.1 % |
| 8-bit 4:4:4 | 197.9 | 24.3 | +0.4 % |
| 10-bit 4:2:0 | 253.3 | 22.0 | +0.3 % |
| 10-bit 4:4:4 | 193.9 | 18.9 | +0.4 % |

- **Speed:** 4:4:4 codes twice the chroma samples of 4:2:0, so av1gpu runs about 24 % slower; 10-bit costs it
  2–3 %. libaom loses 23 % at 10 bits, so av1gpu's lead grows from 9.1× to 11.5×.
- **10-bit vs 8-bit:** −2.2 % for av1gpu and −2.0 % for libaom in 4:2:0, −2.0 % and −1.8 % in 4:4:4, at equal
  SSIMULACRA2. The sources are 8-bit JPEGs, so this comes from the extra precision inside the codec.
- **4:4:4 vs 4:2:0:** +0.4 % for av1gpu and −0.8 % for libaom (8-bit). These photos were 4:2:0 JPEGs (or
  upsampled to 4:4:4 from them), so there is little full-resolution colour to keep; 4:4:4 is for sources that
  have it, such as PNG screenshots and graphics.
- SVT-AV1 has no 4:4:4. At 10-bit 4:2:0, preset 6 ran at 43.5 MP/s and +1.1 % against libaom speed 6.

### Lossless

`--lossless` (2026-10-31) codes AV1 lossless: base_q_idx 0, 4×4 Walsh-Hadamard transforms only, no deblocking
or CDEF. The reconstruction is the source, so av1gpu computes every prediction from source pixels in parallel
(no wavefront), estimates each 4×4's WHT-residual bits for every mode, runs the existing partition search on
those costs, then makes the final mode and CfL choices per superblock with exact edge availability. Lossless
frames use tiles of at most 256K pixels so the entropy coder has more work in flight.

- **Colour:** PNG/WebP RGB is coded as GBR 4:4:4 at the source depth, as `avifenc -l` does. 16-bit PNGs are
  coded at 10 bits only if they hold exactly 10 significant bits, and refused otherwise. `--ycgco-re` codes
  RGB as YCgCo-Re (CICP matrix 16, libavif 1.4's integer transform), 8-bit RGB to 10-bit 4:4:4.
- **JPEGs are refused** (skipped in batch mode). At first `--lossless` coded a JPEG's decoded YCbCr planes
  (44.8 MP/s on the size-range photos), but the files came out a median 2.2× the size of the JPEGs (1.1–3.7× for
  80 % of the 63 accepted photos, up to 6.2×; 1.75× in total).
- **GPU memory:** lossless batches need several times the memory of lossy ones. On the 40 size-range photos of
  5–24 MP as PNGs, the L4's memory use (`nvidia-smi`, 23,034 MiB in all) peaked at 14,553 MiB with
  `GP_ENCODE_MP=12`, 17,305 MiB with 16 and 21,945 MiB with 24, and ran out with the default 48 (lossy: 3,463
  MiB). av1gpu then exits with status 1 and suggests a smaller `GP_ENCODE_MP`. Exact alpha next to lossy colour
  (`--alpha-quality 100`) peaked at 8,711 MiB on the same photos with transparency.
- **Proof:** 189 of 189 files (Kodak and graphics in GBR and YCgCo-Re, 16-bit PNGs, gray, palette, odd sizes,
  forced tiles, a 16.7 MP image, WebP, and 22 JPEGs while JPEGs were accepted) decode to exactly the source with
  `avifdec -c aom` and `-c dav1d` (libavif 1.4.0). Decoder gates 17/17,
  6,787 lossless CPU fuzz cases and the GPU unit tests pass with 0 mismatches.
- **Lossy output unchanged:** 800/800 files byte-identical (default, turbo3, `--yuv 444`, `--depth 10`),
  every pre-existing GPU kernel compiles to identical SASS, and speed is within noise (115.7 vs 114.9 MP/s,
  18 runs each).
- **Not affected by the 4 × 4 default:** lossless frames are tiled by `--ll-tile-area`, and all 197 lossless test
  files (the 63 size-range JPEGs it accepts, the comparison PNGs in GBR and YCgCo-Re, the misc and JPEG sets,
  including a 25 MP PNG) are byte-identical to v0.3.0's.

On 47 8-bit RGB PNGs, 42.7 MP: the 24 Kodak photos and 23 graphics (this repository's 16 SVG charts rendered at
2× and 7 synthetic 1920 × 1080 screenshots). CPU encoders ran as 16 single-threaded processes on the
g6.4xlarge's 16 vCPUs. av1gpu ran in batch mode; its first column is one pass including CUDA start-up, the
second the set repeated 4× (v0.6; v0.5.0 in the same session: 17.4 / 39.3 and 17.8 / 39.9). Lossless is
9–11 % faster. Its frames go through the same entropy coder and batch mode as lossy ones, but not the lossy
analysis, reconstruction or loop filters. v0.9.0 (2026-11-03, `bench/results/aws-g6-g7-v090-check/`) ran the set as
`mk_ll_sets.py` makes it today, 51 PNGs and 49.1 MP (the repository has more charts now), at 21.8 / 45.3 MP/s, and
22.1 / 44.3 with `--ycgco-re`, with palette mode (the default); v0.8.0 in the same session at 20.3 / 43.7 and
20.6 / 42.7. Without palette mode (`--palette off`), as in the table, the set 4× ran at 49.8 and 49.1 MP/s (v0.8.0:
46.8 and 47.2).

| Encoder | Total bytes | vs PNG | Photos | Graphics | MP/s |
|---|---:|---:|---:|---:|---:|
| PNG (the sources) | 18,543,751 | 100 % | 15,394,305 | 3,149,446 | |
| av1gpu `--lossless` | 20,112,025 | 108.5 % | 13,370,817 | 6,741,208 | 18.9 / 43.6 |
| av1gpu `--lossless --ycgco-re` | 16,809,786 | 90.6 % | 11,990,929 | 4,818,857 | 19.4 / 44.5 |
| avifenc `-l -s 6` | 21,516,335 | 116.0 % | 13,686,053 | 7,830,282 | 9.3 |
| avifenc `-l -s d` (libaom's default, cpu-used 0) | 18,861,272 | 101.7 % | 13,324,165 | 5,537,107 | 0.38 |
| avifenc `-l -s 6 --cicp 1/13/16` | 17,103,004 | 92.2 % | 12,396,839 | 4,706,165 | 8.8 |
| cwebp `-lossless` | 12,323,572 | 66.5 % | 11,339,964 | 983,608 | 44.4 |
| cwebp `-lossless -z 9` | 12,187,462 | 65.7 % | 11,252,092 | 935,370 | 1.76 |

Lossless WebP is far smaller on graphics: its back references and colour cache across the whole image do better than
AV1's intra tools there. The table is without palette mode (next section), which makes the graphics 10 % smaller.
One image at a time is limited by the GPU entropy coder (about 126 ms for a Kodak image). The rate estimate behind
the mode choices uses default-CDF costs, so it is approximate.

### Palette mode (screen content and flat colour)

av1gpu codes palettes (AV1's palette mode) in the images that pass a screen-content detection: luma palettes in
lossy frames, luma and chroma palettes in lossless ones. The detection is libaom's own, with one extra condition so
that no photo passes it. Photos are therefore byte-identical to an encoder without palettes, and other images get a
palette search. `--palette on` searches every image, `--palette off` none. Design, scope and the libaom
measurements: [docs/PALETTE.md](PALETTE.md); data: [bench/results/aws-g6-l4-palette-study/](../bench/results/aws-g6-l4-palette-study/summary.md).

BD-rate on SSIMULACRA2 against the same encoder without palettes, the median over images (mean in brackets),
default settings, quality 30–90:

| Images | n | 8-bit 4:2:0 | 10-bit 4:4:4 | vs libaom speed 6 with palette (C) |
|---|---:|---:|---:|---:|
| Screen content: charts, screenshots, web pages | 40 | −23.6 % (−24.0) | −19.5 % (−19.5) | −9.5 % |
| Flat-colour art, PNG: the 11 of 24 that pass detection | 11 | −22.3 % (−16.2) | −12.7 % (−10.3) | +0.7 % |
| Comics, flat-colour JPEGs, painted art, photos | 105 | identical files | identical files | |

`--palette on` (every image, no detection) makes the flat-colour PNGs −6.8 % (−11.4) and their JPEG q90 versions
−5.9 % (−8.9) against main, close to libaom C (+2.7 %, +1.0 %). Detection passes none of the JPEGs.

Lossless: the 23 graphics of the comparison above are 10.1 % smaller in GBR (6.77 → 6.09 MB) and 9.9 % smaller
with `--ycgco-re` (4.84 → 4.36 MB). The whole screen set (40) is 8.7 % smaller and the flat PNGs 3.7 %. Kodak is
unchanged.

Speed, MP/s, with the method of the v0.6 speed check (each set repeated to at least 1,000 MP, 2 rounds with
alternating order, the median of 6 runs):

| Images | main | palette | change |
|---|---:|---:|---:|
| size range, default | 229.3 | 226.3 | −1.3 % |
| size range, `--yuv 444 --depth 10` | 162.2 | 163.8 | +0.9 % |
| standard set, default | 74.4 | 75.0 | +0.8 % |
| main against main (noise) | | | +0.2 % / +0.9 % |
| screen + flat PNG (64 images), default | 152.7 | 57.0 | 0.37× |
| the same, `--palette off` | 155.9 | 150.8 | −3.3 % |
| the same, `--palette on` | 150.6 | 37.2 | 0.25× |
| the same, `--lossless` (`GP_ENCODE_MP=12`) | 51.6 | 49.3 | −4.5 % |
| the same, `--lossless --ycgco-re` | 50.4 | 46.9 | −6.9 % |

Photos run at main's speed: the detection pass is 0.09 ms of a 64 ms photo encode. Screen content runs at 0.37× of
main's speed: the palette search tries up to 14 palettes per block and shape, and palette blocks code one entropy
symbol per pixel.

- **Byte identity, `cmp` of every file against main:** the standard set and the size range, default and
  `--yuv 444 --depth 10`; Kodak `--lossless` and `--lossless --ycgco-re`; the study's comics, flat JPEGs, painted
  pages and photos at five quality points in both formats; and `--palette off` on the images palette is tried on.
- **Decoder gate, dav1d and aomdec, all N/N:** the 64 screen and flat PNGs in lossy 4:2:0 (three qindex values),
  4:4:4, 10-bit, 10-bit 4:4:4, lossless GBR and YCgCo-Re.
- **CPU fuzz** with random palettes: 3,000 cases, 0 mismatches. GPU fuzz test: 500 cases, 0 mismatches. ctest: 9 of 9.

### Transparency

PNG and WebP inputs with transparency (any alpha below the maximum) get an AVIF alpha item (2026-10-31), laid
out as libavif 1.4 writes it:
- **The alpha item:** a second `av01` item holding a monochrome, full-range AV1 image at the colour's size and
  bit depth. It has its own av1C and one-channel pixi, shares the ispe, and carries `auxC` with
  `urn:mpeg:mpegB:cicp:systems:auxiliary:alpha` and an `auxl` reference to the colour item.
- **Straight alpha:** no `prem`. The RGB under transparent pixels is kept as stored.
- **Orientation:** `irot`/`imir` go on both items.
- **Opaque alpha:** a fully opaque alpha channel gets no alpha item, as with avifenc.

Alpha quality defaults to `--quality` through libavif's alpha mapping (q60 → qindex 100), as avifenc does since
libavif 1.2, without the tune=iq tools; `--alpha-quality 100` and `--lossless` use the lossless path. Alpha
frames are encoded as an extra GPU batch next to each batch's colour frames. 16-bit alpha is rounded to 8 bits
in 8-bit mode, kept at 10 bits with `--depth 10`, and must hold 8 or 10 significant bits with `--lossless`.

**Tests** (L4, libavif 1.4.0 `avifdec` with `-c aom` and `-c dav1d`). The test set is 39 generated PNGs plus 4
extras: logos with soft edges, icons with shadows, photo stickers, UI controls, a mostly transparent HUD,
8 Kodak photos with masks, charts, and edge cases (fully transparent, 1 × 1, palette + tRNS, gray + alpha,
16-bit, opaque files, WebP with and without alpha).
- `avifdec --info` shows an alpha item for exactly the 38 files with transparency.
- Decoded alpha equals the encoder's own reconstruction in every configuration. Decoder gates pass in dav1d
  and aomdec, colour and alpha, for the default, `--alpha-quality 80` and `100`, `--lossless`, `--depth 10`
  and `--yuv 444`.
- Alpha equals the source exactly in 38/38 files with `--alpha-quality 100` (also at 10 bits and in
  4:4:4) and in 37/37 with `--lossless`, which also gives exact RGB everywhere, under transparent pixels too.
- Default lossy alpha: PSNR 49.8 dB, mean absolute error 0.21; worst file 35.5 dB. `--alpha-quality 80`:
  53.6 dB.
- Files without transparency (bench and size-range JPEGs; opaque PNG/WebP in every mode) and `--no-alpha` are
  byte-identical to before. Speed on the size-range set is unchanged within noise (114.9 vs 114.4 MP/s
  pooled over 36 runs each).

| Encoder (39 PNGs, 10.1 MP) | Total bytes | Colour | Alpha | Alpha PSNR | MP/s |
|---|---:|---:|---:|---|---:|
| av1gpu default (q60, alpha q60) | 698,892 | 592,901 | 90,523 | 50.2 dB | 50.4 |
| av1gpu `--alpha-quality 100` | 1,088,428 | 592,901 | 480,059 | exact | 43.9 |
| av1gpu `--lossless` | 7,723,961 | 7,222,941 | 485,582 | exact | 30.4 |
| avifenc default (4:4:4, qalpha 60) | 846,207 | 730,647 | 99,452 | 50.9 dB | 14.4 |
| avifenc `--qalpha 100` | 1,274,328 | 730,647 | 527,573 | exact | 12.1 |
| avifenc `-y 420 --qalpha 100` | 1,208,036 | 664,317 | 527,573 | exact | |
| avifenc `-l` | 7,963,112 | 7,419,431 | 527,573 | exact | 8.2 |
| cwebp default (q75, lossless alpha) | 1,059,014 | | | exact | 62.7 |
| cwebp `-lossless` | 4,754,430 | | | exact | 27.4 |

The av1gpu sizes were re-measured with the 4 × 4 default; only `ui_hud_1080p.png` (1920 × 1080) is big enough
to get more tiles, and its colour and lossy alpha grew by 235 and 211 bytes. The lossy colour columns are not
matched for quality. The exact alpha planes are directly comparable:
av1gpu's are about 9 % smaller than libaom's. Speeds are for the set repeated 20× (202 MP), wall time
including start-up, `GP_ENCODERS=4`. The av1gpu rows are without palette mode (`--palette off`), with v0.9.0's
speeds; it ran without alpha (`--no-alpha`) at 57.3 MP/s (v0.8.0 in the same session: 44.4, 39.3, 28.1 and 49.8).
By default palette mode (`--palette auto`, since v0.8.0) takes 15 of the 39 images: the default set is then
685,414 bytes for the default row, and v0.9.0 encodes the set at 31.5, 30.1, 29.3 and 36.8 MP/s (`bench/results/aws-g6-g7-v090-check/`).
Mostly small images and PNG decoding keep the GPU less busy than the photo sets.

**Fixed along the way.** In 8-bit mode, 16-bit PNGs without a gAMA or sRGB chunk were read as linear light by
libpng's simplified API and came out brightened (mid-grey 30000/65535 became 179 instead of 117). They are now
read as sRGB-encoded, as libavif and Pillow read them. 8-bit PNGs and the 10-bit and lossless paths are
unaffected. PNG and WebP ICC profiles and EXIF orientation are still not read.

### CPU threads (PNG and WebP input)

JPEGs are decoded by the L4's hardware JPEG engine, so a JPEG batch keeps only 0.1–0.4 CPU cores busy.
PNG and WebP files are decoded on the CPU. Since 2026-11-01 batch mode gives that decoding one thread per CPU
it may use, instead of 4. JPEG parsing and the CPU entropy decode (`GP_INGEST_DECODER=1`, used on GPUs without
the engine) also get one thread per CPU, as do the writer threads of `--cpu-entropy`. "May use" means the
affinity mask (`taskset`, cpusets), or a cgroup CPU limit (`docker --cpus`, Kubernetes) if that is lower.

Decoded images wait on the GPU until an encoder takes them. At most twice `GP_ENCODE_MP` of them (96 MP)
wait at once, so a fast CPU can't fill the GPU's memory ahead of the encoders. Each encoder batch first takes
the JPEG batches that are ready, then the decoded PNG and WebP frames (see "Mixed lists" below).

**Method.** g6.8xlarge (L4, EPYC 7R13, 32 vCPUs), CUDA 13.2, `sm_89` builds, 4 × 4 tiles (`--tiles 2,2,2`, the
default since 2026-10-31). The 8- and 16-vCPU columns ran under `taskset` on 4 and 8 cores plus their SMT
siblings, which is the core layout of a g6.2xlarge and a g6.4xlarge. A first round on real instances of all
three sizes (below) gave the same PNG gains. Each list repeats its images to a minimum size:
- **JPEG:** the 100 size-range JPEGs, 1,434 MP.
- **PNG:** the same photos saved as RGB PNGs, 956 MP.
- **Mixed:** the same 100 photos, 40 % as PNG or WebP, 956 MP.

The table gives MP/s, main → this version, as the median of 3 runs in alternating order (7 for the 8-vCPU
mixed and PNG lists):

| Input | 8 vCPUs | 16 vCPUs | 32 vCPUs |
|---|---:|---:|---:|
| JPEG | 126.0 → 127.4 | 124.5 → 126.6 | 122.6 → 121.8 |
| JPEG, CPU entropy decode | 120.5 → 121.1 | 117.1 → 120.4 | 115.4 → 121.0 |
| PNG | 95.3 → **107.4** (+13 %) | 98.5 → **112.9** (+15 %) | 98.3 → **116.8** (+19 %) |
| PNG to 10-bit 4:4:4 | 60.6 → **84.3** (+39 %) | 61.1 → **88.6** (+45 %) | 62.7 → **89.4** (+43 %) |
| Mixed | 114.8 → 114.8 | 117.9 → 118.5 | 118.0 → 121.2 |

- **JPEG** doesn't change, because the GPU is the limit; run-to-run noise is about ±3 %. With 32 vCPUs, 5 rounds
  per setting gave 123.4 MP/s for main, 122.7 for one parse thread per CPU, and 123.0 / 124.4 / 124.2 for
  4 / 8 / 16 threads.
- **PNG:** the decode threads now keep the GPU fed. A PNG batch used about 5 CPU cores (3.5 before), and 6.6–6.7
  for 10-bit 4:4:4 (3.6), whatever the number of CPUs.
- **GPU memory:** for 10-bit 4:4:4 PNGs it peaked at 7.1–10.3 GB instead of 4.6–4.7 GB, because the encoders now
  get full 48 MP batches. A JPEG list in 10-bit 4:4:4 already peaks at 12.2 GB. 8-bit PNG peaked at 5.9–6.2 GB
  instead of 4.1–4.3. `GP_ENCODE_MP` lowers it.
- **Mixed lists** were first 2–5 % slower than main, when decoded PNG and WebP frames were taken before JPEG
  batches (as main does). A per-batch log (`GP_BATCH_LOG`, in a diagnostic build) showed why.
  - **Not idle time:** an encoder was running 96–99 % of the time in every version.
  - **The order changed:** fast decoding put 51–56 % of the PNG megapixels in the first half of the run, and only
    19–27 % of the JPEG ones. On main it was 45–48 % and 44 %.
  - **Small JPEGs ended up last:** the JPEG batches of small images run with the slower `base` preset and take
    1.8–2.5 s each, against about 0.9 s for the others. They came near the end, so the run finished with a
    0.8–1.3 s tail with fewer encoders working.
  - **The fix:** take the JPEG batches that are ready first. That moves those batches to the start and leaves
    the large PNG batches for the end. Those are similar in size and spread evenly over the three encoders: a
    0.6 s tail, and all three encoders busy 94 % of the time against 86–93 %.
  - Also tried, at 32 vCPUs (5 rounds; main 113.7, fallback first 115.1, JPEG first 121.7): #23's rule of taking
    fallback frames only as a full batch (113.9), alternating between the two orders (116.7), and at most half a
    batch of fallback frames while JPEGs remain (114.4).
  - Lower priority for the decode threads (nice 10 or 19) and smaller wait queues (12, 24 or 48 MP) made no
    difference either.
- **Thread count** (first round, real 8-, 16- and 32-vCPU instances): twice as many threads as CPUs never
  helped, and was up to 12 % slower. Half as many was 3–14 % slower for 8-bit PNG, and the same for 10-bit
  4:4:4 on 16 and 32 vCPUs. That round's PNG gains were +11 / +13 / +20 % and +36 / +44 / +50 % for 10-bit 4:4:4.
- **Files:** every output was byte-identical to main's at the same settings: 899 runs, 202,500 files,
  including `--cpu-entropy` and the CPU entropy decode.
- **Fixed with it:** `GP_ENCODERS` of 24 or more never finished. No encoder could take an ingest batch,
  with JPEG or PNG input alike. Runs now finish with 24 and 48, and with `GP_INGEST_INFLIGHT=1`.

On a real g6.4xlarge, the size-range JPEGs ran at 128.0 MP/s with the default 3 encoders with v0.4, and run
at 252.4 MP/s with v0.9.0 (the speed tables above). Raw results and scripts: `bench/results/aws-g6-l4-threads/`
(first round) and its `followup/` (the table above, batch logs).

### Crop and resize (contributed by pickae)

`--crop`, `--max-size` / `--max-height`, `--max-mp` / `--max-input-mp` and the per-line `crop=` / `max-size=`
options of `--batch` lists are pickae's ([discussion #16](https://github.com/maddog107/av1gpu/discussions/16)),
written to shrink comic scans to a tablet's 2,960 rows on the GPU. They were ported to v0.6 and checked on a
g6.4xlarge (2026-11-02, the [Setup](#setup-1) instance type; ImageMagick 7.1.2-35 Q16 HDRI). Scripts and raw
results: `bench/results/aws-g6-l4-crop-resize/`.

- **Nothing changes without the options:** 29 runs, 1,748 output files byte-identical to `main` (d7d94b3): the
  size range (default, `turbo3 --small-preset none`, 10-bit, 4:4:4 10-bit; the CPU JPEG decode route too), the
  standard set (default, `slow`), Kodak (lossy, `--lossless`, `--lossless --ycgco-re`), the RGBA set (default,
  `--lossless`), 280 thumbnails, odd sizes from 1 × 1 (8- and 10-bit), gray, WebP (also `--lossless`), 16-bit
  PNGs, a mixed list, 10 single-file runs with three option sets, and the options at values that change nothing
  (a box larger than every image, a whole-image crop). Every existing kernel compiles to the same SASS.
- **Matches ImageMagick:** the encoder's input planes (`--dump-source`) against `magick` with the same crop
  and `-resize 'WxH>'`, its PNG converted by av1gpu the same way. Output sizes were equal in all 14 cases.
  Mean absolute difference (in steps of the output depth) / PSNR of the luma; chroma differs less in 4:4:4, and up
  to 0.4 more in 4:2:0, where av1gpu subsamples before the resize and the reference after it:

  | Case | Output | Luma |
  |---|---|---:|
  | 16-bit RGB PNG of a 24 MP photo, `--max-height 2000`, 4:4:4 10-bit | 3008 × 2000 | 0.19 / 67.2 dB |
  | the same, odd crop (2001 × 1501 + 333 + 777) + resize, 4:2:0 10-bit | 1000 × 750 | 0.19 / 67.0 dB |
  | 24 MP 4:2:0 JPEG, crop only, 4:4:4 10-bit | 1500 × 1000 | 0.73 / 60.5 dB |
  | the same JPEG, `--max-height 2000`, 4:4:4 10-bit | 3008 × 2000 | 0.52 / 62.4 dB |
  | 3840 × 5586 4:4:4 page, `--max-height 2960`, 4:4:4 10-bit | 2035 × 2960 | 0.50 / 62.6 dB |
  | the JPEG, odd crop + resize, 4:2:0 8-bit | 1000 × 750 | 0.10 / 58.1 dB |
  | Kodak PNG, odd crop + resize, 4:2:0 8-bit | 333 × 222 | 0.18 / 55.5 dB |
  | RGBA PNG (1920 × 1080 UI), halved, 4:4:4 8-bit (alpha: identical) | 960 × 540 | 0.06 / 55.4 dB |
  | JPEG with EXIF orientation 6, displayed box 1600 × 1000 | 1000 × 666 stored | 0.20 / 54.7 dB |

  From JPEGs, the difference is mostly ImageMagick's 8-bit RGB decode: a crop alone differs by 0.65–0.73.
  RGBA was compared with `-filter Lanczos` (ImageMagick's default for images with alpha is Mitchell), colour
  where the alpha is visible; both weight colour by alpha.
- **Decoder gate** (dav1d and aomdec), 16 runs, all N/N: 30 photos 4:2:0 8-bit at qindex 60 and 120, 4:4:4
  10-bit, 4:2:0 10-bit and `--preset slow`, each with a crop at an odd offset and an odd box, two of them also
  through the CPU JPEG decode route (`GP_INGEST_DECODER=1`); RGBA (alpha items too) 8-bit and 4:4:4 10-bit;
  WebP; Kodak `--lossless` with crop + resize and with `--ycgco-re`, and 4:4:4 with a crop only; RGBA
  `--lossless`; odd sizes, gray and 280 thumbnails. `--check` is bit-exact on 10 single images (4:2:0, 4:4:4
  10-bit, gray, EXIF-rotated, RGBA, lossless). ctest passes, and compute-sanitizer memcheck finds no access
  errors in six crop + resize runs, the 82 MP source among them (only main's known leak reports of encoders not
  freed at exit).
- **Large pictures** (5587 × 7365 = 41 MP and 11174 × 7365 = 82 MP JPEGs, 4:2:0 and 4:4:4, a 41 MP PNG):
  without the options they are skipped as on `main`, with the same messages. `--max-mp 100` encodes them at
  124 MP/s, written with level 31; GPU memory (`nvidia-smi`) peaked at 4,927 MiB for the five in one batch (3,551 MiB
  for the 82 MP image alone), host memory at 1.9 GB; the gate passes 5/5 (8-bit, and 4:4:4 10-bit) and `--check` is
  bit-exact on 41 and 82 MP. `--max-input-mp 100 --max-height 2960` decodes them at full size and writes
  2245 × 2960 and 4491 × 2960 (levels 5.0 and 6.0), GPU peak 3,135 MiB. PNGs are decoded on the CPU, one per
  `GP_FALLBACK_WORKERS` thread at a time: 16 41 MP PNGs held 7.4 GB of host memory.
- **Speed.** Without the options, size range default (1,434 MP), four rounds of a warm-up and 3 runs per
  binary: 235.8 MP/s for `main`, 233.4 for the branch (median of 12); run by run interleaved, 230.9 against
  231.4, and 232.2 for a copy of `main`'s binary, so the difference is noise. 40 3840 × 5586 4:4:4 pages
  (made from size-range photos) × 4, `--yuv 444 --depth 10 --preset turbo`: 169.0 MP/s of output with
  `--max-height 2960` on the GPU, against 195.2 for the same pages shrunk by ImageMagick beforehand (which then
  costs a CPU pass and a JPEG generation). The resize kernels take 1.3 ms per page; the rest is decoding 3.6×
  the pixels on a GPU that is already busy. Holding more ingest batches per encode (`GP_INGEST_INFLIGHT=48`)
  filled the encodes like the small pages (20 batches instead of 27) but was no faster. pickae measured
  66–73 against 72–79 MP/s on comic pages on an RTX 5090.

### CVVDP: a second metric

SSIMULACRA2 does not model screen size or viewing distance, so the same size-range files were also scored
with CVVDP (ColorVideoVDP 0.5.7) for its `standard_fhd` display: a 24-inch 1920 × 1080 screen at 200 cd/m²
seen from 0.6 m (37.8 pixels per degree), with images shown at 100 % (one image pixel per screen pixel).
All 4,752 files were re-encoded on an L4 and were byte-identical to the published ones, and their
SSIMULACRA2 scores were identical too. Raw results: `bench/results/aws-g6-l4-cvvdp/`
(`bench.py --cvvdp standard_fhd`). For the 4 × 4 default, av1gpu's files were encoded and scored again the same
way (`bench/results/aws-g6-l4-tiles4x4/`); its unchanged files below 2 MP got the same scores to the last digit.

| Encoder | BD-rate vs libaom speed 6, JOD | SSIMULACRA2 | Bits per pixel at 9.5 JOD | Photos reaching 9.5 JOD |
|---|---:|---:|---:|---:|
| libaom speed 6 | 0 | 0 | 0.448 | 99/99 |
| **av1gpu default** | **+4.2 %** | +0.1 % | 0.462 | 99/99 |
| libaom speed 9 | +14.5 % | +15.0 % | 0.522 | 99/99 |
| SVT-AV1 preset 4 | −2.0 % | −2.6 % | 0.430 | 99/99 |
| SVT-AV1 preset 6 | +2.0 % | +2.1 % | 0.468 | 99/99 |
| SVT-AV1 preset 9 | +9.3 % | +9.8 % | 0.491 | 99/99 |
| NVENC AV1 | +28.4 % | +29.5 % | 0.541 | 85/99 |
| WebP | +44.2 % | +42.1 % | 0.666 | 99/99 |

(Median BD-rates over 99 photos, NVENC 97; quality points as published plus q85/q90 and matching higher
WebP/NVENC points. Bits per pixel are medians, interpolated per photo.)

- **The metrics agree on every encoder except av1gpu.** CVVDP rates av1gpu worse than SSIMULACRA2 does on 84
  of the 99 photos, and the gap grows with image size (+2.1 % below 1 MP, +6.1 % at 10 MP and over). Scored
  without colour, CVVDP puts it at +6.2 %, so the difference is in luma. av1gpu's search was tuned with
  SSIMULACRA2; CVVDP suggests its luma decisions on large photos leave some quality behind.
- **Speed at CVVDP's file sizes:** at av1gpu's +4.2 %, interpolating between presets and speeds, SVT-AV1 runs
  at about 73 MP/s (between presets 6 and 9) and libaom at about 49 MP/s, so av1gpu (v0.9.0, 261 MP/s) is about
  3.6× and 5.3× faster. On SSIMULACRA2 the same comparison gives about 8.6×.
- **With v0.3.0's 2 × 2 tiles** av1gpu was +3.9 % on CVVDP (0.459 bits per pixel at 9.5 JOD; +5.9 % at 10 MP and
  over), so the 4 × 4 default costs about a quarter of a point on both metrics.
- 9.5 JOD is a "high fidelity" threshold suggested on Reddit for a 27-inch 1080p screen at about 20 inches,
  which works out to about 28.5 pixels per degree, so artefacts there are easier to see than in this run.

### Correctness

| Check | Result |
|---|---|
| `tests/decoder_gate.py`, standard set, qindex 60/120/180, nvJPEG and `GP_INGEST_DECODER=1` | 300/300 and 300/300 identical in dav1d and aomdec |
| `tests/decoder_gate.py`, size range, qindex 60/120/180 (nvJPEG) and 120 (CPU JPEG route) | 300/300 and 100/100 |
| `ctest` (CPU fuzz, dav1d + aomdec) and `gpu_postfilter_test` | pass; 100 cases, 0 mismatches |
| `gpu_fuzz_test --count 2000` | failed (a test bug, below); after the fix 15/15 runs, 30,000 cases, 0 mismatches |
| compute-sanitizer memcheck: `av1gpu --batch` on 10 images, `gpu_fuzz_test` | 0 errors |
| compute-sanitizer initcheck: `gpu_fuzz_test`, 3 seed ranges | 0 errors |
| compute-sanitizer racecheck: `gpu_fuzz_test` | 4 reports in `k_entropy_win` (below) |
| EXIF orientation 1–8, one-image and batch mode | `irot`/`imir` (and their essential flags) identical to avifenc's on all 8 |
| one-image mode: JPEG, PNG; CMYK JPEG | encode and decode; CMYK rejected with exit status 3, `CMYK/YCCK JPEG` |
| 4:4:4 and 10-bit (2026-10-31): `tests/decoder_gate.py`, 67 runs at q60 and q120 (8-bit default, turbo3, slow, slow `--small-preset none`; `--yuv 444`; `--depth 10`; both; 16-bit PNG, PNG/WebP, 40 large images; CPU JPEG route and CPU entropy) | 67/67 runs identical in dav1d and aomdec, 5,732 encodes |
| 4:4:4 and 10-bit: CPU fuzz (4:2:0, 4:4:4, mono × 8/10-bit), `gpu_fuzz_test`, `gpu_postfilter_test`, `gpu_dsp_test`, `varboost_test` | 0 mismatches (36,000+ CPU fuzz cases; GPU tests per layout and depth) |
| 8-bit 4:2:0 output after the 4:4:4/10-bit merge | 600/600 files byte-identical to before (default, turbo3, slow; both sets) |
| 4 × 4 default (2026-10-31): `tests/decoder_gate.py`, size range, default at qindex 60/120/180, `--yuv 444 --depth 10` and `turbo3 --small-preset none` at 120 | 500/500 identical in dav1d and aomdec |
| 4 × 4 default: files below 2 MP and lossless files against v0.3.0 | byte-identical: 3,996 lossy files (both sets, every format and preset run, RGBA) and 197 lossless |
| v0.6 (main 6a4956e, 2026-11-01) against v0.5.0, every output file: 63 batch runs (size range with 22 option sets including every preset, q30–95, 4:4:4, 10-bit, `--tune none`, `--tiles 2,1,1` and 1 or 6 encoders; standard set; 280 thumbnails; 47 monochrome; Kodak lossy and lossless; RGBA; gray + alpha; odd sizes from 1 × 1 to 4097 × 17; 16-bit PNG; WebP; a mixed list) and 3 × 10 single-file encodes | byte-identical: 4,698 files, 0 differences (`bench/results/aws-g6-l4-speed-v06/`) |
| v0.6: `tests/decoder_gate.py`, 14 runs (30 photos at qindex 13/26/60/120/200, 4:4:4 10-bit, 4:4:4 at 13/26, `--preset slow` in 4:2:0 and 4:4:4, RGBA, Kodak lossless, thumbnails, odd sizes, monochrome) | 676/676 encodes and 38/38 alpha items identical in dav1d and aomdec |
| v0.6: `ctest`, `gpu_fuzz_test` (600 cases), `gpu_postfilter_test` (200), `gpu_dsp_test`, `varboost_test` | pass, 0 mismatches |
| v0.6: compute-sanitizer memcheck and synccheck, `av1gpu --batch` on 3 photos (0.8–2.2 MP), default and `--yuv 444 --depth 10` | 0 errors (memcheck lists only the 256 KB variance-boost tables of each encoder, which are not freed at exit; that code is unchanged since v0.5.0) |
| v0.6: racecheck, same runs | the `k_entropy_win` reports as in v0.5.0, plus 1 report pair in `k_recon` (below) |
| After v0.6.0, #43: racecheck, same runs, and `cmp` against v0.6.0 (13 sets, 1,267 files) | only the `k_entropy_win` reports; byte-identical (`bench/results/aws-g6-l4-racecheck-fix/`) |
| v0.9.0 (main c553ffb, 2026-11-03, the release workflow's Linux packages) against the v0.8.0 tarballs, every output file: 103 runs (v0.6's identity list, the palette sets at four settings, short `--batch` lists with a photo and its thumbnail, single-file mode on 103 files from every set at six settings, error exits), on the L4 and on an RTX PRO 4500 Blackwell | byte-identical: 5,679 files on each GPU, and the same 5,679 files on both GPUs; the CUDA 12 builds (27 runs, 1,220 files) too (`bench/results/aws-g6-g7-v090-check/`) |
| v0.9.0: `tests/decoder_gate.py`, 18 runs on each GPU (30 photos at qindex 60/120/200 and 4:4:4 10-bit, thumbnails, RGBA, odd sizes, monochrome, Kodak lossless, the palette sets at default, `--palette on`, 4:4:4 10-bit and lossless) | 691/691 encodes and 35/35 alpha items identical in dav1d and aomdec, on both GPUs |
| v0.9.0: compute-sanitizer memcheck, racecheck and synccheck on 8 jobs per GPU (3 photos in batch mode at default and 4:4:4 10-bit, a photo and its thumbnail in batch mode, single-file mode on a photo, two thumbnails, a screen PNG and a lossless PNG) | memcheck 0 access errors (batch mode lists only the variance-boost tables, as since v0.6); racecheck only the `k_entropy_win` reports; synccheck 0 errors. On Blackwell, memcheck needs compute-sanitizer 2026.3 (CUDA 13.4): 2026.1 misreports CUB 3.4's scan kernel in the CUDA 13.4 builds there, also in a standalone test of CUB alone |

**`gpu_fuzz_test` failure.** The test failed with `device error 1` (`AV1G_GPU_ERR_SYMBOLS`) at a different
batch from run to run (seed 521, 1481, ...), never under initcheck, which serialises GPU work. The test
uploaded its records with `cudaMemcpy` from pageable memory, which can return before the copy has reached the
GPU, and then ran the entropy coder on a `cudaStreamNonBlocking` stream, which does not wait for the legacy
stream. A batch could therefore be coded from a partly uploaded record, so the count reported for the rerun
was too small. The uploads now go through `cudaMemcpyAsync` on the test's stream, and the failure is gone.
The encoder pipeline produces its records on the GPU and was not affected (the decoder gates above). With
seeds other than the default, the test's swapped-block check can also fail when the two swapped blocks are
the same size (seed 521: two 4×4 blocks), because the record is then still valid; the default seed is not
affected.

**racecheck.** The four reports are all in the windowed entropy coder (`gpu_entropy.cu`: `vload`/`vstore`
of `WinShared::prod`/`cons` and the ring buffer). That kernel hands windows from one warp to the other
through volatile flags and `__threadfence_block()` rather than barriers, and racecheck checks barrier
synchronisation only, so it reports every ring access. The GPU entropy coder's output was byte-identical
to the CPU writer in all fuzz runs and gates above. The "clean" racecheck in the RTX PRO 2000 section was
on representative inputs during development; this kernel is not racecheck-clean as such.

On `av1gpu --batch` (v0.6, 3 photos) racecheck reports those `k_entropy_win` pairs with the same hazard
counts as v0.5.0, and since #37 one more pair in `k_recon`'s concurrent U / V chroma finals
(`k_recon.cuh`, the read of `S.eobUV` after a barrier against thread 0's update of `S.nTxb`). The compiler
reads `eobUV[0..1]` with one 16-byte shared load (`LDS.128`) that also covers `nTxb`. Every thread loads
that word, but only thread 0 uses it, and thread 0 is the one that writes it. No thread uses a value that
races, so the output can't change. The report first appears with #37 (980cc10; 63683ef just before it has
only the entropy pairs). #43, after v0.6.0, reads the two values one word at a time: the report is gone, the
output is byte-identical and the speed unchanged (`bench/results/aws-g6-l4-racecheck-fix/`).

### Compared with the RTX PRO 2000

The image sets differ, so this is a rough comparison. The development set was private forum uploads
(IJG quality 90, mean 4.2 MP); the size-range set here has a similar mean size (4.8 MP) but mostly camera
originals. With that caveat:

- **Speed was about the same with the same code:** 104–116 MP/s with the default settings of v0.3.0 and
  earlier on the L4 against 110.8 MP/s on the RTX PRO 2000. Both cards are power-limited (70–72 W). v0.9.0 runs
  at 226–261 MP/s on the L4.
- **Compression matches:** +0.1 % against libaom speed 6 here (−0.1 % with v0.3.0's 2 × 2 tiles, the tiling
  the held-out set was measured with), −0.6 % on the held-out set. libaom speed 9 is +15.0 % here, +14.0 % there.
- **The CPU here is much faster per thread** (libaom speed 6: 1.8 MP/s per vCPU, against 1.27 on the
  Xeon Silver 4110; speed 9: 11.7 against 8.45). Against the whole 16-vCPU machine, av1gpu (v0.9.0) is 9.1×
  libaom speed 6 at the same file size, and 1.4× libaom speed 9, whose files are 15 % larger.
- **Cost of the run:** 1 h 39 min on a g6.4xlarge spot instance at about $0.80/hour, plus about 2 minutes
  on demand, roughly $1.40.

### Windows

The Windows build (2026-11-01, v0.5.0) on a second g6.4xlarge, against the Linux build of the same commit on a
first one. Scripts, output hashes, exit codes and speed runs: `bench/results/aws-g6-l4-windows/`.

- **Setup:** Windows Server 2022 (21H2, build 20348). av1gpu built with `tools/windows/build.ps1`: Visual
  Studio 2022 Build Tools 17.14 (MSVC 19.44), CUDA 12.9.1, vcpkg 2026.10.16 (libjpeg-turbo 3.2.0, libpng
  1.6.59, zlib 1.3.2, libwebp 1.6.0). Linux: the instance of [Setup](#setup-1) with driver 595.112.07, CUDA
  13.2, gcc 13.3.
- **Driver:** NVIDIA's data-centre driver for Windows (581.42) runs the L4 in TCC mode, which has no CUDA
  memory pools, so `cudaMallocAsync` fails and av1gpu stops with an error that says so. AWS's GRID driver
  (596.86) runs it in WDDM mode, as GeForce and RTX cards always are on Windows; the results below are
  with that driver.
- **Output is byte-identical:** 30 runs over JPEG, PNG, WebP, mixed and RGBA sets, single images and the
  error cases. Among them 4:4:4, 10-bit, `--qindex 40,200`, `--cpu-entropy`, `turbo3`, `slow`, lossless
  (also `--ycgco-re`), alpha at the default quality, exact and dropped, a directory input, nested
  `--outdir`, non-ASCII file names and 24 encoders. All 1,226 output files matched the Linux build's and
  `main`'s, and so did every exit code. Only a libpng warning differs: Windows' libpng 1.6.59 printed no
  "known incorrect sRGB profile" warning for one test PNG. The zip the release workflow built (every
  architecture, from the CI runner) gave the same 1,226 files.
- **Decoder gate:** `tests/decoder_gate.py` needs dav1d and aomdec, so it ran on Linux: 11 runs (size range
  at qindex 60/120/180, 4:4:4 10-bit, `turbo3`, alpha lossy and exact, lossless with and without alpha, the
  CfL edge image), 633/633 encodes and 113/113 alpha items identical in both decoders. The Windows zip,
  given the same commands, wrote the same 2,125 files (AVIFs, OBU streams and reconstructions), byte for
  byte, so the result holds for it too.
- **Speed is lower:** MP/s with v0.5.0 (these runs were not repeated for v0.6 or later, so they compare with each
  other, not with the speeds above), sets repeated to about 800–1,400 MP, one warm-up and the median of 3;
  CPU cores in brackets. "Windows" is WDDM as installed.

| Set | Linux | Windows | Windows, GPU scheduling on |
|---|---:|---:|---:|
| JPEG (size range × 3, 1,434 MP) | 127.5 (0.17) | 76.8 (0.11) | 85.6 (0.12) |
| PNG (40 photos of 5–24 MP × 2, 819 MP) | 137.6 (6.3) | 92.1 (4.9) | 114.2 (6.0) |
| Mixed JPEG / PNG / WebP (× 3) | 123.3 (2.2) | 73.0 (1.8) | 89.2 (2.1) |
| Mixed, `--yuv 444 --depth 10` | 97.4 (2.9) | 59.8 (1.5) | 70.6 (1.7) |

"GPU scheduling on" is Windows' hardware-accelerated GPU scheduling (`HwSchMode=2`, on by default on many
Windows 11 PCs), which helped every set; that run also had the final code's 1 ms timer resolution, which on
its own changed nothing (76.9 against 76.8 MP/s on the JPEG set). The encoders' GPU stages were no slower on Windows, but the
encoders waited longer between them: on the JPEG set they waited 15.3 s in all for decoded JPEGs, against
0.4 s on Linux, although the NVJPG decode itself took as long (6.6 s against 6.3 s). A 1 ms timer
resolution, the other CUDA scheduling modes (spin, yield, auto), fewer hardware queues
(`GP_MAX_CONNECTIONS`) and fewer encoders or entropy threads did not help. A GeForce card under the desktop driver may behave differently; this
is the only Windows measurement so far.

The zips from #51's release-workflow run (v0.7.1's code, 2026-11-02): the CUDA 13 zip and the `-cuda12` zip, on a g6.8xlarge's L4
under Windows Server 2022 with AWS's GRID driver 582.78 (R580, WDDM, GPU scheduling on). See
[Release builds](#release-builds-cuda-13-and-cuda-12).
- **Output:** the files are identical to the Linux tarballs of the same toolkit (size range, defaults and
  `--yuv 444 --depth 10`, and Kodak; 224 files each).
- **Start-up:** 1.1–1.3 s in every cache state.
- **Speed:** 133.9 MP/s (CUDA 13) and 131.5 (`-cuda12`) on the size range × 3. That instance has 32 vCPUs, so these
  numbers do not compare with the table above.

## Release builds: CUDA 13 and CUDA 12

From v0.7.1 each release has a CUDA 13.4 build (driver 580+) and a CUDA 12.9 `-cuda12` build (driver 525+ on
Linux, 528+ on Windows) for Linux and Windows. AWS, 2026-11-02; scripts, logs and hashes are in
`bench/results/aws-g7-g6-cuda13-builds/`.

- **Why ([#49](https://github.com/maddog107/av1gpu/issues/49)):** from R615, the driver re-finalizes every
  Blackwell kernel built by nvcc older than 13.2.2 when it loads. This fixes a reconvergence bug
  ([615.71.09 notes](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-615-71-09/index.html)).
  - The v0.6.0 binary (CUDA 12.9) was recompiled on every start without a warm compute cache, 226 MB of cache.
  - On an RTX PRO 4500 Blackwell (g7.4xlarge) with the reporter's user-mode driver (615.71.10, from
    `cuda-compat-13-4`), one JPEG in `--batch` took 9.3–9.7 s with an empty cache, 1.2–1.3 s warm and 8.8–9.2 s
    with the cache disabled. On 595 it took 1.0 s, and v0.5.0 took 13.1–13.8 s.
  - It isn't PTX JIT: `CUDA_DISABLE_PTX_JIT=1` changed nothing.
  - A one-kernel test got a cache entry when built by nvcc 12.8, 12.9, 13.0 or 13.2 (V13.2.51), as `sm_120`,
    `sm_120a` or `sm_120f`. Built by 13.3 or 13.4, it got none.
  - The L4 is not affected.
  - `CUDA_SELECTIVE_DEVICE_CODE_RECOMPILE=DISABLED` (patching only) starts fast, but its kernels encode 15 % slower
    (298 against 349 MP/s).
- **The CUDA 13.4 build** of the same code starts in 1.2–1.4 s on the RTX PRO 4500 in every cache state, on 595
  and on 615. It also encodes faster, on the size range × 3 (median of 6, 2 alternating rounds):

| GPU, driver | CUDA 12.9 build | CUDA 13.4 build |
|---|---:|---:|
| RTX PRO 4500, 595 | 351.0 MP/s | 406.9 (+16 %) |
| RTX PRO 4500, 615 (12.9 re-finalized, warm cache) | 352.9 | 404.0 (+14 %) |
| L4, 580.196.07 | 212.9 | 223.0 (+5 %) |
| L4, 595.112.07 (#51's tarballs) | 239.7 | 242.9 (+1 %) |

- **Same files:** #51's tarballs (CUDA 13.4 and `-cuda12`, v0.7.1's code) gave identical files (`cmp`) on the L4: the size
  range (defaults and `--yuv 444 --depth 10`), the standard set and Kodak, 324 files.
  - The Windows zips match them ([Windows](#windows)).
  - Earlier, on the RTX PRO 4500 and the L4, the 13.4 build matched v0.6.0 on 300 files per run.
- **v0.9.0 on the RTX PRO 4500** (driver 595, 2026-11-03, `bench/results/aws-g6-g7-v090-check/`): the CUDA 13.4
  package ran the size range × 3 at 434.4 MP/s against v0.8.0's 406.4 (+6.9 %), the 280 thumbnails at 831 against
  776 images/s, and one JPEG as its own process (2.5 MP, CUDA start-up included) in 0.51 s against 1.03 s. Its
  5,679 test files are byte-identical to v0.8.0's and to the L4's, and the `-cuda12` package's to the CUDA 13.4 one's.
- **v0.9.0's published packages** (built from the tag; `bench/results/aws-g6-l4-v090-published/`): since #58,
  `k_an_c.cu`'s SASS is not reproducible between two builds for a few random architectures (same PTX). Against the
  tested build, that hit sm_75 / 86 / 90 in the Linux CUDA 13.4 package (sm_89 and sm_120 identical) and sm_89 among
  others in the `-cuda12` one. Run on an L4, both Linux packages wrote the same files as v0.8.0's and as each other
  (4,394 outputs), and passed the decoder gate. memcheck was clean, and racecheck reported only `k_entropy_win`'s
  known hand-off.
- **Driver 580:** the CUDA 13.4 build ran on 580.196.07 (L4) with identical output.
- **GPUs newer than the SASS:** a forced PTX JIT (`CUDA_FORCE_PTX_JIT=1`) of the 13.4 build was still compiling
  after 10 minutes on the RTX PRO 4500. That is the first run on a GPU with no compiled code (README:
  `CUDA_CACHE_PATH` / `CUDA_CACHE_MAXSIZE`).
- **Build time:** in the release workflow, the CUDA 13.4 jobs took 2 h 36 min (Linux) and 2 h 29 min (Windows).
  The `-cuda12` jobs took 49 and 47 min. Most of the difference is the k_recon files.
- **Not tested:** Windows on Blackwell. AWS's GRID driver bucket stops at 596.86 (R595), before the recompile, and
  the data-centre driver runs data-centre GPUs in TCC mode, where av1gpu stops.

## GeForce RTX 5090 (contributed by pickae)

Contributed by pickae ([discussion #19](https://github.com/maddog107/av1gpu/discussions/19)) and measured on
their machine, not by us. The first run (v0.2.0, below) was speed only: avifenc, libaom and `ssimulacra2` were not
installed there, so it has no CPU baselines or BD-rates. The output was byte-identical to the L4's (see
Correctness), so the L4's compression results apply unchanged. A second run with v0.9.0 added the CPU baselines
and BD-rates ([v0.9.0](#v090-the-full-benchpy)). The raw results are not in `bench/results/`.

### Setup

- **GPU:** NVIDIA GeForce RTX 5090 (Blackwell, compute capability 12.0, 170 SMs, 32 GB, 575 W board power
  limit), driver 615.71.09, CUDA 13.4 (V13.4.92), built with `-DCMAKE_CUDA_ARCHITECTURES=native`. Ubuntu
  26.04.1, kernel 7.0.0-38, gcc 15.2, CMake 4.2.3.
- **CPU:** Intel Core i9-13900K, 32 threads.
- **Code:** `cbea9b8`, the same code as 0.2.0.
- **Images:** the standard set and 99 of the 100 size-range photos (one had changed on Flickr, as in the L4
  `turbo3` run).
- **Speed runs** as on the L4 (`bench.speed_set` / `bench.gpu_speed`): 1,023 MP (standard) and 1,433 MP (size
  range) per run, one warm-up, median of 3. Power and SM clock: `nvidia-smi` every second, averaged over the
  samples above 50 % utilisation.

### Speed

MP/s at quality 60, 8-bit 4:2:0. av1gpu used 0.12–0.36 CPU cores in every run.

| Setting | standard (~1 MP) | size range | L4 standard, same version | L4 size range, same version |
|---|---:|---:|---:|---:|
| default (`GP_ENCODERS=3`) | 121.1 | 225.8 | | |
| default, `GP_ENCODERS=2` | 110.4 | 195.7 | 39.2 | 103.8 |
| default, `GP_ENCODERS=4` | 130.3 | 256.5 | 39.2 | 116.0 |
| `--small-preset none` | 290.6 | 270.5 | 100.9 | 126.8 |
| `--preset turbo3` | 131.8 | 370.4 | | 167.1 |
| `--preset turbo3 --small-preset none` | **416.5** | **462.2** | | 209.8–213.9 |
| default, `--tiles 2,2,2` (4 × 4 tiles from 2 MP; the default since 2026-10-31) | | 314.4 | | |

Run-to-run spread was within ±1 % on the standard set and ±6 % on the size range.

- **About twice the L4** on the size range with the same number of encoders (195.7 against 103.8 with 2,
  256.5 against 116.0 with 4) and 2.8–3.3× on the standard set; 2.2× with `turbo3 --small-preset none`. Here 4
  encoders beat the default 3 by 8–14 %.
- **Not power-limited:** 175–280 W of the 575 W limit at a 2.75 GHz SM clock, 0.83 J/MP on the size range with
  the default settings and 0.44 J/MP with `turbo3 --small-preset none` (the L4 with the same version: about
  0.5 J/MP at 116 MP/s).
- **More tiles help on this GPU:** `--tiles 2,2,2` (4 × 4 tiles instead of v0.2.0's 2 × 2 for images of 2 MP
  and more) was 39 % faster on the size range for 0.21 % more bytes (67,558,719 against 67,417,788). With
  `--prof`, the reconstruction used 5–12 % of the GPU's CTA-cycle capacity, so the wavefront's serial chain is
  the limit there ([discussion #18](https://github.com/maddog107/av1gpu/discussions/18)). 4 × 4 has been the default
  for every `turbo` preset since 2026-10-31; the other rows here use v0.2.0's tiling (`turbo3`: 4 × 2).

### 4:4:4 and 10-bit

Size range, default settings:

| Format | MP/s | vs 8-bit 4:2:0 | L4 MP/s, same version |
|---|---:|---:|---:|
| 8-bit 4:2:0 | 225.8 | | 113.4 |
| 8-bit 4:4:4 | 192.1 | −15 % | 94.5 |
| 10-bit 4:2:0 | 233.8 | +4 % | 109.8 |
| 10-bit 4:4:4 | 186.2 | −18 % | 92.8 |
| 10-bit 4:4:4, `--tiles 2,2,2` | 261.5 | +16 % | |

As on the L4, 4:4:4 costs 15–18 %; 10 bits made no measurable difference.

### Comic pages

Not a bench set: 60 scanned comic pages of 2035 × 2960 (JPEG q94, 4:4:4) against 20 size-range photos of 5–9
MP, `turbo`, MP/s:

| | photos | comic pages |
|---|---:|---:|
| 8-bit 4:2:0 | 191–206 | 124–144 |
| 10-bit 4:4:4 | 147–166 | 89–102 |
| 10-bit 4:4:4, `--tiles 2,2,2` | 246 | 149–150 |

The pages run at 60–70 % of the photos' speed: on line art the reconstruction does about 3.5× more work per
superblock (more, smaller blocks), and the files take more bits per pixel (0.70 against 0.41). Tiles speed both
up by about the same factor.

### Correctness

- **Byte-identical to the L4** where the sets are the same: the standard set gave 5,494,550 bytes per pass with
  the default settings and 71,872,515 bytes for 1,023 MP with `--preset turbo --small-preset none`, the same
  totals as in `bench/results/aws-g6-l4/` and `bench/results/aws-g6-l4-turbo-nosmall/`. The size-range totals
  differ only by the missing photo.
- `tests/decoder_gate.py` was not run (no dav1d or aomdec on that machine).

### v0.9.0: the full `bench.py`

pickae's second run (2026-11-03): v0.9.0 (`7c2bf33`, native build) on the same machine, with `bench.py` as the
README gives it. Baselines: avifenc 1.3.0 with libaom 3.13.1 (`tune=iq`), cwebp 1.5.0, `ssimulacra2` from libjxl
v0.11.1. Standard set, 1,023 MP per speed run, quality 60:

| Encoder | MP/s | CPU cores | L4, v0.9.0 |
|---|---:|---:|---:|
| av1gpu default | **214.3** | 0.3 | 81.2 |
| libaom speed 6, 32 × avifenc `-j 1` | 33.2 | 32 | 21.1 (16 vCPUs) |
| libaom speed 9, 32 × avifenc `-j 1` | 263.8 | 32 | 166.1 (16 vCPUs) |

- **2.6× the L4** on the standard set, and 1.77× this card's v0.2.0 result (121.1 MP/s).
- **Against the whole i9-13900K** (32 threads): 6.5× libaom speed 6 and 0.81× libaom speed 9, at 0.3 CPU cores (the
  L4 against its 16 EPYC vCPUs: 3.9× and 0.49×).
- **av1gpu wrote 71.43 MB**, 13 × 5,494,550 bytes, the same total as on the L4.

BD-rate on SSIMULACRA2 against libaom speed 6, median (mean):

| Encoder | RTX 5090 run | L4 |
|---|---:|---:|
| av1gpu default | −0.5 % (−1.4 %) | +0.3 % (−0.1 %) |
| libaom speed 9 | +16.5 % (+19.5 %) | +17.7 % (+20.7 %) |
| WebP | +40.2 % (+43.1 %) | +40.4 % (+44.3 %)³ |
| av1gpu default vs WebP | −29.7 % (−30.5 %) | −29.7 % (−30.5 %)³ |

av1gpu's files are the same on both machines, and so are WebP's, so the av1gpu-vs-WebP row matches to the digit.
The rows against libaom speed 6 move because the reference differs. avifenc 1.3.0 maps `-q` to the quantizer as
(100 − q) × 63 / 100 for every tuning; 1.4.0, used on the L4, has a table made for `tune=iq`. So `-q 60` is
quantizer 25 in 1.3.0 and 30 in 1.4.0 (30 to 80: 44–13 against 49–15). The same libaom is sampled at lower
quantizers, its quality-60 files are 27 % larger (107.55 against 84.50 MB), and the curves overlap over a
different range. This run's WebP mean led to the standard set's re-score with ICC profiles (³ in
[Compression](#compression-1)).

pickae also timed their media tool end to end, including its CPU-side work (unpacking, page sorting, trim
measurement). It calls `av1gpu --batch` with `--crop` / `--max-size` per line, `--tiles 2,2,2` and `--max-mp 100`:

- **961 photos** of about 17 MP (16,470 MP, up to 48 MP), 10-bit 4:2:0 `turbo`: 22.7 s, over 700 MP/s, with the
  card at 90 % utilisation or more for 85 % of the run.
- **61 comic books** (about 37,250 MP of pages of about 3.3 MP), trimmed and shrunk on the card, 4:4:4 10-bit
  `turbo`: 249 s. The card averaged 75 % utilisation; the tool's CPU-side page sorting was the limit.
- Their CPU path (ImageMagick 7.1.2 / libheif with libaom speed 5 and 4, every thread), with quality settings
  calibrated to the same SSIMULACRA2, took 60× and 25× as long. Those outputs were not scored.
