# Measured results

Two GPUs so far: the RTX PRO 2000 Blackwell the encoder was developed on (private images, below), and an
NVIDIA L4 on AWS with the public `bench/` image sets ([NVIDIA L4](#nvidia-l4-aws-g64xlarge)).

**The L4 run is the reference.** It uses public images, current baselines and a current CPU (AMD EPYC, 16
vCPUs). The development section below compares against the development server's dual Xeon Silver 4110,
which is slow per thread, so its CPU figures flatter the GPU. It is kept for its larger held-out image set.

## RTX PRO 2000 Blackwell (development)

These measurements were taken during development, before the code was extracted into this
repository. The encoder source is the same; only the CLI and build were reorganized. The test images
were private and are not distributed. `bench/` runs the same kind of comparison on public images,
and results from other GPUs are welcome.

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
- **Speed runs** repeat each set up to at least 1,000 MP (`bench.py --min-mp`; one warm-up, median of 3).
  A single pass over the standard set is 79 MP, less than two GPU batches; that measured 28.1 MP/s instead
  of 39.2 MP/s, because start-up and pipeline fill dominate such a short run.

### Speed

MP/s at quality 60. The CPU rows use the whole machine; av1gpu uses about 0.1–0.2 CPU cores.

| Encoder | standard (~1 MP) | size range | CPU cores |
|---|---:|---:|---:|
| av1gpu default, `GP_ENCODERS=2` | 39.2 | 103.8 | 0.1 |
| av1gpu default, `GP_ENCODERS=4` | 39.2 | **116.0** | 0.1 |
| av1gpu `--small-preset none` (`turbo` for every image) | **100.9** | **126.8** | 0.1–0.2 |
| libaom speed 6, 16 × avifenc `-j 1` | 21.1 | 28.8 | 16 |
| libaom speed 9, 16 × avifenc `-j 1` | 166.1 | 187.6 | 16 |

libaom speed 10 gave the same files and speed as speed 9. While encoding, the L4 averaged 97–100 % SM
utilisation at 54–59 W (`nvidia-smi dmon`, 1 s samples), about 0.5 J/MP at 116 MP/s. On the standard set
the number of concurrent encoders made no difference; on the size range 4 beat 2 by 12 %. The default
(3) was not run at 1,000 MP, so 2 → 4 is all this run shows.

The default sends images below 1 MP to the slower `base` preset (`--small-preset`), for better compression
on small images. On a set of nothing but ~1 MP images that dominates the speed: `--small-preset none` is
2.6× faster there, for files 3 % larger (below).

`--preset turbo3 --small-preset none` was timed later in a separate session on a new g6.4xlarge, the same
way (`bench.gpu_speed`, `GP_ENCODERS=4`, 1,433 MP per run, median of 3 after a warm-up), on 99 of the 100
size-range photos (one has since changed on Flickr). Two interleaved rounds:

| Setting | round 1 | round 2 | CPU cores |
|---|---:|---:|---:|
| av1gpu default | 120.4 | 118.0 | 0.12 |
| `--preset turbo3 --small-preset none` | **213.9** | **209.8** | 0.19 |
| `--preset turbo3` | 167.0 | 167.1 | 0.16 |

So `turbo3 --small-preset none` runs at about 212 MP/s, 1.8× the default and a little faster than libaom
speed 9 (188).

### Compression

BD-rate on SSIMULACRA2 against libaom speed 6, median over images (mean in brackets):

| Encoder | standard | size range |
|---|---:|---:|
| **av1gpu default** | **+0.3 %** (−0.1 %) | **−0.1 %** (−0.4 %) |
| av1gpu `--small-preset none` | +3.4 % (+2.9 %) | +2.1 % (+1.1 %) |
| av1gpu `--preset turbo3 --small-preset none` | | +5.4 % (+5.4 %)¹ |
| libaom speed 9 | +17.7 % (+20.7 %) | +15.0 % (+17.2 %) |
| SVT-AV1 preset 2 / 4 / 6 / 8 / 9 (below) | | −5.1 / −2.6 / +2.1 / +7.7 / +9.8 % |
| WebP | +41.3 % (+55.3 %) | +42.1 % (+44.9 %)³ |
| NVENC AV1 (ffmpeg `av1_nvenc` p7, below) | | +29.5 % (+46.6 %)² |

¹ 99 images, encoded on the RTX PRO 2000 (byte-identical to the L4) and scored with the same SSIMULACRA2
build on another machine. That machine scored the default 0.55 points more favourably than the L4 run on
the same files (−0.65 % instead of −0.1 %), so its turbo3 numbers are shifted by the same amount. `turbo3`
with the small-image preset left on scored +1.3 %.

² 94 of the 99 photos (on the other 5 its quality range does not overlap libaom's), scored on the L4 with
the same tools as the rest of the table; see [NVENC AV1 on the same L4](#nvenc-av1-on-the-same-l4). The image
pipeline used for NVENC decodes photos with an odd height with wrong chroma; on the 76 even-height photos
NVENC is +27.9 %.

³ Scored with each .webp's own ICC profile. The first published run decoded WebP with dwebp, which leaves
the profile out of its PNG, so the 13 photos with Adobe RGB, Camera RGB or Generic RGB profiles were scored
as colour-shifted: +45.1 % (+61.2 %). `bench.py` now copies the profile back. The standard set has not been
re-checked.

Against WebP at each image's WebP quality, the default is 30.0 % (standard) and 30.7 % (size range)
smaller (31.2 % before the WebP ICC correction). Size range by image size (median, images in brackets):

| Encoder | < 1 MP | 1–5 MP | 5–10 MP | ≥ 10 MP |
|---|---:|---:|---:|---:|
| av1gpu default | +0.1 % (40) | +1.1 % (20) | −0.6 % (23) | −2.4 % (17) |
| av1gpu `--small-preset none` | +3.9 % (40) | +1.1 % (20) | −0.6 % (23) | −2.4 % (17) |
| libaom speed 9 | +20.1 % (40) | +18.5 % (20) | +12.8 % (23) | +13.4 % (17) |
| NVENC AV1 | +35.7 % (38) | +29.2 % (18) | +28.5 % (21) | +22.2 % (17) |
| SVT-AV1 preset 4 | −2.0 % (39) | −2.2 % (20) | −3.4 % (23) | −3.9 % (17) |
| SVT-AV1 preset 6 | +5.1 % (39) | +2.9 % (20) | +0.7 % (23) | +0.2 % (17) |
| SVT-AV1 preset 9 | +14.9 % (39) | +12.5 % (20) | +6.3 % (23) | +7.2 % (17) |

(At 1 MP and above both av1gpu rows are the same `turbo` encodes.) The groups are small, but they show the
same trend as the held-out set: av1gpu compares best on large images.

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

- at libaom speed 6's file size, av1gpu's default (116 MP/s, −0.1 %) is about 4× faster;
- at +2.1 %, av1gpu with `--small-preset none` (127 MP/s) is 2.6× faster than preset 6 (49 MP/s);
- at the fast end, `turbo3` (212 MP/s, +5.4 %) beats preset 9 (183 MP/s, +9.8 %) on both counts;
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
  av1gpu uses one tile unless AV1's limits need more.

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
| av1gpu default | −0.1 % | +0.5 % |
| av1gpu `--small-preset none` | +2.1 % | +2.5 % |
| libaom speed 9, one tile | +14.3 % | +14.7 % |
| SVT-AV1 preset 4, one tile | −3.2 % | −2.9 % |
| SVT-AV1 preset 6, one tile | +1.8 % | +2.6 % |
| SVT-AV1 preset 9, one tile | +9.4 % | +9.8 % |
| NVENC AV1 | +29.5 % | +30.3 % |
| WebP | +45.1 % | +46.0 % |

**What it means for av1gpu.** At its own file size (+0.5 % against untiled libaom speed 6), av1gpu's
116 MP/s is 4.0× stock libaom speed 6, 3.9× the tuned build and 3.7× the tuned build at 8 × `-j 2`
(31.1 MP/s, tiled). Interpolating between SVT-AV1 presets 4 and 6 at the same size gives 31.5 MP/s for the
stock build (3.7×) and 35.1 MP/s with PGO (3.3×), both untiled at 16 × `-j 1`; with tiling, PGO and
8 × `-j 2`, 32.8 MP/s (3.5×). The fastest CPU configuration we found is about 3.3× slower than one L4.

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

It levels off at about 16 MP/s once both engines are busy. av1gpu's default, timed in the same session
with `bench.gpu_speed`, ran at 115.6 and 117.1 MP/s on 0.12 cores, **7.4× faster**. The CPU cores are
ffmpeg decoding JPEGs and converting to 4:2:0 for the encoder; av1gpu decodes on the GPU.

**Compression:** six points, CQ 18, 25, 32, 38, 44 and 50, decoded with avifdec 1.4.0 and scored with the
same `ssimulacra2` as everything else. Against libaom speed 6 its files are **+29.5 %** (median; mean
+46.6 %) at equal quality, worse than libaom speed 9 (+15.0 %), and still 11.3 % smaller than WebP (85
photos, with WebP's ICC profiles); measured the other way, av1gpu's default files are 22.8 % smaller than
NVENC's (96 photos). Part of that is a bug in the image pipeline, not in NVENC: photos with an odd height
decode with wrong chroma (below). On the 76 even-height photos NVENC is +27.9 % against libaom speed 6, and
av1gpu's files are 20.9 % smaller than NVENC's. The 61 photos without an ICC profile (the pipeline drops ICC)
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

### 4:4:4 and 10-bit

`--yuv 444` (AV1 High profile) and `--depth 10` arrived on 2026-10-31. They were measured on a g6.4xlarge with
`bench/formats.py`: the size-range photos, bench.py's six quality points, SSIMULACRA2 against the source JPEG
(10-bit outputs decoded to 16-bit PNGs), and libaom speed 6 through avifenc 1.4.0 in the same format
(`-y 444`, `-d 10`; `tune=iq`, avifenc defaults, 16 × `-j 1`). The libaom 8-bit 4:2:0 encodes in the same
session reproduced the published files and scores (600/600). Raw results: `bench/results/aws-g6-l4-formats/`.

| Format | av1gpu MP/s | libaom speed 6 MP/s | av1gpu vs libaom, same format (median BD-rate) |
|---|---:|---:|---:|
| 8-bit 4:2:0 | 113.4 | 28.6 | −0.1 % |
| 8-bit 4:4:4 | 94.5 | 24.3 | +0.3 % |
| 10-bit 4:2:0 | 109.8 | 22.0 | +0.1 % |
| 10-bit 4:4:4 | 92.8 | 18.9 | +0.0 % |

- **Speed:** 4:4:4 codes twice the chroma samples of 4:2:0, so av1gpu runs about 17 % slower; 10-bit costs it
  about 3 %. libaom loses more at 10 bits (−23 %), so av1gpu's lead grows from 4.0× to 5.0×.
- **10-bit vs 8-bit:** −2.3 % for av1gpu and −2.0 % for libaom in 4:2:0, −1.9 % and −1.8 % in 4:4:4, at equal
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
  RGB as YCgCo-Re (CICP matrix 16, libavif 1.4's integer transform), 8-bit RGB to 10-bit 4:4:4. JPEGs keep
  their decoded YCbCr planes at their own subsampling; 4:2:2, 4:4:0 and 4:1:1 JPEGs are refused.
- **Proof:** 189 of 189 files (Kodak and graphics in GBR and YCgCo-Re, 16-bit PNGs, gray, palette, odd sizes,
  forced tiles, a 16.7 MP image, WebP, 22 JPEGs) decode to exactly the source with `avifdec -c aom` and
  `-c dav1d` (libavif 1.4.0); the JPEG cases equal libjpeg-turbo's raw planes. Decoder gates 17/17,
  6,787 lossless CPU fuzz cases and the GPU unit tests pass with 0 mismatches.
- **Lossy output unchanged:** 800/800 files byte-identical (default, turbo3, `--yuv 444`, `--depth 10`),
  every pre-existing GPU kernel compiles to identical SASS, and speed is within noise (115.7 vs 114.9 MP/s,
  18 runs each).

On 47 8-bit RGB PNGs, 42.7 MP: the 24 Kodak photos and 23 graphics (this repository's 16 SVG charts rendered at
2× and 7 synthetic 1920 × 1080 screenshots). CPU encoders ran as 16 single-threaded processes on the
g6.4xlarge's 16 vCPUs. av1gpu ran in batch mode; its first column is one pass including CUDA start-up, the
second the set repeated 4×.

| Encoder | Total bytes | vs PNG | Photos | Graphics | MP/s |
|---|---:|---:|---:|---:|---:|
| PNG (the sources) | 18,543,751 | 100 % | 15,394,305 | 3,149,446 | |
| av1gpu `--lossless` | 20,112,025 | 108.5 % | 13,370,817 | 6,741,208 | 16.8 / 41–43 |
| av1gpu `--lossless --ycgco-re` | 16,809,786 | 90.6 % | 11,990,929 | 4,818,857 | 16.4 / 41–43 |
| avifenc `-l -s 6` | 21,516,335 | 116.0 % | 13,686,053 | 7,830,282 | 9.3 |
| avifenc `-l -s d` (libaom's default, cpu-used 0) | 18,861,272 | 101.7 % | 13,324,165 | 5,537,107 | 0.38 |
| avifenc `-l -s 6 --cicp 1/13/16` | 17,103,004 | 92.2 % | 12,396,839 | 4,706,165 | 8.8 |
| cwebp `-lossless` | 12,323,572 | 66.5 % | 11,339,964 | 983,608 | 44.4 |
| cwebp `-lossless -z 9` | 12,187,462 | 65.7 % | 11,252,092 | 935,370 | 1.76 |

Lossless WebP is far smaller on graphics because av1gpu (like libaom at these settings) has no palette mode or
intra block copy. On lossless JPEG input (the size-range photos) av1gpu ran at 44.8 MP/s. One image at a time
is limited by the GPU entropy coder (about 126 ms for a Kodak image). The rate estimate behind the mode
choices uses default-CDF costs, so it is approximate.

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
| av1gpu default (q60, alpha q60) | 697,115 | 591,335 | 90,312 | 50.2 dB | 27.1 |
| av1gpu `--alpha-quality 100` | 1,086,862 | 591,335 | 480,059 | exact | 24.9 |
| av1gpu `--lossless` | 7,723,961 | 7,222,941 | 485,582 | exact | 26.4 |
| avifenc default (4:4:4, qalpha 60) | 846,207 | 730,647 | 99,452 | 50.9 dB | 14.4 |
| avifenc `--qalpha 100` | 1,274,328 | 730,647 | 527,573 | exact | 12.1 |
| avifenc `-y 420 --qalpha 100` | 1,208,036 | 664,317 | 527,573 | exact | |
| avifenc `-l` | 7,963,112 | 7,419,431 | 527,573 | exact | 8.2 |
| cwebp default (q75, lossless alpha) | 1,059,014 | | | exact | 62.7 |
| cwebp `-lossless` | 4,754,430 | | | exact | 27.4 |

The lossy colour columns are not matched for quality. The exact alpha planes are directly comparable:
av1gpu's are about 9 % smaller than libaom's. Speeds are for the set repeated 20× (202 MP); av1gpu
without alpha (`--no-alpha`) runs at 29.9 MP/s on it. Mostly small images keep the GPU less busy than
the photo sets.

**Fixed along the way.** In 8-bit mode, 16-bit PNGs without a gAMA or sRGB chunk were read as linear light by
libpng's simplified API and came out brightened (mid-grey 30000/65535 became 179 instead of 117). They are now
read as sRGB-encoded, as libavif and Pillow read them. 8-bit PNGs and the 10-bit and lossless paths are
unaffected. PNG and WebP ICC profiles and EXIF orientation are still not read.

### CVVDP: a second metric

SSIMULACRA2 does not model screen size or viewing distance, so the same size-range files were also scored
with CVVDP (ColorVideoVDP 0.5.7) for its `standard_fhd` display: a 24-inch 1920 × 1080 screen at 200 cd/m²
seen from 0.6 m (37.8 pixels per degree), with images shown at 100 % (one image pixel per screen pixel).
All 4,752 files were re-encoded on an L4 and were byte-identical to the published ones, and their
SSIMULACRA2 scores were identical too. Raw results: `bench/results/aws-g6-l4-cvvdp/`
(`bench.py --cvvdp standard_fhd`).

| Encoder | BD-rate vs libaom speed 6, JOD | SSIMULACRA2 | Bits per pixel at 9.5 JOD | Photos reaching 9.5 JOD |
|---|---:|---:|---:|---:|
| libaom speed 6 | 0 | 0 | 0.448 | 99/99 |
| **av1gpu default** | **+3.9 %** | −0.1 % | 0.459 | 99/99 |
| libaom speed 9 | +14.5 % | +15.0 % | 0.522 | 99/99 |
| SVT-AV1 preset 4 | −2.0 % | −2.6 % | 0.430 | 99/99 |
| SVT-AV1 preset 6 | +2.0 % | +2.1 % | 0.468 | 99/99 |
| SVT-AV1 preset 9 | +9.3 % | +9.8 % | 0.491 | 99/99 |
| NVENC AV1 | +28.4 % | +29.5 % | 0.541 | 85/99 |
| WebP | +44.2 % | +42.1 % | 0.666 | 99/99 |

(Median BD-rates over 99 photos, NVENC 97; quality points as published plus q85/q90 and matching higher
WebP/NVENC points. Bits per pixel are medians, interpolated per photo.)

- **The metrics agree on every encoder except av1gpu.** CVVDP rates av1gpu worse than SSIMULACRA2 does on 84
  of the 99 photos, and the gap grows with image size (+2.1 % below 1 MP, +5.9 % at 10 MP and over). Scored
  without colour, CVVDP puts it at +6.2 %, so the difference is in luma. av1gpu's search was tuned with
  SSIMULACRA2; CVVDP suggests its luma decisions on large photos leave some quality behind.
- **Speed at CVVDP's file sizes:** at av1gpu's +3.9 %, interpolating between presets and speeds, SVT-AV1 runs
  at about 70 MP/s (between presets 6 and 9) and libaom at about 47 MP/s, so av1gpu is about 1.7× and 2.4×
  faster. On SSIMULACRA2 the same comparison gives 4×.
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

### Compared with the RTX PRO 2000

The image sets differ, so this is a rough comparison. The development set was private forum uploads
(IJG quality 90, mean 4.2 MP); the size-range set here has a similar mean size (4.8 MP) but mostly camera
originals. With that caveat:

- **Speed is about the same:** 104–116 MP/s with the default settings on the L4 against 110.8 MP/s on the
  RTX PRO 2000. Both cards are power-limited (70–72 W).
- **Compression matches:** −0.1 % against libaom speed 6 here, −0.6 % on the held-out set. libaom speed 9
  is +15.0 % here, +14.0 % there.
- **The CPU here is much faster per thread** (libaom speed 6: 1.8 MP/s per vCPU, against 1.27 on the
  Xeon Silver 4110; speed 9: 11.7 against 8.45). Against the whole 16-vCPU machine, av1gpu is 4.0× libaom
  speed 6 at the same file size, while libaom speed 9 is 1.6× faster than av1gpu with files 15 % larger.
- **Cost of the run:** 1 h 39 min on a g6.4xlarge spot instance at about $0.80/hour, plus about 2 minutes
  on demand, roughly $1.40.
