# Measured results

Two GPUs so far: the RTX PRO 2000 Blackwell the encoder was developed on (private images, below), and an
NVIDIA L4 on AWS with the public `bench/` image sets ([NVIDIA L4](#nvidia-l4-aws-g64xlarge)).

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
  avifenc per vCPU, all 16 at once. WebP is cwebp 1.3.2 `-m 4`. Decoders: dav1d 1.4.1, aomdec 3.13.1.
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

### Compression

BD-rate on SSIMULACRA2 against libaom speed 6, median over images (mean in brackets):

| Encoder | standard | size range |
|---|---:|---:|
| **av1gpu default** | **+0.3 %** (−0.1 %) | **−0.1 %** (−0.4 %) |
| av1gpu `--small-preset none` | +3.4 % (+2.9 %) | +2.1 % (+1.1 %) |
| libaom speed 9 | +17.7 % (+20.7 %) | +15.0 % (+17.2 %) |
| WebP | +41.3 % (+55.3 %) | +45.1 % (+61.2 %) |

Against WebP at each image's WebP quality, the default is 30.0 % (standard) and 31.2 % (size range)
smaller. Size range by image size (median, images in brackets):

| Encoder | < 1 MP | 1–5 MP | 5–10 MP | ≥ 10 MP |
|---|---:|---:|---:|---:|
| av1gpu default | +0.1 % (40) | +1.1 % (20) | −0.6 % (23) | −2.4 % (17) |
| av1gpu `--small-preset none` | +3.9 % (40) | +1.1 % (20) | −0.6 % (23) | −2.4 % (17) |
| libaom speed 9 | +20.1 % (40) | +18.5 % (20) | +12.8 % (23) | +13.4 % (17) |

(At 1 MP and above both av1gpu rows are the same `turbo` encodes.) The groups are small, but they show the
same trend as the held-out set: av1gpu compares best on large images.

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
- **Cost of the run:** about 1.5 hours on a g6.4xlarge spot instance at about $0.80/hour, roughly $1.30.
