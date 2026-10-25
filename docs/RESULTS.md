# Measured results (RTX PRO 2000 Blackwell)

These measurements were taken during development, before the code was extracted into this
repository. The encoder source is the same; only the CLI and build were reorganized. The test images
were private and are not distributed. `bench/` runs the same kind of comparison on public images,
and results from other GPUs are welcome.

## Setup

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

## Compression

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

## Speed

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

## Correctness

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

## In production

Since October 2026 av1gpu has converted every new upload of the community site the encoder was
developed for. At one snapshot it had processed 316 uploads (632 files including thumbnails): 236 MB
of source JPEGs became 63 MB of AVIF, 73 % smaller. That figure depends on the source files and is not
a forecast.
