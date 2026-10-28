<h1 align="center">av1gpu</h1>

<p align="center">
  <b>AVIF encoding on the GPU.</b> libaom-speed-6 file sizes, four times faster than libaom speed 6 on a 16-thread CPU,<br>
  while using a tenth of one CPU core.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: BSD 2-Clause" src="https://img.shields.io/badge/license-BSD--2--Clause-blue"></a>
  <img alt="CUDA 12+" src="https://img.shields.io/badge/CUDA-12%2B-76b900">
  <img alt="Output: AV1 / AVIF" src="https://img.shields.io/badge/output-AV1%20%2F%20AVIF-informational">
  <a href="docs/RESULTS.md"><img alt="Tested GPUs: RTX PRO 2000, L4" src="https://img.shields.io/badge/tested%20on-RTX%20PRO%202000%20%C2%B7%20L4-lightgrey"></a>
</p>

A CUDA encoder for AV1 still images that writes AVIF. Mode search, partitioning, reconstruction,
loop-filter decisions and entropy coding all run on the GPU's general-purpose CUDA cores. It does not
use NVENC. JPEG input is decoded on the GPU too, so a whole batch of images stays in GPU memory from
the JPEG bitstream to the finished AVIF bytes. It was built to convert a large photo archive.

| | |
|---|---|
| **116 MP/s** | on one NVIDIA L4, a 72 W data-centre card (with `GP_ENCODERS=4`); 111 MP/s on an RTX PRO 2000 |
| **−0.1 %** | file size against libaom speed 6 at equal quality (median BD-rate, SSIMULACRA2) |
| **−31 %** | file size against WebP at equal quality |
| **0.1 cores** | of CPU while encoding, so the CPU stays free for other work |
| **bit-exact** | every test encode decodes to the encoder's own reconstruction in dav1d and libaom |

```bash
build/av1gpu --batch photos/ --outdir avif/        # every JPEG in photos/ -> avif/<name>.jpg.avif
```

## Results

Measured on an AWS g6.4xlarge (NVIDIA L4, AMD EPYC 7R13 with 16 vCPUs) with 100 public
[Open Images](https://storage.googleapis.com/openimages/web/index.html) photos from 0.18 to 24 MP, against
avifenc 1.4 / libaom 3.13.1 with `tune=iq`, one avifenc per vCPU. Full numbers, method and logs:
[docs/RESULTS.md](docs/RESULTS.md#nvidia-l4-aws-g64xlarge); reproduce them with [Benchmarks](#benchmarks).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/speed-dark.svg">
  <img alt="Bar chart of encoding speed in megapixels per second: libaom speed 9 188, av1gpu with the small-image preset off 127, av1gpu default 116, libaom speed 6 29." src="docs/charts/speed-light.svg" width="760">
</picture>

**At the same file size, av1gpu is 4× faster than the whole CPU.** libaom speed 9 is faster still
(1.6× av1gpu), but its files are 15 % bigger. libaom speed 10 is no faster than 9 for still images.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/quality-dark.svg">
  <img alt="Line chart of mean SSIMULACRA2 against bits per pixel. The av1gpu and libaom speed 6 curves coincide; libaom speed 9 is below them and WebP well below." src="docs/charts/quality-light.svg" width="760">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/by-size-dark.svg">
  <img alt="Column chart of median BD-rate against libaom speed 6 by image size. av1gpu: +0.1, +1.1, −0.6 and −2.4 percent for under 1 MP, 1–5, 5–10 and 10 MP and over. libaom speed 9: +20.1, +18.5, +12.8 and +13.4 percent." src="docs/charts/by-size-light.svg" width="760">
</picture>

| Encoder | MP/s | Files vs libaom s6 | Files vs WebP | CPU cores |
|---|---:|---:|---:|---:|
| **av1gpu** default, `GP_ENCODERS=4` | **116** | **−0.1 %** | **−31.2 %** | 0.1 |
| av1gpu `--small-preset none` | 127 | +2.1 % | | 0.1 |
| libaom speed 6, 16 × avifenc | 29 | 0 | −31.1 % | 16 |
| libaom speed 9, 16 × avifenc | 188 | +15.0 % | −19.3 % | 16 |
| WebP (cwebp `-m 4`) | | +45.1 % | 0 | |

The same picture held on the RTX PRO 2000 it was developed on (a 70 W workstation card), with 300
held-out uploads from thumbnails to 35 MP: 110.8 MP/s, −0.6 % against libaom speed 6, +14.0 % for speed 9
and −30.7 % against WebP ([details](docs/RESULTS.md#rtx-pro-2000-blackwell-development)).

### What the files look like

The default quality (`--quality 60`, the avifenc scale) turned the 100 test JPEGs, 160.4 MB, into
22.5 MB of AVIF, at a mean SSIMULACRA2 of 72 ("high quality"; 90 is visually lossless). `--quality 70`
gave 33.4 MB at 78. Five of the photos, one from each size class:

| Photo (CC BY 2.0) | Pixels | JPEG | AVIF, quality 60 | Smaller by | SSIMULACRA2 |
|---|---:|---:|---:|---:|---:|
| [DSC08815](https://www.flickr.com/photos/stupiddingo/4016599113) by Brent | 640 × 427 | 90.3 KB | 27.2 KB | 70 % | 67.9 |
| [Fresh Red Peppers at La Boqueria](https://www.flickr.com/photos/oh-barcelona/4811363097) by Oh-Barcelona.com | 1024 × 768 | 177.1 KB | 59.9 KB | 66 % | 77.3 |
| [Cock](https://www.flickr.com/photos/kallegustafsson/5866854319) by Kalle Gustafsson | 1920 × 1280 | 2.06 MB | 176.9 KB | 91 % | 73.7 |
| [St.Armands Circle](https://www.flickr.com/photos/philliecasablanca/2037727892) by Phil Whitehouse | 2304 × 3072 | 1.88 MB | 212.3 KB | 89 % | 71.1 |
| [DSC_2946](https://www.flickr.com/photos/hockeyholic/7881222604) by Andy Blackledge | 2848 × 4288 | 6.08 MB | 578.2 KB | 90 % | 69.0 |

Camera originals shrink the most, because they are stored at high JPEG quality. SSIMULACRA2 here compares
the AVIF with the decoded JPEG, not with the scene.

**Every output is a standard AV1 bitstream, and the tests check that by decoding.** The encoder's
own reconstruction must match what dav1d and libaom's decoder produce, byte for byte. That held for
all 1,475 held-out encodes on the development GPU and all 1,000 encodes of the L4 run, and the test
suite keeps checking it (see [Correctness](#correctness)).

## Status

av1gpu is a working encoder that runs in production on one site. It is also a young, single-developer
project. Before you rely on it, note:

- **It has been tested on two GPU models:** the RTX PRO 2000 Blackwell it was developed on (compute
  capability 12.0) and an NVIDIA L4 (8.9, on AWS; see [docs/RESULTS.md](docs/RESULTS.md#nvidia-l4-aws-g64xlarge)).
  The code uses only standard CUDA features, with no tensor cores, no architecture-specific intrinsics and
  no large shared-memory carve-outs. It builds for compute capability 7.5 through 12.0. Other GPUs should
  work but have not been run yet. Reports are welcome, especially `tests/decoder_gate.py` results.
- The output format is fixed: 8-bit 4:2:0 (or monochrome) AV1, intra only. 4:4:4 and 4:2:2 JPEGs are
  downsampled to 4:2:0. There is no alpha, 10-bit, animation or lossless mode.
- Some inputs are not handled on the GPU path. The batch mode skips them and lists them on stderr, so
  you can send them to another encoder: CMYK/YCCK JPEGs, 12-bit JPEGs, images over 35.6 MP and
  unreadable files. PNG and WebP inputs are decoded on the CPU.
- Tools not implemented: palette, intra block copy, loop restoration, superres, film grain.

## Building

You need Linux, CUDA 12.0 or newer (12.8 or newer to target Blackwell), CMake 3.24 or newer, a C++17
compiler, and libjpeg(-turbo), libpng and libwebp development packages.

```bash
# Ubuntu / Debian (plus the CUDA toolkit from NVIDIA)
sudo apt install cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev libwebp-dev

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build
```

`-DCMAKE_CUDA_ARCHITECTURES=native` builds only for the GPU in the machine, which is much quicker.
Leave it out to build a portable binary for compute capability 7.5 through 12.0. The binary links the
CUDA runtime statically, so at run time it needs only the NVIDIA driver and `libnvjpeg`.

`-DAV1GPU_CUDA=OFF` builds just the CPU pieces: the bitstream writer, the AVIF muxer and the fuzz
test. That build needs no CUDA toolkit.

## Usage

```bash
# one image
build/av1gpu photo.jpg photo.avif

# a directory (or a text file with one path per line): many images in flight at once
build/av1gpu --batch photos/ --outdir avif/            # writes avif/<name>.jpg.avif

# quality uses the avifenc -q scale (0..100; default 60)
build/av1gpu --batch photos/ --outdir avif/ --quality 70

# slower, smaller files
build/av1gpu --batch photos/ --outdir avif/ --preset slow
```

Batch mode is the fast path. It decodes, encodes and entropy-codes several batches at once, so the
GPU stays busy. Encoding images one process at a time is much slower, because each process pays
for CUDA start-up.

| Option | Meaning |
|---|---|
| `--quality Q` | 0–100 on avifenc's `-q` scale with `tune=iq` (60 → AV1 qindex 120) |
| `--qindex Q[,Q2,…]` | AV1 `base_q_idx` directly. In batch mode, several values run one pass each (add `--name-q` to suffix outputs with `.q<Q>`) |
| `--preset P` | `slow`, `base`, `fast`, `faster`, `turbo` (default), `turbo2`, `turbo3` |
| `--small-preset P` | preset for images below 1 MP (default `base`; `none` disables) |
| `--tune iq` / `none` | libaom `tune=iq` tools: quantization matrices, variance-boost delta-q, chroma delta-q, deblocking and CDEF strength choice (default `iq`) |
| `--cpu-entropy` | entropy-code on the CPU (produces bytes identical to the GPU coder) |
| `--stats` | print a JSON line with timings, throughput and CPU use |
| `--debug-outputs` | also write `<out>.obu` and `<out>.recon.yuv` for decoder comparisons |
| `--no-defaults` | start from the bare encoder instead of the defaults (research use) |

`av1gpu --help` lists the expert tuning options. The defaults are `--quality 60 --tune iq
--edge-filter --preset turbo --small-preset base --gpu-entropy`.

Batch-mode environment variables: `GP_ENCODERS` (concurrent encoders, default 3), `GP_ENCODE_MP`
(megapixels per GPU batch, default 48) and `GP_INGEST_WORKERS` (JPEG parsing threads, default 4).
The encoder needs about 1 GB of GPU memory with these settings, and the JPEG decoder about 0.6 GB more.
`GP_INGEST_DECODER=1` forces the CPU-entropy-decode + GPU-IDCT JPEG path on GPUs without a hardware
JPEG engine. av1gpu also falls back to that path automatically.

AVIF files carry the source's ICC profile. EXIF orientation is signalled with `irot`/`imir` rather
than by rotating pixels.

[examples/](examples/) has a directory converter with an avifenc fallback for skipped files, and an
nginx configuration that serves `image.jpg.avif` to browsers that accept AVIF.

## Correctness

The encoder keeps its own copy of the decoded picture. It needs that copy for prediction, and it is
also used to check every output against independent decoders:

```bash
# CPU only (no GPU needed): random legal encoder decisions -> av1core bitstream writer -> dav1d and aomdec
build/av1g_fuzz --count 1000            # or: ctest --test-dir build

# full GPU encoder, batch mode: every image must decode bit-exactly in dav1d and aomdec
tests/decoder_gate.py --bin build/av1gpu --images bench/images --q 60,120,180

# GPU entropy coder and GPU loop filters against the CPU reference
build/gpu_fuzz_test --count 2000
build/gpu_postfilter_test
```

`dav1d` and `aomdec` must be on `PATH` (Ubuntu: `apt install dav1d aom-tools`), or set `AV1G_DAV1D` /
`AV1G_AOMDEC`. During development the CUDA kernels were also run under compute-sanitizer
(memcheck, racecheck, synccheck) with clean results.

## Benchmarks

`bench/` reproduces the comparison on 100 public JPEGs from the
[Open Images](https://storage.googleapis.com/openimages/web/index.html) validation set (CC BY 2.0,
attribution in the CSV, about 1 MP each):

```bash
bench/fetch_images.py                                   # -> bench/images (33 MB, SHA-256 checked)
bench/bench.py --images bench/images --out bench/results/$(hostname)
```

It measures av1gpu throughput (the median of 3 runs after a warm-up) against all CPU threads running
avifenc at speeds 6 and 9. It also encodes every image at six quality points with av1gpu, libaom s6,
libaom s9 and WebP, scores them with SSIMULACRA2 and reports per-image BD-rates. It needs `avifenc`,
`avifdec`, `cwebp`, `dwebp` and libjxl's `ssimulacra2`; the script header shows how to get them. Results
from other GPUs are very welcome.

These images are small (about 1 MP), where the speed preset matters most. On the held-out set, small
images were where av1gpu compared worst against libaom and large images where it compared best. For a
spread of sizes there is a second set of 100 images from 0.18 to 24 MP (478 MP; the summary also breaks
the BD-rates down by size):

```bash
bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out bench/images_gamut
bench/bench.py --images bench/images_gamut --out bench/results/$(hostname)-gamut
```

Speed runs repeat the set up to 1,000 MP (`--min-mp`), so that CUDA start-up does not dominate. Other
av1gpu settings can be timed through a wrapper script (`--av1gpu`, `--gpu-label`), and `--aom-speeds`
picks the libaom speeds to compare.

## How it works

Short version (details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)):

1. **Ingest.** JPEGs are parsed on the CPU and decoded with nvJPEG's hardware engine where one exists.
   Otherwise libjpeg-turbo does the entropy decoding on the CPU and the GPU does the IDCT. Either
   way the planes are bit-exact with libjpeg-turbo, and they are packed into size-classed batches in
   GPU memory.
2. **Analysis.** Every block shape from 4×4 to 64×64 is evaluated at every position. An open-loop
   SATD screen of the luma intra modes (directional with angle deltas, smooth, Paeth, DC) keeps the
   top K. Those K get an exact-rate RD check, then a transform-type search.
3. **Partition DP.** A dynamic program over each 64×64 superblock picks the partition tree.
4. **Closed loop.** Superblocks are reconstructed in wavefront order with the real, already-decoded
   neighbours. The top candidates are re-checked with exact entropy contexts. Chroma modes, CfL,
   quantization (with quantization matrices and per-superblock delta-q) and the exact integer inverse
   transforms run here, giving a bit-exact reconstruction. Many images are in flight at once, which
   hides the wavefront's serial dependencies.
5. **Loop filters.** Deblocking levels and CDEF strengths are chosen and applied on the GPU. The
   choice follows libaom's search for `tune=iq`.
6. **Entropy coding.** Symbols are generated and arithmetic-coded on the GPU with AV1's adaptive
   CDFs, coding the tiles of many images in parallel. The output is byte-identical to the CPU
   writer. Then OBUs are assembled and muxed into AVIF.

## Repository layout

| Path | Contents |
|---|---|
| `av1core/` | AV1 bitstream writer (CPU), GPU symbol generation and range coder, GPU deblocking/CDEF, AVIF muxer, spec tables |
| `encoder/` | GPU encoder: analysis, partition DP, closed-loop reconstruction, batching, the `av1gpu` CLI |
| `ingest/` | JPEG → GPU 4:2:0 planes (nvJPEG hardware path, or CPU entropy decode + GPU IDCT) |
| `av1f/` | libaom `tune=iq` decisions, variance boost, deblocking/CDEF strength search |
| `tests/` | decoder-identity fuzzers and gates |
| `bench/` | public-image benchmark |
| `tools/` | generators for the spec and libaom tables (outputs are checked in) |

## License

BSD 2-Clause, see [LICENSE](LICENSE). Parts of `av1f/` and the range coder are derived from libaom
(BSD 2-Clause, Alliance for Open Media); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). AV1 is
royalty-free under the AOMedia Patent License 1.0.
