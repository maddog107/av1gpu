<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/charts/banner-dark.svg">
    <img alt="av1gpu: AVIF encoding on the GPU" src="docs/charts/banner-light.svg" width="900">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/maddog107/av1gpu/releases"><img alt="Latest release" src="https://img.shields.io/github/v/release/maddog107/av1gpu?include_prereleases&style=flat-square&color=2a78d6"></a>
  <a href="https://github.com/maddog107/av1gpu/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/maddog107/av1gpu/ci.yml?style=flat-square&label=CI"></a>
  <a href="LICENSE"><img alt="License: BSD 2-Clause" src="https://img.shields.io/badge/license-BSD--2--Clause-1baf7a?style=flat-square"></a>
  <img alt="CUDA 12+" src="https://img.shields.io/badge/CUDA-12%2B-76b900?style=flat-square&logo=nvidia&logoColor=white">
  <img alt="Output: AV1 / AVIF" src="https://img.shields.io/badge/output-AV1%20%2F%20AVIF-eb6834?style=flat-square">
  <a href="docs/RESULTS.md"><img alt="Tested GPUs: RTX PRO 2000, L4" src="https://img.shields.io/badge/tested%20on-RTX%20PRO%202000%20%C2%B7%20L4-6250d6?style=flat-square"></a>
  <a href="https://github.com/maddog107/av1gpu/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/maddog107/av1gpu?style=flat-square&color=eda100"></a>
</p>

<p align="center">
  <img alt="A 12-megapixel photo of a maroon 1962 Opel Kapitän split down the middle: the left half is the 6,031 KB original JPEG, the right half av1gpu's 419 KB AVIF. The seam is invisible." src="docs/img/hero-car.jpg" width="900"><br>
  <sub>Left: the original 12 MP JPEG, 6,031 KB. Right: av1gpu's AVIF at the default quality, 419 KB.
  Photo: <a href="https://www.flickr.com/photos/yakobusan/20438833739">Untitled</a> by Jakob Montrasio, CC BY 2.0.</sub>
</p>

> [!TIP]
> **TL;DR:** av1gpu turns JPEGs into AVIFs on an NVIDIA GPU **4× faster than libaom running on all 16 CPU
> threads at once** (3.7× against a tuned build), with the **same file sizes** on SSIMULACRA2 (libaom speed 6:
> −0.1 % as avifenc runs it, +0.5 % with its tiling turned off; a second metric, CVVDP, says +3.9 %), while
> using only **0.1 of a CPU core**. The 100 test photos went from **160 MB of JPEG to 22.5 MB of AVIF
> (−86 %)**, and its files are **31 % smaller than WebP** at the same quality. SVT-AV1 needs presets 4–6 on
> all 16 threads (16–49 MP/s) for the same sizes; built with PGO, it is still 3.3× slower. On the same GPU
> av1gpu is **7× faster than NVIDIA's own NVENC AV1 encoder**, whose files are about 28 % bigger than
> libaom's. It also writes **4:4:4, 10-bit, lossless and transparent** AVIFs (4:4:4 and 10-bit 4–5× faster
> than libaom in the same format). One prebuilt binary, no CPU farm.

## ⚡ Overview

A CUDA encoder for AV1 still images that writes AVIF. Mode search, partitioning, reconstruction,
loop-filter decisions and entropy coding all run on the GPU's general-purpose CUDA cores. It does not
use NVENC, the GPU's fixed-function video encoder, which on the same card is 7× slower and makes about 26 %
bigger files ([below](#-same-gpu-av1gpu-vs-nvenc)). JPEG input is decoded on the GPU too, so a whole
batch of images stays in GPU memory from the JPEG bitstream to the finished AVIF bytes.

It was built for [Bellazon](https://www.bellazon.com), a forum with more than **15 million photos**:
32 million files with their thumbnails, 50 trillion pixels and 8.9 TB of originals. On one 70 W card
the whole archive is about 7 days of GPU time, while the server's CPUs carry on serving the site. At
the quality the forum chose, the AVIF files are 40 % smaller than the WebP it serves today, 1.5 TB less
to store and send (at equal quality, av1gpu's files are 31 % smaller than WebP's).

| | |
|---|---|
| **15 M+ photos** | Bellazon's archive (32 M files, 50 trillion pixels) in about 7 GPU-days on one card |
| **116 MP/s** | on one NVIDIA L4, a 72 W data-centre card (with `GP_ENCODERS=4`) |
| **−0.1 %** | file size against libaom speed 6 at equal quality (median BD-rate, SSIMULACRA2; +0.5 % against libaom with its tiling off) |
| **−31 %** | file size against WebP at equal quality |
| **0.1 cores** | of CPU while encoding, so the CPU stays free for other work |
| **bit-exact** | every test encode decodes to the encoder's own reconstruction in dav1d and libaom |

## 🎬 See it run

99 camera photos (478 MP, 154 MB of JPEG) converted on the RTX PRO 2000, recorded in real time:

<p align="center">
  <img alt="Terminal recording: av1gpu --batch converts 99 photos (477.7 MP) in 7.6 seconds of wall time at 83.5 MP/s using 0.35 CPU cores; 154 MB of JPEG becomes 22 MB of AVIF." src="docs/img/demo.webp" width="900">
</p>

That run includes CUDA start-up; on longer batches the speed settles at the figures below.

## 📊 Results

Measured on an AWS g6.4xlarge (NVIDIA L4, AMD EPYC 7R13 with 16 vCPUs) with 100 public
[Open Images](https://storage.googleapis.com/openimages/web/index.html) photos from 0.18 to 24 MP, against
avifenc 1.4 with libaom 3.13.1 (`tune=iq`) and SVT-AV1 4.2.0 (`tune=3`, its still-image mode), one
avifenc per vCPU with avifenc's defaults (which tile images automatically). Full numbers, method and logs:
[docs/RESULTS.md](docs/RESULTS.md#nvidia-l4-aws-g64xlarge); reproduce them with [Benchmarks](#-benchmarks).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/tradeoff-dark.svg">
  <img alt="Scatter chart of speed against file size vs libaom speed 6, with an arrow marking av1gpu 4× faster than libaom speed 6 at the same file size. av1gpu: default 116 MP/s at −0.1 percent, no small preset 127 at +2.1, turbo3 212 at +5.4. SVT-AV1 on 16 threads: preset 2 4.6 MP/s at −5.1 percent, preset 4 15.5 at −2.6, preset 6 49 at +2.1, preset 8 136 at +7.7, preset 9 183 at +9.8. libaom on 16 threads: speed 6 29 MP/s at 0, speed 9 188 at +15.0. NVENC AV1 16 MP/s at +29.5 percent." src="docs/charts/tradeoff-light.svg" width="760">
</picture>

**Speed against file size:** up is faster, left is smaller files. From libaom-speed-6 sizes up to +5 %, av1gpu
on one GPU is 2.4–4× faster than the best CPU encoder on all 16 threads (interpolating between SVT-AV1
presets). For the smallest files, SVT-AV1's slow presets go further (preset 2: −5.1 %, at 4.6 MP/s);
av1gpu's own `--preset slow` has not been timed on the L4 yet. Tuning the CPU side (`-march=native`, LTO,
PGO for SVT-AV1, the best thread split, no tiling) makes it 1–15 % faster with the same files; at
libaom-speed-6 sizes av1gpu is then 3.3–3.7× faster
([details](docs/RESULTS.md#cpu-builds-tuning-threads-and-tiling)).

Each encoder has a "small files" setting and a "fast" setting, so there are two fair match-ups. A third
compares av1gpu with NVENC, the AV1 hardware encoder built into the same GPU.

### 🐢 Smallest files: av1gpu default vs libaom speed 6 and SVT-AV1

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/match-small-dark.svg">
  <img alt="Bar chart of encoding speed: av1gpu default 116 MP/s, libaom speed 6 29 MP/s, SVT-AV1 preset 6 49 MP/s with files 2.1 percent bigger, SVT-AV1 preset 4 16 MP/s with files 2.6 percent smaller, NVENC AV1 on the same L4 16 MP/s with files 30 percent bigger." src="docs/charts/match-small-light.svg" width="760">
</picture>

**At the same file size, av1gpu is 4× faster than libaom running on all 16 CPU threads at once**,
and it leaves the CPU almost idle (0.1 of a core). SVT-AV1 brackets the same size with preset 6 (49 MP/s,
files 2.1 % bigger) and preset 4 (16 MP/s, 2.6 % smaller).

### 🐇 Fastest: av1gpu turbo3 vs SVT-AV1 preset 9 and libaom speed 9

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/match-fast-dark.svg">
  <img alt="Two bar panels. Speed: av1gpu turbo3 with the small-image preset off 212 MP/s, SVT-AV1 preset 9 183 MP/s, libaom speed 9 188 MP/s. Files vs libaom speed 6: av1gpu turbo3 +5.4 percent, SVT-AV1 preset 9 +9.8 percent, libaom speed 9 +15.0 percent." src="docs/charts/match-fast-light.svg" width="760">
</picture>

**A little faster than both, and av1gpu's files grow about half as much as SVT-AV1's and a third as much
as libaom's:** +5.4 % against libaom speed 6, where SVT-AV1 preset 9 pays +9.8 % and libaom speed 9 +15 %.
av1gpu uses 0.2 of a CPU core; the CPU encoders need all 16 threads. libaom speed 10 is no faster than 9
for still images, and SVT-AV1 runs presets 10–13 as 9 when every frame is intra.

The `turbo3` speed was measured on the same L4 in a separate session, on 99 of the 100 photos (the default
gave 119 MP/s there, against 116 in the main run).

### 🆚 Same GPU: av1gpu vs NVENC

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/match-nvenc-dark.svg">
  <img alt="Two bar panels. Speed: av1gpu default 116 MP/s, NVENC AV1 preset p7 16 MP/s. Files vs libaom speed 6: av1gpu −0.1 percent, NVENC +29.5 percent." src="docs/charts/match-nvenc-light.svg" width="760">
</picture>

**7× faster than the GPU's own AV1 encoder, with files 21 % smaller.** NVENC is a fixed-function block
(the L4 has two), so it tops out at about 16 MP/s however many encodes you run at once, and it keeps
4 CPU cores busy decoding JPEGs for it. Its files are 27.9 % bigger than libaom speed 6's at equal
quality, worse than libaom speed 9 (on the 76 photos with an even height; the image pipeline we used
decodes odd heights with wrong chroma, which makes it +29.5 % over all 99). Tested through ffmpeg `av1_nvenc` at its best-quality setting (`-preset p7
-tune uhq`) on 99 of the photos; details in
[docs/RESULTS.md](docs/RESULTS.md#nvenc-av1-on-the-same-l4).

### 📈 Quality and file size

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/quality-dark.svg">
  <img alt="Line chart of mean SSIMULACRA2 against bits per pixel. The av1gpu and libaom speed 6 curves coincide; libaom speed 9 is below them, and WebP and NVENC AV1 well below." src="docs/charts/quality-light.svg" width="760">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/by-size-dark.svg">
  <img alt="Column chart of median BD-rate against libaom speed 6 by image size. av1gpu: +0.1, +1.1, −0.6 and −2.4 percent for under 1 MP, 1–5, 5–10 and 10 MP and over. libaom speed 9: +20.1, +18.5, +12.8 and +13.4 percent. NVENC AV1: +35.7, +29.2, +28.5 and +22.2 percent." src="docs/charts/by-size-light.svg" width="760">
</picture>

| Encoder | MP/s | Files vs libaom s6 | Files vs WebP | CPU cores |
|---|---:|---:|---:|---:|
| **av1gpu** default, `GP_ENCODERS=4` | **116** | **−0.1 %** | **−30.7 %** | 0.1 |
| av1gpu `--small-preset none` | 127 | +2.1 % | | 0.1 |
| av1gpu `--preset turbo3 --small-preset none` | 212¹ | +5.4 % | | 0.2 |
| libaom speed 6, 16 × avifenc | 29 | 0 | −29.6 % | 16 |
| libaom speed 9, 16 × avifenc | 188 | +15.0 % | −18.7 % | 16 |
| SVT-AV1 preset 2, 16 × avifenc | 4.6³ | −5.1 % | −33.8 % | 16 |
| SVT-AV1 preset 4, 16 × avifenc | 15.5³ | −2.6 % | −31.7 % | 16 |
| SVT-AV1 preset 6, 16 × avifenc | 49³ | +2.1 % | −27.4 % | 16 |
| SVT-AV1 preset 8, 16 × avifenc | 136³ | +7.7 % | −23.6 % | 16 |
| SVT-AV1 preset 9, 16 × avifenc | 183³ | +9.8 % | −22.4 % | 16 |
| WebP (cwebp `-m 4`) | | +42.1 %⁴ | 0 | |
| NVENC AV1 (ffmpeg `av1_nvenc -preset p7 -tune uhq`), 8 at once | 16² | +29.5 %⁵ | −11.3 % | 4 |

¹ Measured in a separate L4 session on 99 of the 100 photos, where the default ran at 119 MP/s.
² Same L4, separate session, 99 photos; the av1gpu default ran at 116 MP/s alongside it.
³ On a c6a.4xlarge, the same EPYC 7R13 with 16 vCPUs, 99 photos; libaom ran alongside at 28.7 and 186.4 MP/s,
and its speed 6 files and scores were identical to the main run's.
⁴ Scored with each WebP's own ICC profile. The first published numbers (+45.1 %, and av1gpu −31.2 % against
WebP) lost it in dwebp's PNG, which made 13 photos with non-sRGB profiles look colour-shifted.
⁵ The image pipeline used for NVENC decodes photos with an odd height with wrong chroma; on the 76 even-height
photos NVENC is +27.9 % ([details](docs/RESULTS.md#nvenc-av1-on-the-same-l4)).

The CPU encoders ran with avifenc's defaults, which split most of these photos into up to 8 tiles. With one
tile they run at the same speed and make files about 0.5 % smaller, so against untiled libaom speed 6
av1gpu's default is +0.5 %, and SVT-AV1 presets 4, 6 and 9 are −2.9, +2.6 and +9.8 %. Builds with
`-march=native` and LTO write the same files 1–4 % faster, and SVT-AV1 with PGO 9–13 % faster
([details](docs/RESULTS.md#cpu-builds-tuning-threads-and-tiling)).

### 🎨 4:4:4 and 10-bit

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/formats-dark.svg">
  <img alt="Bar chart of speed in each output format. 8-bit 4:2:0: av1gpu 113 MP/s, libaom speed 6 29. 8-bit 4:4:4: 95 and 24. 10-bit 4:2:0: 110 and 22. 10-bit 4:4:4: 93 and 19. In each format av1gpu's files are within 0.4 percent of libaom's in the same format." src="docs/charts/formats-light.svg" width="760">
</picture>

`--yuv 444` (full-resolution colour, AV1 High profile) and `--depth 10`, alone or together, keep the same
picture: in every format av1gpu's files are within 0.4 % of libaom speed 6's in that format, at 3.9–5× the
speed. On these photos 10-bit makes files about 2 % smaller at equal quality for both encoders (av1gpu
−2.3 %, libaom −2.0 %), at a 3 % speed cost for av1gpu. 4:4:4 makes little difference here, because the
sources are 4:2:0 JPEGs; it is for sources with full-resolution colour, such as PNG screenshots and graphics.
Details: [docs/RESULTS.md](docs/RESULTS.md#444-and-10-bit).

### 💎 Lossless

`--lossless` writes AVIFs that decode to exactly the input. On 47 PNGs (24 Kodak photos and 23 charts and
screenshots, 42.7 MP), with every output checked pixel for pixel in libavif with both decoders:

| Lossless encoder | Size vs the PNGs | Photos | Graphics | MP/s |
|---|---:|---:|---:|---:|
| **av1gpu `--lossless --ycgco-re`** | **90.6 %** | 11.99 MB | 4.82 MB | 41–43 |
| av1gpu `--lossless` (GBR) | 108.5 % | 13.37 MB | 6.74 MB | 41–43 |
| avifenc `-l --cicp 1/13/16` (speed 6) | 92.2 % | 12.40 MB | 4.71 MB | 8.8 |
| avifenc `-l` (speed 6) | 116.0 % | 13.69 MB | 7.83 MB | 9.3 |
| cwebp `-lossless` | 66.5 % | 11.34 MB | 0.98 MB | 44 |

av1gpu beats libaom's lossless AVIF on size and is about 5× faster, but lossless WebP is much smaller,
especially on graphics: av1gpu has no palette or intra block copy, the AV1 tools for flat colours and
repeated patterns. (av1gpu's speed is for the set repeated 4×; a single pass of 42.7 MP, CUDA start-up
included, ran at 17 MP/s. CPU encoders: 16 single-threaded processes on the same 16-vCPU host.) Details:
[docs/RESULTS.md](docs/RESULTS.md#lossless).

### 🪟 Transparency

PNG and WebP files with transparency get an AVIF alpha channel: a second, monochrome AV1 image, laid out as
libavif writes it, and decoded with exact alpha by libavif with both dav1d and libaom. Alpha quality follows
`--quality` by default, as in avifenc 1.2+; `--alpha-quality 100` (or `--lossless`) keeps it exact. Files
without transparency are unchanged, byte for byte. On 39 test PNGs (logos, icons, stickers, UI elements,
masked photos, charts):

| Encoder | Total | Alpha part | Alpha | MP/s |
|---|---:|---:|---|---:|
| av1gpu default (q60, alpha q60) | 697 KB | 90 KB | 50.2 dB | 27 |
| av1gpu `--alpha-quality 100` | 1,087 KB | 480 KB | exact | 25 |
| avifenc `--qalpha 100` (its default 4:4:4) | 1,274 KB | 528 KB | exact | 12 |
| avifenc `-y 420 --qalpha 100` | 1,208 KB | 528 KB | exact | |
| av1gpu `--lossless` | 7.72 MB | 486 KB | exact | 26 |
| avifenc `-l` | 7.96 MB | 528 KB | exact | 8.2 |
| cwebp `-lossless` | 4.75 MB | | exact | 27 |

Exact alpha comes out about 9 % smaller than libaom's. The lossy colour rows are not matched for quality
(the alpha rows are). Details: [docs/RESULTS.md](docs/RESULTS.md#transparency).

### 👁️ A second metric: CVVDP

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/metrics-dark.svg">
  <img alt="Dot chart of BD-rate against libaom speed 6 on SSIMULACRA2 and on CVVDP for each encoder: av1gpu −0.1 and +3.9 percent, SVT-AV1 preset 4 −2.6 and −2.0, preset 6 +2.1 and +2.0, preset 9 +9.8 and +9.3, libaom speed 9 +15.0 and +14.5, NVENC AV1 +29.5 and +28.4." src="docs/charts/metrics-light.svg" width="760">
</picture>

SSIMULACRA2 does not model screen size or viewing distance, so we also scored the same files with CVVDP
(ColorVideoVDP, `standard_fhd`: a 24-inch 1080p screen at 60 cm, photos shown at 100 %). It agrees with
SSIMULACRA2 within about a point for every encoder except av1gpu, which it puts at **+3.9 %** against libaom
speed 6 instead of −0.1 %. The difference is in luma and grows with image size. At that file size av1gpu is
still about 1.7× faster than SVT-AV1 and 2.4× faster than libaom (interpolating between presets). Every AVIF
encoder here reaches 9.5 JOD, a high-fidelity level, on all 99 photos: av1gpu at 0.46 bits per pixel, libaom
speed 6 at 0.45, SVT-AV1 at 0.43–0.49 and WebP at 0.67. Details:
[docs/RESULTS.md](docs/RESULTS.md#cvvdp-a-second-metric).

### 🖼️ What the files look like

The default quality (`--quality 60`, the avifenc scale) turned the 100 test JPEGs, 160.4 MB, into
22.5 MB of AVIF, at a mean SSIMULACRA2 of 72 ("high quality"; 90 is visually lossless). `--quality 70`
gave 33.4 MB at 78. Five of the photos, one from each size class:

<table>
  <tr>
    <td align="center" width="20%"><img src="docs/img/thumb-tiny.jpg" alt="A display board" width="100%"><br><b>90 KB → 27 KB</b><br>−70 % · 640 × 427</td>
    <td align="center" width="20%"><img src="docs/img/thumb-small.jpg" alt="Red peppers" width="100%"><br><b>177 KB → 60 KB</b><br>−66 % · 1024 × 768</td>
    <td align="center" width="20%"><img src="docs/img/thumb-typical.jpg" alt="A rooster" width="100%"><br><b>2.06 MB → 177 KB</b><br>−91 % · 1920 × 1280</td>
    <td align="center" width="20%"><img src="docs/img/thumb-large.jpg" alt="A shop window with a crib" width="100%"><br><b>1.88 MB → 212 KB</b><br>−89 % · 2304 × 3072</td>
    <td align="center" width="20%"><img src="docs/img/thumb-huge.jpg" alt="A dog" width="100%"><br><b>6.08 MB → 578 KB</b><br>−90 % · 2848 × 4288</td>
  </tr>
</table>

SSIMULACRA2 against the source: 67.9, 77.3, 73.7, 71.1 and 69.0. Thumbnails are cropped from av1gpu's
output. Photos (CC BY 2.0): [DSC08815](https://www.flickr.com/photos/stupiddingo/4016599113) by Brent,
[Fresh Red Peppers at La Boqueria](https://www.flickr.com/photos/oh-barcelona/4811363097) by Oh-Barcelona.com,
[Cock](https://www.flickr.com/photos/kallegustafsson/5866854319) by Kalle Gustafsson,
[St.Armands Circle](https://www.flickr.com/photos/philliecasablanca/2037727892) by Phil Whitehouse and
[DSC_2946](https://www.flickr.com/photos/hockeyholic/7881222604) by Andy Blackledge.

Camera originals shrink the most, because they are stored at high JPEG quality. SSIMULACRA2 here compares
the AVIF with the decoded JPEG, not with the scene.

The RTX PRO 2000 and the L4 produced byte-identical files for all five.

### 🔍 Same file size, side by side

Each encoder was pushed to the size of av1gpu's file (the highest quality setting that fits), then the
same spot was cropped and enlarged 2×. av1gpu and libaom speed 6 keep the chrome bars of the grille
crisp; speed 9 and WebP soften them, and JPEG turns them into blocks.

<p align="center">
  <img alt="Six 2x crops of the car's chrome grille and badge: the original JPEG (6,031 KB), and av1gpu (419.4 KB, SSIMULACRA2 72), libaom speed 6 (412.1 KB, 71), libaom speed 9 (395.9 KB, 66), WebP (417.6 KB, 62) and JPEG (409.5 KB, 34) at the same file size." src="docs/img/compare-car.png" width="900">
</p>

<details>
<summary>A second example: a dog's eye and fur, 12 MP (578 KB each)</summary>
<p align="center">
  <img alt="Six 2x crops of a dog's eye: the original JPEG (6,083 KB), and av1gpu (578.2 KB, SSIMULACRA2 69), libaom speed 6 (577.8 KB, 68), libaom speed 9 (565.0 KB, 66), WebP (576.7 KB, 60) and JPEG (558.7 KB, 49) at the same file size." src="docs/img/compare-dog.png" width="900">
</p>
</details>

Crops of [Untitled](https://www.flickr.com/photos/yakobusan/20438833739) by Jakob Montrasio and
[DSC_2946](https://www.flickr.com/photos/hockeyholic/7881222604) by Andy Blackledge (CC BY 2.0), re-encoded
with av1gpu on the RTX PRO 2000, avifenc 1.4 / libaom 3.13 (`tune=iq`), cwebp `-m 4` and libjpeg 4:2:0.

**Every output is a standard AV1 bitstream, and the tests check that by decoding.** The encoder's
own reconstruction must match what dav1d and libaom's decoder produce, byte for byte. That held for
all 1,475 held-out encodes on the development GPU and all 1,000 encodes of the L4 run, and the test
suite keeps checking it (see [Correctness](#-correctness)).

## 🚦 Status

av1gpu is a working encoder that runs in production on Bellazon. It is also a young, single-developer
project. Before you rely on it, note:

- **It has been tested on two GPU models:** the RTX PRO 2000 Blackwell it was developed on (compute
  capability 12.0) and an NVIDIA L4 (8.9, on AWS; see [docs/RESULTS.md](docs/RESULTS.md#nvidia-l4-aws-g64xlarge)).
  The code uses only standard CUDA features, with no tensor cores, no architecture-specific intrinsics and
  no large shared-memory carve-outs. It builds for compute capability 7.5 through 12.0. Other GPUs should
  work but have not been run yet. Reports are welcome, especially `tests/decoder_gate.py` results.
- Output: 8- or 10-bit (`--depth 10`), 4:2:0, 4:4:4 (`--yuv 444`) or monochrome AV1, intra only. In 4:2:0,
  4:4:4 and 4:2:2 JPEGs are downsampled; in 4:4:4, subsampled JPEGs are upsampled exactly as libjpeg-turbo
  does. `--lossless` codes exactly (RGB as GBR or YCgCo-Re, JPEGs as their decoded YCbCr). PNG and WebP
  transparency becomes an AVIF alpha channel. There is no 4:2:2 output or animation.
- Some inputs are not handled on the GPU path. The batch mode skips them and lists them on stderr, so
  you can send them to another encoder: CMYK/YCCK JPEGs, 12-bit JPEGs, images over 35.6 MP and
  unreadable files. PNG and WebP inputs are decoded on the CPU.
- Tools not implemented: palette, intra block copy, loop restoration, superres, film grain.

## 🛠️ Building

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

## 🚀 Usage

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
| `--yuv 420` / `444` | chroma format (default `420`; `444` is AV1 High profile and keeps full-resolution colour) |
| `--lossless` | exact (AV1 lossless). PNG/WebP RGB as GBR 4:4:4 at the source depth (as `avifenc -l`); JPEG as its decoded YCbCr at its own subsampling (4:2:2 JPEGs are refused) |
| `--ycgco-re` | with `--lossless`: RGB as YCgCo-Re (`--cicp 1/13/16` in avifenc), 8-bit RGB → 10-bit 4:4:4; usually smaller |
| `--alpha-quality Q` | quality of the alpha channel of PNG/WebP files with transparency: defaults to `--quality`, as in avifenc 1.2+; `100` is lossless (also with `--lossless`) |
| `--no-alpha` | drop transparency as before alpha support (PNG colour composited onto black) |
| `--depth 8` / `10` | bit depth (default 8; 16-bit PNGs keep 10 bits, 8-bit sources are scaled to the 10-bit range) |
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

AVIF files made from JPEGs carry the source's ICC profile (PNG and WebP profiles are not read yet). EXIF orientation is signalled with `irot`/`imir` rather
than by rotating pixels.

[examples/](examples/) has a directory converter with an avifenc fallback for skipped files, and an
nginx configuration that serves `image.jpg.avif` to browsers that accept AVIF.

## ✅ Correctness

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

The fuzzers and gates cover 4:2:0, 4:4:4 and monochrome at 8 and 10 bits (pass the av1gpu options after
`--`, e.g. `tests/decoder_gate.py ... -- --yuv 444 --depth 10`). `dav1d` and `aomdec` must be on `PATH`
(Ubuntu: `apt install dav1d aom-tools`), or set `AV1G_DAV1D` / `AV1G_AOMDEC`. During development the CUDA kernels were also run under compute-sanitizer
(memcheck, racecheck, synccheck) with clean results.

## 📏 Benchmarks

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

One original in that set (`9fba6f263f3c4f3e`) has since been changed on Flickr, so `fetch_images.py`
skipped it with a hash mismatch and the published runs used 99 images; since 2026-10-31 it downloads
again.

`bench/formats.py` runs the same comparison in every output format (`--yuv 444`, `--depth 10`) against libaom
in the same format, and `bench.py --cvvdp standard_fhd` adds CVVDP scores next to SSIMULACRA2
(`pip install cvvdp`; it runs on the GPU through PyTorch).

Speed runs repeat the set up to 1,000 MP (`--min-mp`), so that CUDA start-up does not dominate. Other
av1gpu settings can be timed through a wrapper script (`--av1gpu`, `--gpu-label`), and `--aom-speeds`
picks the libaom speeds to compare. With an avifenc built with SVT-AV1, `--svt-presets 2,4,6,8,9
--svt-tune 3` adds SVT-AV1 (`--skip-gpu` runs the CPU encoders alone). `--avifenc-threads N` runs
avifenc `-j N`, 16 / N at once, and `--no-tiling` encodes one tile per image.

## ⚙️ How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/pipeline-dark.svg">
  <img alt="Pipeline: the CPU reads JPEGs and writes AVIF files; the GPU decodes the JPEG, searches modes, picks partitions, reconstructs, loop-filters and entropy-codes." src="docs/charts/pipeline-light.svg" width="900">
</picture>

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

## 🗂️ Repository layout

| Path | Contents |
|---|---|
| `av1core/` | AV1 bitstream writer (CPU), GPU symbol generation and range coder, GPU deblocking/CDEF, AVIF muxer, spec tables |
| `encoder/` | GPU encoder: analysis, partition DP, closed-loop reconstruction, batching, the `av1gpu` CLI |
| `ingest/` | JPEG → GPU 4:2:0 planes (nvJPEG hardware path, or CPU entropy decode + GPU IDCT) |
| `av1f/` | libaom `tune=iq` decisions, variance boost, deblocking/CDEF strength search |
| `tests/` | decoder-identity fuzzers and gates |
| `bench/` | public-image benchmark |
| `tools/` | generators for the spec and libaom tables (outputs are checked in) |

## 📄 License

BSD 2-Clause, see [LICENSE](LICENSE). Parts of `av1f/` and the range coder are derived from libaom
(BSD 2-Clause, Alliance for Open Media); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). AV1 is
royalty-free under the AOMedia Patent License 1.0.
