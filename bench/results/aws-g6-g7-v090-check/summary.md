# v0.9.0 release check: main c553ffb against v0.8.0 (NVIDIA L4 and RTX PRO 4500 Blackwell, 2026-11-03)

Agent rc09, branch `docs/v090-check`. Since v0.8.0 (cde1f74) main has merged four byte-identical speed PRs, each
checked on its own against v0.8.0: #56 (start-up, single image, pipeline), #57 (k_recon: 32-bit inverse transforms,
unrolled forward loops), #58 (analysis edge filter, shared memory), #59 (`k_entropy_win` coder on 3 warps). This
checks the combination as it ships: the Linux packages of main c553ffb from the Release workflow, against the
v0.8.0 release tarballs, on an L4 and on a Blackwell GPU (sm_120), which none of the four PRs could test.

**Result:** v0.9.0 writes the same files as v0.8.0. 5,679 of 5,679 output files are identical (`cmp`) on the L4 and
on the RTX PRO 4500 (103 runs: speed-v06's whole identity list, the palette sets, short `--batch` lists with the
production pair, single-file mode on every set), and v0.9.0's files on the two GPUs are identical too. The CUDA 12
packages give the same files as v0.8.0's and as the CUDA 13 package (1,220 / 1,220 on each GPU). The decoder gate
passes on both GPUs (691 / 691 encodes and 35 / 35 alpha items identical in dav1d and aomdec).
Compute-sanitizer finds no access error (memcheck), no synchronisation error (synccheck) and no race outside
the known `k_entropy_win` ring hand-off (racecheck) on either GPU. On Blackwell, memcheck needs compute-sanitizer
2026.3 (CUDA 13.4): the DLAMI's 2026.1 reports an invalid shared read inside CUB's scan kernel there, a tool false
positive (it reports the same for v0.8.0 and for a standalone CUB test, and 2026.3 reports nothing).
Speed on the L4: size range **+9.1 %** (257.3 against 235.9 MP/s), standard set +6.2 %, thumbnails +8.1 % (450 against
416 img/s), size range 4:4:4 10-bit +11.4 %; noise 0.2–0.7 %. A cold single JPEG (`av1gpu in.jpg out.avif`) takes
**359 instead of 1,059 ms** (2.5 MP; 2.95×), a `--batch` of a photo and its thumbnail 575 instead of 1,235 ms. On the
RTX PRO 4500: +4.9 to +10.9 % in batch mode, no regression anywhere; a cold single JPEG 506 instead of 1,027 ms.
k_recon's sm_120 code spills no more than v0.8.0's. One finding for production (Blackwell): a cold process there
spends about 150 ms more than on the L4 loading kernel modules, mostly the four 38 MB k_recon cubins, of which a
photo uses one (below).

## Binaries

| | file | sha256 |
|---|---|---|
| A13 | v0.8.0 `av1gpu-0.8.0-linux-x86_64.tar.gz` (CUDA 13.4), from the release page | ad98b846… (matches its `.sha256`) |
| A12 | v0.8.0 `av1gpu-0.8.0-linux-x86_64-cuda12.tar.gz` (CUDA 12.9) | 30d50f04… (matches) |
| B13 | main c553ffb `av1gpu-dev-c553ffb-linux-x86_64.tar.gz`, Release run 41596172304 | c77e109e… (matches) |
| B12 | main c553ffb `av1gpu-dev-c553ffb-linux-x86_64-cuda12.tar.gz`, same run | 11e9d401… (matches) |

Both instances downloaded the v0.8.0 tarballs themselves and checked them with `sha256sum -c` against the
release's `.sha256` files. The c553ffb artifacts were downloaded with `gh run download`, checked, copied with scp and
checked again on each instance (`logs/*/b_sha256.txt`). "A13c" in the speed tables is A13 copied to a second path,
the noise run.

## Setup

| | L4 (`av1gpu-rc09l4`) | RTX PRO 4500 (`av1gpu-rc09bw`) |
|---|---|---|
| instance | g6.4xlarge, on demand (no spot capacity in any subnet) | g7.4xlarge, spot |
| GPU | NVIDIA L4, sm_89, 23 GB, 72 W, 2,040 MHz max | RTX PRO 4500 Blackwell Server Edition, sm_120, 32 GB, 165 W, 2,415 MHz max |
| CPU | AMD EPYC 7R13, 16 vCPUs, 60 GB | Intel Xeon 6975P-C, 16 vCPUs, 60 GB |
| driver, tools | 595.112.07; compute-sanitizer 2026.1.0 (CUDA 13.2) | the same; from apt for the checks below: cuobjdump, nvdisasm, nvcc 13.4 and compute-sanitizer 2026.3.0 (CUDA 13.4) |
| OS | Ubuntu 24.04.5, kernel 7.0.0-1014-aws, DLAMI ami-00dc060a4dfb01df5 | the same |
| decoders | dav1d 1.4.1, aomdec 3.8.2 (Ubuntu packages) | the same |

The input sets are speed-v06's (`bench/results/aws-g6-l4-speed-v06/scripts`: size range, standard set, `mk_ll_sets.py`,
`mk_sets.py`, `mk_mono.sh`) and the palette study's (`prep_sets.py`: screen, flat_png, flat_jpg), made on each
instance by `scripts/setup.sh`. All 959 input files have the same SHA-256 on both instances
(`logs/*/inputs.sha256`). The single-image inputs (`~/st`): `photo.jpg` = 1068f4f5ac7e8d01_typical.jpg (1280 × 1920,
2.46 MP), `big.jpg` = 16aaa6421eaf9905_huge.jpg (4320 × 2432, 10.5 MP), `thumb512.jpg` = photo.jpg as a 512 × 384
thumbnail (ImageMagick `-resize 512x384^ -extent 512x384 -quality 85`, like `mk_sets.py`'s `d512_`), `t320.jpg` =
t320_037774ce28834ba2_tiny.jpg (320 × 196); the production pair is `photo.jpg` + `thumb512.jpg`.

Scripts (`scripts/`): `setup.sh`; `mktable.sh` (the run list, `~/sets9.tsv`), `gen.sh` (encode every run with one
binary), `cmp9.py` (compare two output trees byte by byte, or two SHA-256 manifests across machines); `checks9.sh`
(decoder gate, compute-sanitizer); `spd9.py`, `wall9.py`, `speed9.sh` (speed); `resusage9.py` (k_recon resources);
`modload.c`, `modload.sh` (module-load times); `post.sh`, `after.sh`, `bg.sh` (the order of it all). The AWS
launch / teardown scripts are not committed (they hold the account's network IDs).

## Byte identity

`gen.sh` encodes every run of the list with one binary; `cmp9.py` compares the two trees file by file (`cmp`), and
also the exit status of every run (per file in single-file mode) and the error messages of failed single-file
runs. Across the two machines it compares SHA-256 manifests of every output file. The list
(`scripts/mktable.sh`):

- speed-v06's whole `identity.sh`: the size range at 22 settings (presets, qualities, 4:4:4, 10-bit, 4:4:4 10-bit,
  tiles, `--tune none`, 1 and 6 encoders), monochrome (47) at 7, standard set, 280 thumbnails, Kodak lossy and
  lossless, RGBA (39) at 4, gray + alpha, odd sizes 1 × 1 … 4097 × 17, ll/misc, 16-bit PNGs, graphics, WebP, the
  mixed `--batch` list, ll/jpeg, and single-file mode (10 files × 3 settings);
- the palette sets (screen 40, flat_png 24, flat_jpg 24) at default, `--palette on`, `--yuv 444 --depth 10` and
  `--lossless` (JPEGs are refused with `--lossless`, by both versions);
- short `--batch` lists (≤ 8 encodes: one hardware queue in v0.9.0): the mixed list in 5 slices, default and 4:4:4
  10-bit; the production pair at default, 4:4:4 10-bit, `--preset slow` and `--quality 90`;
- single-file mode on 103 files from every set (thumbnails, size range, standard, RGBA, gray, odd, WebP, screen,
  flat_png, flat_jpg, mono, 16-bit, misc, ll/jpeg, Kodak) at default, `--palette on`, `--yuv 444 --depth 10`,
  `--preset slow`, `--quality 90`, and the 53 non-JPEGs at `--lossless`; the four single-image inputs at default and
  4:4:4 10-bit;
- three single-file error exits (a missing file, a JPEG with `--lossless`, a crop outside the image).

The CUDA 12 subset: the size range (default, 4:4:4 10-bit), standard set, thumbnails, Kodak / RGBA / graphics
lossless, RGBA, the 12 palette runs, the production pair, and single-file mode (10 files × 2 settings, 103 files at
default, 53 at `--lossless`, the four single-image inputs).

**Every run is identical:**

| comparison | runs | files | identical |
|---|---:|---:|---:|
| L4: v0.8.0 → v0.9.0 (CUDA 13) | 103 | 5,679 | **5,679** |
| RTX PRO 4500: v0.8.0 → v0.9.0 (CUDA 13) | 103 | 5,679 | **5,679** |
| v0.9.0 on the L4 → v0.9.0 on the RTX PRO 4500 | 103 | 5,679 | **5,679** |
| v0.8.0 on the L4 → v0.8.0 on the RTX PRO 4500 | 103 | 5,679 | 5,679 |
| L4: v0.8.0 → v0.9.0, CUDA 12 | 27 | 1,220 | **1,220** |
| L4: v0.9.0 CUDA 13 → CUDA 12 | 27 | 1,220 | **1,220** |
| RTX PRO 4500: v0.8.0 → v0.9.0, CUDA 12 | 27 | 1,220 | **1,220** |
| RTX PRO 4500: v0.9.0 CUDA 13 → CUDA 12 | 27 | 1,220 | **1,220** |
| L4: v0.8.0 CUDA 13 → CUDA 12 | 27 | 1,220 | 1,220 |

**av1gpu's output does not depend on the GPU** here: the L4 (Ada, sm_89 code) and the RTX PRO 4500 (Blackwell,
sm_120 code) write the same bytes for every one of the 5,679 files, in both versions.

Per run below (the skipped column counts inputs both versions skip: `--lossless` refuses JPEGs, one gray PNG is not
supported lossless; every error exit has the same status, 3, and the same message in both versions;
`logs/*/ident_*.md` and `.json`, `logs/*/cmp_*.txt`).

<details><summary>Every run</summary>

| run | options | inputs | skipped | L4: v0.8.0 → v0.9.0 | RTX PRO 4500: v0.8.0 → v0.9.0 | v0.9.0: L4 → RTX PRO 4500 |
|---|---|---:|---:|---:|---:|---:|
| sr_default | default | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_turbo3_nosmall | `--preset turbo3 --small-preset none` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_nosmall | `--small-preset none` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_444 | `--yuv 444` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_10 | `--depth 10` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_444_10 | `--yuv 444 --depth 10` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_base_nosmall | `--preset base --small-preset none` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_slow_nosmall | `--preset slow --small-preset none` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_fast_nosmall | `--preset fast --small-preset none` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_q30 | `--quality 30` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_q90 | `--quality 90` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_q95 | `--quality 95` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_tune_none | `--tune none` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_tiles_2_1_1 | `--tiles 2,1,1` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_default_enc1 | `GP_ENCODERS=1` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_default_enc6 | `GP_ENCODERS=6` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_slow | `--preset slow` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_444_q90 | `--yuv 444 --quality 90` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_444_q95 | `--yuv 444 --quality 95` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_444_slow | `--yuv 444 --preset slow` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_444_10_q95 | `--yuv 444 --depth 10 --quality 95` | 100 |  | 100/100 | 100/100 | 100/100 |
| sr_10_q95 | `--depth 10 --quality 95` | 100 |  | 100/100 | 100/100 | 100/100 |
| mono_default | default | 47 |  | 47/47 | 47/47 | 47/47 |
| mono_q90 | `--quality 90` | 47 |  | 47/47 | 47/47 | 47/47 |
| mono_q95 | `--quality 95` | 47 |  | 47/47 | 47/47 | 47/47 |
| mono_slow | `--preset slow` | 47 |  | 47/47 | 47/47 | 47/47 |
| mono_nosmall | `--small-preset none` | 47 |  | 47/47 | 47/47 | 47/47 |
| mono_10 | `--depth 10` | 47 |  | 47/47 | 47/47 | 47/47 |
| mono_tune_none | `--tune none` | 47 |  | 47/47 | 47/47 | 47/47 |
| std_q95 | `--quality 95` | 100 |  | 100/100 | 100/100 | 100/100 |
| std_slow | `--preset slow` | 100 |  | 100/100 | 100/100 | 100/100 |
| thumbs_q95 | `--quality 95` | 280 |  | 280/280 | 280/280 | 280/280 |
| thumbs_slow | `--preset slow` | 280 |  | 280/280 | 280/280 | 280/280 |
| kodak_q95 | `--quality 95` | 24 |  | 24/24 | 24/24 | 24/24 |
| kodak_444_q90 | `--yuv 444 --quality 90` | 24 |  | 24/24 | 24/24 | 24/24 |
| rgba_q95 | `--quality 95` | 39 |  | 39/39 | 39/39 | 39/39 |
| odd_q95 | `--quality 95` | 30 |  | 30/30 | 30/30 | 30/30 |
| odd_slow | `--preset slow` | 30 |  | 30/30 | 30/30 | 30/30 |
| std_default | default | 100 |  | 100/100 | 100/100 | 100/100 |
| std_nosmall | `--small-preset none` | 100 |  | 100/100 | 100/100 | 100/100 |
| thumbs_default | default | 280 |  | 280/280 | 280/280 | 280/280 |
| thumbs_nosmall | `--small-preset none` | 280 |  | 280/280 | 280/280 | 280/280 |
| kodak_default | default | 24 |  | 24/24 | 24/24 | 24/24 |
| kodak_lossless | `--lossless` | 24 |  | 24/24 | 24/24 | 24/24 |
| kodak_lossless_ycgco | `--lossless --ycgco-re` | 24 |  | 24/24 | 24/24 | 24/24 |
| rgba_default | default | 39 |  | 39/39 | 39/39 | 39/39 |
| rgba_aq100 | `--alpha-quality 100` | 39 |  | 39/39 | 39/39 | 39/39 |
| rgba_lossless | `--lossless` | 39 |  | 39/39 | 39/39 | 39/39 |
| rgba_no_alpha | `--no-alpha` | 39 |  | 39/39 | 39/39 | 39/39 |
| gray_default | default | 9 |  | 9/9 | 9/9 | 9/9 |
| gray_lossless | `--lossless` | 9 | 1 | 8/8 | 8/8 | 8/8 |
| odd_default | default | 30 |  | 30/30 | 30/30 | 30/30 |
| odd_444_10 | `--yuv 444 --depth 10` | 30 |  | 30/30 | 30/30 | 30/30 |
| odd_lossless | `--lossless` | 30 | 20 | 10/10 | 10/10 | 10/10 |
| misc_default | default | 13 |  | 13/13 | 13/13 | 13/13 |
| png16_10_default | `--depth 10` | 8 |  | 8/8 | 8/8 | 8/8 |
| png16_10_lossless | `--lossless` | 8 |  | 8/8 | 8/8 | 8/8 |
| llgraphics_lossless | `--lossless --ycgco-re` | 27 |  | 27/27 | 27/27 | 27/27 |
| webp_default | default | 7 |  | 7/7 | 7/7 | 7/7 |
| webp_lossless | `--lossless` | 7 |  | 7/7 | 7/7 | 7/7 |
| mixed_list | default | 33 |  | 33/33 | 33/33 | 33/33 |
| mixed_list_444_10 | `--yuv 444 --depth 10` | 33 |  | 33/33 | 33/33 | 33/33 |
| lljpeg_set | default | 25 |  | 25/25 | 25/25 | 25/25 |
| pal_screen_default | default | 40 |  | 40/40 | 40/40 | 40/40 |
| pal_screen_palette_on | `--palette on` | 40 |  | 40/40 | 40/40 | 40/40 |
| pal_screen_444_10 | `--yuv 444 --depth 10` | 40 |  | 40/40 | 40/40 | 40/40 |
| pal_screen_lossless | `--lossless` | 40 |  | 40/40 | 40/40 | 40/40 |
| pal_flat_png_default | default | 24 |  | 24/24 | 24/24 | 24/24 |
| pal_flat_png_palette_on | `--palette on` | 24 |  | 24/24 | 24/24 | 24/24 |
| pal_flat_png_444_10 | `--yuv 444 --depth 10` | 24 |  | 24/24 | 24/24 | 24/24 |
| pal_flat_png_lossless | `--lossless` | 24 |  | 24/24 | 24/24 | 24/24 |
| pal_flat_jpg_default | default | 24 |  | 24/24 | 24/24 | 24/24 |
| pal_flat_jpg_palette_on | `--palette on` | 24 |  | 24/24 | 24/24 | 24/24 |
| pal_flat_jpg_444_10 | `--yuv 444 --depth 10` | 24 |  | 24/24 | 24/24 | 24/24 |
| pal_flat_jpg_lossless | `--lossless` | 24 | 24 | 0/0 | 0/0 | 0/0 |
| short_mixed_00 | default | 8 |  | 8/8 | 8/8 | 8/8 |
| short_mixed_00_444_10 | `--yuv 444 --depth 10` | 8 |  | 8/8 | 8/8 | 8/8 |
| short_mixed_01 | default | 8 |  | 8/8 | 8/8 | 8/8 |
| short_mixed_01_444_10 | `--yuv 444 --depth 10` | 8 |  | 8/8 | 8/8 | 8/8 |
| short_mixed_02 | default | 8 |  | 8/8 | 8/8 | 8/8 |
| short_mixed_02_444_10 | `--yuv 444 --depth 10` | 8 |  | 8/8 | 8/8 | 8/8 |
| short_mixed_03 | default | 8 |  | 8/8 | 8/8 | 8/8 |
| short_mixed_03_444_10 | `--yuv 444 --depth 10` | 8 |  | 8/8 | 8/8 | 8/8 |
| short_mixed_04 | default | 1 |  | 1/1 | 1/1 | 1/1 |
| short_mixed_04_444_10 | `--yuv 444 --depth 10` | 1 |  | 1/1 | 1/1 | 1/1 |
| short_pair | default | 2 |  | 2/2 | 2/2 | 2/2 |
| short_pair_444_10 | `--yuv 444 --depth 10` | 2 |  | 2/2 | 2/2 | 2/2 |
| short_pair_slow | `--preset slow` | 2 |  | 2/2 | 2/2 | 2/2 |
| short_pair_q90 | `--quality 90` | 2 |  | 2/2 | 2/2 | 2/2 |
| single10_default | default | 10 |  | 10/10 | 10/10 | 10/10 |
| single10_444_10 | `--yuv 444 --depth 10` | 10 |  | 10/10 | 10/10 | 10/10 |
| single10_slow | `--preset slow` | 10 |  | 10/10 | 10/10 | 10/10 |
| single_all_default | default | 103 |  | 103/103 | 103/103 | 103/103 |
| single_all_lossless | `--lossless` | 53 | 1 | 52/52 | 52/52 | 52/52 |
| single_all_palette_on | `--palette on` | 103 |  | 103/103 | 103/103 | 103/103 |
| single_all_444_10 | `--yuv 444 --depth 10` | 103 |  | 103/103 | 103/103 | 103/103 |
| single_all_slow | `--preset slow` | 103 |  | 103/103 | 103/103 | 103/103 |
| single_all_q90 | `--quality 90` | 103 |  | 103/103 | 103/103 | 103/103 |
| single_st_default | default | 4 |  | 4/4 | 4/4 | 4/4 |
| single_st_444_10 | `--yuv 444 --depth 10` | 4 |  | 4/4 | 4/4 | 4/4 |
| err_missing | default | 1 |  | exit 3 both, same message | exit 3 both, same message | exit 3 both, same message |
| err_jpeg_lossless | `--lossless` | 1 |  | exit 3 both, same message | exit 3 both, same message | exit 3 both, same message |
| err_crop | `--crop 999x999+2000+2000` | 1 |  | exit 3 both, same message | exit 3 both, same message | exit 3 both, same message |
| **all (103 runs)** | | | | **5679/5679** | **5679/5679** | **5679/5679** |

| run | options | inputs | skipped | L4: v0.8.0 → v0.9.0, CUDA 12 | L4: v0.9.0 CUDA 13 → CUDA 12 | RTX PRO 4500: v0.8.0 → v0.9.0, CUDA 12 | RTX PRO 4500: v0.9.0 CUDA 13 → CUDA 12 |
|---|---|---:|---:|---:|---:|---:|---:|
| sr_default | default | 100 |  | 100/100 | 100/100 | 100/100 | 100/100 |
| sr_444_10 | `--yuv 444 --depth 10` | 100 |  | 100/100 | 100/100 | 100/100 | 100/100 |
| std_default | default | 100 |  | 100/100 | 100/100 | 100/100 | 100/100 |
| thumbs_default | default | 280 |  | 280/280 | 280/280 | 280/280 | 280/280 |
| kodak_lossless | `--lossless` | 24 |  | 24/24 | 24/24 | 24/24 | 24/24 |
| rgba_default | default | 39 |  | 39/39 | 39/39 | 39/39 | 39/39 |
| rgba_lossless | `--lossless` | 39 |  | 39/39 | 39/39 | 39/39 | 39/39 |
| llgraphics_lossless | `--lossless --ycgco-re` | 27 |  | 27/27 | 27/27 | 27/27 | 27/27 |
| pal_screen_default | default | 40 |  | 40/40 | 40/40 | 40/40 | 40/40 |
| pal_screen_palette_on | `--palette on` | 40 |  | 40/40 | 40/40 | 40/40 | 40/40 |
| pal_screen_444_10 | `--yuv 444 --depth 10` | 40 |  | 40/40 | 40/40 | 40/40 | 40/40 |
| pal_screen_lossless | `--lossless` | 40 |  | 40/40 | 40/40 | 40/40 | 40/40 |
| pal_flat_png_default | default | 24 |  | 24/24 | 24/24 | 24/24 | 24/24 |
| pal_flat_png_palette_on | `--palette on` | 24 |  | 24/24 | 24/24 | 24/24 | 24/24 |
| pal_flat_png_444_10 | `--yuv 444 --depth 10` | 24 |  | 24/24 | 24/24 | 24/24 | 24/24 |
| pal_flat_png_lossless | `--lossless` | 24 |  | 24/24 | 24/24 | 24/24 | 24/24 |
| pal_flat_jpg_default | default | 24 |  | 24/24 | 24/24 | 24/24 | 24/24 |
| pal_flat_jpg_palette_on | `--palette on` | 24 |  | 24/24 | 24/24 | 24/24 | 24/24 |
| pal_flat_jpg_444_10 | `--yuv 444 --depth 10` | 24 |  | 24/24 | 24/24 | 24/24 | 24/24 |
| pal_flat_jpg_lossless | `--lossless` | 24 | 24 | 0/0 | 0/0 | 0/0 | 0/0 |
| short_pair | default | 2 |  | 2/2 | 2/2 | 2/2 | 2/2 |
| short_pair_444_10 | `--yuv 444 --depth 10` | 2 |  | 2/2 | 2/2 | 2/2 | 2/2 |
| single10_default | default | 10 |  | 10/10 | 10/10 | 10/10 | 10/10 |
| single10_444_10 | `--yuv 444 --depth 10` | 10 |  | 10/10 | 10/10 | 10/10 | 10/10 |
| single_all_default | default | 103 |  | 103/103 | 103/103 | 103/103 | 103/103 |
| single_all_lossless | `--lossless` | 53 | 1 | 52/52 | 52/52 | 52/52 | 52/52 |
| single_st_default | default | 4 |  | 4/4 | 4/4 | 4/4 | 4/4 |
| **all (27 runs)** | | | | **1220/1220** | **1220/1220** | **1220/1220** | **1220/1220** |

</details>

## Decoder gate

`tests/decoder_gate.py` of c553ffb with B13 (`--batch --debug-outputs`, every encode decoded by dav1d 1.4.1 and aomdec
3.8.2 and compared with the encoder's own reconstruction), 18 runs on each GPU (`logs/*/gate.txt`):

| run | encodes |
|---|---:|
| 30 size-range photos, qindex 60 / 120 / 200 | 90 |
| the same, `--yuv 444 --depth 10`, qindex 120 | 30 |
| 28 thumbnails, qindex 60 / 120 | 56 |
| RGBA (39 PNGs, `--expect-alpha`) | 39 + 35 alpha items |
| odd sizes (30) | 30 |
| monochrome (47), qindex 26 / 120 | 94 |
| Kodak `--lossless` | 24 |
| screen (40): default, `--palette on`, `--yuv 444 --depth 10`, `--lossless` | 160 |
| flat_png (24): the same four | 96 |
| flat_jpg (24): default, `--palette on`, `--yuv 444 --depth 10` | 72 |

**PASS on both GPUs: 691 / 691 encodes and 35 / 35 alpha items identical in both decoders**, with the same byte
counts on the two GPUs.

## compute-sanitizer

B13 with memcheck (`--leak-check full`), racecheck (analysis mode) and synccheck on 8 jobs per GPU (`scripts/checks9.sh`):
`--batch` on speed-v06's 3 photos (2.16 MP turbo with 4 × 4 tiles, 1.71 MP turbo, 0.78 MP base preset) at default and
at `--yuv 444 --depth 10`, `--batch` on the production pair, and single-file mode on `photo.jpg`, `thumb512.jpg`,
`t320.jpg`, a screen PNG (screen_ai_source_verification.png, palette mode) and Kodak kodim01 with `--lossless`. Every
output of a sanitizer run that completes is compared with the plain run's.

| tool | L4 (compute-sanitizer 2026.1, CUDA 13.2) | RTX PRO 4500 |
|---|---|---|
| memcheck | **0 access errors** in all 8, outputs identical; batch mode lists only the 256 KB variance-boost tables of each encoder (5–6 × 262,144 B, allocated in `GpEncoder::init`, not freed at exit; known since v0.6, docs/RESULTS.md); single-file mode leaks nothing | 2026.3 (CUDA 13.4): **0 access errors** in all 8, outputs identical; batch mode lists only the 256 KB variance-boost tables of each encoder (5–6 × 262,144 B, allocated in `GpEncoder::init`, not freed at exit; known since v0.6, docs/RESULTS.md) |
| | | 2026.1 (CUDA 13.2): a false report in all 8, below |
| racecheck | only the known `k_entropy_win` ring hand-off pairs, all 8 (below) | only the known `k_entropy_win` ring hand-off pairs, all 8 (below) |
| synccheck | 0 errors, all 8, outputs identical | 0 errors, all 8, outputs identical |

**racecheck** reports only pairs inside the windowed entropy coder in gpu_entropy.cu (`logs/*/race_pairs.txt`):
`win_group` (1219, 1224) and the producer's ring writes (1303, 1344, 1347) against the coder warps' reads (1334, 1338,
1369, 1373, 1380), and `vload` / `vstore` / `__nanosleep` of the window counters (1261, 1262). That is the documented
hand-off through volatile flags and `__threadfence_block()` (docs/RESULTS.md, Correctness), with #59's second ring
(B1 → B2: 1344 / 1347 against 1369 / 1373 / 1380), as ent08 found. No other kernel reports anything.

**The Blackwell memcheck report is a false positive of compute-sanitizer 2026.1.** With the DLAMI's sanitizer (2026.1,
CUDA 13.2) every memcheck run on the RTX PRO 4500 stops at one `Invalid __shared__ read of size 16 bytes` in
`cuda::ptx::cp_async_bulk_cp_mask` (cp_async_bulk.h:205), inside CUB's `DeviceScan::ExclusiveSum<unsigned*,unsigned*,int>`
(its sm_100+ "warpspeed" kernel, CCCL `_V_300403`), called from `av1g_gpu::run_batch_device` (gpu_entropy.cu:1734).
memcheck then aborts the kernel, and the rest of each log is the sticky-error cascade (so those runs wrote no output).
What decides it (`logs/bw/mc_ab.txt`, `logs/bw/mc134.txt`; `scripts/mc_ab.sh`, `scripts/mc134.sh`, `scripts/cubsweep.cu`):

- v0.8.0's CUDA 13.4 binary gets the same report; both CUDA 12.9 builds (v0.8.0 and v0.9.0, CUB 2.8) are clean. The
  L4 does not run that kernel. So it comes with the CUDA 13.4 builds (from v0.7.1), not with #56–#59.
- compute-sanitizer 2026.3 (CUDA 13.4, apt `cuda-sanitizer-13-4`) finds nothing: B13 and A13, photo and pair, 0 errors;
  then all 8 jobs above.
- A standalone repro built by nvcc 13.4 for sm_120 (CUB 3.4.3, the package's), the call of gpu_entropy.cu:1734 (the
  scratch size queried with a `const uint32_t *` input as at line 1569, an aligned sub-buffer of exactly that size,
  `ExclusiveSum` on a non-blocking stream, results checked): every n from 1 to 5,000 and every 7th to 200,000. Clean
  under 2026.3; under 2026.1 the same invalid `__shared__` read already at n = 1.

So neither av1gpu's call nor CUB reads out of bounds; the older sanitizer misreads CUB 3.4's TMA bulk copy with a byte
mask on sm_120. Sanitizer runs of a CUDA 13.4 build on Blackwell need compute-sanitizer 2026.3 or newer.

## k_recon on sm_120: registers, stack, spills

recon08 found that #57's unrolled forward loops grew `rc_luma`'s spills on sm_89 (ptxas with CUDA 13.2). From the
shipped binaries (`scripts/resusage9.py`: `cuobjdump --dump-resource-usage`, and `nvdisasm` of the four k_recon cubins
per architecture, counting local-memory stores and loads, STL / LDL, per function; `logs/bw/resusage.txt`):

| kernel (sm_120) | v0.8.0 registers / stack | v0.9.0 registers / stack | STL v0.8.0 → v0.9.0 | LDL v0.8.0 → v0.9.0 |
|---|---:|---:|---:|---:|
| `k_recon<1,8>` (4:2:0 8-bit) | 128 / 1,792 B | 128 / 1,632 B | 6,338 → 6,070 (−4 %) | 18,182 → 15,243 (−16 %) |
| `k_recon<0,8>` (4:4:4 8-bit) | 128 / 1,952 B | 128 / 1,664 B | 6,572 → 6,258 (−5 %) | 18,446 → 15,274 (−17 %) |
| `k_recon<1,10>` (4:2:0 10-bit) | 128 / 1,904 B | 128 / 1,792 B | 7,142 → 6,717 (−6 %) | 19,457 → 16,019 (−18 %) |
| `k_recon<0,10>` (4:4:4 10-bit) | 128 / 2,096 B | 128 / 1,792 B | 7,052 → 6,744 (−4 %) | 19,412 → 16,002 (−18 %) |

The STL / LDL counts are summed over the kernel and its non-inlined functions. Per function (4:2:0 8-bit, sm_120):
`rc_luma` 5,493 / 16,587 → 5,328 / 13,794, `cta_inverse_add` 177 / 263 → 41 / 79, `rc_chroma` 271 / 767 →
293 / 806 (the one function with more, +8 % / +5 %). Shared memory is unchanged (24,496 B for 4:2:0 8-bit), and
every k_recon kernel still uses 128 registers.

**On sm_120 nothing spills more than in v0.8.0 in total**: the stack frames are 6–15 % smaller and the static local
loads 16–18 % fewer. The growth recon08 saw is sm_89's: in the shipped sm_89 code (CUDA 13.4) the frame grows from
3,104 to 3,312 B and `rc_luma`'s LDL count from 2,554 to 2,981, while its STL count stays (1,020 → 1,019) and the
kernel's total STL count falls 22 %. These are static instruction counts, not the number executed; the batch speeds
below include all of it.

## Speed

The stored method (speed-v04 / speed-v06): each set repeated to at least 1,000 MP (`bench.speed_set`), a warm-up and 3
timed runs per binary (`--stats`), 2 rounds with the binary order alternating (v0.9.0, v0.8.0, v0.8.0 again; then
reversed), median of the 6 timed runs. "v0.8.0 again" (A13c, the same binary from a second path) is the noise run.
Default settings (3 encoders). MP/s and img/s are av1gpu's `--stats`; wall MP/s counts the whole process, CUDA start-up
included. Nothing else ran on the GPU (`logs/*/speed9.log`, `logs/*/spd9_batch.jsonl`, `scripts/spd9.py`). Output bytes
were the same in every run of a set and for both versions.

### NVIDIA L4, batch mode

| set | v0.9.0 | v0.8.0 | change | v0.8.0 again (noise) | rounds v0.9.0 | rounds v0.8.0 | wall MP/s v0.9.0 / v0.8.0 |
|---|---:|---:|---:|---:|---|---|---:|
| size range (100 JPEGs × 3, 1,434 MP), MP/s | **257.3** | 235.9 | **+9.1 %** | 234.4 (−0.7 %) | 260.3 / 254.3 | 237.1 / 234.4 | 220.7 / 203.6 |
| standard (100 JPEGs × 13, 1,023 MP), MP/s | **81.1** | 76.4 | **+6.2 %** | 76.6 (+0.3 %) | 81.1 / 81.2 | 76.3 / 76.5 | 75.5 / 71.3 |
| thumbnails (280 JPEGs × 24, 1,019 MP), img/s | **450.1** | 416.4 | **+8.1 %** | 413.3 (−0.7 %) | 450.2 / 450.1 | 413.6 / 418.5 | 64.3 / 59.6 |
| size range `--yuv 444 --depth 10`, MP/s | **189.9** | 170.4 | **+11.4 %** | 170.1 (−0.2 %) | 189.6 / 191.7 | 168.5 / 171.9 | 168.9 / 152.6 |

Every gain is outside the noise: v0.9.0 is ahead of both v0.8.0 runs in every round, and v0.8.0 against itself moved
by at most 0.7 %. The four PRs measured on their own (each against cde1f74 on its own instance; size range / standard /
thumbnails): #56 +5.1 / +1.6 / +1.1 % (the last two within the noise), #57 +4.1 / +3.4 / +3.2 %, #58 +1.1 / +4.2 /
+3.1 %, #59 within the noise in batch mode. Together, here: +9.1 / +6.2 / +8.1 %.

### RTX PRO 4500 Blackwell, batch mode

| set | v0.9.0 | v0.8.0 | change | v0.8.0 again (noise) | rounds v0.9.0 | rounds v0.8.0 | wall MP/s v0.9.0 / v0.8.0 |
|---|---:|---:|---:|---:|---|---|---:|
| size range (1,434 MP), MP/s | **434.4** | 406.4 | **+6.9 %** | 408.5 (+0.5 %) | 437.1 / 420.9 | 411.2 / 401.6 | 336.5 / 318.6 |
| standard (1,023 MP), MP/s | **148.3** | 141.4 | **+4.9 %** | 141.5 (+0.1 %) | 148.7 / 147.6 | 141.8 / 141.1 | 130.1 / 124.7 |
| thumbnails (1,019 MP), img/s | **830.7** | 775.6 | **+7.1 %** | 774.8 (−0.1 %) | 833.6 / 827.0 | 778.4 / 774.8 | 112.7 / 105.8 |
| size range `--yuv 444 --depth 10`, MP/s | **324.6** | 292.6 | **+10.9 %** | 292.6 (0.0 %) | 327.1 / 321.9 | 295.5 / 291.3 | 267.0 / 244.1 |

**No regression on Blackwell**; every set is faster, outside the noise (at most 0.5 %). The gains are a little smaller
than on the L4 for the 4:2:0 sets. The 4500 ran at 2,400–2,407 MHz, below its 165 W limit
(`logs/bw/clocks_*.txt`); the L4 at its 72 W cap.

### Cold single process

`av1gpu IN OUT` (no `--batch`), one process per run, wall time from the launcher (`scripts/wall9.py`): a warm-up per
binary, then 12 runs each with the four binaries alternating run by run (forward, then reversed); median in ms
(`logs/*/spd9_single.txt`). The fourth binary is v0.8.0 with `GP_MAX_CONNECTIONS=1`, its own setting for one hardware
queue (v0.8.0 asks for 32 by default; v0.9.0 asks for 1 for a single image and for short lists, #56).

**NVIDIA L4:**

| input | v0.9.0 | v0.8.0 | v0.8.0, `GP_MAX_CONNECTIONS=1` | v0.8.0 again (noise) | v0.9.0 against v0.8.0 |
|---|---:|---:|---:|---:|---:|
| 2.46 MP JPEG (1280 × 1920) | **358.7** | 1,059.4 | 543.4 | 1,057.7 (−0.2 %) | −66 % (2.95×) |
| 10.5 MP JPEG (4320 × 2432) | **467.1** | 1,214.1 | 700.8 | 1,213.4 (−0.1 %) | −62 % (2.60×) |
| 512 × 384 thumbnail | **349.5** | 1,059.9 | 554.2 | 1,059.6 (0.0 %) | −67 % (3.03×) |
| 320 × 196 thumbnail | **376.7** | 1,098.4 | 578.9 | 1,091.9 (−0.6 %) | −66 % (2.92×) |
| `--batch` photo + 512 × 384 thumbnail | **575.3** | 1,235.3 | 722.4 | 1,231.5 (−0.3 %) | −53 % (2.15×) |

**RTX PRO 4500:**

| input | v0.9.0 | v0.8.0 | v0.8.0, `GP_MAX_CONNECTIONS=1` | v0.8.0 again (noise) | v0.9.0 against v0.8.0 |
|---|---:|---:|---:|---:|---:|
| 2.46 MP JPEG | **506.3** | 1,026.7 | 629.3 | 1,032.6 (+0.6 %) | −51 % (2.03×) |
| 10.5 MP JPEG | **577.8** | 1,139.4 | 745.8 | 1,143.2 (+0.3 %) | −49 % (1.97×) |
| 512 × 384 thumbnail | **498.0** | 1,035.6 | 643.4 | 1,037.6 (+0.2 %) | −52 % (2.08×) |
| 320 × 196 thumbnail | **523.2** | 1,058.5 | 660.2 | 1,060.8 (+0.2 %) | −51 % (2.02×) |
| `--batch` photo + 512 × 384 thumbnail | **677.1** | 1,190.0 | 788.7 | 1,185.9 (−0.3 %) | −43 % (1.76×) |

- **Per upload** (a JPEG and its thumbnail), production's case: one process per file went from 1,059 + 1,060 =
  2,119 ms to 359 + 350 = 708 ms on the L4, and from 2,062 to 1,004 ms on the RTX PRO 4500. A 2-entry `--batch`:
  1,235 → 575 ms (L4) and 1,190 → 677 ms (4500). On the L4 two single-file runs now cost about as much as the batch
  (708 against 575 ms).
- **v0.8.0 with `GP_MAX_CONNECTIONS=1`** gets about half of the gain on the L4 (1,059 → 543 ms) and less on the 4500
  (1,027 → 629); v0.9.0's other start-up changes (#56: lazy arena, decode overlapped with the encoder's set-up, the
  LUT row, `_exit`; #59: one frame uploaded without a host copy) take another third off on the L4 (543 → 359 ms) and a fifth on the 4500 (629 → 506).
- **The 4500 starts slower than the L4** with v0.9.0 (506 against 359 ms for the photo), while v0.8.0 takes about the
  same on both: its 32 hardware queues cost 516 ms on the L4 and 397 ms on the 4500 (v0.8.0 against v0.8.0 with
  `GP_MAX_CONNECTIONS=1`). The 4500's kernel code is larger. `scripts/modload.c` times the driver loading each
  cubin of the binary for the GPU's architecture (extracted with cuobjdump 13.4; warm page cache; 3 runs;
  `logs/*/modload_*.txt`):

  | | L4 (sm_89) | RTX PRO 4500 (sm_120) |
  |---|---:|---:|
  | context creation (driver API, 8 queues) | 141 ms | 120 ms |
  | all 29 modules of v0.9.0, lazy loading (the default) | 94 MB, **48 ms** | 240 MB, **200 ms** |
  | of which the four k_recon modules | 7–8 MB each, 8.5 ms | 38 MB each, **144 ms** |

  The difference, 152 ms, is about the 147 ms between the two GPUs' cold runs. Lazy loading does not help: loading
  a k_recon module itself takes 33–37 ms on sm_120, its one kernel then 1.4 ms more. nvcc 13.4 emits about 8× the
  instructions for `rc_luma` on sm_120 (498,853 against 63,499 for 4:2:0 8-bit; `logs/bw/resusage.txt`). v0.9.0
  still uploads its tables into every kernel module at start (pipe08's "not done" list), so a photo loads all four
  k_recon variants and uses one. Loading only the variant a run needs would save roughly 100 ms per process on
  Blackwell, production's GPU (an estimate from these load times, not measured in av1gpu).

### Every av1gpu speed of the docs, L4

speed-v06's `spd.py` unchanged except for the binaries and paths (`scripts/speed9.sh`: new = v0.9.0's CUDA 13.4
package, old = v0.8.0's tarball): every av1gpu speed of README.md and docs/RESULTS.md, measured the way it was
published (sets repeated to ≥ 1,000 MP, the lossless set 1× and 4×, the RGBA set 20×; a warm-up and 3 timed runs,
2 rounds with the order alternating, median of 6). `logs/l4/docs/*.jsonl`, `logs/l4/docs_speed.txt` (with power,
SM utilisation and clock from `nvidia-smi dmon`), and `results.json` (`speed`, the layout of speed-v06's, which
`docs/charts/make_charts.py` now reads). This run has no noise binary; the batch runs above (same session) put
the noise at 0.2–0.7 %.

| setting | v0.6 (speed-v06, another instance) | v0.8.0 | v0.9.0 | v0.9.0 vs v0.8.0 |
|---|---:|---:|---:|---:|
| size range, default (3 encoders), MP/s | 224.7 | 231.5 | **252.4** | +9.0 % |
| size range, `GP_ENCODERS=2` | 207.1 | 213.8 | **225.6** | +5.5 % |
| size range, `GP_ENCODERS=4` | 229.7 | 236.4 | **260.7** | +10.3 % |
| size range, `--small-preset none` | 261.7 | 267.7 | **289.8** | +8.3 % |
| size range, `--small-preset none`, 4 encoders | 271.5 | 276.9 | **298.9** | +7.9 % |
| size range, `--preset turbo3`, 4 encoders | 291.1 | 294.6 | **331.0** | +12.4 % |
| size range, `turbo3 --small-preset none`, 4 encoders | 366.8 | 376.9 | **419.5** | +11.3 % |
| size range, `--tiles 2,1,1` (2 × 2) | 205.0 | 212.4 | **230.0** | +8.3 % |
| size range, `--tiles 2,3,3` (8 × 8) | 236.7 | 241.2 | **269.0** | +11.5 % |
| standard, default | 72.4 | 76.2 | **81.2** | +6.6 % |
| standard, `GP_ENCODERS=2` | 71.2 | 72.5 | **77.1** | +6.3 % |
| standard, `GP_ENCODERS=4` | 72.7 | 76.1 | **80.9** | +6.3 % |
| standard, `--small-preset none` | 168.6 | 178.8 | **190.8** | +6.7 % |
| standard, `--small-preset none`, 4 encoders | 172.1 | 179.0 | **191.0** | +6.7 % |
| size range `--yuv 444`, 4 encoders | 169.6 | 174.7 | **197.9** | +13.3 % |
| size range `--depth 10`, 4 encoders | 229.5 | 231.4 | **253.3** | +9.5 % |
| size range `--yuv 444 --depth 10`, 4 encoders | 166.4 | 172.3 | **193.9** | +12.6 % |
| thumbnails, default, MP/s (img/s) | 59.7 (393) | 62.8 (414) | **67.5 (445)** | +7.5 % |
| thumbnails, `--small-preset none` | 143.8 (948) | 148.3 (977) | **161.0 (1062)** | +8.6 % |
| Kodak, default, MP/s (img/s) | 54.4 (138) | 55.4 (141) | **59.6 (152)** | +7.7 % |
| Kodak, `--small-preset none` | 129.9 (330) | 135.2 (344) | **143.4 (365)** | +6.0 % |
| Kodak, `turbo3 --small-preset none` | 219.1 (557) | 229.9 (585) | **253.8 (645)** | +10.4 % |
| lossless set, one pass, wall MP/s | 18.9 | 20.3 | **21.8** | +7.7 % |
| lossless set, one pass, `--ycgco-re`, wall | 19.4 | 20.6 | **22.1** | +7.3 % |
| lossless set × 4, MP/s | 43.6 | 43.7 | **45.3** | +3.7 % |
| lossless set × 4, `--ycgco-re` | 44.5 | 42.7 | **44.3** | +3.7 % |
| RGBA × 20, default, 4 encoders, wall MP/s | 43.9 | 29.1 | **31.5** | +8.2 % |
| RGBA × 20, `--alpha-quality 100`, wall | 38.8 | 28.0 | **30.1** | +7.6 % |
| RGBA × 20, `--lossless`, wall | 27.6 | 27.6 | **29.3** | +6.3 % |
| RGBA × 20, `--no-alpha`, wall | 49.4 | 33.8 | **36.8** | +8.9 % |

- The batch A/B above measured the same size range at 257.3 / 235.9 MP/s (3 encoders) and this run at 252.4 / 231.5:
  the L4 drifts by about 2 % over an hour, and both versions move together.
- **The lossless set has 51 PNGs here, not 47**: `mk_ll_sets.py` renders every SVG of `docs/charts/`, and the
  repository has more charts now (27 graphics instead of 23; 49.1 MP instead of 42.7). The README's lossless table
  keeps the 47-PNG figures; v0.8.0 and v0.9.0 compare with each other here.
- **The RGBA set is slower than with v0.6** in both v0.8.0 and v0.9.0 (31.5 against 43.9 MP/s, wall): palette mode,
  on by default since v0.8.0 (`--palette auto`), takes 15 of the 39 images
  (`logs/l4/rgba_palette_count.txt`). With `--palette off` (`scripts/spd_paloff.py`, `spd_paloff2.py`, the same
  method; `logs/l4/paloff_speed.txt`) v0.9.0 runs the RGBA set at 50.4 MP/s (default), 43.9 (`--alpha-quality 100`), 30.4 (`--lossless`) and 57.3 (`--no-alpha`),
  v0.8.0 at 44.4, 39.3, 28.1 and 49.8, as v0.6 did; the lossless set × 4 at 49.8 and 49.1 MP/s (GBR, YCgCo-Re;
  v0.8.0: 46.8 and 47.2). The docs' transparency table (sizes without palette mode) now carries the `--palette off`
  speeds, with the default's in the text.

## Docs updated with these numbers

- `docs/RESULTS.md`, NVIDIA L4: the speed section (every av1gpu row, power and clocks, v0.8.0 in the same session,
  thumbnails, Kodak), a new "Single-image start-up" section (both GPUs), the ratios that use av1gpu's speed (SVT-AV1,
  CPU builds, NVENC, CVVDP, the RTX PRO 2000 comparison), FPGA (Kodak), 4:4:4 / 10-bit, lossless (a v0.9.0 sentence;
  the table keeps its 47-PNG figures), transparency (`--palette off` speeds, the default's in the text), the
  correctness table (three v0.9.0 rows), and the release-build section (v0.9.0 on the RTX PRO 4500).
- `README.md`: the headline speed (261 MP/s with 4 encoders) and a cold single-image row (0.36 s), the TL;DR ratios
  (9× libaom, 17× NVENC), the speed table, the chart descriptions, the match-ups, 4:4:4 / 10-bit, CVVDP, FPGA,
  transparency, the tile speeds, the single-image note in Usage, Download (start-up on Blackwell) and Status (the
  RTX PRO 4500's identical output).
- `docs/charts/make_charts.py` reads av1gpu's speeds from this directory's `results.json` (was speed-v06's); the
  speed axes of four charts are wider. The charts are regenerated; only the av1gpu speeds and the ratios computed
  from them change.
- Not changed: the lossless table's speeds (measured on the 47-PNG set), the palette, crop / resize, CPU-thread and
  Windows sections (their own studies), and the README's terminal recording (v0.6, recorded).

## Files

- `results.json`: `speed` (the L4 docs run, speed-v06's layout: `<phase>/<set>/<setting>/new|old`, new = v0.9.0),
  `batch` and `single` (spd9.py on both GPUs), `identity` (per comparison and run), `gate`, binaries and environment.
- `logs/l4/`, `logs/bw/`: `env.txt`, `inputs.sha256`, `b_sha256.txt`; `gen_*.txt` (every encode run), `cmp_*.txt` and
  `ident_*.md` / `.json` (identity per run), `outputs_*_digests.txt` (per run: file count and a SHA-256 over every
  output's SHA-256; the same 103 lines for v0.8.0 and v0.9.0 on both GPUs), `err_messages.txt`; `gate.txt`, `gate/`;
  `sanitizer*.txt`, `san/`, `san134/`, `race_pairs.txt`, `mc_ab.txt`, `mc134.txt`; `resusage.txt`; `modload_*.txt`;
  `spd9_*` (batch and cold runs), `speed9.log`, `dmon.txt`, `clocks_*.txt`; `docs/` and `docs_speed.txt` (L4);
  `paloff/`, `paloff_speed.txt`, `rgba_palette_count.txt` (L4).
- `scripts/`: listed under Setup; `spd_paloff.py`, `spd_paloff2.py`, `after2.sh` for the palette-off runs.

## AWS

`av1gpu-rc09l4` (g6.4xlarge, on demand) and `av1gpu-rc09bw` (g7.4xlarge, spot), us-east-1, each with its own key
pair and security group (SSH from one address), a 150 GB gp3 root volume deleted on termination, and a dead-man
shutdown (+240 min; the L4 re-armed to +300 min). Both are terminated; `NOTES.md` has the describe calls.
