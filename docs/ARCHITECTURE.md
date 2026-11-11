# How av1gpu works

av1gpu encodes AV1 key frames, intra only, and wraps them in AVIF. The design aims at **throughput on
many images**, not latency on one. A batch of images, typically 48 MP worth, is encoded at once, and
several batches overlap in different pipeline stages.

```
 files ─► ingest ─► analysis ─► partition DP ─► closed-loop recon ─► loop filters ─► entropy ─► AVIF
          (CPU parse,  (open loop,   (per 64×64)    (SB wavefront,      (deblock +       (GPU range
           GPU decode)  every shape)                 exact)              CDEF)            coder)
```

All per-image state stays in GPU memory from the decoded JPEG planes to the finished tile bytes. The
host only schedules work, assembles OBUs and writes files.

## Ingest (`ingest/`)

CPU worker threads read each file and parse its JPEG markers (subsampling, quantization tables, EXIF
orientation, ICC profile). Then each file takes one of two routes:

- **Hardware route.** Structurally clean baseline JPEGs are decoded by nvJPEG on the GPU's hardware
  JPEG engine (NVJPG), when the GPU has one.
- **CPU route.** For everything else (progressive, arithmetic, restart quirks, or no NVJPG engine),
  libjpeg-turbo does the entropy decoding. Its IDCT is replaced by a hook that stores the quantized
  coefficient blocks straight into a pinned staging arena. The GPU then dequantizes and runs the IDCT
  (libjpeg's islow algorithm, bit-exact).

A finalize kernel converts the native sampling to 4:2:0 or monochrome and replicates edges to a
multiple of 64. Images are packed into size-classed batches. The JPEG's YCbCr planes are used directly
(BT.601, full range, as JPEG defines them), so there is no RGB round trip.

## Image operations (`encoder/src/k_imageops.cu`, `imageops.cpp`, `pipeline.cpp`)

Optional stages between the decode and the encoder, on the decoded planes, always in this order:
`--crop`, `--trim`, the tone operators, `--resize` / `--max-size`, `--grayscale`, then the 10-bit
conversion. Without these options none of them runs. `--trim` measures every frame of a batch in one
launch per sample size and waits for them once; the host decides each frame's crop (or skips it as
blank), and the crop and resize run as one transform. The tone operators (per pixel, so they can run
before the transform) and the gray conversion rewrite the planes; at 4:2:0 they read a 4:2:0 JPEG's
chroma upsampled as libjpeg-turbo does, and other chroma as its nearest sample. The rules follow
ImageMagick 7 (README, Options).

## Analysis (`encoder/src/k_analysis*.cuh`, `k_an_*.cu`)

For every block shape from 4×4 to 64×64 at every aligned position, a warp predicts the block from
**source** edges (open loop) and screens the luma intra modes with a SATD-based cost. It covers DC,
directional modes with angle deltas around the best directions, smooth, and Paeth. The best K
candidates get an exact-rate RD evaluation with the real AV1 coefficient cost model, followed by a
transform-type search. The output is a per-shape grid of RD costs.

Images that pass a screen-content detection (`k_sc_count`, libaom's count of few-colour 16×16 blocks) also get
palette candidates (`k_pal.cu`): for each block of 8×8 to 64×64 with at most 64 distinct luma values, palettes of 8
down to 2 colours (the most frequent values, and k-means) with their index-map and residual cost. The closed loop
rebuilds the winning palette from the source. Photos do not pass the detection and skip all of this. Details:
[PALETTE.md](PALETTE.md).

### Staged analysis (opt-in: `--topdown`, `--prune-41`; `an_stage.h`, `k_an_list.cuh`)

Two expert options trade a little compression for analysis speed. Both are off in every preset and change
decisions (not just speed); lossless frames ignore them. With either, the shapes are analysed in groups, and
between groups a DP stage (the partition DP below, over the shapes analysed so far) decides on the GPU which
blocks of the next group are worth analysing. A list kernel (`k_an_list`) collects those blocks per shape, and
the analysis kernels' list-driven instances take only them. Nothing is read back to the host.

- `--topdown 1`: the shapes whose smaller side is 16 or more first; the shapes with a side of 8 only in the 16×16
  cells whose leaf in that DP has a side of 16 (or next to such a cell), then the shapes with a side of 4 likewise
  in 8×8 cells. Cells that the frame edge cuts are always analysed in full.
- `--prune-41 PCT`: a node's 4:1 shapes (16×4 … 64×16) only where its 2:1 partition costs at most PCT % more
  than the best of its other partitions. Only `slow` and `base` have 4:1 shapes (with the defaults: the images
  below `--small-mp`).

A stage leaves a cell unrefined only under a leaf of finite cost, so neither option can leave a superblock
without a finite partition where the full analysis would find one. Off, the default analysis kernels and the DP
are the same code as without the options. On an L4 (development numbers): `--preset base` +30 % throughput with
`--topdown`, +36 % with both, for +0.2 to +0.3 % BD-rate (PSNR-Y, quality 30 to 80); the default preset +5 % / +6 %
for +0.14 %.

## Partition DP (`k_dp.cuh`)

For each 64×64 superblock, a bottom-up dynamic program over the cost grids picks the partition tree.
It considers all ten AV1 partition types, including the 4-way and 3-way rectangular splits.

## Closed loop (`k_recon.cuh`)

Intra prediction depends on reconstructed neighbours. Superblocks are therefore processed in
wavefront order: superblock (r, c) can start once (r−1, c+1) is done. Each wavefront step is one
kernel launch over the ready superblocks of every image in the batch, with one thread block per
superblock. Each thread block:

- re-evaluates the top candidates with the **real** reconstructed edges and exact entropy contexts
- chooses chroma modes, including CfL
- quantizes with quantization matrices and per-superblock delta-q (variance boost, libaom `tune=iq`)
- runs the exact integer inverse transforms of the AV1 spec to produce the reconstruction

The reconstruction is bit-exact with what any conforming decoder produces, and the tests check that.

The wavefront has long serial chains. They are hidden by having many images in flight and by
running several encoders (CUDA streams) at once. The wavefront kernels go on a high-priority stream,
so the next batch's analysis fills the gaps.

## Loop filters (`av1core/src/gpu_postfilter.cu`, `av1f/`)

Deblocking levels and CDEF strengths are chosen on the GPU. With `--tune iq` this reproduces libaom's
all-intra choices: loop-filter level from q, and CDEF strength search with libaom's preset list.
`--preset slow` searches both instead. The filters are then applied to the reconstruction.
AV1's third loop filter, loop restoration, is not implemented: on photos it is worth nothing at equal
SSIMULACRA2 or CVVDP, and libaom turns it off at the speeds av1gpu is compared with
([LOOP_RESTORATION.md](LOOP_RESTORATION.md)).

## Entropy coding (`av1core/src/gpu_entropy.cu`)

The block decisions of a batch are turned into per-tile symbol streams on the GPU. For its own lossy
frames the encoder hands over what the closed loop already worked out: each tx block's end of block, the
coefficient context maps, the 4×4 mode-info maps, and the tx blocks' symbol counts, which `k_pack` adds
up while it copies the levels into the record; the symbolizer does not derive them again (external
records and lossless frames still do; `GP_CHECK_TRUSTED=1` compares the two). Each tile is then
range-coded with AV1's adaptive CDFs: one warp adapts the CDFs and a second warp runs the arithmetic
coder. A single tile is serial, so throughput comes from coding hundreds of tiles of many images at
once. The output is byte-identical to the CPU bitstream writer in `av1core/src/encoder.cpp`.
`--cpu-entropy` uses that writer instead. Finally the host assembles OBUs (sequence header, frame) and
muxes the AVIF: `ispe`, `av1C`, `pixi`, `colr` (ICC and nclx), and `irot`/`imir` from the EXIF
orientation.

## Presets

Presets change only the search effort; every preset produces valid, decoder-exact output.

| Preset | Search |
|---|---|
| `slow` | tx-size search, all tx types for every re-checked candidate, K = 6, filter intra, deblocking/CDEF search |
| `base` | all 19 block shapes, K = 4 candidates (2 re-checked in the closed loop), tx-type search |
| `fast` | no 4:1 shapes, K = 2, no tx search in the analysis |
| `faster` | as fast, plus pruning of flat areas and a single closed-loop candidate |
| `turbo` | as faster, without 32×64/64×32, 2× subsampled SATD for blocks ≥ 16×16, K = 1, 4×4 tiles for images ≥ 2 MP |
| `turbo2` | as turbo, without 4×4/4×8/8×4, no closed-loop tx search, one chroma candidate |
| `turbo3` | as turbo2, without angle deltas |

The default is `turbo` for images of 1 MP and above and `base` for smaller ones (`--small-preset`).
In the original test set, images under 1 MP were 30 % of the files but only 3.5 % of the pixels.
So there the slower preset cost about 7 % throughput and recovered most of turbo's loss on small images.
On the L4's size-range photos (40 of the 100 below 1 MP), v0.6's default runs 14 % slower than
`--small-preset none`.

## Source map

| File | Role |
|---|---|
| `encoder/apps/av1gpu.cpp` | CLI, presets, single-image path |
| `encoder/src/batch.cpp` | batch mode: ingest ↔ encoders ↔ entropy threads |
| `encoder/src/pipeline.cpp` | one GPU batch: allocation, kernel sequence, record packing |
| `encoder/src/imageops.cpp`, `k_imageops.cu` | `--trim`, tone and gray: option parsing, the trim decision, kernels |
| `encoder/src/gp.h` | shared host/device structures and tunables |
| `encoder/src/k_pal.cu`, `pal_dev.cuh` | palette candidates of screen-content images (analysis and closed loop) |
| `av1core/include/av1g/decisions.h` | the DecisionRecord: every coding decision of a frame |
| `av1core/src/encoder.cpp` | CPU bitstream writer and reference reconstruction |
| `av1core/include/av1g/dsp.h` | prediction, transforms, quantization shared by CPU and GPU |
| `av1f/include/av1f/tuneiq.h` | libaom `tune=iq` frame decisions |
