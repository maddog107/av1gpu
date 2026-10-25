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

## Analysis (`encoder/src/k_analysis*.cuh`, `k_an_*.cu`)

For every block shape from 4×4 to 64×64 at every aligned position, a warp predicts the block from
**source** edges (open loop) and screens the luma intra modes with a SATD-based cost. It covers DC,
directional modes with angle deltas around the best directions, smooth, and Paeth. The best K
candidates get an exact-rate RD evaluation with the real AV1 coefficient cost model, followed by a
transform-type search. The output is a per-shape grid of RD costs.

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

## Entropy coding (`av1core/src/gpu_entropy.cu`)

The block decisions of a batch are turned into per-tile symbol streams on the GPU. Each tile is then
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
| `turbo` | as faster, without 32×64/64×32, 2× subsampled SATD for blocks ≥ 16×16, K = 1, 2×2 tiles for images ≥ 2 MP |
| `turbo2` | as turbo, without 4×4/4×8/8×4, no closed-loop tx search, one chroma candidate |
| `turbo3` | as turbo2, without angle deltas, 4×2 tiles |

The default is `turbo` for images of 1 MP and above and `base` for smaller ones (`--small-preset`).
In the original test set, images under 1 MP were 30 % of the files but only 3.5 % of the pixels.
The slower preset therefore costs about 7 % throughput and recovers most of turbo's loss on small
images.

## Source map

| File | Role |
|---|---|
| `encoder/apps/av1gpu.cpp` | CLI, presets, single-image path |
| `encoder/src/batch.cpp` | batch mode: ingest ↔ encoders ↔ entropy threads |
| `encoder/src/pipeline.cpp` | one GPU batch: allocation, kernel sequence, record packing |
| `encoder/src/gp.h` | shared host/device structures and tunables |
| `av1core/include/av1g/decisions.h` | the DecisionRecord: every coding decision of a frame |
| `av1core/src/encoder.cpp` | CPU bitstream writer and reference reconstruction |
| `av1core/include/av1g/dsp.h` | prediction, transforms, quantization shared by CPU and GPU |
| `av1f/include/av1f/tuneiq.h` | libaom `tune=iq` frame decisions |
