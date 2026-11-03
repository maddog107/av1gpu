# Loop restoration

av1gpu has no AV1 loop restoration (LR): the Wiener and self-guided filters that run after CDEF. This note
answers whether to build it, in four parts:

- what libaom does with LR when it encodes stills;
- what LR would save on photos at equal quality, measured with libaom;
- what it would cost: encoder time, decoder time, and a design sketch with a GPU estimate for av1gpu;
- the recommendation.

**Short answer: don't build it.** av1gpu reports two perceptual metrics, SSIMULACRA2 (which it is tuned on) and
CVVDP. On both, LR saves 0.0 % on the median photo, and 0.3–1.1 % on photos of 4 MP and up. Butteraugli and
PSNR see more: 1–1.6 % overall, and 2–3.6 % on the largest photos. Against that, LR would cost an estimated 3–6 %
of GPU time, 4–7 % more decode time, and a sizeable amount of bitstream, entropy-coder and test work.

Measurements: [`bench/results/aws-c7a-lr-study/`](../bench/results/aws-c7a-lr-study/summary.md) (AWS
c7a.16xlarge, 2026-11-03; libaom 3.13.1 through avifenc 1.4.0; SSIMULACRA2 and butteraugli from libjxl 0.11.1 and
CVVDP 0.5.7, all on the dav1d 1.4.1 decode).

## What loop restoration is

LR is AV1's third in-loop filter, after deblocking and CDEF.
- **Restoration units.** The frame is split per plane into units of 64×64, 128×128 or 256×256 (chroma can be
  halved). The frame header picks a type per plane: none, Wiener, self-guided, or switchable per unit.
- **Wiener:** a separable symmetric filter, 7 taps for luma and 5 for chroma, whose coefficients are chosen per unit
  by a least-squares fit to the source.
- **Self-guided:** two box filters (radius 1 and 2) of a guided-filter type, with one of 16 preset parameter sets.
  It is followed by a projection with two weights per unit, also fitted by least squares.
- **Signalling.** The parameters are coded in the tile data, at the start of the 64×64 superblock that contains
  each unit's top-left corner. They are coded relative to the previous unit's parameters.
- **Stripes.** The filters run in 64-row stripes. At stripe boundaries they read the deblocked rows from before
  CDEF.

Both filters minimise squared error against the source: LR is a PSNR tool.

## What libaom does with stills

From libaom 3.13.1 (`av1/encoder/speed_features.c`, `av1/av1_cx_iface.c`):
- `enable_restoration` defaults to 1 in all-intra mode, as in good-quality mode. Only realtime mode forces 0.
- **All-intra speed 5 and up turns both filters off** (`disable_wiener_filter = disable_sgr_filter = true`). The
  encoder then clears `enable_restoration` in the sequence header. So at speed 6, the reference av1gpu is measured
  against, there is no LR, and `-a enable-restoration=1` changes nothing. All 1,120 photo files (224 photos × 5
  qualities) were byte-identical with the flag at 0 and 1.
- **Speeds 1–4 search LR, with pruning:**
  - speed 1: a 1 % rate penalty on the self-guided sets that use both box filters, and a binary-search-like
    pruning of the self-guided parameter search;
  - speed 2: Wiener skipped on units of low source variance, and self-guided skipped where Wiener did not help;
  - speed 3: stronger versions of both, and a 5×5 luma Wiener window instead of 7×7.

  The unit size is not searched at any speed above 0. It is 128 at qindex ≤ 96 below 1440p, and 256 otherwise.
  Key frames are exempt from the speed-4 rule that turns luma LR off at low and high qindex: at q30 (qindex 196),
  speed 4 still uses luma LR in 211 of 224 photos.
- **Multi-threaded encodes** (`-j` > 1) also turn LR off from speed 5 up, in every mode. av1gpu's reference runs
  `-j 1`, so this does not matter here.

So libaom's own choice for stills at speed 5 and above is no LR. av1gpu matches libaom speed 6 (+0.1 % median
BD-rate), so it is not behind libaom on LR. LR would have to be a gain on top of speed 6.

## What it is worth

libaom ran as `bench/bench.py` runs it: all-intra, `-a tune=iq`, `-j 1`, one avifenc per vCPU, at quality 30, 45,
60, 75 and 90. q60 is qindex 120, av1gpu's default. To get LR at speed 6, libaom was patched with an environment
switch (`scripts/patch_lrstudy.py`; with the switch unset the files are byte-identical to stock libaom).
Configurations:
- **s6:** stock speed 6, no LR, the reference.
- **s6 + LR:** speed 6 with LR searched as at speed 4 (the speed-5 rule undone).
- **s6 + full LR:** speed 6 with every LR pruning off and the speed-0 unit-size search. This is the ceiling of
  libaom's LR search.
- **Wiener only** and **self-guided only:** the full search with one filter disabled.
- **s4 / s4 + LR:** stock speed 4 with `enable-restoration` 0 and 1. Speed 4 is libaom's fastest all-intra speed
  that uses LR.

The photo sets are the 100 standard JPEGs, the 100 size-range JPEGs (20 per size class) and the 24 Kodak PNGs.

BD-rate on SSIMULACRA2 per image, median over images (mean in brackets); negative means smaller files at equal
quality. 8-bit 4:2:0:

| Images | n | s6 + LR | s6 + full LR | Wiener only | self-guided only | s4 + LR vs s4 |
|---|---:|---:|---:|---:|---:|---:|
| All photos | 224 | +0.1 % (−0.0) | −0.0 % (−0.1) | −0.0 % (−0.1) | +0.3 % (+0.2) | +0.0 % (+0.0) |
| Standard JPEGs (~1 MP) | 100 | +0.1 % (+0.0) | −0.0 % (−0.0) | −0.0 % (−0.1) | +0.3 % (+0.3) | +0.1 % (+0.1) |
| Kodak (0.4 MP) | 24 | +0.3 % (+0.3) | +0.3 % (+0.3) | +0.1 % (+0.2) | +0.6 % (+0.7) | +0.1 % (+0.1) |
| Size range: all | 100 | −0.1 % (−0.1) | −0.2 % (−0.3) | −0.1 % (−0.2) | +0.1 % (−0.0) | −0.0 % (−0.0) |
| Size range: tiny (< 0.3 MP) | 20 | +0.2 % (+0.2) | +0.2 % (+0.2) | −0.0 % (+0.1) | +0.4 % (+0.4) | +0.1 % (+0.1) |
| Size range: small (0.3–1 MP) | 20 | +0.1 % (+0.1) | +0.1 % (+0.0) | +0.0 % (−0.0) | +0.3 % (+0.2) | +0.1 % (+0.1) |
| Size range: typical (1–4 MP) | 20 | −0.1 % (−0.1) | −0.3 % (−0.2) | −0.1 % (−0.2) | +0.2 % (+0.2) | −0.1 % (−0.0) |
| Size range: large (4–9.4 MP) | 20 | −0.3 % (−0.4) | −0.6 % (−0.8) | −0.4 % (−0.6) | −0.3 % (−0.5) | −0.1 % (−0.2) |
| Size range: huge (9.4–35.6 MP) | 20 | −0.3 % (−0.4) | −0.5 % (−0.7) | −0.4 % (−0.5) | −0.3 % (−0.4) | −0.1 % (−0.2) |

- **Nothing on SSIMULACRA2.** With LR as libaom searches it, the median is +0.1 %: slightly worse. Even the full
  search gains nothing in the median. Only photos of 4 MP and up gain, about 0.3–0.6 %. Thumbnails and Kodak get
  slightly worse.
- **Per image** (s6 + full LR): 16 of 224 photos gain more than 1 %, 3 more than 2 %, and the best gains 4.1 % (a
  huge size-range photo). 27 photos lose more than 0.5 %.
- **libaom does use the filters.** In s6 + LR, 1,030 of the 1,120 files turn LR on in some plane (luma in 953).
  The units are 20,190 Wiener, 7,883 self-guided and 18,693 off (luma). The filters are on; SSIMULACRA2 just does
  not reward what they do.
- **Speed 4 shows the same:** LR on or off at speed 4 is +0.0 %. The 5.9 % that speed 4 gains over speed 6 (median,
  all photos) comes entirely from its other tools.
- **10-bit and 4:4:4.** 4:4:4 (8-bit, standard set and Kodak, 124 photos) shows the same: s6 + LR +0.0 %,
  s6 + full LR −0.0 % on SSIMULACRA2, against −1.8 % and −2.3 % on PSNR. 10-bit was not measured.

The same files on four metrics: BD-rate, median over images (mean in brackets), against s6 (the last column
against s4):

| Images | Metric | s6 + LR | s6 + full LR | s4 + LR vs s4 |
|---|---|---:|---:|---:|
| All photos (224) | SSIMULACRA2 | +0.1 % (−0.0) | −0.0 % (−0.1) | +0.0 % (+0.0) |
| | CVVDP | −0.0 % (−0.1) | −0.0 % (−0.2) | +0.1 % (+0.1)¹ |
| | butteraugli | −0.9 % (−1.3) | −1.6 % (−1.9) | −0.3 % (−0.4) |
| | PSNR | −1.0 % (−1.2) | −1.3 % (−1.6) | −0.7 % (−0.8) |
| Photos under 1 MP (164) | SSIMULACRA2 | +0.1 to +0.3 % | 0.0 to +0.3 % | +0.1 % |
| | CVVDP | −0.1 to +0.2 % | −0.1 to +0.3 % | 0.0 to +0.1 % |
| | butteraugli | −0.5 to −0.9 % | −1.0 to −1.6 % | +0.3 to −0.4 % |
| Size range, large (4–9.4 MP) | SSIMULACRA2 | −0.3 % (−0.4) | −0.6 % (−0.8) | −0.1 % (−0.2) |
| | CVVDP | −0.8 % (−0.8) | −1.1 % (−1.1) | |
| | butteraugli | −2.7 % (−2.7) | −3.6 % (−3.8) | −1.2 % (−1.4) |
| | PSNR | −1.5 % (−1.9) | −2.3 % (−2.3) | −1.1 % (−1.2) |
| Size range, huge (9.4–35.6 MP) | SSIMULACRA2 | −0.3 % (−0.4) | −0.5 % (−0.7) | −0.1 % (−0.2) |
| | CVVDP | −0.4 % (−0.6) | −0.6 % (−0.8) | |
| | butteraugli | −2.1 % (−2.7) | −3.0 % (−3.7) | −1.1 % (−1.2) |
| | PSNR | −2.3 % (−2.2) | −2.9 % (−2.8) | −1.7 % (−1.6) |

¹ CVVDP at speed 4 covers the 184 photos under 5 MP; for lack of time it was not run on speed 4's large and huge
photos. Under 1 MP means the standard set, Kodak, and the tiny and small classes; each range runs over those four
groups' medians.

- **The metrics split in two.** LR's filters minimise squared error, so PSNR gains, as designed: −1.0 % with s6 +
  LR, and half the photos gain more than 1 %.
  - Butteraugli, which weighs the largest local errors, also gains: about 1–1.6 % overall and 2–3.6 % on photos of
    4 MP and up.
  - SSIMULACRA2 and CVVDP see nothing on the median photo. They are the two metrics the README reports, and
    av1gpu is tuned on SSIMULACRA2. They see 0.3–1.1 % on photos of 4 MP and up.
- **Below 1 MP, all the perceptual metrics are flat or slightly worse,** except butteraugli. That covers thumbnails
  (one with every upload) and small photos.
- **libaom's own default LR at speed 4 is worth even less** (−0.3 % butteraugli, +0.0 % SSIMULACRA2). Its LR search
  is the same as s6 + LR's, so the difference is the rest of speed 4: its better reconstruction probably leaves less
  for LR to fix.

At the default operating point (q60), LR makes the files 0.06 % (s6 + LR) to 0.10 % (full) larger. Mean
SSIMULACRA2 rises by 0.02 and 0.07 points, from 76.02. At equal quality the size changes by −0.01 % and −0.17 %
(median over photos), and by −0.6 % and −1.1 % on the huge photos.

What this means for av1gpu's +0.1 % against libaom speed 6:
- LR searched as libaom searches it would change nothing.
- The full search would move it by about 0.1–0.2 % overall, and by about 0.5 % on photos of 4 MP and up.

This assumes LR gains as much on av1gpu's reconstruction as on libaom's, which was not measured.

## What it costs

### libaom encode time (CPU proxy)

CPU seconds (user + system) of all 5 quality points × 224 photos, 4:2:0:

| Configuration | CPU s | vs its reference |
|---|---:|---:|
| s6 | 1,006 | |
| s6 + LR (speed-4 search) | 1,166 | +16 % |
| s6 + full LR | 2,023 | +101 % |
| s6 + Wiener only (full search) | 1,324 | +32 % |
| s6 + self-guided only (full search) | 1,691 | +68 % |
| s4 | 5,656 | (5.6× s6) |
| s4 + LR | 5,806 | +2.6 % of s4 |
| s4 + full LR | 6,657 | +18 % of s4 |

- **Fixed cost.** The LR search costs about the same CPU time at speed 4 as at speed 6: 150–160 s here, about
  0.055 s per MP encoded. That is 16 % of a speed-6 encode, and 3 % of a speed-4 encode.
- **Grows with size.** On the huge photos it is +18 % (full search: +116 %). On tiny photos it is +10 %.
- **The full search costs as much as the encode itself.** The self-guided parameter search is the expensive part.

### Decoder time

dav1d 1.4.1, one thread, best of 3 runs on the q60 files:

| Images | s6 | s6 + LR | s6 + full LR |
|---|---:|---:|---:|
| Size range (478 MP) | 3.95 s | +4.2 % | +7.2 % |
| Standard JPEGs (79 MP) | 0.89 s | +3.6 % | +5.4 % |

Every viewer of every image would pay this.

## How it would fit into av1gpu (design sketch)

LR would be a new stage between CDEF and entropy coding. It would sit next to `av1f_lf_search_async_pf` and
`av1g_gpu_cdef` in `encoder/src/pipeline.cpp`. Everything it needs is already on the GPU at that point:
- the source (`dsrc`);
- the deblocked frame (`I.rec`, which CDEF does not overwrite);
- the CDEF output.

- **The CDEF apply comes back.** LR filters the CDEF output, and its search needs that output as input. In batch mode
  the CDEF apply now only feeds `GP_WANT_RECON` downloads, so it may soon be skipped when nothing reads it. With LR
  it has to run for every frame with CDEF on again, so its cost counts toward LR.
- **Stripes and boundaries.** Above and below each 64-row stripe boundary, LR reads deblocked rows from before CDEF.
  libaom saves these lines before running CDEF. av1gpu can read them straight from `I.rec`, because CDEF writes to
  a separate buffer.
- **Statistics (most of the work).** One CTA per 64×64 block, summed into the restoration units (256×256, or
  128×128 at qindex ≤ 96, libaom's all-intra sizes).
  - *Wiener:* per plane, the cross-correlation M of the source with the windowed CDEF output (49 values for the 7×7
    luma window, 25 for 5×5 chroma), and the autocorrelation H (49×49, 1,225 distinct values). That is about
    1,225 multiply-adds per luma pixel and 325 per chroma pixel. H is a Gram matrix (Xᵀ X, where X is the
    pixels × 49 matrix of windows). So it fits int8 dp4a or the int8 tensor cores (u8 × u8 → s32), with the mean
    subtracted afterwards in 64-bit.
  - *Self-guided:* the radius-1 and radius-2 box sums of the CDEF output, from prefix sums in shared memory. Then,
    per parameter set (16, or libaom's pruned subset), the two filtered planes and the 2×2 normal equations of the
    projection: 5 sums per set and unit. That is about 140 integer operations per pixel and set: about 2,200 per
    luma pixel for all 16 sets, and 3,000 with chroma.
- **Solve and decide:** one warp per restoration unit.
  - *Wiener:* libaom's `wiener_decompose_sep_sym` (5 alternating least-squares passes for the separable symmetric
    filter), then quantisation to the spec's tap ranges. libaom's coefficient refinement re-filters the unit at
    every step; it would be left out.
  - *Self-guided:* per set, the 2×2 solve, and the two quantised, clamped `xqd`.
  - *Error:* each candidate's squared error comes from the statistics in closed form (e = Σs² − 2hᵀM + hᵀHh).
    libaom filters each candidate to get the exact error; this needs no trial filtering.
  - *Rate:* coefficients are coded relative to the previous unit's in the same tile and plane, and the type uses
    adaptive CDFs. So the RD choice is a short serial chain per tile and plane, a few to a few dozen units long. It
    runs once per frame-level type (none, Wiener, self-guided, switchable), and the sums pick the type.
- **Apply.** A decoder-exact LR kernel: Wiener with the spec's intermediate rounding, self-guided from the box
  sums, per stripe with the saved boundary rows.
  - The bitstream does not need it.
  - `GP_WANT_RECON` downloads, the tests and the reference reconstruction do.
  - Its cost would be about that of CDEF.
- **Bitstream and records.**
  - *Sequence header:* `enable_restoration` = 1 only for images that use LR. Each AVIF has its own sequence header,
    so images without LR, and everything with LR off, stay byte-identical.
  - *Frame header:* `lr_params`: the type per plane, `lr_unit_shift` and `lr_uv_shift`.
  - *DecisionRecord version 6:* per-plane unit-parameter arrays, as side arrays like the palettes.
  - *CPU writer, reference decoder and parser* (`av1core/src/encoder.cpp`): `read_lr` at the start of each 64×64
    superblock, before the partition. For each plane it covers the units whose top-left corner falls in that
    superblock: the `use_wiener` / `use_sgrproj` / `restoration_type` symbol, then the taps, or the set index and
    the `xqd`.
  - *GPU entropy coder* (`av1core/src/gpu_entropy.cu`): `k_sbsym` codes the same symbols before its partition walk.
    Every value and context is a function of the record (the reference is just the previous unit in coding order),
    so symbolization stays parallel. At a few dozen bits per 256×256 unit, the entropy time is negligible. The code
    must stay outside the recursive walk, whose stack is already at 1,424 bytes (see [PALETTE.md](PALETTE.md)).
- **Tests:**
  - the decoder gate (dav1d's and aomdec's reconstructions must equal av1gpu's);
  - CPU fuzz with random LR parameters: every type, unit sizes 64–256, units at tile and frame edges, 4:2:0, 4:4:4,
    mono, 8 and 10 bits;
  - `gpu_fuzz_test` for the entropy side;
  - byte identity against main with LR off.

### GPU time (estimate, not measured)

For reference, av1gpu v0.6 on the L4 encodes the size range (477.7 MP) with stage times, summed over the
concurrent encoders, of:

| Stage | ms |
|---|---:|
| analysis | 1,434 |
| reconstruction | 3,435 |
| post-filter (deblocking and CDEF, search and apply) | 380 |
| GPU total | 5,400 |
| entropy (own stream) | 1,507 |

Source: `bench/results/aws-g6-l4-speed-v06/results.json`. The post-filter is 0.8 ms per MP.

Rough LR costs on the same set:
- **Wiener statistics:** about 1,400 multiply-adds per luma pixel with chroma, so 0.67 T for 478 MP. That is about
  25 ms at the L4's dp4a peak (about 25 T MAC/s at its power-capped clock). At a realistic 25–50 % of peak it is
  50–100 ms. Tensor cores would cut the arithmetic further, but forming the windows (im2col in shared memory) and
  the memory traffic would then dominate.
- **Self-guided search:** about 140 operations per pixel and parameter set: a and b for each radius, the weighted
  neighbour sums, the projection sums. Over all 16 sets, with chroma, that is about 3,000 per luma pixel, 1.4 T
  operations in all. That is 120–230 ms at the L4's integer and FP32 peak, and 250–450 ms realistic. A pruned
  subset of 4–6 sets, as libaom uses, takes about a third of that.
- **Solve and RD chain:** a few ms (about 7,300 luma units of 256×256 on this set).
- **The CDEF apply,** if it has been dropped from batch mode by then: part of the 380 ms.

Against the 5,400 ms of GPU time, that is about:
- **1–3 %** for Wiener only;
- **3–6 %** for libaom-style pruning (Wiener plus a pruned self-guided search);
- **6–10 %** for the full search.

Every image would pay that, for a 0.0 % median gain. In relative terms it is the same order of cost as on the
CPU: +16 % (pruned) to +101 % (full) of a libaom speed-6 encode.

## Recommendation

**Don't build it**, neither as the default nor, for now, as an opt-in.
- **No gain on the metrics av1gpu is tuned and reported on.** At equal SSIMULACRA2 and at equal CVVDP it is worth
  0.0 % on the median photo. Thumbnails (one with every upload) and photos under 1 MP get slightly worse.
- **The only consistent gain is on photos of 4 MP and up.** There all four metrics agree, but on SSIMULACRA2 and
  CVVDP it is 0.3–1.1 %, mostly with the full (most expensive) search.
- **Only butteraugli and PSNR see 1–1.6 % overall,** and they are not what av1gpu is tuned for.
- **libaom agrees:** its own all-intra speed 5 and 6 turn LR off, and at speed 4 LR is worth +0.0 % on
  SSIMULACRA2 here.
- **Real costs:**
  - an estimated 3–6 % of GPU time per image (1–3 % for Wiener only, 6–10 % for the full search);
  - 4–7 % more decode time for every viewer;
  - a large implementation: DecisionRecord v6, the CPU writer, parser and reference decoder, the GPU entropy coder,
    a decoder-exact filter, a statistics and solve stage, and new fuzz and decoder-gate coverage.

For an encoder that puts speed first, a few per cent of throughput for 0 % on the median photo is a bad trade.

If it is ever wanted, the shape that fits the numbers is an opt-in for photos of 4 MP and up, for example in
`--preset slow`:
- Wiener plus a pruned self-guided search, at an estimated 3–6 % of GPU time;
- worth about 0.3–0.8 % on SSIMULACRA2 and CVVDP, and 2–3 % on butteraugli.

For compression on SSIMULACRA2, the 5.9 % between libaom speeds 6 and 4 comes from tools other than LR, so that is
where to look.
