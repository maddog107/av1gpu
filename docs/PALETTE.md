# Palette mode

Before this change av1gpu had neither palette mode nor intra block copy (IntraBC), the two AV1 tools for flat
colours and repeated patterns. That was a scope choice for photos. This note does three things:

- measures with libaom what the two tools are worth on comics, flat-colour art, screen content and photos;
- describes the palette mode av1gpu now has: lossy luma palettes, and lossless luma and chroma palettes, in images
  that pass a screen-content detection;
- lists what the first version leaves out.

Photos do not pass the detection, so their output is unchanged byte for byte.

Measurements: [`bench/results/aws-g6-l4-palette-study/`](../bench/results/aws-g6-l4-palette-study/summary.md)
(NVIDIA L4, 2026-11-02; libaom 3.13.1 through avifenc 1.4.0, av1gpu main 60f94ff and the feat/palette branch).

## What the tools are worth (libaom)

libaom ran at speed 6, all-intra, `tune=iq`, as `bench/bench.py` runs it, at quality 30, 45, 60, 75 and 90, in four
configurations that differ only in the screen-content switches:

- **A:** palette and IntraBC off.
- **B:** libaom's default. Its own detection decides.
- **C:** screen-content tools forced on, IntraBC off. Palette only.
- **D:** screen-content tools forced on, IntraBC allowed.

BD-rate is computed on SSIMULACRA2: the median over images, with the mean in brackets. Each cell is against A, and
negative means smaller files at equal quality. All runs are 8-bit 4:2:0.

| Images | n | C: palette | D: palette + IntraBC | B: libaom's choice | av1gpu main |
|---|---:|---:|---:|---:|---:|
| Comics (scanned public-domain pages, 3–11 MP) | 31 | −0.5 % | +1.3 % | 0.0 % | +1.3 % |
| Flat-colour / anime art, PNG | 24 | −4.4 % (−8.3) | +0.4 % | 0.0 % (−3.9) | +6.7 % |
| The same art as JPEG q90 | 24 | −5.7 % (−8.5) | +1.0 % | 0.0 % | +1.4 % |
| Painted webcomic (Pepper&Carrot) | 6 | −2.5 % | +3.8 % | 0.0 % | +2.0 % |
| Rendered graphics (the 23 of the lossless comparison) | 23 | −14.7 % (−19.6) | −19.6 % (−29.0) | = D | −2.4 % |
| Real screenshots | 12 | −18.3 % | −35.5 % | = D | −2.4 % |
| Web page renders | 5 | −17.8 % | −33.2 % | = D | −0.1 % |
| Photos (Kodak, size range) | 44 | −0.2 % | +2.6 % | 0.0 % | +1.2 % |

- **Palette** is worth 15–18 % on screen content and graphics, and about 5 % on flat-colour art (8 % on average:
  the gain is concentrated in some of the images). On comics, painted art and photos it is worth nothing.
- **IntraBC** adds another 5–17 points on screen content. Everywhere else it loses 1–5 %, because a frame with
  IntraBC allowed has no deblocking and no CDEF.
- **libaom's own choice (B)** turns both tools on for every screen image. It also turns them on for 14 of the 24 flat
  PNGs, 4 of the flat JPEGs, 1 painted page and 2 of the 44 photos (kodim20 and a tiny JPEG), but not for any comic.
- **10-bit 4:4:4** shows the same pattern: on screen content C is −13.5 % and D −24.4 %, on the flat PNGs C is −3.3 %.
  Butteraugli agrees in direction: on screen content C is −14 % and D −19 %.
- **av1gpu against libaom without the tools** is within 2.5 % on every class except the flat PNGs (+6.7 %; +9.8 % on
  butteraugli). The general gap from being tuned on photos is small next to what palette is worth on screen content.

Lossless (`avifenc -l`), total bytes against A in GBR:

| Images | n | C | D | D, YCgCo-R | av1gpu main | av1gpu main, YCgCo-R | cwebp `-lossless -z 9` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flat-colour PNG | 24 | 0.85 | 0.85 | 0.61 | 0.79 | 0.62 | 0.45 |
| Screen and graphics | 40 | 0.66 | 0.53 | 0.39 | 0.71 | 0.56 | 0.16 |
| Kodak | 24 | 1.00 | 1.00 | 0.90 | 0.98 | 0.88 | 0.82 |

Lossless WebP stays smaller than the best AV1 configuration (D in YCgCo-R): 2.4× on screen content, 1.4× on the
flat-colour art. WebP has tools AV1 lacks: back references and a colour cache over the whole image, and predictors
chosen per tile.

## What av1gpu does

### Detection: per image

A palette is only possible in frames with `allow_screen_content_tools = 1`, and with the flag at 0 no palette syntax
is coded at all. So the switch is per image: an image that does not pass detection is encoded exactly as before.

- **libaom's measure.** av1gpu ports `estimate_screen_content_antialiasing_aware` (libaom 3.13.1, the mode 2 that
  `tune=iq` selects) exactly, on 16×16 luma blocks counted at 8 bits:
  - a block with 2–4 colours counts as palette-friendly;
  - a block with 5–40 colours counts too if, after a dilation with its dominant value, it has at most 6 colours and
    its variance is above 5;
  - a block with more than 40 colours counts as photo-like.
  - The image passes when (palette-friendly − photo-like / 16) × 2560 > the picture area.

  On the study's images the port picks exactly the images libaom picks.
- **One addition:** a palette-friendly block must also span at least 8 levels. libaom's rule also fires on photos
  with large smooth areas (sky, studio backgrounds), whose 16×16 blocks hold 2–4 neighbouring levels. It fires on 7
  of the 100 images of the size range, 8 of the 100 of the standard set and 2 of the study's 44 photos. With the
  extra condition none of these 244 passes: the highest score is 0.12, against a threshold of 1.
  - Screen content keeps 38 of 40. The two misses are screenshots with photo wallpapers.
  - The flat PNGs keep 11 of 24.
  - The flat JPEGs (0 of 24) and the painted page drop out: JPEG noise turns their flat areas photo-like.
- **Cost:** one warp per 16×16 block (`k_sc_count`, `encoder/src/k_misc.cu`): 0.09 ms of the 64 ms GPU time of a
  size-range photo (ncu).
  - In lossy batches there is no host sync. The decision goes into the device copy of each image, and the palette
    kernels read it there: in a batch without screen content every CTA of `k_pal` returns at once.
  - The batch's palette slots (12 KB per SB) are allocated in every lossy batch.
  - Lossless batches sync once, because their kernels are compiled with and without palettes.
  - `GP_SC_DEBUG=1` prints each image's counts and both scores.
- **`--palette auto|on|off`:** `auto` (the default) uses the detection, `on` treats every image as screen content,
  `off` never does.

### Lossless search (`k_lossless.cu`)

A block whose coded samples take at most 8 values per plane gets that exact palette and no residual.
- `k_ll_shape` (one warp per shape and block) counts the distinct values with a warp ballot. If the block fits, the
  palette is the sorted values and the map follows directly. Chroma palettes hold (U, V) pairs, for blocks of 8×8
  and up.
- The rate is the palette header and colour bits plus every index at its exact context (`get_palette_color_context`)
  with the default CDFs. One lane per pixel works out its context, and the warp sums the bits. That J competes with
  the block's other candidates in `k_dp`.
- `k_ll_decide` decides again for the winning partition with exact availability and writes the palettes and maps;
  `k_ll_coeffs` predicts from them (all-zero residual).

### Lossy search (`k_pal.cu`, `pal_dev.cuh`)

The search adds one more analysis candidate per block of 8×8 to 64×64 (and 4×16 / 16×4, which the spec allows), in
frames that passed detection:
- **Colours:** the block's histogram. Blocks with more than 64 distinct values are skipped (libaom's limit). For
  each size from 8 down to 2, two palettes are tried, as libaom tries them: the most frequent values exactly ("top
  colours"), and 1-D k-means started from them, rounded and merged.
- **Map and rate:** each pixel's nearest colour. The index bits come at their exact contexts with the default CDFs,
  as in the lossless case, plus the colour bits.
- **Residual:** the source minus the palette goes through the analysis' exact-rate path with `DCT_DCT` and `IDTX`.
  Distortion and lambda are the same as for the intra modes, so a palette wins only where it is cheaper at equal
  distortion. The search stops after two sizes in a row without improvement.
- **The candidate:** the best palette, if cheaper than the block's best mode, becomes candidate 0
  (`PAL_CAND + size − 2`, + 8 for top colours) for the partition DP and the closed loop.

### Closed loop (`k_recon.cuh`)

Palette prediction reads no neighbouring pixels, so it adds no dependency to the SB wavefront.
- The closed loop rebuilds the candidate's palette and map from the source (the same code as the analysis, so the
  same palette) into the SB's colour-map slot.
- It evaluates the palette like any other candidate: transform types, exact rate, and the palette bits in the mode
  cost.
- A palette block keeps one transform block (no tx split). Its neighbours see `DC_PRED` and its reconstruction, as
  for any intra block.
- Chosen palettes go to the SB's palette slots (`GpSbOut::nPal` / `nMap`). `k_pack` gathers them per image.

### Coding

- **DecisionRecord version 5** (`av1core/include/av1g/decisions.h`). `av1g_block` is full, so palettes live in side
  arrays: `palettes[]` (block index, sizes, Y/U/V colours, map offset) and `color_map[]`. Records of version 1–4
  have none.
- **CPU writer, reference decoder and parser** (`av1core/src/encoder.cpp`, one template walk). They handle
  `palette_mode_info` with the palette cache, and the colour coding with cache flags, the literal and delta rules
  and the V delta-or-raw choice (as libaom writes it). They also handle `palette_tokens` in anti-diagonal order,
  palette prediction in `predict_txb`, and the per-4×4 palette maps. The CDF tables come from the spec
  (`tools/gen_tables.py`). `av1g_validate` checks sizes, order and modes.
- **GPU entropy coder** (`av1core/src/gpu_entropy.cu`). Every context of the palette syntax is a function of the
  record, so symbolization stays parallel:
  - `k_maps` adds the per-4×4 palette sizes and palette indices;
  - `k_sbsym` codes the mode info and colours, and reserves the token slots;
  - `k_palsym` (one CUDA block per palette and plane) writes the index tokens into those slots in wavefront order.

  The palette helpers are not inlined into the recursive partition walk, because every level would then carry
  their locals. The walk's deepest path is still 1,424 bytes of stack (ptxas): k_sbsym, 5 levels of 128×128
  partitions, and the palette helpers. That is above CUDA's default 1 KB, so `run_batch_device` raises the limit to
  2 KB once. The lossy encoder already sets more.

### Tests

- **Decoder gate** (`tests/decoder_gate.py`, dav1d and aomdec, reconstruction byte for byte). It ran on the 40 screen
  images and 24 flat PNGs: lossy 4:2:0 at three qindex values, 4:4:4, 10-bit, 10-bit 4:4:4, lossless GBR and
  YCgCo-R. The flat JPEGs and painted pages ran too, with no palette. Every run passed N/N. The images are up to
  12 MP, and 45 of the 64 are over 2 MP and so get 4×4 tiles. Palettes therefore meet SB-row boundaries (where the
  cache drops the above block) and tile boundaries.
- **CPU fuzz** (`tests/fuzz.cpp --palette P`, default 50 % of cases). Random palettes cover every size, colours with
  and without cache hits, and maps with runs and noise. They also cover blocks at an SB's top row, partly off-screen
  maps, and 4×16 / 16×4 blocks (whose 4:2:0 chroma maps need the spec's +2 rule), in lossy and lossless, 4:2:0 and
  4:4:4, 8- and 10-bit, with and without tiles. All of it goes through av1core, dav1d and aomdec, plus the parser
  round trip. 3,000 cases (400 with palettes: 41,421 luma and 20,169 chroma palettes) had no mismatch. ctest's fuzz
  runs (200 lossy and 120 lossless cases) include palettes too, and ctest passes 9 of 9.
- **GPU fuzz test** (`tests/gpu_fuzz_test.cu`): the GPU entropy coder against the CPU writer, on records with
  palettes. 300 cases (seeds 1–300) and 200 lossless cases (seeds 1000–1199) had 0 mismatches. This test found the
  stack overflow above.
- **Byte identity against main.** No photo passes detection, so photos are unchanged. Every file was compared with
  `cmp`:
  - the standard set (100) and the size range (100), default and `--yuv 444 --depth 10`;
  - Kodak `--lossless` and `--lossless --ycgco-re`;
  - the study's comics, flat JPEGs, painted pages and photos at five quality points;
  - `--palette off` on the 64 screen and flat PNGs (lossy, 10-bit 4:4:4, lossless GBR and YCgCo-R).

## Results

BD-rate on SSIMULACRA2 of the feat/palette av1gpu, median over images (mean in brackets). The default settings,
quality 30–90, 8-bit 4:2:0 unless the column says otherwise:

| Images | n | vs av1gpu main | vs libaom A | vs libaom C | 10-bit 4:4:4 vs main |
|---|---:|---:|---:|---:|---:|
| Screen: all | 40 | −23.6 % (−24.0) | −24.9 % (−26.0) | −9.5 % (−8.7) | −19.5 % (−19.5) |
| Screen: 23 graphics | 23 | −23.3 % (−25.3) | −25.1 % (−27.7) | −10.3 % (−10.1) | −18.5 % (−20.6) |
| Screen: 12 screenshots | 12 | −24.7 % (−21.7) | −23.9 % (−22.9) | −6.3 % (−5.8) | −18.7 % (−16.6) |
| Screen: 5 web renders | 5 | −24.6 % (−23.5) | −27.2 % (−25.3) | −8.4 % (−9.0) | −21.9 % (−21.3) |
| Flat-colour PNG: all | 24 | 0.0 % (−7.4) | +3.8 % (−1.2) | +5.6 % (+7.2) | 0.0 % (−4.7) |
| Flat-colour PNG: the 11 that pass detection | 11 | −22.3 % (−16.2) | −12.5 % (−10.0) | +0.7 % (+0.9) | −12.7 % (−10.3) |
| Comics, flat JPEG, painted, photos | 105 | identical files | | | identical files |

- **Screen content:** about −24 % at equal SSIMULACRA2 (butteraugli −23 %). That is ahead of libaom's palette
  search (C) by 9.5 %. Against libaom's palette + IntraBC (D) it is +3.3 % (median; +12.8 % on the real
  screenshots, where IntraBC counts most, −3.9 % on the graphics).
- **Flat-colour art:** where detection passes, about the same as libaom C. Elsewhere unchanged.
- **`--palette on`** skips detection, so every image gets the search. The flat PNGs then come out −6.8 % (−11.4)
  against main, and the flat JPEGs −5.9 % (−8.9); detection passes none of the JPEGs. Both are close to libaom C
  (+2.7 %, +1.0 %). In 10-bit 4:4:4 both are −3 %. The cost is the search's speed on every image (below).
- **The first version** used k-means palettes of 8 colours only, and gained just 3 % on screen content. Trying the
  exact top colours, and every size from 8 down, made the difference, as in libaom.
- **Lossless:** −8.7 % on the screen set, −7.8 % with `--ycgco-re`, and −3.7 % on the flat PNGs, against main.
  That puts av1gpu's GBR lossless (20.5 MB on the screen set) below libaom's palette configuration C (21.0 MB).

### Speed

Measured with the method of `bench/results/aws-g6-l4-speed-v06/scripts/spd.py` (each set repeated to at least
1,000 MP, a warm-up, then 3 timed runs, in 2 rounds with alternating order; the median of the 6), on the L4. MP/s:

| Images | main | feat/palette | change |
|---|---:|---:|---:|
| size range, default | 229.3 | 226.3 | −1.3 % |
| size range, `--yuv 444 --depth 10` | 162.2 | 163.8 | +0.9 % |
| standard set, default | 74.4 | 75.0 | +0.8 % |
| main against main (noise) | | | +0.2 % (+0.9 % earlier the same day) |
| size range, `--palette off` | 227.5 | 225.3 | −1.0 % |
| screen + flat PNG (64 images), default | 152.7 | 57.0 | 0.37× |
| the same, `--palette off` | 155.9 | 150.8 | −3.3 % |
| the same, `--palette on` | 150.6 | 37.2 | 0.25× |
| the same, `--lossless` (`GP_ENCODE_MP=12`) | 51.6 | 49.3 | −4.5 % |
| the same, `--lossless --ycgco-re` | 50.4 | 46.9 | −6.9 % |

- **Photos:** within noise. The detection pass is 0.09 ms of a 64 ms photo encode (ncu). The first build was 2–3 %
  slower, for two reasons: a thread-per-block detection kernel (1.03 ms of the 64 ms), and a palette branch inside
  the closed loop's per-pixel predictor. That kernel runs at its 128-register cap. Both are fixed; `--palette off`
  measures what is left of the code itself.
- **Screen content is 2.7× slower** (`--palette on`: 4×), in two places:
  - the palette search: the analysis stage grows from 3.2 to 20.9 s of GPU time per 1,400 MP, for up to 14
    palettes per block and shape, each with its exact-rate transform;
  - the entropy coder: a palette block codes one adaptive symbol per pixel. Its GPU time grows from 3.1 to 17.2 s,
    on its own stream, so it is not the bottleneck.

  A cheaper search was tried and is not in the branch: k-means only at the largest size, and a stop at the first
  size without improvement (`scripts/chain6.sh`). It ran at 102.5 MP/s on the same images (0.67×, against main's
  153.0), but saved only 9.8 % on screen content instead of 23.6 %: the smaller palettes carry much of the gain.
  Screening the palettes with a SATD estimate before the exact rate, as the analysis does for the intra modes, is
  the next thing to try.

## What the first version leaves out

- **Chroma palettes in lossy mode.** Lossy frames get luma palettes only; lossless frames get both. In 4:2:0 libaom
  C puts chroma palettes on about 1 % of the area of screen content. The smaller gain in 4:4:4 (−19.5 % against
  −23.6 %) is probably where they would count.
- **Lossless palettes with a residual.** A lossless block gets a palette only if it has at most 8 values per plane.
  Blocks with more (anti-aliased text) could use a palette plus a lossless residual, as lossy blocks do.
- **The colour cache in the search.** Colours come from each block's own pixels. The writer codes the cache flags
  for every colour that matches a neighbour's exactly. libaom also moves k-means colours within one level of a
  cached colour onto it.
- **Rate estimates:** the index map is costed with the default CDFs. The adapted CDFs code long runs more cheaply,
  so palettes are costed somewhat too high.
- **No transform split** in palette blocks: one transform block of the block's size.
- **Detection** passes no JPEG of the flat-colour art and no painted page: JPEG noise makes their flat areas
  photo-like. libaom C still gains 5.7 % on the flat JPEGs when palette is forced. `--palette on` forces it in
  av1gpu too, at the cost of the search time on every image.
- **IntraBC** (next section).

## IntraBC

IntraBC is the bigger tool on screenshots: libaom's D beats C by another 10–17 points there. It is a separate
project:
- **Search:** IntraBC needs a block-matching search over the already-coded part of the frame (libaom hashes the
  source). On the GPU that is a global hash table per frame, built before the wavefront.
- **Dependencies:** the reference must be a reconstructed area at least one SB row (or 256 pixels) behind the
  wavefront, under the spec's delay constraints. The wavefront order (step = column + 2 × row) gives that only to
  references well above and to the left.
- **Loop filters:** a frame with `allow_intrabc` has no deblocking, CDEF or loop restoration. That costs 1–5 % on
  everything that is not screen content (D against C above), so IntraBC needs a detection of its own that is
  stricter than palette's. libaom uses a stricter variant of the same counts.
- **Coding:** the MV coding (`dv`), and IntraBC's own rules for the transform and filter syntax in the record,
  CPU writer and GPU entropy coder.
