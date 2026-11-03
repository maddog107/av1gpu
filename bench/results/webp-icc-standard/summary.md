# Standard set: WebP re-scored with its ICC profiles (2026-11-03)

The L4 run (`../aws-g6-l4/`) scored WebP before `bench.py` put each .webp's ICC profile back into dwebp's PNG, so
photos with non-sRGB profiles were scored as colour-shifted. The size range was re-scored in `../aws-g6-l4-cvvdp/`;
the standard set was not. pickae's v0.9.0 run on an RTX 5090
([discussion #19](https://github.com/maddog107/av1gpu/discussions/19)) gave WebP a mean of +43.1 % against the
published +55.3 % and suggested the profile fix as the cause. This re-score checks it.

## What was run

`scripts/webp_icc_std.py` on a Mac (CPU only): the 100 standard-set JPEGs (`bench/fetch_images.py`), each encoded
with `cwebp -q Q -m 4 -metadata icc` at the run's six qualities (30–80), as `bench.py` does. Each file is scored
with `ssimulacra2` twice: from dwebp's PNG as it is (as the L4 run did) and with the .webp's profile put back
(`bench.score`). The BD-rates use the L4 run's libaom speed 6 and av1gpu rows unchanged.

- cwebp / dwebp 1.5.0 (the L4 run: 1.3.2), Pillow 11.3.0. **All 600 .webp files were byte-identical to the L4
  run's**, and all 600 scores without the profile equalled the L4 run's.
- `results.json`: every file's bytes and both scores.

## Result

BD-rate on SSIMULACRA2, median (mean), 100 images:

| | as published | with the WebP profiles |
|---|---:|---:|
| WebP vs libaom speed 6 | +41.3 % (+55.3 %) | **+40.4 % (+44.3 %)** |
| av1gpu default vs WebP | −30.0 % (−32.9 %) | **−29.7 % (−30.5 %)** |
| libaom speed 6 vs WebP | −29.2 % (−32.5 %) | −28.8 % (−29.9 %) |
| libaom speed 9 vs WebP | −17.6 % (−19.5 %) | −16.7 % (−16.4 %) |

The rows against libaom speed 6 without WebP (av1gpu +0.3 % (−0.1 %), libaom speed 9 +17.7 % (+20.7 %)) do not
change. The av1gpu-vs-WebP row equals pickae's RTX 5090 result to the digit. That comparison uses only av1gpu and
WebP files, which are byte-identical on both machines, so it does not depend on their older avifenc.

The change comes from 12 photos (a 13th, with a compact sRGB profile, moved by 0.1 point):

| Image | JPEG profile | WebP vs libaom s6, as published | with the profile |
|---|---|---:|---:|
| `c83f5f9e96640769.jpg` | Adobe RGB (1998) | +396.7 % | +42.9 % |
| `d5b340573162192c.jpg` | Generic RGB Profile | +235.3 % | +38.9 % |
| `bd18b0390f876337.jpg` | Adobe RGB (1998) | +194.9 % | +57.4 % |
| `0981622cc4d4ac2b.jpg` | Adobe RGB (1998) | +132.2 % | +60.8 % |
| `9b5c3985f2ae29bb.jpg` | Adobe RGB (1998) | +121.6 % | +59.2 % |
| `d05da312b9e2fe39.jpg` | Camera RGB Profile | +120.0 % | +56.3 % |
| `b2dc8e2437e8803f.jpg` | Camera RGB Profile | +105.3 % | +45.7 % |
| `c4c2bb0181ff7fb9.jpg` | Adobe RGB (1998) | +92.3 % | +40.1 % |
| `bfbf87f0676be455.jpg` | Adobe RGB (1998) | +83.2 % | +47.0 % |
| `711ab87f57d2b5c3.jpg` | Camera RGB Profile | +80.9 % | +34.0 % |
| `3fa5da93895d2de8.jpg` | Camera RGB Profile | +56.4 % | +44.5 % |
| `ec024f284026b083.jpg` | Display | +44.3 % | +35.2 % |

## Also corrected

The tiling table in RESULTS.md (CPU builds: tuning, threads and tiling) still had the size range's WebP row from
before the profile fix, +45.1 % / +46.0 %. With the profiles (`../aws-g6-l4-cvvdp/` scores) it is +42.1 % against the
published, tiled libaom speed 6 and +43.5 % against one tile per image (`../aws-c6a-cpu-tuning/`,
`stock_aom_s6_notiles`).
