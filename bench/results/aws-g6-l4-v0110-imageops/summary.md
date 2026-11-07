# v0.11.0: the image operations of #65, checked on an L4 (2026-11-06/07)

The review of pickae's PR #65 (`--trim` and its conditions, `--resize` geometry, `--auto-level` / `--level` /
`--gamma` / `--negate`, `--grayscale` / `--colorspace Gray`) on AWS g6.4xlarge instances (NVIDIA L4, driver
595.112.07, CUDA 13.2, AMD EPYC 7R13 16 vCPU). Results and their reading: `docs/RESULTS.md`, "Trim, resize, tone and
gray: checked on an L4".

Builds:
- **main** 4721191: the same encoder source as v0.10.0;
- **PR head** c466e88;
- **release head:** b08c856, the PR plus three fix commits; e5cc7ce adds a help-text change, nothing else.

Each run built main and the head under test with identical flags (sm_89, Release, tests on). ImageMagick 7.1.2-18
Q16-HDRI was built from its tag without OpenMP. Decoders: dav1d 1.4.1 and aomdec 3.8.2.

## correctness/ (release head b08c856)

| File | What |
|---|---|
| `identity_main_vs_v0110.txt` | without the options: 999 comparisons against main, 22,231 outputs; the differences are the size-limit refusals and the `--max-mp` hint (10 rows are harness cases using `--resize`, which main lacks) |
| `limits.txt` | sides of 65,472–65,537, crops, shrinks and `--cpu-entropy` at the limits, against main |
| `tone_gray_v0110.txt`, `tone_gray_pr_head*.txt` | tone and gray against magick: plane metric and decoded luma, 4:2:0 / 4:4:4 / 10-bit, per source |
| `alr.txt`, `alr2.txt` | `--auto-level` per comic page, PNG and JPEG |
| `trimref_*.out` / `.jsonl` | trim boxes against magick and against a model of the documented rule: real pages and photos, adversarial JPEGs, synthetic PNGs |
| `cropgray.txt` | crop then gray on a 4:2:0 JPEG (edge rows and columns) |
| `san4.out` | compute-sanitizer memcheck / synccheck / racecheck summary |
| `fz13.out`, `fz21.out` | fuzzing, 2 × 3,000 iterations |
| `ctest.txt`, `probes.txt` | ctest and the single-case probes |
| `suite/` | geometry (`geom`, `exifgeo`, `exifpix`), conditions (`cond`), batch against single (`bvs`), batch under pressure (`batchp`), lossless tone (`tone_ll`, `deep*`), the tone kernel model (`tonemodel*`), alpha (`alpharz`, `n3g5`) |

## speed-pr-head/ (PR head c466e88 against main)

Options-off A/B (`spd_off*`), cold start (`cold_*`), memory, release-style packages (`spd_rel`, `cold_rel2`, `mods/`),
cost per option (`spd_on*`, `spd_stdlv` for the `--level` palette effect), kernel times (`pk.txt`), and the
byte-identical prototypes for a later PR (`spd_wpd`, `idn*`).

## speed-release-head/ (release head b08c856 against main)

The options-off A/B on the code as it ships, on a second L4 (i-0699ca24f9375ab14): `spd_fix3off.*` (24 interleaved
rounds per set; A = main, C = a byte copy of main, the noise control, F = b08c856), `cold_fix3.*` (single-image cold
start, 24 runs each), `env.txt`. e5cc7ce changes only the `--auto-level` help text and docs, so these hold for it.

## scripts/

`correctness/`: the test harness (`avt.py` parses the AVIF and decodes both items in both decoders), generators and
checks. `speed/`: `spd65.py` (interleaved rounds, rotating order), `cold65.py`, `mem65.py`, the set builders and the
analysis.
