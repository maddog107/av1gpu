# av1gpu speed, v0.4 against v0.3.0 (NVIDIA L4, g6.4xlarge, 2026-11-01)

v0.4 = fdb0628 (4×4 tiles, one decode thread per usable CPU); v0.3.0 = f8f1ecc, alternating in the same session.
Each set repeated to ≥ 1,000 MP (lossless and RGBA sets as published), a warm-up then 3 timed runs, two rounds with
the binary order alternating; MP/s is the median of the 6 timed runs (av1gpu's `--stats`, or the wall time where
noted). CPU cores: process CPU time / wall time. Power: `nvidia-smi dmon`, 1 s samples during the timed runs.

## JPEG, size range (1,433 MP)

| setting | v0.4 MP/s | v0.3.0 MP/s | change | v0.4 CPU cores | v0.3.0 CPU cores | v0.4 W |
|---|---:|---:|---:|---:|---:|---:|
| default (3 encoders) | 128.0 | 112.3 | +14.0 % | 0.17 | 0.11 | 66.7 |
| `GP_ENCODERS=2` | 122.3 | 102.6 | +19.1 % | 0.16 | 0.10 | 66.0 |
| `GP_ENCODERS=4` | 126.6 | 115.4 | +9.7 % | 0.18 | 0.12 | 66.9 |
| `--small-preset none`, 3 encoders | 145.0 | 126.4 | +14.7 % | 0.19 | 0.12 | 69.3 |
| `--small-preset none`, 4 encoders | 147.3 | 129.0 | +14.2 % | 0.19 | 0.12 | 68.2 |
| `--preset turbo3`, 4 encoders | 166.1 | 164.2 | +1.1 % | 0.23 | 0.17 | 66.5 |
| `--preset turbo3 --small-preset none`, 4 encoders | 218.3 | 204.0 | +7.0 % | 0.27 | 0.19 | 69.2 |

## JPEG, standard (1,023 MP)

| setting | v0.4 MP/s | v0.3.0 MP/s | change | v0.4 CPU cores | v0.3.0 CPU cores | v0.4 W |
|---|---:|---:|---:|---:|---:|---:|
| default (3 encoders) | 39.4 | | | 0.15 | | 69.6 |
| `GP_ENCODERS=2` | 38.6 | | | 0.12 | | 69.7 |
| `GP_ENCODERS=4` | 39.3 | 39.4 | -0.4 % | 0.16 | 0.13 | 69.5 |
| `--small-preset none`, 3 encoders | 98.6 | 98.4 | +0.3 % | 0.26 | 0.21 | 68.2 |
| `--small-preset none`, 4 encoders | 98.1 | 98.7 | -0.6 % | 0.25 | 0.21 | 68.7 |

## Formats, size range, 4 encoders

| setting | v0.4 MP/s | v0.3.0 MP/s | change | v0.4 CPU cores | v0.3.0 CPU cores | v0.4 W |
|---|---:|---:|---:|---:|---:|---:|
| 8-bit 4:4:4 | 101.7 | 96.3 | +5.6 % | 0.15 | 0.10 | 67.8 |
| 10-bit 4:2:0 | 121.7 | 111.1 | +9.6 % | 0.19 | 0.13 | 66.4 |
| 10-bit 4:4:4 | 97.8 | 92.8 | +5.4 % | 0.16 | 0.11 | 68.0 |

## Lossless, 47 PNGs (42.7 MP), default encoders

| setting | v0.4 MP/s | v0.3.0 MP/s | change | v0.4 CPU cores | v0.3.0 CPU cores | v0.4 W |
|---|---:|---:|---:|---:|---:|---:|
| `--lossless`, one pass (wall incl. start-up) | 18.0 | 15.9 | +13.1 % | 1.35 | 0.89 |  |
| `--lossless --ycgco-re`, one pass (wall) | 18.2 | 16.2 | +12.4 % | 1.41 | 0.84 |  |
| `--lossless`, set 4× | 41.2 | 43.0 | -4.1 % | 1.43 | 1.04 |  |
| `--lossless --ycgco-re`, set 4× | 41.5 | 42.9 | -3.2 % | 1.45 | 0.96 |  |

## RGBA, 39 PNGs × 20 (202 MP), 4 encoders, wall incl. start-up

| setting | v0.4 MP/s | v0.3.0 MP/s | change | v0.4 CPU cores | v0.3.0 CPU cores | v0.4 W |
|---|---:|---:|---:|---:|---:|---:|
| default | 27.1 | 27.1 | -0.2 % | 1.68 | 1.54 |  |
| `--alpha-quality 100` | 25.8 | 25.1 | +2.9 % | 1.70 | 1.52 |  |
| `--lossless` | 25.5 | 26.0 | -2.0 % | 1.28 | 0.95 |  |
| `--no-alpha` | 30.2 | 30.0 | +0.7 % | 1.98 | 1.81 |  |

Bytes per run were identical across runs for every setting. The demo recording (README) is `logs/rec.json`:
99 photos, default settings, 98.9 MP/s in av1gpu's count, 5.8 s wall time including start-up, 0.20 cores.

**Lossless JPEG input was not timed.** With all 100 size-range JPEGs and `--lossless`, v0.4 stopped after reporting
75 of the refusals and never finished (`runs/lossless_jpeg_hang_partial.jsonl` holds the runs before it); with 3,
10 or 30 JPEGs it finished. The speed in RESULTS.md (44.8 MP/s) is from before JPEGs were refused, so it stays.
