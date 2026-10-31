# av1gpu formats benchmark

- GPU: NVIDIA L4, 595.112.07, 23034 MiB, 72.00 W; `GP_ENCODERS=4`
- CPU: AMD EPYC 7R13 Processor (16 logical CPUs)
- avifenc 1.4.0 (dav1d [dec]:1.4.1, aom [enc/dec]:3.13.1, svt [enc]:v4.2.0); tune=iq: yes; libaom speed 6, one avifenc -j 1 per CPU
- decoding: avifdec 1.4.0 (dav1d [dec]:1.4.1, aom [enc/dec]:3.13.1, svt [enc]:v4.2.0), -c dav1d (16-bit PNGs for 10-bit); SSIMULACRA2 against the source JPEG
- 100 images, 477.8 MP; quality points 30,40,50,60,70,80
- SVT-AV1: tune 3
- speed runs: the set repeated 3x (300 images, 1434 MP), quality 60

## Throughput (quality 60)

av1gpu: median of all timed runs over the interleaved rounds (round medians in brackets).

| Encoder | Format | MP/s | images/s | CPU cores | output MB |
|---|---|---:|---:|---:|---:|
| av1gpu | 8-bit 4:2:0 | 113.4 (114.2, 112.8, 111.4) | 23.7 | 0.1 | 67.47 |
| av1gpu | 8-bit 4:4:4 | 94.5 (93.5, 94.5, 94.6) | 19.8 | 0.1 | 70.67 |
| av1gpu | 10-bit 4:2:0 | 109.8 (112.5, 109.0, 109.8) | 23.0 | 0.1 | 66.64 |
| av1gpu | 10-bit 4:4:4 | 92.8 (93.8, 92.3, 92.9) | 19.4 | 0.1 | 69.26 |
| libaom s6 | 8-bit 4:2:0 | 28.6 | 6.0 | 16.0 | 86.61 |
| libaom s6 | 8-bit 4:4:4 | 24.3 | 5.1 | 16.0 | 90.11 |
| libaom s6 | 10-bit 4:2:0 | 22.0 | 4.6 | 16.0 | 87.83 |
| libaom s6 | 10-bit 4:4:4 | 18.9 | 4.0 | 16.0 | 91.69 |
| SVT-AV1 preset 6 | 10-bit 4:2:0 | 43.5 | 9.1 | 16.0 | 112.74 |

## BD-rate (SSIMULACRA2, per image; negative = smaller files at equal quality)

| Test | vs | images | median | mean |
|---|---|---:|---:|---:|
| av1gpu 8-bit 4:2:0 | libaom s6 8-bit 4:2:0 | 100 | -0.12% | -0.42% |
| av1gpu 8-bit 4:4:4 | libaom s6 8-bit 4:4:4 | 100 | +0.34% | +0.27% |
| av1gpu 10-bit 4:2:0 | libaom s6 10-bit 4:2:0 | 100 | +0.07% | -0.51% |
| av1gpu 10-bit 4:4:4 | libaom s6 10-bit 4:4:4 | 100 | +0.02% | +0.06% |
| av1gpu 8-bit 4:4:4 | av1gpu 8-bit 4:2:0 | 100 | +0.41% | +0.43% |
| av1gpu 10-bit 4:4:4 | av1gpu 10-bit 4:2:0 | 100 | +0.35% | +0.40% |
| av1gpu 10-bit 4:2:0 | av1gpu 8-bit 4:2:0 | 100 | -2.25% | -3.06% |
| av1gpu 10-bit 4:4:4 | av1gpu 8-bit 4:4:4 | 100 | -1.94% | -2.98% |
| libaom s6 8-bit 4:4:4 | libaom s6 8-bit 4:2:0 | 100 | -0.76% | -0.85% |
| libaom s6 10-bit 4:4:4 | libaom s6 10-bit 4:2:0 | 100 | -0.53% | -0.76% |
| libaom s6 10-bit 4:2:0 | libaom s6 8-bit 4:2:0 | 100 | -2.01% | -3.21% |
| libaom s6 10-bit 4:4:4 | libaom s6 8-bit 4:4:4 | 100 | -1.84% | -3.16% |
| SVT-AV1 p6 10-bit 4:2:0 | libaom s6 10-bit 4:2:0 | 100 | +1.14% | +1.86% |
| SVT-AV1 p6 10-bit 4:2:0 | SVT-AV1 8-bit 4:2:0 (published) | 99 | -3.68% | -5.51% |
| av1gpu 8-bit 4:2:0 | libaom s6 8-bit 4:2:0 (published) | 100 | -0.12% | -0.42% |
| libaom s6 8-bit 4:2:0 | libaom s6 8-bit 4:2:0 (published) | 100 | +0.00% | +0.00% |
| av1gpu 8-bit 4:4:4 | libaom s6 8-bit 4:2:0 (published) | 100 | -0.71% | -0.42% |
| libaom s6 8-bit 4:4:4 | libaom s6 8-bit 4:2:0 (published) | 100 | -0.76% | -0.85% |
| av1gpu 10-bit 4:2:0 | libaom s6 8-bit 4:2:0 (published) | 100 | -2.54% | -3.69% |
| libaom s6 10-bit 4:2:0 | libaom s6 8-bit 4:2:0 (published) | 100 | -2.01% | -3.21% |
| av1gpu 10-bit 4:4:4 | libaom s6 8-bit 4:2:0 (published) | 100 | -2.95% | -3.70% |
| libaom s6 10-bit 4:4:4 | libaom s6 8-bit 4:2:0 (published) | 100 | -3.23% | -3.96% |
| SVT-AV1 p6 10-bit 4:2:0 | libaom s6 8-bit 4:2:0 (published) | 100 | -0.84% | -2.29% |

## av1gpu vs libaom s6 in the same format, by image size (median; images in brackets)

| Format | < 1 MP | 1-5 MP | 5-10 MP | >= 10 MP |
|---|---:|---:|---:|---:|
| 8-bit 4:2:0 | +0.1% (40) | +1.1% (20) | -0.6% (23) | -2.4% (17) |
| 8-bit 4:4:4 | +0.1% (40) | +2.8% (20) | +1.0% (23) | -1.2% (17) |
| 10-bit 4:2:0 | +0.1% (40) | +1.1% (20) | -1.4% (23) | -0.9% (17) |
| 10-bit 4:4:4 | -0.1% (40) | +1.6% (20) | -0.2% (23) | +0.2% (17) |

## Bytes and mean SSIMULACRA2 per quality point (all images)

| Encoder | q30 | q40 | q50 | q60 | q70 | q80 |
|---|---:|---:|---:|---:|---:|---:|
| av1gpu 8-bit 4:2:0 | 5.27 MB, 32.72 | 8.49 MB, 48.99 | 14.35 MB, 62.70 | 22.49 MB, 72.00 | 33.37 MB, 78.43 | 49.34 MB, 83.75 |
| libaom s6 8-bit 4:2:0 | 7.10 MB, 42.88 | 11.39 MB, 56.63 | 19.04 MB, 67.87 | 28.87 MB, 75.88 | 40.98 MB, 81.31 | 57.02 MB, 85.56 |
| av1gpu 8-bit 4:4:4 | 5.51 MB, 33.82 | 8.81 MB, 50.00 | 14.92 MB, 63.73 | 23.56 MB, 72.97 | 34.34 MB, 79.25 | 49.00 MB, 84.10 |
| libaom s6 8-bit 4:4:4 | 7.34 MB, 44.02 | 11.70 MB, 57.58 | 19.65 MB, 68.83 | 30.04 MB, 76.79 | 42.00 MB, 82.13 | 56.83 MB, 86.09 |
| av1gpu 10-bit 4:2:0 | 5.16 MB, 31.95 | 8.25 MB, 48.76 | 14.00 MB, 62.77 | 22.21 MB, 72.46 | 33.65 MB, 79.27 | 53.54 MB, 85.22 |
| libaom s6 10-bit 4:2:0 | 7.25 MB, 42.65 | 11.53 MB, 57.19 | 19.04 MB, 68.61 | 29.28 MB, 76.79 | 42.36 MB, 82.48 | 61.94 MB, 87.16 |
| av1gpu 10-bit 4:4:4 | 5.37 MB, 32.95 | 8.53 MB, 49.62 | 14.49 MB, 63.65 | 23.09 MB, 73.28 | 34.25 MB, 79.89 | 51.68 MB, 85.33 |
| libaom s6 10-bit 4:4:4 | 7.55 MB, 44.06 | 11.95 MB, 58.33 | 19.78 MB, 69.68 | 30.56 MB, 77.77 | 43.42 MB, 83.34 | 60.85 MB, 87.75 |
| SVT-AV1 p6 10-bit 4:2:0 | 10.07 MB, 53.13 | 16.75 MB, 65.81 | 25.01 MB, 73.99 | 37.58 MB, 80.89 | 49.15 MB, 84.75 | 67.50 MB, 88.04 |

- libaom s6 8-bit 4:2:0 against the published `aom_s6` rows (bench/results/aws-g6-l4-gamut/results.json): 600 common files, 600 with the same bytes, 600 with the same SSIMULACRA2 score
- av1gpu 8-bit 4:2:0 against the published `av1gpu` rows (bench/results/aws-g6-l4-gamut/results.json): 600 common files, 600 with the same bytes, 600 with the same SSIMULACRA2 score
