# av1gpu benchmark

- GPU: NVIDIA L4, 595.91.07, 23034 MiB, 72.00 W
- CPU: AMD EPYC 7R13 Processor (16 logical CPUs)
- avifenc 1.4.0 (aom [enc/dec]:3.13.1); tune=iq: yes
- cwebp 1.3.2
- 100 images, 477.8 MP
- speed runs: the set repeated 3x (300 images, 1434 MP)

## Throughput (quality 60)

| Encoder | MP/s | images/s | CPU cores used | output MB |
|---|---:|---:|---:|---:|
| av1gpu (--preset turbo --small-preset none) | 126.8 | 26.5 | 0.1 | 67.52 |
| libaom speed 6, 16 parallel avifenc -j 1 | 28.9 | 6.0 | 16.0 | 86.61 |
| libaom speed 9, 16 parallel avifenc -j 1 | 186.5 | 39.0 | 16.0 | 92.66 |
| libaom speed 10, 16 parallel avifenc -j 1 | 188.7 | 39.5 | 16.0 | 92.66 |

## Compression (BD-rate on SSIMULACRA2, per image; negative = smaller files at equal quality)

| Codec | vs | images | median | mean |
|---|---|---:|---:|---:|
| av1gpu | aom_s6 | 100 | +2.1% | +1.1% |
| aom_s9 | aom_s6 | 100 | +15.0% | +17.2% |
| aom_s10 | aom_s6 | 100 | +15.0% | +17.2% |
| webp | aom_s6 | 99 | +45.1% | +61.2% |
| av1gpu | webp | 99 | -30.9% | -32.7% |
| aom_s6 | webp | 99 | -31.1% | -33.1% |
| aom_s9 | webp | 99 | -19.3% | -22.6% |
| aom_s10 | webp | 99 | -19.3% | -22.6% |

## BD-rate vs libaom s6 by image size (median; images in brackets)

| Codec | < 1 MP | 1-5 MP | 5-10 MP | >= 10 MP |
|---|---:|---:|---:|---:|
| av1gpu | +3.9% (40) | +1.1% (20) | -0.6% (23) | -2.4% (17) |
| aom_s9 | +20.1% (40) | +18.5% (20) | +12.8% (23) | +13.4% (17) |
| aom_s10 | +20.1% (40) | +18.5% (20) | +12.8% (23) | +13.4% (17) |
| webp | +38.9% (40) | +49.9% (20) | +49.4% (22) | +58.9% (17) |
