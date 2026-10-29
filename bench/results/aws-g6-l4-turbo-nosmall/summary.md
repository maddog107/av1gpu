# av1gpu benchmark

- GPU: NVIDIA L4, 595.91.07, 23034 MiB, 72.00 W
- CPU: AMD EPYC 7R13 Processor (16 logical CPUs)
- avifenc 1.4.0 (aom [enc/dec]:3.13.1); tune=iq: yes
- cwebp 1.3.2
- 100 images, 78.7 MP
- speed runs: the set repeated 13x (1300 images, 1023 MP)

## Throughput (quality 60)

| Encoder | MP/s | images/s | CPU cores used | output MB |
|---|---:|---:|---:|---:|
| av1gpu (--preset turbo --small-preset none) | 100.9 | 128.1 | 0.2 | 71.87 |
| libaom speed 6, 16 parallel avifenc -j 1 | 21.1 | 26.8 | 16.0 | 84.50 |
| libaom speed 9, 16 parallel avifenc -j 1 | 166.1 | 210.9 | 16.0 | 92.85 |
| libaom speed 10, 16 parallel avifenc -j 1 | 166.1 | 211.0 | 16.0 | 92.85 |

## Compression (BD-rate on SSIMULACRA2, per image; negative = smaller files at equal quality)

| Codec | vs | images | median | mean |
|---|---|---:|---:|---:|
| av1gpu | aom_s6 | 100 | +3.4% | +2.9% |
| aom_s9 | aom_s6 | 100 | +17.7% | +20.7% |
| aom_s10 | aom_s6 | 100 | +17.7% | +20.7% |
| webp | aom_s6 | 100 | +41.3% | +55.3% |
| av1gpu | webp | 100 | -27.9% | -30.8% |
| aom_s6 | webp | 100 | -29.2% | -32.5% |
| aom_s9 | webp | 100 | -17.6% | -19.5% |
| aom_s10 | webp | 100 | -17.6% | -19.5% |

## BD-rate vs libaom s6 by image size (median; images in brackets)

| Codec | < 1 MP | 1-5 MP | 5-10 MP | >= 10 MP |
|---|---:|---:|---:|---:|
| av1gpu | +3.4% (86) | +3.8% (14) | — | — |
| aom_s9 | +17.5% (86) | +23.8% (14) | — | — |
| aom_s10 | +17.5% (86) | +23.8% (14) | — | — |
| webp | +40.4% (86) | +57.4% (14) | — | — |
