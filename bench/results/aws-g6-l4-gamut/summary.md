# av1gpu benchmark

- GPU: NVIDIA L4, 595.91.07, 23034 MiB, 72.00 W
- CPU: AMD EPYC 7R13 Processor (16 logical CPUs)
- avifenc 1.4.0 (aom [enc/dec]:3.13.1); tune=iq: yes
- cwebp 1.3.2
- 100 images, 477.8 MP

## Throughput (quality 60)

| Encoder | MP/s | images/s | CPU cores used | output MB |
|---|---:|---:|---:|---:|
| av1gpu (default settings) | 82.4 | 17.2 | 0.1 | 22.49 |
| libaom speed 6, 16 parallel avifenc -j 1 | 24.7 | 5.2 | 16.0 | 28.87 |
| libaom speed 9, 16 parallel avifenc -j 1 | 169.3 | 35.4 | 16.0 | 30.89 |

## Compression (BD-rate on SSIMULACRA2, per image; negative = smaller files at equal quality)

| Codec | vs | images | median | mean |
|---|---|---:|---:|---:|
| av1gpu | aom_s6 | 100 | -0.1% | -0.4% |
| aom_s9 | aom_s6 | 100 | +15.0% | +17.2% |
| webp | aom_s6 | 99 | +45.1% | +61.2% |
| av1gpu | webp | 99 | -31.2% | -33.7% |
| aom_s6 | webp | 99 | -31.1% | -33.1% |
| aom_s9 | webp | 99 | -19.3% | -22.6% |
