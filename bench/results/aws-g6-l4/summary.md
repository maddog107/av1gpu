# av1gpu benchmark

- GPU: NVIDIA L4, 595.91.07, 23034 MiB, 72.00 W
- CPU: AMD EPYC 7R13 Processor (16 logical CPUs)
- avifenc 1.4.0 (aom [enc/dec]:3.13.1); tune=iq: yes
- cwebp 1.3.2
- 100 images, 78.7 MP

## Throughput (quality 60)

| Encoder | MP/s | images/s | CPU cores used | output MB |
|---|---:|---:|---:|---:|
| av1gpu (default settings) | 28.1 | 35.7 | 0.2 | 5.49 |
| libaom speed 6, 16 parallel avifenc -j 1 | 19.7 | 25.0 | 16.0 | 6.50 |
| libaom speed 9, 16 parallel avifenc -j 1 | 151.7 | 192.7 | 16.0 | 7.14 |

## Compression (BD-rate on SSIMULACRA2, per image; negative = smaller files at equal quality)

| Codec | vs | images | median | mean |
|---|---|---:|---:|---:|
| av1gpu | aom_s6 | 100 | +0.3% | -0.1% |
| aom_s9 | aom_s6 | 100 | +17.7% | +20.7% |
| webp | aom_s6 | 100 | +41.3% | +55.3% |
| av1gpu | webp | 100 | -30.0% | -32.9% |
| aom_s6 | webp | 100 | -29.2% | -32.5% |
| aom_s9 | webp | 100 | -17.6% | -19.5% |
