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
| av1gpu (default settings) | 103.8 | 21.7 | 0.1 | 67.47 |
| libaom speed 6, 16 parallel avifenc -j 1 | 28.8 | 6.0 | 16.0 | 86.61 |
| libaom speed 9, 16 parallel avifenc -j 1 | 187.6 | 39.3 | 16.0 | 92.66 |
