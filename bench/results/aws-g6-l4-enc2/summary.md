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
| av1gpu (default settings) | 39.2 | 49.7 | 0.1 | 71.43 |
| libaom speed 6, 16 parallel avifenc -j 1 | 21.0 | 26.7 | 16.0 | 84.50 |
| libaom speed 9, 16 parallel avifenc -j 1 | 166.0 | 210.9 | 16.0 | 92.85 |
