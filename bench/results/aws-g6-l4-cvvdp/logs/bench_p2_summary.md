# av1gpu benchmark

- GPU: NVIDIA L4, 595.112.07, 23034 MiB, 72.00 W
- CPU: AMD EPYC 7R13 Processor (16 logical CPUs)
- avifenc 1.4.0 (dav1d [dec]:1.4.1, aom [enc/dec]:3.13.1, svt [enc]:v4.2.0); tune=iq: yes
- cwebp 1.3.2
- 99 images, 477.7 MP

## Compression (BD-rate on SSIMULACRA2, per image; negative = smaller files at equal quality)

| Codec | vs | images | median | mean |
|---|---|---:|---:|---:|
| av1gpu | aom_s6 | 1 | -5.2% | -5.2% |
| aom_s9 | aom_s6 | 12 | +6.2% | +7.4% |

## BD-rate vs libaom s6 by image size (median; images in brackets)

| Codec | < 1 MP | 1-5 MP | 5-10 MP | >= 10 MP |
|---|---:|---:|---:|---:|
| av1gpu | -5.2% (1) | — | — | — |
| aom_s9 | +5.7% (4) | +5.9% (3) | +7.4% (3) | +6.4% (2) |

## CVVDP

- cvvdp 0.5.7 (parameters 0.5.7), torch 2.15.0+cu132 (CUDA 13.2) on NVIDIA L4
- display `standard_fhd`: 24-inch FullHD monitor, peak luminance 200 cd/m^2, viewed under office light levels (250 lux), seen from 2 x display height; 1920 x 1080, 200 cd/m^2, contrast 1000:1, 250 lux, 0.6 m, 37.8 pixels per degree; EOTF sRGB
- inputs: 8-bit sRGB (ICC profiles converted with LittleCMS, relative colorimetric), 1:1 pixel mapping

## Bits per pixel to reach 9.5 JOD

Interpolated per image within the tested qualities. The last column is over the 98 images that every codec reaches.

| Codec | images reaching | median bpp | median bpp, common images |
|---|---:|---:|---:|
| av1gpu | 98/99 | 1.116 | 1.116 |
| aom_s6 | 98/99 | 1.202 | 1.202 |
| aom_s9 | 98/99 | 1.290 | 1.290 |
| svt_p4 | 98/99 | 1.412 | 1.412 |
| svt_p6 | 98/99 | 1.431 | 1.431 |
| svt_p9 | 98/99 | 1.436 | 1.436 |
| webp | 98/99 | 2.335 | 2.335 |

## Mean scores per quality point

| Codec | q | images | MB | mean SSIMULACRA2 | mean JOD | images >= 9.5 JOD |
|---|---:|---:|---:|---:|---:|---:|
| av1gpu | 85 | 99 | 57.83 | 85.48 | 9.853 | 98 |
| av1gpu | 90 | 99 | 76.59 | 87.86 | 9.910 | 98 |
| aom_s6 | 85 | 99 | 65.13 | 86.91 | 9.889 | 98 |
| aom_s6 | 90 | 99 | 82.31 | 88.60 | 9.924 | 98 |
| aom_s9 | 85 | 99 | 68.52 | 86.35 | 9.881 | 98 |
| aom_s9 | 90 | 99 | 87.36 | 88.32 | 9.919 | 98 |
| svt_p4 | 85 | 99 | 76.67 | 88.48 | 9.921 | 98 |
| svt_p4 | 90 | 99 | 92.05 | 89.55 | 9.942 | 98 |
| svt_p6 | 85 | 99 | 77.55 | 88.14 | 9.916 | 98 |
| svt_p6 | 90 | 99 | 92.42 | 89.39 | 9.941 | 98 |
| svt_p9 | 85 | 99 | 79.01 | 88.02 | 9.912 | 98 |
| svt_p9 | 90 | 99 | 93.45 | 89.25 | 9.938 | 98 |
| webp | 98 | 99 | 125.43 | 86.19 | 9.927 | 98 |
| webp | 100 | 99 | 135.61 | 86.51 | 9.934 | 98 |
