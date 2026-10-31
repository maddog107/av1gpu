# CVVDP and SSIMULACRA2, size-range photos, NVIDIA L4

- GPU: NVIDIA L4, 595.112.07, 23034 MiB, 72.00 W; CPU: AMD EPYC 7R13 Processor (16 vCPUs); Linux-7.0.0-1014-aws-x86_64-with-glibc2.39
- avifenc 1.4.0 (dav1d [dec]:1.4.1, aom [enc/dec]:3.13.1, svt [enc]:v4.2.0); tune=iq; cwebp 1.3.2; NVENC through ffmpeg n8.1.3 (BtbN static) and the hybrid_nvenc_libavif muxer, as published
- decoded with avifdec -c dav1d (AVIF) and dwebp 1.3.2 (WebP); SSIMULACRA2: libjxl v0.11.1 `ssimulacra2`
- CVVDP: cvvdp 0.5.7 (pip), torch 2.15.0+cu132 on the NVIDIA L4; display `standard_fhd`: 1920 x 1080, 200 cd/m^2, contrast 1000:1, 250 lux, 0.6 m, 37.8 pixels per degree, EOTF sRGB
- inputs: 8-bit sRGB (ICC profiles converted with LittleCMS, relative colorimetric), 1:1 pixel mapping; reference = the source JPEG decoded by Pillow (identical to libjxl's decode)
- 99 photos, 477.7 MP; quality points: av1gpu 30,40,50,60,70,80,85,90; aom_s6 30,40,50,60,70,80,85,90; aom_s9 30,40,50,60,70,80,85,90; svt_p4 30,40,50,60,70,80,85,90; svt_p6 30,40,50,60,70,80,85,90; svt_p9 30,40,50,60,70,80,85,90; webp 50,60,70,80,90,95,98,100; nvenc 6,12,18,25,32,38,44,50

## How it was run

One g6.4xlarge spot instance (2026-10-31), the published builds (`scripts/setup.sh`: libaom 3.13.1, SVT-AV1 4.2.0 and libavif 1.4.0 from source, libjxl v0.11.1 `ssimulacra2`, av1gpu from this branch) and the 99 photos of the published SVT-AV1 and NVENC runs (`fetch_images.py` now downloads all 100; 9fba6f263f3c4f3e_tiny.jpg is left out as before).

```
bench/bench.py --images ~/images_gamut --out p1 --av1gpu <GP_ENCODERS=4 wrapper> --skip-speed --aom-speeds 6,9 \
  --svt-presets 4,6,9 --svt-tune 3 --avifdec-codec dav1d --cvvdp standard_fhd --keep        # published points
bench/bench.py ... --quality 85,90 --webp-quality 98,100                                   # extra points
```

NVENC: `scripts/nvenc_encode.py` (the published hybrid_nvenc_libavif pipeline, unchanged) and `scripts/nvenc_score.py` (SSIMULACRA2 exactly as published, CVVDP through `bench.Cvvdp`). `scripts/analyze.py` merged the phases' outputs (not committed; every row is in `results.json`) into `results.json` and this file; `logs/` has the console output (the bench.py summaries there used a 0.1 JOD overlap; 0.05 and 0.1 give the same BD-rates here).

**ICC fix.** Four photos (370f7c330a75892f_large.jpg, bd4ff3b451c3ba6b_large.jpg, ca12d94f3a1dca2c_large.jpg, fe00a8d898a01166_large.jpg) store their ICC profile (Camera RGB or Generic RGB) after the JPEG frame header. libjpeg, and so ssimulacra2, avifenc and cwebp, read it, but Pillow does not, so the first two phases gave CVVDP an untagged (sRGB) reference against tagged encodes. bench.py now assembles the profile itself (commit `bench.py: CVVDP reads JPEG ICC profiles that Pillow misses`). A second g6.4xlarge (on demand; `scripts/run3.sh`) re-encoded those photos at every point and re-scored them: 256 files, 256 with the same bytes and 256 with the same SSIMULACRA2 as before. Their CVVDP scores in `results.json` and below are the re-scored ones.

## Identity with the published runs

| Encoder | files compared | same bytes | same SSIMULACRA2 | max difference |
|---|---:|---:|---:|---:|
| libaom speed 6 | 594 | 594 | 594 | 0 |
| av1gpu default (`GP_ENCODERS=4`) | 594 | 594 | 594 | 0 |
| libaom speed 9 | 594 | 594 | 594 | 0 |
| SVT-AV1 preset 4 | 594 | 594 | 594 | 0 |
| SVT-AV1 preset 6 | 594 | 594 | 594 | 0 |
| SVT-AV1 preset 9 | 594 | 594 | 594 | 0 |
| NVENC AV1 (p7, `tune uhq`) | 594 | 594 | 594 | 0 |
| WebP (cwebp `-m 4`) | 594 | 594 | 594 | 0 |

## BD-rate against libaom speed 6 (median over images, mean in brackets)

JOD curves must share at least 0.05 JOD (SSIMULACRA2: 2 points). Images in the BD-rate: last column.

| Encoder | CVVDP JOD, all points | CVVDP JOD, published points | SSIMULACRA2 as published | SSIMULACRA2 this run, published points | SSIMULACRA2 this run, all points | images (JOD all / SSIM2 published) |
|---|---:|---:|---:|---:|---:|---:|
| av1gpu default (`GP_ENCODERS=4`) | +3.9% (+4.0%) | +3.9% (+4.1%) | -0.1% (-0.4%) | -0.0% (-0.4%) | +0.3% (-0.2%) | 99 / 100 |
| libaom speed 9 | +14.5% (+16.3%) | +14.9% (+16.7%) | +15.0% (+17.2%) | +15.0% (+17.1%) | +14.7% (+16.7%) | 99 / 100 |
| SVT-AV1 preset 4 | -2.0% (-2.3%) | -2.0% (-2.2%) | -2.6% (-2.9%) | -2.6% (-2.9%) | -2.7% (-3.0%) | 99 / 99 |
| SVT-AV1 preset 6 | +2.0% (+2.7%) | +1.9% (+2.7%) | +2.1% (+3.2%) | +2.1% (+3.2%) | +2.1% (+3.2%) | 99 / 99 |
| SVT-AV1 preset 9 | +9.3% (+11.1%) | +10.0% (+11.3%) | +9.8% (+11.9%) | +9.8% (+11.9%) | +9.4% (+11.5%) | 99 / 99 |
| NVENC AV1 (p7, `tune uhq`) | +28.4% (+35.6%) | +27.4% (+32.8%) | +29.5% (+46.6%) | +29.5% (+46.6%) | +30.0% (+54.0%) | 97 / 94 |
| WebP (cwebp `-m 4`) | +44.2% (+44.6%) | +45.1% (+45.5%) | +45.1% (+61.2%) | +45.2% (+61.6%) | +45.1% (+62.9%) | 99 / 99 |

## By image size (median BD-rate against libaom speed 6, images in brackets)

| Encoder | metric | < 1 MP | 1-5 MP | 5-10 MP | >= 10 MP |
|---|---|---:|---:|---:|---:|
| av1gpu default (`GP_ENCODERS=4`) | CVVDP JOD | +2.1% (39) | +5.2% (20) | +5.8% (23) | +5.9% (17) |
| av1gpu default (`GP_ENCODERS=4`) | SSIMULACRA2 (this run, all points) | +0.4% (39) | +1.4% (20) | +0.2% (23) | -2.0% (17) |
| av1gpu default (`GP_ENCODERS=4`) | SSIMULACRA2 this run, published points (= published scores) | +0.2% (39) | +1.1% (20) | -0.6% (23) | -2.4% (17) |
| libaom speed 9 | CVVDP JOD | +17.9% (39) | +17.3% (20) | +12.3% (23) | +12.0% (17) |
| libaom speed 9 | SSIMULACRA2 (this run, all points) | +19.5% (39) | +18.0% (20) | +12.3% (23) | +13.0% (17) |
| libaom speed 9 | SSIMULACRA2 as published | +20.1% (39) | +18.5% (20) | +12.8% (23) | +13.4% (17) |
| SVT-AV1 preset 4 | CVVDP JOD | -1.9% (39) | -2.4% (20) | -2.4% (23) | -1.7% (17) |
| SVT-AV1 preset 4 | SSIMULACRA2 (this run, all points) | -2.0% (39) | -2.4% (20) | -3.6% (23) | -3.7% (17) |
| SVT-AV1 preset 4 | SSIMULACRA2 as published | -2.0% (39) | -2.2% (20) | -3.4% (23) | -3.9% (17) |
| SVT-AV1 preset 6 | CVVDP JOD | +4.4% (39) | +2.0% (20) | +0.8% (23) | +1.2% (17) |
| SVT-AV1 preset 6 | SSIMULACRA2 (this run, all points) | +4.8% (39) | +2.8% (20) | +0.8% (23) | +0.2% (17) |
| SVT-AV1 preset 6 | SSIMULACRA2 as published | +5.1% (39) | +2.9% (20) | +0.7% (23) | +0.2% (17) |
| SVT-AV1 preset 9 | CVVDP JOD | +13.5% (39) | +10.7% (20) | +5.3% (23) | +8.0% (17) |
| SVT-AV1 preset 9 | SSIMULACRA2 (this run, all points) | +14.1% (39) | +12.3% (20) | +6.3% (23) | +7.1% (17) |
| SVT-AV1 preset 9 | SSIMULACRA2 as published | +14.9% (39) | +12.5% (20) | +6.3% (23) | +7.2% (17) |
| NVENC AV1 (p7, `tune uhq`) | CVVDP JOD | +33.1% (38) | +25.3% (19) | +23.8% (23) | +20.7% (17) |
| NVENC AV1 (p7, `tune uhq`) | SSIMULACRA2 (this run, all points) | +35.8% (38) | +28.8% (19) | +28.4% (21) | +22.0% (17) |
| NVENC AV1 (p7, `tune uhq`) | SSIMULACRA2 as published | +35.7% (38) | +29.2% (18) | +28.5% (21) | +22.2% (17) |
| WebP (cwebp `-m 4`) | CVVDP JOD | +36.3% (39) | +48.6% (20) | +44.5% (23) | +57.9% (17) |
| WebP (cwebp `-m 4`) | SSIMULACRA2 (this run, all points) | +38.9% (39) | +48.0% (20) | +49.1% (22) | +59.1% (17) |
| WebP (cwebp `-m 4`) | SSIMULACRA2 as published | +39.1% (39) | +49.9% (20) | +49.4% (22) | +58.9% (17) |

## Reaching 9.5 JOD

Bits per pixel at 9.5 JOD, interpolated per image (log size against JOD, as in the BD-rate) within the tested points. 'Common' = the 85 photos that every encoder reaches.

| Encoder | photos reaching it | median bpp | median bpp, common photos | photos reaching it, published points only |
|---|---:|---:|---:|---:|
| libaom speed 6 | 99/99 | 0.448 | 0.418 | 99/99 |
| av1gpu default (`GP_ENCODERS=4`) | 99/99 | 0.459 | 0.443 | 99/99 |
| libaom speed 9 | 99/99 | 0.522 | 0.504 | 99/99 |
| SVT-AV1 preset 4 | 99/99 | 0.430 | 0.410 | 99/99 |
| SVT-AV1 preset 6 | 99/99 | 0.468 | 0.433 | 99/99 |
| SVT-AV1 preset 9 | 99/99 | 0.491 | 0.461 | 99/99 |
| NVENC AV1 (p7, `tune uhq`) | 85/99 | 0.541 | 0.541 | 85/99 |
| WebP (cwebp `-m 4`) | 99/99 | 0.666 | 0.621 | 99/99 |

## Mean JOD per quality point

| Encoder | q30 | q40 | q50 | q60 | q70 | q80 | q85 | q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| libaom speed 6 | 8.665 | 9.059 | 9.409 | 9.638 | 9.766 | 9.865 | 9.897 | 9.933 |
| av1gpu default (`GP_ENCODERS=4`) | 8.386 | 8.807 | 9.220 | 9.498 | 9.674 | 9.812 | 9.860 | 9.918 |
| libaom speed 9 | 8.607 | 9.012 | 9.386 | 9.610 | 9.748 | 9.854 | 9.888 | 9.928 |
| SVT-AV1 preset 4 | 8.959 | 9.331 | 9.571 | 9.734 | 9.820 | 9.891 | 9.930 | 9.951 |
| SVT-AV1 preset 6 | 8.965 | 9.335 | 9.576 | 9.736 | 9.819 | 9.886 | 9.924 | 9.950 |
| SVT-AV1 preset 9 | 8.957 | 9.329 | 9.570 | 9.717 | 9.805 | 9.878 | 9.920 | 9.947 |

| WebP (cwebp `-m 4`) | q50 | q60 | q70 | q80 | q90 | q95 | q98 | q100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean JOD | 9.126 | 9.238 | 9.345 | 9.542 | 9.806 | 9.908 | 9.936 | 9.942 |

| NVENC AV1 (p7, `tune uhq`) | CQ 6 | CQ 12 | CQ 18 | CQ 25 | CQ 32 | CQ 38 | CQ 44 | CQ 50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean JOD | 9.806 | 9.799 | 9.770 | 9.684 | 9.489 | 9.215 | 8.884 | 8.432 |

## Every quality point

| Encoder | point | photos | MB | mean SSIMULACRA2 | mean JOD | photos >= 9.5 JOD |
|---|---:|---:|---:|---:|---:|---:|
| libaom speed 6 | 30 | 99 | 7.09 | 42.88 | 8.665 | 0 |
| libaom speed 6 | 40 | 99 | 11.38 | 56.64 | 9.059 | 2 |
| libaom speed 6 | 50 | 99 | 19.03 | 67.89 | 9.409 | 22 |
| libaom speed 6 | 60 | 99 | 28.85 | 75.89 | 9.638 | 92 |
| libaom speed 6 | 70 | 99 | 40.95 | 81.29 | 9.766 | 99 |
| libaom speed 6 | 80 | 99 | 56.97 | 85.54 | 9.865 | 99 |
| libaom speed 6 | 85 | 99 | 65.13 | 86.91 | 9.897 | 99 |
| libaom speed 6 | 90 | 99 | 82.31 | 88.60 | 9.933 | 99 |
| av1gpu default (`GP_ENCODERS=4`) | 30 | 99 | 5.27 | 32.67 | 8.386 | 0 |
| av1gpu default (`GP_ENCODERS=4`) | 40 | 99 | 8.48 | 48.98 | 8.807 | 0 |
| av1gpu default (`GP_ENCODERS=4`) | 50 | 99 | 14.34 | 62.71 | 9.220 | 8 |
| av1gpu default (`GP_ENCODERS=4`) | 60 | 99 | 22.47 | 72.00 | 9.498 | 58 |
| av1gpu default (`GP_ENCODERS=4`) | 70 | 99 | 33.34 | 78.42 | 9.674 | 90 |
| av1gpu default (`GP_ENCODERS=4`) | 80 | 99 | 49.30 | 83.73 | 9.812 | 99 |
| av1gpu default (`GP_ENCODERS=4`) | 85 | 99 | 57.83 | 85.48 | 9.860 | 99 |
| av1gpu default (`GP_ENCODERS=4`) | 90 | 99 | 76.59 | 87.86 | 9.918 | 99 |
| libaom speed 9 | 30 | 99 | 7.99 | 40.47 | 8.607 | 0 |
| libaom speed 9 | 40 | 99 | 12.53 | 54.94 | 9.012 | 1 |
| libaom speed 9 | 50 | 99 | 20.74 | 66.82 | 9.386 | 18 |
| libaom speed 9 | 60 | 99 | 30.86 | 74.70 | 9.610 | 81 |
| libaom speed 9 | 70 | 99 | 43.22 | 80.45 | 9.748 | 99 |
| libaom speed 9 | 80 | 99 | 59.90 | 84.92 | 9.854 | 99 |
| libaom speed 9 | 85 | 99 | 68.52 | 86.35 | 9.888 | 99 |
| libaom speed 9 | 90 | 99 | 87.36 | 88.32 | 9.928 | 99 |
| SVT-AV1 preset 4 | 30 | 99 | 9.56 | 53.21 | 8.959 | 1 |
| SVT-AV1 preset 4 | 40 | 99 | 16.18 | 65.54 | 9.331 | 12 |
| SVT-AV1 preset 4 | 50 | 99 | 24.39 | 73.53 | 9.571 | 71 |
| SVT-AV1 preset 4 | 60 | 99 | 36.59 | 80.07 | 9.734 | 99 |
| SVT-AV1 preset 4 | 70 | 99 | 46.80 | 83.56 | 9.820 | 99 |
| SVT-AV1 preset 4 | 80 | 99 | 61.03 | 86.73 | 9.891 | 99 |
| SVT-AV1 preset 4 | 85 | 99 | 76.67 | 88.48 | 9.930 | 99 |
| SVT-AV1 preset 4 | 90 | 99 | 92.05 | 89.55 | 9.951 | 99 |
| SVT-AV1 preset 6 | 30 | 99 | 10.21 | 52.98 | 8.965 | 1 |
| SVT-AV1 preset 6 | 40 | 99 | 17.13 | 65.39 | 9.335 | 10 |
| SVT-AV1 preset 6 | 50 | 99 | 25.45 | 73.39 | 9.576 | 75 |
| SVT-AV1 preset 6 | 60 | 99 | 37.59 | 79.91 | 9.736 | 99 |
| SVT-AV1 preset 6 | 70 | 99 | 47.89 | 83.37 | 9.819 | 99 |
| SVT-AV1 preset 6 | 80 | 99 | 62.25 | 86.39 | 9.886 | 99 |
| SVT-AV1 preset 6 | 85 | 99 | 77.55 | 88.14 | 9.924 | 99 |
| SVT-AV1 preset 6 | 90 | 99 | 92.42 | 89.39 | 9.950 | 99 |
| SVT-AV1 preset 9 | 30 | 99 | 11.07 | 52.82 | 8.957 | 1 |
| SVT-AV1 preset 9 | 40 | 99 | 18.28 | 65.04 | 9.329 | 11 |
| SVT-AV1 preset 9 | 50 | 99 | 26.74 | 73.02 | 9.570 | 74 |
| SVT-AV1 preset 9 | 60 | 99 | 39.03 | 79.38 | 9.717 | 99 |
| SVT-AV1 preset 9 | 70 | 99 | 49.43 | 82.96 | 9.805 | 99 |
| SVT-AV1 preset 9 | 80 | 99 | 63.77 | 86.06 | 9.878 | 99 |
| SVT-AV1 preset 9 | 85 | 99 | 79.01 | 88.02 | 9.920 | 99 |
| SVT-AV1 preset 9 | 90 | 99 | 93.45 | 89.25 | 9.947 | 99 |
| NVENC AV1 (p7, `tune uhq`) | 6 | 99 | 132.48 | 81.14 | 9.806 | 85 |
| NVENC AV1 (p7, `tune uhq`) | 12 | 99 | 112.72 | 80.63 | 9.799 | 85 |
| NVENC AV1 (p7, `tune uhq`) | 18 | 99 | 85.55 | 79.08 | 9.770 | 85 |
| NVENC AV1 (p7, `tune uhq`) | 25 | 99 | 54.24 | 75.63 | 9.684 | 83 |
| NVENC AV1 (p7, `tune uhq`) | 32 | 99 | 31.80 | 68.74 | 9.489 | 49 |
| NVENC AV1 (p7, `tune uhq`) | 38 | 99 | 18.36 | 59.06 | 9.215 | 10 |
| NVENC AV1 (p7, `tune uhq`) | 44 | 99 | 10.81 | 46.47 | 8.884 | 1 |
| NVENC AV1 (p7, `tune uhq`) | 50 | 99 | 6.13 | 28.48 | 8.432 | 0 |
| WebP (cwebp `-m 4`) | 50 | 99 | 18.30 | 57.29 | 9.126 | 7 |
| WebP (cwebp `-m 4`) | 60 | 99 | 21.25 | 61.28 | 9.238 | 14 |
| WebP (cwebp `-m 4`) | 70 | 99 | 24.66 | 64.98 | 9.345 | 28 |
| WebP (cwebp `-m 4`) | 80 | 99 | 34.28 | 71.48 | 9.542 | 59 |
| WebP (cwebp `-m 4`) | 90 | 99 | 62.94 | 80.76 | 9.806 | 99 |
| WebP (cwebp `-m 4`) | 95 | 99 | 98.04 | 84.93 | 9.908 | 99 |
| WebP (cwebp `-m 4`) | 98 | 99 | 125.43 | 86.19 | 9.936 | 99 |
| WebP (cwebp `-m 4`) | 100 | 99 | 135.61 | 86.51 | 9.942 | 99 |

## NVENC by image height

In this pipeline, photos with an odd height decode with a chroma error that stays at every CQ (luma is fine on the two checked, `logs/nvenc_shift2.log`); both metrics see it. BD-rate against libaom speed 6, median (mean):

| Photos | count | CVVDP JOD, all points | SSIMULACRA2, published points | reach 9.5 JOD |
|---|---:|---:|---:|---:|
| even height | 76 | +27.2% (+28.0%, 76) | +27.9% (+33.4%, 75) | 75/76 |
| odd height | 23 | +46.6% (+63.0%, 21) | +81.8% (+98.7%, 19) | 10/23 |

## Diagnostic: CVVDP on luma only

Both images converted to grey after the sRGB conversion (CVVDP sees no colour). av1gpu against libaom speed 6: +6.2% median (+6.6% mean, 99 photos), against +3.9% (+4.0%) in colour. By size: < 1 MP +3.5% (39), 1-5 MP +8.7% (20), 5-10 MP +9.0% (23), >= 10 MP +9.6% (17).

## Side check: WebP and ICC profiles

dwebp's PNG drops the ICC profile that `cwebp -metadata icc` stores in the .webp, so SSIMULACRA2 compares an untagged PNG with the tagged source. With the profile put back (312 files of 39 photos with a profile), WebP's SSIMULACRA2 BD-rate against libaom speed 6 is +42.5% (mean +45.1%; published points) instead of +45.1% (mean +61.2%) as published. 112 files change: those of the 13 photos with an Adobe RGB, Camera RGB or Generic RGB profile (by 0.6 to 37 points) and of one with an sRGB-equivalent profile (by 0.01). CVVDP reads the profile from the .webp, so its WebP numbers do not have this. NVENC's pipeline drops the profile from the file itself, so both metrics count that against it.

## Notes

- The encodes are the published files: every compared file has the published size, and this run's SSIMULACRA2 scores equal the published ones to the last digit. The extra points (q85/q90, WebP q98/q100, NVENC CQ 12/6) are new.
- CVVDP and SSIMULACRA2 agree within about a point on libaom speed 9 and on SVT-AV1, and on WebP's median.
- They disagree on av1gpu: +3.9% (+4.0%) on JOD against +0.3% (-0.2%) on SSIMULACRA2 (same points). JOD rates av1gpu worse than SSIMULACRA2 does on 84 of 99 photos (per-photo correlation 0.45); the gap grows with image size, the opposite of SSIMULACRA2's trend. With colour removed, CVVDP puts av1gpu at +6.2%, so the gap is in luma, not chroma.
- 9.5 JOD: libaom speed 6 reaches it on every photo from q70; av1gpu default reaches it on every photo from q80; libaom speed 9 reaches it on every photo from q70; SVT-AV1 preset 4 reaches it on every photo from q60; SVT-AV1 preset 6 reaches it on every photo from q60; SVT-AV1 preset 9 reaches it on every photo from q60; WebP reaches it on every photo from q90 (the published top points already suffice). NVENC never reaches it on 14 photos.
- NVENC: 23 photos have an odd height; in this pipeline they decode with a chroma error that stays at every CQ (luma is fine on the two checked: `scripts/nvenc_shift2.py`, `logs/nvenc_shift2.log`). 13 of the 14 photos on which NVENC never reaches 9.5 JOD are among them; the other (bd4ff3b451c3ba6b_large.jpg) has a Camera RGB profile, which the pipeline drops. On the 76 even-height photos NVENC is +27.2% (JOD) and +27.9% (SSIMULACRA2) against libaom speed 6.
- `standard_fhd` is a 24-inch 1080p screen at 0.6 m (37.8 pixels per degree). The commenter's 27-inch 1080p at about 20 inches is about 28.5 pixels per degree, with larger pixels, so artefacts there are easier to see than here. CVVDP shows image pixels 1:1 on display pixels: a 24 MP photo is judged as if viewed at 100 % zoom, not scaled to the screen.
- On the Deep Learning Base AMI, LD_LIBRARY_PATH holds CUDA 13.2's cuDNN 9.23, which breaks the cuDNN 9.26 that PyTorch 2.15 ships (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED); the runs put the pip cuDNN first on LD_LIBRARY_PATH (scripts/run1.sh). bench.py now runs a small CVVDP self-test at start-up so such a failure stops the run.
