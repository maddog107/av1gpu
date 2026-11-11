# v0.13.0 on the NVIDIA L4: speed and file sizes for the docs

Scorer check (ssimulacra2): all 5160 files identical to the published ones (v0.12.0's, whose sizes all equal the stored rows', and v0.13.0's with v0.12.0's md5) score exactly as published. 10 other v0.13.0 files merely have the byte count of a stored row (different bitstreams, md5 unlike v0.12.0's) and score differently (by -0.6641 to +0.1771), as different files do: not a scorer difference.

Scorer check (cvvdp): all 64 files identical to the published ones (v0.12.0's, whose sizes all equal the stored rows', and v0.13.0's with v0.12.0's md5) score exactly as published. 2 other v0.13.0 files merely have the byte count of a stored row (different bitstreams, md5 unlike v0.12.0's) and score differently (by -0.004773 to -0.000164), as different files do: not a scorer difference.

new: `new /home/ubuntu/rel/av1gpu-0.13.0-linux-x86_64/av1gpu d3decbb27a3fd5a2a02dba3e07179d97d5341d26d4177e74c2d8e7ded1badeb9`  
old: `old /home/ubuntu/rel/av1gpu-0.12.0-linux-x86_64/av1gpu bafc8c1d1bf7050e2b7e95ffdcd4d8a15144d3b0718e281458244f62e5d891dc`  
CPU: AMD EPYC 7R13 Processor, 16 vCPUs

## Speed (MP/s, median of 6; av1gpu --stats)

| setting | v0.13.0 | v0.12.0 | change | img/s new | CPU cores new |
|---|---:|---:|---:|---:|---:|
| alpha/default | 43.3 | 40.0 | +8.3 % | 166.9 | 2.80 |
| alpha/lossless | 34.2 | 34.2 | -0.0 % | 131.8 | 1.54 |
| alpha/no_alpha | 54.1 | 47.5 | +14.0 % | 208.6 | 3.35 |
| alpha/q100 | 40.2 | 36.9 | +9.0 % | 154.9 | 2.70 |
| formats/420_10 | 326.8 | 291.1 | +12.3 % | 68.4 | 0.35 |
| formats/444_10 | 241.9 | 224.6 | +7.7 % | 50.6 | 0.27 |
| formats/444_8 | 248.0 | 228.3 | +8.6 % | 51.9 | 0.27 |
| jpeg/size_range/default_enc2 | 295.3 | 266.6 | +10.7 % | 61.8 | 0.32 |
| jpeg/size_range/default_enc3 | 325.4 | 292.7 | +11.2 % | 68.1 | 0.34 |
| jpeg/size_range/default_enc4 | 337.4 | 299.7 | +12.6 % | 70.6 | 0.36 |
| jpeg/size_range/nosmall_enc3 | 370.1 | 340.0 | +8.9 % | 77.5 | 0.39 |
| jpeg/size_range/nosmall_enc4 | 379.1 | 349.3 | +8.5 % | 79.3 | 0.40 |
| jpeg/size_range/tiles211_enc3 | 290.7 | 263.6 | +10.3 % | 60.8 | 0.31 |
| jpeg/size_range/tiles233_enc3 | 339.4 | 304.0 | +11.6 % | 71.0 | 0.37 |
| jpeg/size_range/turbo3_enc4 | 387.9 | 371.0 | +4.6 % | 81.2 | 0.41 |
| jpeg/size_range/turbo3ns_enc4 | 459.7 | 459.1 | +0.1 % | 96.2 | 0.48 |
| jpeg/standard/default_enc2 | 108.4 | 86.2 | +25.8 % | 137.7 | 0.20 |
| jpeg/standard/default_enc3 | 113.7 | 90.2 | +26.0 % | 144.4 | 0.20 |
| jpeg/standard/default_enc4 | 114.3 | 89.7 | +27.4 % | 145.2 | 0.20 |
| jpeg/standard/nosmall_enc3 | 236.8 | 217.4 | +8.9 % | 300.8 | 0.30 |
| jpeg/standard/nosmall_enc4 | 237.4 | 217.2 | +9.3 % | 301.6 | 0.31 |
| kodak/default | 84.4 | 69.9 | +20.6 % | 214.6 | 4.37 |
| kodak/nosmall | 193.9 | 180.6 | +7.3 % | 493.1 | 8.97 |
| kodak/turbo3_nosmall | 275.8 | 272.9 | +1.1 % | 701.4 | 12.94 |
| lossless/cmp_1pass | 37.3 | 38.2 | -2.5 % | 38.8 | 1.33 |
| lossless/cmp_1pass_ycgco | 38.9 | 39.4 | -1.3 % | 40.4 | 1.42 |
| lossless/cmp_4x | 44.7 | 45.6 | -1.9 % | 46.5 | 1.47 |
| lossless/cmp_4x_ycgco | 44.2 | 44.6 | -1.0 % | 45.9 | 1.35 |
| thumbs/default | 94.8 | 75.5 | +25.5 % | 625.2 | 1.54 |
| thumbs/nosmall | 202.4 | 186.3 | +8.6 % | 1334.1 | 1.59 |

## File size: median BD-rate on SSIMULACRA2 (mean in brackets), 100 size-range photos

| av1gpu | v0.13.0 | v0.12.0 | published (tiles4x4) |
|---|---:|---:|---:|
| default vs libaom speed 6 | +0.46 % (+0.23 %) | +0.08 % (−0.13 %) | +0.08 % (−0.13 %) |
| `--small-preset none` | +2.49 % (+1.69 %) | +2.23 % (+1.36 %) | +2.23 % (+1.36 %) |
| turbo3 nosmall | +5.74 % (+5.63 %) | +5.74 % (+5.63 %) | +5.74 % (+5.63 %) |
| turbo3 | +1.55 % (+2.18 %) | +1.25 % (+1.96 %) | +1.25 % (+1.96 %) |
| default vs one-tile libaom speed 6 | +1.15 % (+0.87 %) | +0.72 % (+0.50 %) | +0.72 % (+0.50 %) |
| default vs WebP | −30.49 % (−30.60 %) | −30.60 % (−30.80 %) | −30.60 % (−30.80 %) |
| default vs NVENC AV1 | −22.54 % (−26.28 %) | −22.74 % (−26.55 %) | −22.74 % (−26.55 %) |
| 420 / 8-bit vs libaom speed 6, same format | +0.46 % (+0.23 %) | +0.08 % (−0.13 %) | +0.08 % (−0.13 %) |
| 444 / 8-bit vs libaom speed 6, same format | +1.05 % (+0.96 %) | +0.40 % (+0.56 %) | +0.40 % (+0.56 %) |
| 420 / 10-bit vs libaom speed 6, same format | +0.70 % (+0.22 %) | +0.27 % (−0.20 %) | +0.27 % (−0.20 %) |
| 444 / 10-bit vs libaom speed 6, same format | +0.82 % (+0.69 %) | +0.35 % (+0.36 %) | +0.35 % (+0.36 %) |
| CVVDP (JOD), default vs libaom speed 6 (99 photos, 8 points) | +4.22 % (+4.51 %) | — | +4.16 % (+4.29 %) |

By size, default vs libaom speed 6: < 1 MP +0.5 % (n 40), 1-5 MP +2.1 % (n 20), 5-10 MP +0.5 % (n 23), >= 10 MP −1.9 % (n 17)

Standard set (100 photos) vs aws-g6-l4's libaom speed 6: av1gpu_vs_aom_s6 +1.0 %, av1gpu_nosmall_vs_aom_s6 +3.8 %, av1gpu_turbo3_nosmall_vs_aom_s6 +8.2 %, av1gpu_vs_webp −29.4 %

## Derived

- default (GP_ENCODERS=4) / libaom speed 6 (16 threads): 11.74×
- default / NVENC AV1 best: 21.55×; turbo3 nosmall / SVT-AV1 p9 2.51×, / libaom speed 9 2.45×
- vs SVT-AV1 interpolated at av1gpu's file size: default 10.15×, nosmall 7.11×, turbo3ns 4.78×
- q60 totals (99 photos): 22,529,390 bytes, mean SSIMULACRA2 71.94

## v0.13.0 against v0.12.0, per config (bytes of the common files; BD-rate NEW vs OLD on SSIMULACRA2)

| config | identical | bytes | BD-rate | < 1 MP | 1-2 MP | ≥ 2 MP identical |
|---|---:|---:|---:|---:|---:|---:|
| gamut_default | 0/800 | +0.00 % | +0.23 % (+0.35 %) | 0/320 | 0/64 | 0/416 |
| gamut_nosmall | 0/600 | +0.02 % | +0.21 % (+0.31 %) | 0/240 | 0/48 | 0/312 |
| gamut_turbo3ns | 600/600 | +0.00 % | +0.00 % (+0.00 %) | 240/240 | 48/48 | 312/312 |
| gamut_turbo3 | 360/600 | -0.00 % | +0.00 % (+0.22 %) | 0/240 | 48/48 | 312/312 |
| fmt_444_8 | 0/600 | +0.08 % | +0.26 % (+0.38 %) | 0/240 | 0/48 | 0/312 |
| fmt_420_10 | 0/600 | +0.00 % | +0.30 % (+0.41 %) | 0/240 | 0/48 | 0/312 |
| fmt_444_10 | 0/600 | +0.06 % | +0.26 % (+0.32 %) | 0/240 | 0/48 | 0/312 |
| std_default | 0/600 | -0.05 % | +0.39 % (+0.46 %) | 0/516 | 0/84 | 0/0 |
| std_nosmall | 0/600 | +0.14 % | +0.33 % (+0.45 %) | 0/516 | 0/84 | 0/0 |
| std_turbo3ns | 600/600 | +0.00 % | +0.00 % (+0.00 %) | 516/516 | 84/84 | 0/0 |
| ll_jpeg_gamut | 0/0 | — | — | 0/0 | 0/0 | 0/0 |
| ll_cmp | 51/51 | +0.00 % | — | 30/30 | 14/14 | 7/7 |
| ll_cmp_ycgco | 51/51 | +0.00 % | — | 30/30 | 14/14 | 7/7 |
| ll_misc | 13/13 | +0.00 % | — | 12/12 | 0/0 | 1/1 |
| ll_jpeg_misc | 0/0 | — | — | 0/0 | 0/0 | 0/0 |
| alpha_default | 2/43 | +0.25 % | — | 2/38 | 0/4 | 0/1 |
| alpha_q100 | 2/43 | +0.10 % | — | 2/38 | 0/4 | 0/1 |
| alpha_lossless | 42/42 | +0.00 % | — | 37/37 | 4/4 | 1/1 |

## Checks

- sizes new/gamut_default vs the stored rows: 3/800 same size
- sizes old/gamut_default vs the stored rows: 800/800 same size
- sizes new/gamut_nosmall vs the stored rows: 3/600 same size
- sizes old/gamut_nosmall vs the stored rows: 600/600 same size
- sizes new/gamut_turbo3ns vs the stored rows: 600/600 same size
- sizes old/gamut_turbo3ns vs the stored rows: 600/600 same size
- sizes new/gamut_turbo3 vs the stored rows: 362/600 same size
- sizes old/gamut_turbo3 vs the stored rows: 600/600 same size
- sizes new/fmt_444_8 vs the stored rows: 1/600 same size
- sizes old/fmt_444_8 vs the stored rows: 600/600 same size
- sizes new/fmt_420_10 vs the stored rows: 0/600 same size
- sizes old/fmt_420_10 vs the stored rows: 600/600 same size
- sizes new/fmt_444_10 vs the stored rows: 1/600 same size
- sizes old/fmt_444_10 vs the stored rows: 600/600 same size
- sizes new/std_default vs the stored rows: 4/600 same size
- sizes old/std_default vs the stored rows: 600/600 same size
- sizes new/std_nosmall vs the stored rows: 5/600 same size
- sizes old/std_nosmall vs the stored rows: 600/600 same size
- scorer ssimulacra2: 5160 files identical to the published ones, 5160 equal, max |diff| 0.0; 10 other bitstreams of a stored row's byte count
- scorer cvvdp: 64 files identical to the published ones, 64 equal, max |diff| 0.0; 2 other bitstreams of a stored row's byte count
- inputs vs the v0.9.0 check: data/rgba 39/39; images 100/100; images_std 100/100; ll/cmp 39/51; ll/graphics 15/27; ll/gray 5/5; ll/jpeg 25/25; ll/kodak 24/24; ll/misc 13/13; ll/png16_10 8/8; ll/png16_8 3/3; ll/png16_full 2/2; ll/webp 4/4; sets/gray 9/9; sets/mixed.txt 0/0; sets/mono 47/47; sets/odd 30/30; sets/single 10/10; sets/thumbs 280/280; sets/webp 7/7; st 5/5

## Decoder gates (NEW, tests/decoder_gate.py, 100 size-range photos)

- default_q60_120_180: [gate] q60: 100/100 identical in dav1d and aomdec (420 100; 49396235 bytes), 0 not encoded (unsupported input), rc=0 PASS
- default_q60_120_180: [gate] q120: 100/100 identical in dav1d and aomdec (420 100; 22548495 bytes), 0 not encoded (unsupported input), rc=0 PASS
- default_q60_120_180: [gate] q180: 100/100 identical in dav1d and aomdec (420 100; 7231302 bytes), 0 not encoded (unsupported input), rc=0 PASS
- turbo3_nosmall_q120: [gate] q120: 100/100 identical in dav1d and aomdec (420 100; 23197757 bytes), 0 not encoded (unsupported input), rc=0 PASS
- yuv444_10_q120: [gate] q120: 100/100 identical in dav1d and aomdec (444 10-bit 100; 23156313 bytes), 0 not encoded (unsupported input), rc=0 PASS

## Cold start, one process per image (ms, median of 12)

| input | v0.13.0 | v0.12.0 | v0.12.0 again |
|---|---:|---:|---:|
| photo.jpg | 380.4 | 361.4 | 360.5 |
| big.jpg | 458.2 | 463.9 | 454.7 |
| thumb512.jpg | 366.6 | 346.9 | 342.8 |
| t320.jpg | 381.9 | 365.4 | 363.8 |
| pair batch (photo.jpg + thumb512.jpg) | 593.1 | 564.8 | 562.5 |

## RGBA set (39 PNGs)

- new/alpha_default: 687,225 bytes (alpha 91,344), alpha exact 7/39
- new/alpha_q100: 1,073,452 bytes (alpha 477,571), alpha exact 39/39
- new/alpha_lossless: 7,489,058 bytes (alpha 483,094), alpha exact 39/39
- old/alpha_default: 685,414 bytes (alpha 90,611), alpha exact 7/39
- old/alpha_q100: 1,072,374 bytes (alpha 477,571), alpha exact 39/39
- old/alpha_lossless: 7,489,058 bytes (alpha 483,094), alpha exact 39/39

