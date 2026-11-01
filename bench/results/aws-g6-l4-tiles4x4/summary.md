# 4x4-tile default: av1gpu re-encoded (NVIDIA L4)

av1gpu 50b62cb (perf/tiles-4x4: 4x4 tiles for images of 2 MP and more in every turbo preset) against the stored CPU baselines, 100 size-range photos unless noted. Median BD-rate on SSIMULACRA2 (mean in brackets); v0.3.0 (2x2 tiles; turbo3 4x2) for comparison.

| av1gpu | 4x4 (new default) | v0.3.0 |
|---|---:|---:|
| default vs libaom speed 6 | +0.1 % (−0.1 %) | −0.1 % (−0.4 %) |
| `--small-preset none` | +2.2 % (+1.4 %) | +2.1 % (+1.1 %) |
| `--preset turbo3 --small-preset none` | +5.7 % (+5.6 %) | +5.6 % (+5.4 %) |
| `--preset turbo3` | +1.2 % (+2.0 %) | +1.1 % (+1.8 %) |
| default vs one-tile libaom speed 6 (99 photos) | +0.7 % (+0.5 %) | +0.5 % (+0.2 %) |
| default vs WebP (ICC re-scored) | −30.6 % (−30.8 %) | −30.7 % (−31.0 %) |
| default vs NVENC AV1 (96 photos) | −22.7 % (−26.5 %) | −22.8 % (−26.8 %) |
| CVVDP (JOD), default vs libaom speed 6 (99 photos, 8 points) | +4.2 % (+4.3 %) | +3.9 % (+4.0 %) |
| 420 / 8-bit vs libaom speed 6 in the same format | +0.1 % (−0.1 %) | −0.1 % (−0.4 %) |
| 444 / 8-bit vs libaom speed 6 in the same format | +0.4 % (+0.6 %) | +0.3 % (+0.3 %) |
| 420 / 10-bit vs libaom speed 6 in the same format | +0.3 % (−0.2 %) | +0.1 % (−0.5 %) |
| 444 / 10-bit vs libaom speed 6 in the same format | +0.4 % (+0.4 %) | +0.0 % (+0.1 %) |

Byte identity with v0.3.0 (files below / at or above 2 MP):

- gamut_default: below 2 MP 384/384 identical; 2 MP and more 0/416 identical, +437,207 bytes
- gamut_nosmall: below 2 MP 288/288 identical; 2 MP and more 0/312 identical, +301,727 bytes
- gamut_turbo3ns: below 2 MP 288/288 identical; 2 MP and more 0/312 identical, +213,408 bytes
- gamut_turbo3: below 2 MP 288/288 identical; 2 MP and more 0/312 identical, +213,408 bytes
- fmt_444_8: below 2 MP 288/288 identical; 2 MP and more 0/312 identical, +305,523 bytes
- fmt_420_10: below 2 MP 288/288 identical; 2 MP and more 0/312 identical, +324,190 bytes
- fmt_444_10: below 2 MP 288/288 identical; 2 MP and more 0/312 identical, +321,917 bytes
- std_default: below 2 MP 600/600 identical; 2 MP and more 0/0 identical, +0 bytes
- std_nosmall: below 2 MP 600/600 identical; 2 MP and more 0/0 identical, +0 bytes
- std_turbo3ns: below 2 MP 600/600 identical; 2 MP and more 0/0 identical, +0 bytes
- ll_jpeg_gamut: below 2 MP 32/32 identical; 2 MP and more 31/31 identical, +0 bytes
- ll_cmp: below 2 MP 44/44 identical; 2 MP and more 7/7 identical, +0 bytes
- ll_cmp_ycgco: below 2 MP 44/44 identical; 2 MP and more 7/7 identical, +0 bytes
- ll_misc: below 2 MP 12/12 identical; 2 MP and more 1/1 identical, +0 bytes
- ll_jpeg_misc: below 2 MP 16/16 identical; 2 MP and more 3/3 identical, +0 bytes
- alpha_default: below 2 MP 42/42 identical; 2 MP and more 0/1 identical, +446 bytes
- alpha_q100: below 2 MP 42/42 identical; 2 MP and more 0/1 identical, +235 bytes
- alpha_lossless: below 2 MP 41/41 identical; 2 MP and more 1/1 identical, +0 bytes

Decoder gates (NEW, `tests/decoder_gate.py`, 100 size-range photos):

- default_q60_120_180: [gate] q60: 100/100 identical in dav1d and aomdec (420 100; 49410801 bytes), 0 not encoded (unsupported input), rc=0 PASS
- default_q60_120_180: [gate] q120: 100/100 identical in dav1d and aomdec (420 100; 22538643 bytes), 0 not encoded (unsupported input), rc=0 PASS
- default_q60_120_180: [gate] q180: 100/100 identical in dav1d and aomdec (420 100; 7228580 bytes), 0 not encoded (unsupported input), rc=0 PASS
- turbo3_nosmall_q120: [gate] q120: 100/100 identical in dav1d and aomdec (420 100; 23197757 bytes), 0 not encoded (unsupported input), rc=0 PASS
- yuv444_10_q120: [gate] q120: 100/100 identical in dav1d and aomdec (444 10-bit 100; 23139619 bytes), 0 not encoded (unsupported input), rc=0 PASS

