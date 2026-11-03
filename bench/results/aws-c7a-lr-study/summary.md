# Loop-restoration study (AWS c7a.16xlarge, 2026-11-03)

What AV1 loop restoration (Wiener and self-guided filters after CDEF) would be worth for av1gpu's photos, measured with
libaom. The write-up and recommendation are in [docs/LOOP_RESTORATION.md](../../../docs/LOOP_RESTORATION.md). This
directory holds the method, the tables (`tables.md` and `results.json`, from `scripts/analyze_lr.py`), per-image
BD-rates (`rows/bd_per_image.csv`), every encode's row (`rows/rows.csv.gz`: bytes, CPU time, SSIMULACRA2, PSNR,
LR use), the butteraugli and CVVDP rows, the run logs, and the scripts. There are no images.

## Environment

- c7a.16xlarge spot: 64 vCPU AMD EPYC 9R14, 123 GB, Ubuntu 24.04.5, kernel 7.0.0-1014-aws, gcc 13.3.0
  (`env.txt`). No GPU: this is a libaom-only study.
- libaom v3.13.1 + libavif v1.4.0 (avifenc / avifdec), dav1d 1.4.1 (Ubuntu), libjxl v0.11.1 ssimulacra2 and
  butteraugli_main (`scripts/setup.sh`). These are the versions of the stored baselines.
- A second libaom 3.13.1, patched with `scripts/patch_lrstudy.py`, with its own avifenc: environment switch
  `LRSTUDY`. Unset, the encoder is unchanged: the files are byte-identical to the stock build's.
  - `keep`: undoes the all-intra speed >= 5 rule that disables both LR filters.
  - `full`: also resets every LR pruning speed feature and searches the unit size, as at speed 0.
  - `wiener` / `sgr`: the full search with one filter only.
- `scripts/lrstat.c`: libaom's inspection decoder (as the palette study's `pstat.c`). Per plane, it prints the
  frame's LR type, the unit size, and the units by type.

## Method

- **Images.** `bench/fetch_images.py`: the standard 100 (`openimages_v4_validation_100.csv`), the size range 100
  (`openimages_v4_gamut.csv`, 20 per size class) and Kodak (the 24 `photo_kodim*` rows of
  `bench/palette_study_images.csv`).
- **Encoding.** `scripts/lrstudy.py encode`: `avifenc -q Q -y 420 -j 1 -s S -a tune=iq -a enable-restoration=R
  --ignore-exif --ignore-xmp`, all-intra, as `bench/bench.py` runs libaom, with one avifenc per vCPU.
  - All 13,180 jobs ran in one pool, largest first, so every configuration ran under the same machine load.
  - Q = 30, 45, 60, 75, 90; q60 is qindex 120, av1gpu's default.
  - 4:2:0: 9 configurations x 224 photos x 5 Q. 4:4:4: 5 configurations x 124 photos (standard + Kodak) x 5 Q.
  - Encode CPU time is each avifenc's rusage (user + system).
- **Scoring.** `scripts/lrstudy.py score`:
  - `avifdec -c dav1d` to PNG, then SSIMULACRA2 and RGB PSNR against the source;
  - butteraugli 3-norm (`scripts/bscore.py`) and CVVDP (`scripts/cvscore.py`, cvvdp 0.5.7 on the CPU, display
    standard_fhd, through `bench.py`'s `Cvvdp`) for s6, p6keep, p6full, s4 and s4r1;
  - lrstat on every file.
- **BD-rate.** `bench.py`'s `bd_rate`: PCHIP of log size against the metric, per image over the 5 points, with
  an overlap of at least 2 SSIMULACRA2 points, 0.2 dB PSNR, 0.1 butteraugli or 0.05 JOD. Then the median (and
  mean) over images. Negative means smaller at equal quality.
- **Decoder time.** `scripts/dectime.py`: dav1d `--threads 1`, the q60 files, best of 3 runs in rotating order.

Configurations: s6 / s6r1 (stock speed 6, enable-restoration 0 / 1), p6keep, p6full, p6wiener, p6sgr (patched
speed 6), s4 / s4r1 (stock speed 4, 0 / 1), p4full (patched speed 4).

## Results

**libaom's default.** At all-intra speed 5 and 6, LR is off in the sequence header, whatever `enable-restoration`
says (`speed_features.c`). `scripts/identity.sh`: s6r1 against s6 gives **1120/1120 identical** files. At speed 4
LR is on (s4r1: LR in some plane in 1,024 of 1,120 files).

**BD-rate on SSIMULACRA2**, 4:2:0, median (mean):

| Images | n | p6keep vs s6 | p6full vs s6 | p6wiener vs s6 | p6sgr vs s6 | s4r1 vs s4 | s4 vs s6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| All photos | 224 | +0.1 % (−0.0) | −0.0 % (−0.1) | −0.0 % (−0.1) | +0.3 % (+0.2) | +0.0 % (+0.0) | −5.9 % (−6.2) |
| Standard | 100 | +0.1 % (+0.0) | −0.0 % (−0.0) | −0.0 % (−0.1) | +0.3 % (+0.3) | +0.1 % (+0.1) | −5.6 % (−6.1) |
| Kodak | 24 | +0.3 % (+0.3) | +0.3 % (+0.3) | +0.1 % (+0.2) | +0.6 % (+0.7) | +0.1 % (+0.1) | −4.7 % (−4.9) |
| Size range | 100 | −0.1 % (−0.1) | −0.2 % (−0.3) | −0.1 % (−0.2) | +0.1 % (−0.0) | −0.0 % (−0.0) | −6.5 % (−6.7) |
| tiny | 20 | +0.2 % (+0.2) | +0.2 % (+0.2) | −0.0 % (+0.1) | +0.4 % (+0.4) | +0.1 % (+0.1) | −4.4 % (−4.6) |
| small | 20 | +0.1 % (+0.1) | +0.1 % (+0.0) | +0.0 % (−0.0) | +0.3 % (+0.2) | +0.1 % (+0.1) | −7.5 % (−6.5) |
| typical | 20 | −0.1 % (−0.1) | −0.3 % (−0.2) | −0.1 % (−0.2) | +0.2 % (+0.2) | −0.1 % (−0.0) | −5.8 % (−6.7) |
| large | 20 | −0.3 % (−0.4) | −0.6 % (−0.8) | −0.4 % (−0.6) | −0.3 % (−0.5) | −0.1 % (−0.2) | −7.5 % (−7.5) |
| huge | 20 | −0.3 % (−0.4) | −0.5 % (−0.7) | −0.4 % (−0.5) | −0.3 % (−0.4) | −0.1 % (−0.2) | −8.0 % (−8.2) |

**On RGB PSNR** (all photos, median): p6keep −1.0 %, p6full −1.3 %, p6wiener −1.0 %, p6sgr −0.9 %, s4r1 vs s4
−0.7 %. The huge photos: p6full −2.9 %.

**On butteraugli** (3-norm, s6 / p6keep / p6full / s4 / s4r1, all photos, median): p6keep −0.9 %, p6full −1.6 %,
s4r1 vs s4 −0.3 %. Large: −2.7 % / −3.6 %; huge: −2.1 % / −3.0 %.

**On CVVDP** (JOD, standard_fhd, the same configurations, all photos; speed 4 only below 5 MP), median: p6keep
−0.0 %, p6full −0.0 %, s4r1 vs s4 +0.1 % (184 photos). Large: −0.8 % / −1.1 %; huge: −0.4 % / −0.6 %.
`scripts/cvscore.py` ran CVVDP on the CPU (torch 2.15.0+cpu), in three size bands. The band of photos over 5 MP
was restarted wider (`cv_run2.sh` + `cv_run3.sh`), skipping rows already done; duplicated rows have identical
inputs, and the analysis keeps one per file.

**4:4:4** (124 photos, SSIMULACRA2): p6keep +0.0 %, p6full −0.0 %, s4r1 vs s4 +0.0 %. PSNR: −1.8 %, −2.3 %, −1.7 %.

**q60** (all photos), against s6:
- p6keep: bytes +0.06 %, SSIMULACRA2 76.02 → 76.04, equal-quality size −0.01 % (median);
- p6full: bytes +0.10 %, SSIMULACRA2 76.02 → 76.09, equal-quality size −0.17 %;
- the huge photos: −0.58 % (p6keep) and −1.09 % (p6full).

**LR use** (p6keep, 4:2:0): 1,030 of 1,120 files use LR in some plane (luma 953, U 646, V 627). The luma units
are 20,190 Wiener, 7,883 self-guided and 18,693 off. The unit is 256, or 128 at q75 and q90 below 1440p.

**Encode CPU** (4:2:0, all photos and Q):
- s6: 1,006 s;
- p6keep: 1.159x, p6full: 2.010x, p6wiener: 1.315x, p6sgr: 1.680x;
- s4: 5,656 s (5.6x s6); s4r1: 1.026x s4; p4full: 1.177x s4.

**Decoder** (dav1d, one thread, q60), against s6:
- size range: 3.95 s; p6keep +4.2 %, p6full +7.2 %;
- standard set: 0.89 s; p6keep +3.6 %, p6full +5.4 %.

## Notes

- The first butteraugli run (64 at once) ran the instance out of memory on the huge photos: about 4 GB per process
  at 24 MP. The instance was rebooted, the dead-man shutdown was re-armed (`shutdown -h 08:26`), and
  `scripts/bscore.py` was rerun with 16 at a time for sources of 5 MP and more. No other data was affected; the
  encodes and scores were already on disk.
- `NOTES.md` is the working log.
