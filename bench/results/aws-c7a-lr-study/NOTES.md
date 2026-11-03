# Loop-restoration study: working notes (agent lr08, 2026-11-03)

Instance: av1gpu-lr08, c7a.16xlarge spot (64 vCPU AMD EPYC 9R14, 123 GB), Canonical Ubuntu 24.04, us-east-1,
launched 05:26 UTC with a dead-man `shutdown -h +180` (terminate on shutdown).

## Steps

1. `scripts/setup.sh`: libaom 3.13.1 + libavif 1.4.0 (`~/opt/bin/avifenc`), libjxl 0.11.1 ssimulacra2, system dav1d
   1.4.1. Images: `bench/fetch_images.py` for the standard 100, the size range 100 (`openimages_v4_gamut.csv`) and
   Kodak (the 24 `photo_kodim*` rows of `bench/palette_study_images.csv`). Done 05:30.
2. Read libaom's speed features (findings below).
3. `scripts/build_lr.sh`: libaom patched with `scripts/patch_lrstudy.py` (env switch `LRSTUDY`) + avifenc against it
   (`~/opt-lr/bin/avifenc`), and `lrstat` (`scripts/lrstat.c`, libaom inspection decoder: per-plane LR type, unit size,
   units by type). Checked: the patched avifenc with LRSTUDY unset gives the same bytes as the stock one.
4. `scripts/lrstudy.py encode`: 13,180 encodes (420: 9 configurations × 224 images × 5 q; 444: 5 configurations ×
   124 images × 5 q), all in one pool, one avifenc -j 1 per vCPU. Started 05:36.
5. `scripts/chain.sh` -> `lrstudy.py score` once the encodes finish: dav1d decode, SSIMULACRA2, RGB PSNR, lrstat.
6. `scripts/analyze_lr.py` -> tables.md, results.json, rows/.

## libaom facts (3.13.1, av1/encoder/speed_features.c, av1_cx_iface.c)

- `enable_restoration` defaults to 1 in every non-realtime usage, all-intra included (av1_cx_iface.c:254;
  realtime forces 0 at :1217).
- All-intra speed >= 5: `disable_wiener_filter = disable_sgr_filter = true` (set_allintra_speed_features_framesize_
  independent, "speed >= 5" block), and then `seq_params->enable_restoration &= (!disable_wiener || !disable_sgr)`.
  So at speed 5 and 6 the sequence header has enable_restoration = 0 and `-a enable-restoration=1` changes nothing
  (checked: identical bytes).
- With -j > 1 (num_workers > 1), speed >= 5 also disables both (for every mode); irrelevant at -j 1.
- All-intra pruning that applies at speed 1-4: speed 1 `dual_sgr_penalty_level = 1`, `enable_sgr_ep_pruning = 1`;
  speed 2 `prune_wiener_based_on_src_var = 1`, `prune_sgr_based_on_wiener = 1`; speed 3
  `prune_sgr_based_on_wiener = 2` (1 for screen content), `reduce_wiener_window_size = 1`,
  `prune_wiener_based_on_src_var = 2`.
- Unit size (qindex dependent, all-intra speed >= 1): 128 if base_qindex <= 96 and min(w, h) < 1440, else 256 (no
  size search). Speed 0 searches 64..256.
- qindex-dependent `disable_loop_restoration_luma` at speed >= 4 applies to non-boosted frames only; key frames are
  boosted, so not to stills.

## First look (one 1 MP JPEG, q60)

s6: 166,418 B, LR off. p6keep: +21 B, luma none, chroma Wiener (3/4 and 4/4 units), +7 % time. p6full: +59 B, luma
self-guided on all 16 units, +47 % time. s4: 166,140 B; s4r1 +21 B (same LR choice as p6keep).

## Encode results (no scores needed)

- 13,180 encodes in 477 s, 0 failures.
- `scripts/identity.sh`: s6r1 vs s6 (stock speed 6, enable-restoration 1 vs 0) 1120/1120 identical.
- CPU (`scripts/cpu_quick.py`), 4:2:0, all 5 q, all 224 photos: s6 1006 s; p6keep 1.159x, p6full 2.010x,
  p6wiener 1.315x, p6sgr 1.680x; s4 5656 s (5.6x s6), s4r1 1.026x of s4, p4full 1.177x of s4.
  q60 only: p6keep 1.194x, p6full 2.020x. 4:4:4: p6keep 1.183x, p6full 2.249x, s4r1 1.029x.

## Scores and later steps

- 05:51 scoring done (446 s, 0 missing scores, 0 missing lrstat). `analyze_lr.py` -> tables.md / results.json / rows.
  SSIMULACRA2 BD-rate, all 224 photos, median: p6keep +0.1 %, p6full -0.0 %, s4r1 vs s4 +0.0 %; huge photos
  p6full -0.5 %. PSNR: p6keep -1.0 %, p6full -1.3 %.
- 05:52 `dectime.py`: dav1d one thread q60, p6keep +4.2 % / p6full +7.2 % on the size range.
- 05:53 `bscore.py` first run with 64 at once: OOM on the huge photos (butteraugli_main ~4 GB at 24 MP; console
  log shows the OOM killer), sshd unreachable. 05:58 rebooted with `aws ec2 reboot-instances`; the user-data dead-man
  does not rerun after a reboot, so re-armed by hand: `sudo shutdown -h 08:26` (checked in
  /run/systemd/shutdown/scheduled). Disk intact. 06:01 bscore.py rerun: 16 at a time for sources >= 5 MP, then 64.
- Side finding sent to the lead: batch mode runs `av1g_gpu_cdef` though only `GP_WANT_RECON` reads its output
  (handed to pipe08).

- 06:10 butteraugli done (5,600 rows, 0 missing): p6keep -0.9 %, p6full -1.6 % median; large/huge -2 to -3.6 %.
- 06:12 CVVDP: cvvdp 0.5.7 + torch 2.15.0+cpu in a venv (`~/cv`; torchvision had to come from the CPU index too).
  One CPU score: 1.9 s at 1 MP, 41.7 s and 9 GB at 17.9 MP. `cv_run.sh` in three size bands with a memory watchdog;
  the > 5 MP band (7 x 9 threads) used only ~3 cores per worker, so it was stopped and rerun wider
  (`cv_run2.sh` 14 x 4, plus `cv_run3.sh` for p6keep); `cvscore.py` skips rows already done. s4 / s4r1 were not
  run above 5 MP (time). Note: `pkill -f` in an ssh command line matches the ssh's own shell; use `stop_cv.sh`.
- 06:33 CVVDP done: all photos -0.0 % (p6keep and p6full); large -0.8 / -1.1 %, huge -0.4 / -0.6 %.
- 06:36 final analyze_lr.py; results copied (tables.md, results.json, rows/, logs/, raw_jsonl.tgz = the raw rows).
- 06:37 teardown.sh: instance terminated, SG and key pair deleted, no volume; spot request sir-vhyqqgcg cancelled.

## Status

Done: docs/LOOP_RESTORATION.md, summary.md. Recommendation: don't build LR (0.0 % median on SSIMULACRA2 and CVVDP;
0.3-1.1 % on photos >= 4 MP; 1-1.6 % on butteraugli / PSNR; est. 3-6 % GPU time, +4-7 % decode).
