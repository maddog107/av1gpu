# ent08 working notes (GPU entropy + loop filters, 2026-11-03)

Instance: av1gpu-ent08, g6.4xlarge on-demand (spot had no capacity in any subnet), i-0f3efc1bdc83a8fa8, launched
05:26 UTC, dead-man `shutdown -h +180` (terminate). DLAMI base-oss-nvidia-driver-gpu-ubuntu-24.04, driver 595.112.07,
CUDA 13.2; nsys from apt (cuda-nsight-systems-13-2), cuobjdump/nvdisasm from apt.

Trees on the instance: ~/main (cde1f74, tests ON, the reference), ~/br (commit 1), ~/br2 (commits 1 + 2), all built
for sm_89 with the same toolkit; br / br2 link main's k_recon objects (AV1GPU_RECON_OBJECTS; k_recon is untouched).
~/exp, ~/expp: experiment builds (exp_patch.py / prof_patch.py in the session scratchpad, never committed).

## Timeline
- 05:26 instance up; setup.sh (deps, main build 183 s, sets: size range, standard, ll, sets, palette sets).
- 05:35-05:45 profiling: nsys on size range / std x3 / thumbs x3 / one 8 MP single file (nsys_an.py).
- 05:39-05:45 speed-of-light experiment (sol.sh): entropy work removed: sr +2-4 %, std / thumbs 0; k_select_small
  removed: within noise everywhere. So the batch speed is not bound by entropy or the CDEF search; the entropy
  tail (last batch) is the visible part.
- 05:50-06:07 k_entropy_win: per-symbol profile (AV1G_ENTROPY_PROFILE build + a reader in run_batch, prof_patch.py),
  ncu source counters; coder thread 112-115 cycles/symbol = the bottleneck of photo tiles.
  Tried: tree select in win_group (A slower: dropped), unrolled single coder loop + branch hint (no change: dropped),
  coder split into range thread + low thread (kept), producer prefetch of the next group member (kept).
- 06:10 commit 1 (637f113); identity (identity8.sh) + checks (checks8.sh) started.
- 06:20 commit 2 (a8d2969, single-file upload without the host copy); br2 identity (identity8b.sh = identity8.sh with
  ~/work2) + gpu_fuzz_test started; speed8.sh armed (starts when identities and checks are done).

## State / to resume
- Results on the instance in ~/results (identity.txt, identity_br2.txt, gate.txt, ctest.txt, sanitizer.txt,
  fuzz_br2.txt, spd_*.txt, speed.log). Copy into logs/ here.
- Known set gaps: ~/sets/gray has only the 4 gray + alpha PNGs (ll/gray's gray PNGs are missing on this instance:
  mk_ll_sets.py made none), so single/g0.png fails to decode in every binary (main too).
- `pal_flat_jpg_lossless` encodes nothing: av1gpu refuses JPEGs with --lossless (both binaries; the gate counts them
  as unsupported).

## Progress 06:20-06:40
- 06:20 br2's full identity stopped after 8 sets (too slow next to br1's run); br2 = br1 in batch mode (batch mode
  never calls run_batch), so br2 gets the single-file identity (identity_single.sh) + fuzz + --check-entropy.
- The set-up's mk_ll_sets.py had failed (it reads ~/br, which did not exist yet): only ll/kodak was there.
  fix_llsets.sh made the rest; rerun8.sh re-ran the affected sets (mono 47, gray 9, misc, png16_10, graphics, WebP,
  ll/jpeg) after identity8.sh: all identical.
- Identity: 75 batch runs + 6 single-file runs, 5,052 files, 0 differ (logs/identity_table.md).
- Gate all PASS (flat_jpg --lossless: nothing to encode); ctest 9/9; gpu_fuzz_test 0 mismatches; memcheck /
  synccheck 0 errors; racecheck: only the ring / counter hand-offs in k_entropy_win (logs/racecheck_summary.txt).
- 06:29 speed8.sh started (ab: main vs br2; noise: main vs main copy; single_br1).
  First numbers: sr 228.9 -> 225.8 MP/s (B/A 0.987), std 73.9 -> 74.4 (1.007). Noise run pending.

## Final (06:57)
- Speed (logs/speed_analysis.txt): batch sets within noise (sr 0.987 / per round 0.993, 1.001; std 1.007; thumbs
  1.000; noise run 1.002 / 0.996 / 0.997); single 8 MP cold -15.8 ms (0.987, noise 1.003); commit 1 alone in single
  mode +0.4 % (noise). k_entropy_win alone in single-file mode: 8 MP 18.43 -> 15.37 ms, 1.8 MP one tile 26.47 ->
  24.11 ms (logs/prof/kern_single.txt). nsys batch sr: k_entropy_win total unchanged (825 ms), tail 88 -> 84 ms.
- Ideas not done: the producer is now the limit (83-128 cycles/symbol); the 16-entry groups (eob_pt, modes,
  partitions) and the divergence between the 4- and 16-entry paths in one window are the next target. A 1-2 MP
  one-tile photo is producer-bound (26 ms of k_entropy_win in single-file mode). k_select_small could run in one
  warp-synchronous pass for small frames (thumbnails: 0.21 ms / image of serial latency) but showed no throughput
  effect in the speed-of-light run.
