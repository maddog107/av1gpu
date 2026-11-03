# rc09 notes (working log; summary.md has the results)

Agent rc09, branch `docs/v090-check`: the release check of main c553ffb (v0.8.0 + #56 #57 #58 #59) before v0.9.0.

## Binaries

- A13 / A12: v0.8.0's Linux release tarballs (CUDA 13.4 / 12.9), downloaded on the instances, `sha256sum -c` against
  the release's `.sha256` files: OK on both instances (ad98b846... and 30d50f04...).
- B13 / B12: main c553ffb's packages from the manual Release run 41596172304 (`av1gpu-dev-c553ffb-linux-x86_64`,
  `-cuda12`), downloaded with `gh run download`, copied with scp.
- A13c: a copy of A13 at a second path, for the noise runs.

## Instances

- `av1gpu-rc09l4`: g6.4xlarge on demand (no spot capacity in any subnet), us-east-1a subnet, NVIDIA L4, driver
  595.112.07, CUDA 13.2 tools (compute-sanitizer 2026.1.0), launched 07:31 UTC.
- `av1gpu-rc09bw`: g7.4xlarge spot, NVIDIA RTX PRO 4500 Blackwell Server Edition (sm_120), driver 595.112.07,
  launched 07:32 UTC. cuobjdump / nvdisasm 13.4 from apt (cuda-cuobjdump-13-4, cuda-nvdisasm-13-4): the DLAMI has none.
- Dead-man shutdown +240 min on both (user data).

## Progress (UTC)

- 07:35 setup on both (sets as speed-v06 + palette study; inputs.sha256 on each: all 959 inputs identical on the two
  instances); 07:43 A13 full list on the L4 (1,010 s), 07:45 on the Blackwell (939 s); A12 subset on the L4.
- 07:54 L4 dead-man re-armed: `shutdown -h +300` (12:54 UTC), since the full speed list runs there.
- 08:04 v0.8.0 across GPUs: A13 on the L4 against A13 on the RTX PRO 4500, 103/103 runs, 5,679 / 5,679 files identical;
  A13 against A12 on the L4 (subset): 27/27 runs, 1,220 files identical.
- post.sh (both instances) waits for the B tarballs, then identity, gate, speed, sanitizers.
- 08:16 run 41596172304's Linux packages done; 08:17-08:21 downloaded one at a time into an empty scratch dir,
  `shasum -c` OK, scp to both, local copies deleted. On the instances `sha256sum -c` OK again:
  c77e109e... `av1gpu-dev-c553ffb-linux-x86_64.tar.gz`, 11e9d401... `-cuda12`. 08:22 post.sh started on both.
- 08:24 resusage9.py (Blackwell instance, cuobjdump/nvdisasm 13.4 on the binaries): sm_120 k_recon stack frames
  shrink (1792 -> 1632 B for <1,8>), rc_luma's static LDL count drops 17 %; sm_89 stack grows 3104 -> 3312 B.
- 08:34 L4 identity: A13 -> B13 103/103 runs, 5,679 / 5,679 files; A12 -> B12 27/27, 1,220; B13 -> B12 27/27, 1,220.
- 08:38 Blackwell: A13 -> B13 103/103, 5,679 / 5,679. B13 on the L4 against B13 on the RTX PRO 4500 (SHA-256
  manifests): 103/103, 5,679 / 5,679 identical (v0.8.0 likewise).
- 08:35-08:41 gate on both: 18 runs each, 691/691 encodes + 35/35 alpha items, PASS. (During the L4's cold-start
  timing, 08:36-08:39, I ran the JSON-vs-JSON cross-GPU compare and an scp on the L4: < 1 s of CPU; noise run unaffected.)
- 08:39 L4 cold single (median of 12): photo 358.7 vs 1059.4 ms, big 467.1 vs 1214.1, thumb512 349.5 vs 1059.9,
  t320 376.7 vs 1098.4, pair batch 575.3 vs 1235.3; v0.8.0 with GP_MAX_CONNECTIONS=1 543 / 701 / 554 / 579 / 722.
- 08:44 Blackwell cold single: 506 / 578 / 498 / 523 / 677 against 1027 / 1139 / 1036 / 1059 / 1190 (1q: 629 / 746 /
  643 / 660 / 789). v0.9.0 is slower on the 4500 than on the L4 (506 vs 359 ms): modload.sh to look at module loads.
- 08:56 Blackwell batch: size range +6.9 % (noise +0.5), standard +4.9 (+0.1), thumbs +7.1 (-0.1), 444_10 +10.9 (0.0).
- 09:00 L4 batch: size range +9.1 % (257.3 / 235.9; noise -0.7), standard +6.2 (+0.3), thumbs +8.1 img/s (-0.7),
  444_10 +11.4 (-0.2). Then the docs speed list (speed-v06's spd.py, new = B13, old = A13) on the L4.
- 09:00 **Blackwell memcheck: every run fails** with one `Invalid __shared__ read of size 16 bytes` in CUB's
  DeviceScan::ExclusiveSum warpspeed kernel (`cp_async_bulk_cp_mask`, cp_async_bulk.h:205), from run_batch_device
  (gpu_entropy.cu:1734), then the kernel is aborted. mc_ab.sh: v0.8.0's CUDA 13.4 binary the same; both CUDA 12.9
  builds (A12, B12) 0 errors. So: CUDA 13.4's CUB on sm_120, pre-existing since v0.7.1, not from #56-#59. Reported
  to the lead. synccheck 0 errors on the 4500.
- 09:06 racecheck on the 4500 done: only the k_entropy_win / win_group / vload / vstore pairs (gpu_entropy.cu
  1219-1380). Installing nvcc 13.4 from apt (cubscan.sh) turned /usr/local/cuda into an alternatives link to 13.4;
  restored to the DLAMI's 13.2 at 09:07 (every sanitizer job had started before; all rc 0 / 99).
- 09:10 lead's follow-up (mc134.sh): compute-sanitizer 2026.3 (apt cuda-sanitizer-13-4) memcheck: B13 and A13,
  photo and pair: 0 errors. Standalone repro (cubsweep.cu, nvcc 13.4 = CUB 3.4.3, sm_120, the call of
  gpu_entropy.cu:1734): n 1..5000 and 5001..200000 step 7, clean under 2026.3; under 2026.1 the same invalid
  __shared__ read at n = 1. => a false positive of compute-sanitizer 2026.1 on CUB 3.4.3's TMA scan.
  mc134_all.sh: the 8 memcheck jobs again with 2026.3.
- 09:07 modload (4500, EAGER, warm): 29 sm_120 cubins, 241 MB, ~205 ms; the four k_recon cubins 38-39 MB, 35-40 ms
  each. (Partly concurrent with cubscan: rerun with LAZY and EAGER after mc134_all.)
- 09:12 mc134_all (2026.3, 8 jobs): 0 access errors, outputs identical; batch runs list only the 5-6 x 256 KB
  variance-boost tables (GpEncoder::init, not freed at exit; known since v0.6).
- 09:14 modload rerun (idle GPU): even LAZY, cuModuleLoad of each k_recon sm_120 cubin (38-39 MB) takes 33-36 ms
  (+1.4 ms for its function); all 29 sm_120 cubins (241 MB) 185-210 ms. Context creation ~120 ms.
- 09:15 Blackwell results pulled; 09:16-09:19 `av1gpu-rc09bw` torn down: instance terminated, spot request
  cancelled, SG / key pair / volumes gone (describe calls).
- 10:01 L4 docs speed list done (speed-v06's spd.py, ~60 min); 10:13 L4 sanitizers: memcheck 0 access errors (batch
  lists the 256 KB variance-boost tables), racecheck only the k_entropy_win pairs, synccheck 0.
- 10:14 L4 modload: 29 sm_89 cubins, 94 MB, 48 ms (lazy); k_recon 7-8 MB each, 8.5 ms for four.
- 10:14-10:24 palette-off speeds (spd_paloff*.py): RGBA x20 with --palette off 50.4 / 43.9 / 30.4 / 57.3 MP/s wall
  (v0.8.0 44.4 / 39.3 / 28.1 / 49.8, = v0.6); palette auto changes 15 of the 39 RGBA outputs (685,414 vs 698,892 B).
- 10:25 L4 results pulled; L4 teardown started; 10:33 `av1gpu-rc09l4` terminated, its SG and key pair deleted.
- Teardown verified (describe calls, name filter `av1gpu-rc09*`, 10:35): `av1gpu-rc09l4` i-041c0debd2cb670f1
  terminated; `av1gpu-rc09bw` i-0fab245d6674b3792 terminated at 09:19 (no longer listed); spot request
  sir-6mc7nnfj cancelled; no security group, key pair or volume with either name. Local key files deleted.
- Docs: README / RESULTS speed figures and ratios updated from the docs run; make_charts.py reads results.json here.
