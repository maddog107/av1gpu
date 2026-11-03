# v0.9.0 published Linux packages on an L4 (2026-11-03)

The v0.9.0 release check (`../aws-g6-g7-v090-check/`) ran the Linux packages of main c553ffb from a manual Release
run. This run uses the files published on the v0.9.0 release page, built from the tag (7c2bf33: c553ffb plus the
docs, the version number and a README line). The tag build was not byte-for-byte the tested build.

## Why: k_an_c.cu's SASS is not reproducible

Comparing the GPU code of the two builds entry by entry (`scripts/fatbin_hashes.py`, 152 fatbin entries per
package) showed this:

- **CUDA 13.4 packages:** 149 of 152 entries are identical. The three that differ are all `k_an_c.cu` SASS: sm_75,
  86 and 90 on Linux, and sm_75, 80 and 86 on Windows. The PTX of `k_an_c.cu` is identical in both builds, and so is
  the SASS for sm_89 (L4) and sm_120 (Blackwell).
- **CUDA 12.9 packages:** every other difference lies inside the source-file mtimes that nvcc 12.9 writes into its
  line tables. The exception is three `k_an_c.cu` cubins per package, sm_89 among them (Linux 75 / 86 / 89,
  Windows 86 / 89 / 90).

So two clean builds of the same PTX give different SASS for a few architectures, picked at random. v0.8.0's
builds were reproducible, so this started with #58, which changed the analysis kernels in that file (16 × 32,
32 × 16 and 32 × 32 blocks). The build that was tested was not stale: its `k_an_c.cu` differs from v0.8.0's for
every architecture, and its PTX equals the tag's.

## What was run

`scripts/run.sh` on a g6.2xlarge (NVIDIA L4, driver 595.112.07, spot). It downloads the v0.8.0 and v0.9.0 Linux
packages from the release pages and checks them against `scripts/expect.sha256`. The inputs are the standard set,
the size range, 200 thumbnails made from the size range (512 × 384 and within 320 × 320, JPEG q85), Kodak, and the
palette study's sets (`prep_sets.py`). It then compares every output with `cmp`, together with the exit status of
every run.

| Comparison | Files | Identical |
|---|---:|---:|
| v0.8.0 → v0.9.0, CUDA 13.4 packages: 21 batch runs (standard set at default and `--small-preset none`; size range at default, 4:4:4, 10-bit, 4:4:4 10-bit, `--preset slow`, `--quality 90` / `30`, `turbo3 --small-preset none`; thumbnails at default and 4:4:4 10-bit; Kodak lossless; photo, comics, painted, flat_jpg; screen + flat_png at default, `--palette on`, 4:4:4 10-bit and lossless), plus single-file mode on 23 files at default and 4:4:4 10-bit | 1,833 | **1,833** |
| v0.8.0 → v0.9.0, CUDA 12.9 packages, the same runs | 1,833 | **1,833** |
| v0.9.0: CUDA 13.4 → CUDA 12.9 (all sets, thumbnails, size range 4:4:4 10-bit, screen + flat_png lossless, single-file mode) | 733 | **733** |
| **Total** (comparisons; 4,394 distinct outputs, see Notes) | **4,399** | **4,399** |

- **Decoder gate** (`tests/decoder_gate.py`, dav1d 1.4.1 and aomdec), both v0.9.0 packages: 30 size-range photos at
  qindex 60 / 120 / 200 and at 4:4:4 10-bit, 40 thumbnails at 60 / 120, and screen + flat_png at 120 / 40 and
  lossless. **PASS, 10 of 10 runs**, with the same byte counts for the two packages (`logs/gate/`).
- **compute-sanitizer 2026.1**, both v0.9.0 packages, on 3 photos in batch mode, one photo and one thumbnail in
  single-file mode:
  - memcheck: 0 errors.
  - racecheck: only `k_entropy_win`'s ring hand-off, at the gpu_entropy.cu lines the release check lists (1219–1380).
  - No report in any analysis or reconstruction kernel (`logs/cs_*.log`).

**The published Linux packages write the same files as v0.8.0 and as each other on the L4.** The SASS that differs
in the CUDA 13.4 packages (sm_75 / 80 / 86 / 90) is for GPUs no release check has run on (T4, A100, RTX 30, H100).

## Notes

- `run.sh` as run has a variable clash: `pair()` sets a global `t`, which renames every run after the first of each
  set from `c13_*` / `c12_*` to `B_*`. So the CUDA 12.9 runs reused the CUDA 13.4 runs' output directories. Each
  comparison is made right after its own two encodes. Every exit status was 0, and every A / B file count matched.
  All 3,462 files in the `B_*` directories were newer than the first CUDA 12.9 run's, so no file was left over from
  the CUDA 13.4 runs. The log lines' run names are therefore `B_*` for both packages; `SUBTOTAL` separates them.
- The single-file list (`logs/lists/single.txt`) has 24 lines but 23 files: 12 size-range photos, 6 thumbnails
  (512 × 384), 3 screen PNGs and 2 flat_jpg JPEGs. One thumbnail is listed twice. So the 5 single-file runs count
  one comparison twice each, and the 4,399 comparisons are 4,394 distinct outputs. That is also why 3,462 files
  (not 3,466) were in the `B_*` directories.
- The AWS launch and teardown scripts are not committed (they hold the account's network IDs). The instance, its
  key pair and its security group were deleted after the run, and the spot request was cancelled.
