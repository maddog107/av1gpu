# v0.10.0 published Linux packages on an L4 (2026-11-03)

`../aws-g6-g7-recon-lazy/` tested the Linux packages of 3739ede from a manual Release run on an L4 and an RTX PRO
4500. This run checks the files published on the v0.10.0 release page, built from the tag (6d886ab: 3739ede plus the
results, the docs and the version number).

**Result:** the published v0.10.0 packages write the same files as v0.9.0: **1,155 of 1,155 identical** on the L4
(CUDA 13.4 and CUDA 12.9 packages), and their cold start-up matches the tested build's.

## GPU code: published against tested

`scripts/fatbin_compare.py` hashes every GPU code entry of both packages by source file and architecture
(`logs/fatbin_compare.txt`):

- **CUDA 13.4:** 271 of 272 entries are identical, all sm_120 (Blackwell) code among them, so
  `aws-g6-g7-recon-lazy`'s RTX PRO 4500 results hold for the published package. The one that differs is
  `k_an_c.cu`'s sm_89 SASS, the analysis kernel whose SASS is not reproducible between builds (as found for v0.9.0,
  `../aws-g6-l4-v090-published/`). sm_89 is the L4's architecture, so this run tests that code.
- **CUDA 12.9:** 193 of 312 entries are identical, every PTX entry among them. The other 119 are the SASS of 17
  source files for all seven architectures. v0.9.0's check traced such differences to the source-file mtimes that
  nvcc 12.9 writes into its line tables; that was not checked here. This run tests the package's sm_89 code; its
  sm_120 code was not run.

## What was run

`scripts/run.sh` (rc11's run with B taken from the release page) on a g6.2xlarge (NVIDIA L4, driver 595.112.07,
8 vCPUs, spot). It downloads the v0.9.0 and v0.10.0 Linux packages from the release pages and checks each against
its published `.sha256`. The inputs are made as in `aws-g6-g7-recon-lazy` (same SHA-256s, `logs/inputs.sha256`).

| Comparison, v0.9.0 → v0.10.0 | Files | Identical |
|---|---:|---:|
| `--batch`, CUDA 13.4: size range at 4:2:0 / 4:4:4 × 8 / 10 bits, standard set, thumbnails, grayscale (default, 4:4:4, 4:4:4 with `GP_ENCODERS=4` at 8 and 10 bits), photos + grayscale at 4:4:4, RGBA (default, 4:4:4, 4:4:4 10-bit), lossless RGB and RGBA | 845 | **845** |
| single file, CUDA 13.4: 26 files at four settings, 14 lossless; the photo ×20, grayscale `--yuv 444` ×10, RGBA `--yuv 444` ×10 | 158 | **158** |
| CUDA 12.9: single file at default and 4:4:4 10-bit, size range `--batch` | 152 | **152** |
| **Total** | **1,155** | **1,155** |

Cold single-image start-up (`scripts/wall9.py`: `av1gpu IN OUT`, one process per run, 12 runs per binary
alternating, median in ms):

| Input | v0.9.0 | v0.9.0 again | **v0.10.0** | tested build (`aws-g6-g7-recon-lazy`) |
|---|---:|---:|---:|---:|
| 2.5 MP JPEG | 357.1 | 356.9 | **353.8** | 354.8 |
| 10.5 MP JPEG | 475.1 | 466.4 | **464.7** | 468.6 |
| 512 × 384 thumbnail | 352.6 | 345.1 | **341.5** | 343.2 |
| 213 × 320 thumbnail | 345.2 | 338.1 | **335.1** | 337.7 |
| the photo, `--yuv 444 --depth 10` | 381.3 | 376.9 | **373.6** | 374.9 |
| grayscale JPEG, `--yuv 444` | 339.3 | 338.8 | **340.7** | 340.9 |
| 683 × 1024 RGBA PNG | 479.4 | 471.8 | **462.2** | 459.9 |
| the photo and its thumbnail, `--batch` | 574.6 | 570.6 | **571.1** | 573.5 |

As on the L4 before, the change is a few ms, within v0.9.0's own spread (0.1–2.1 %) except the RGBA PNG. The
saving is on Blackwell (about 150 ms per image, `aws-g6-g7-recon-lazy`).

The instance waited for the release to be published (`run.sh` polls for the last asset; the 60 404 lines of that
wait are left out of `logs/run.log`). The AWS launch / teardown scripts are not committed (they hold the account's
network IDs).

## Files

- `scripts/run.sh`, `scripts/wall9.py`, `scripts/medians.py` (the table above from `run.log`),
  `scripts/fatbin_compare.py`.
- `results.json`: identity counts and every timing run. `logs/`: `run.log`, `gpu.txt`, `inputs.sha256`, the lists,
  each identity run's logs and exit codes, `fatbin_compare.txt`.
