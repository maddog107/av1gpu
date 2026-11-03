# Loading only the k_recon instantiation a run uses (NVIDIA L4 and RTX PRO 4500 Blackwell, 2026-11-03)

Agent rc11, branch `perf/recon-lazy-load` (3739ede). v0.9.0 uploaded the closed loop's tables to all four k_recon
instantiations (4:2:0 and 4:4:4, 8 and 10 bits) when the encoder started. Each is its own kernel module, and with lazy
module loading (the default outside batch mode) the first upload to a module loads it: 39 MB of sm_120 code each, so
a single photo loaded three reconstruction modules it never used. The branch uploads those tables per instantiation,
once per process, before its first launch (`gp_recon_prepare`): the encoder's init prepares the one its configuration
predicts (4:2:0 or 4:4:4 at the configured depth) while the JPEG decodes, and an encode whose frames need another one
prepares it first. Host code only; the GPU code is unchanged.

**Result:** the branch writes the same files as v0.9.0: **1,155 of 1,155 identical on each GPU**, in batch mode,
single-file mode and the CUDA 12 build. A cold single image on the **RTX PRO 4500 starts about 150 ms sooner**: the
2.5 MP photo in **342 instead of 492 ms** (−31 %), the 10.5 MP photo in 421 instead of 571, the thumbnails in 336–337
instead of 485–492. The RTX PRO 4500 now starts a single image as quickly as the L4 (355 ms). On the L4, whose
modules are 7–8 MB, the change is 3–22 ms. `--batch` lists are unchanged: batch mode loads every module when the
CUDA context is created (`CUDA_MODULE_LOADING=EAGER`, #36).

## Binaries

| | file | sha256 |
|---|---|---|
| A | v0.9.0 `av1gpu-0.9.0-linux-x86_64.tar.gz` (CUDA 13.4) and `-cuda12.tar.gz` (CUDA 12.9), from the release page | checked against the release's `.sha256` files on each instance (`logs/*/run.log`) |
| Ac | A's binary copied to a second path: the noise of the timing | |
| B | 3739ede `av1gpu-dev-3739ede-linux-x86_64.tar.gz` and `-cuda12.tar.gz`, Release run 41705802559 | 19445b59… and a370058f… (match the artifacts' `.sha256`, checked locally and on each instance) |

`scripts/run.sh` expects the branch's packages at `~/pkg/av1gpu-0.9.0-linux-x86_64{,-cuda12}`; there those were
symlinks to the extracted `av1gpu-dev-3739ede-*` directories (the log's `/home/ubuntu/pkg/...` paths are B).

## Setup

| | L4 (`av1gpu-rc11l4`) | RTX PRO 4500 (`av1gpu-rc11bw`) |
|---|---|---|
| instance | g6.2xlarge, spot, 8 vCPUs | g7.4xlarge, spot, 16 vCPUs |
| GPU | NVIDIA L4, sm_89, 72 W | RTX PRO 4500 Blackwell Server Edition, sm_120, 165 W |
| driver, OS | 595.112.07; Ubuntu 24.04, kernel 7.0.0-1014-aws, DLAMI ami-00dc060a4dfb01df5 | the same |

`scripts/run.sh` fetches the size range (100 photos, 477.8 MP) and the standard set (100, 78.7 MP) with the repo's
fetch scripts, and makes the rest with PIL: 200 thumbnails (512 × 384 and 320-pixel), 17 grayscale JPEGs, 10 RGBA PNGs
with a radial alpha, 10 small RGB PNGs for `--lossless`, and the single-image inputs: `photo.jpg` =
1068f4f5ac7e8d01_typical.jpg (1280 × 1920, 2.46 MP), `big.jpg` = 16aaa6421eaf9905_huge.jpg (4320 × 2432, 10.5 MP),
`d512.jpg` and `t320.jpg` (the photo as a 512 × 384 and a 213 × 320 thumbnail), `mono.jpg` (a 1280 × 1920 grayscale
JPEG) and `alpha.png` (683 × 1024 RGBA). The inputs have the same SHA-256 on both instances (`logs/*/inputs.sha256`).
The AWS launch / teardown scripts are not committed (they hold the account's network IDs).

## Byte identity

Every output file of B against A's (`cmp`), and the exit status of every run. The runs pick every k_recon
instantiation, alone and side by side, and the paths where the predicted one is not the one used:

| mode | runs | files | identical, L4 | identical, RTX PRO 4500 |
|---|---|---:|---:|---:|
| `--batch`, CUDA 13 | size range at default, 4:4:4, 10-bit, 4:4:4 10-bit; standard set; thumbnails; grayscale at default and 4:4:4, and 4:4:4 8- and 10-bit with `GP_ENCODERS=4` (encoders preparing the same instantiation at once); 10 photos + 17 grayscale at 4:4:4 (both groups in one batch); RGBA at default, 4:4:4, 4:4:4 10-bit; `--lossless` RGB and RGBA | 845 | **845** | **845** |
| single file, CUDA 13 | 26 files (12 photos, 6 thumbnails, 4 grayscale, 4 RGBA) at default, 4:4:4, 10-bit, 4:4:4 10-bit; 14 at `--lossless` | 118 | **118** | **118** |
| single file, repeated | the photo 20 times; grayscale `--yuv 444` 10 times (init prepares 4:4:4, the frame needs 4:2:0); RGBA `--yuv 444` 10 times (alpha joins the 4:4:4 group) | 40 | **40** | **40** |
| CUDA 12 | single file at default and 4:4:4 10-bit; size range `--batch` | 152 | **152** | **152** |
| **total** | | **1,155** | **1,155** | **1,155** |

## Cold single-image start-up

`scripts/wall9.py` (as in `aws-g6-g7-v090-check`): `av1gpu IN OUT`, one process per run, wall time including CUDA
start-up, 12 runs per binary with A, Ac and B alternating; the median in ms.

| Input | RTX PRO 4500, v0.9.0 | v0.9.0 again | **branch** | saved | L4, v0.9.0 | v0.9.0 again | **branch** | saved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.5 MP JPEG (1280 × 1920) | 492.3 | 488.4 | **341.9** | 150 (31 %) | 360.1 | 358.9 | **354.8** | 5 |
| 10.5 MP JPEG (4320 × 2432) | 571.4 | 572.8 | **420.5** | 151 (26 %) | 471.4 | 470.9 | **468.6** | 3 |
| 512 × 384 thumbnail | 491.6 | 488.4 | **336.4** | 155 (32 %) | 354.3 | 349.4 | **343.2** | 11 |
| 213 × 320 thumbnail | 484.5 | 485.7 | **336.6** | 148 (31 %) | 347.1 | 339.2 | **337.7** | 9 |
| the photo, `--yuv 444 --depth 10` | 515.1 | 511.1 | **371.4** | 144 (28 %) | 382.3 | 378.7 | **374.9** | 7 |
| grayscale JPEG, `--yuv 444` | 483.1 | 481.9 | **372.8** | 110 (23 %) | 344.0 | 338.6 | **340.9** | 3 |
| 683 × 1024 RGBA PNG | 609.5 | 612.1 | **457.0** | 153 (25 %) | 481.5 | 475.5 | **459.9** | 22 |
| `--batch` of the photo and its 512 × 384 thumbnail | 670.4 | 667.4 | **665.8** | 5 | 577.2 | 573.6 | **573.5** | 4 |

- **About 150 ms per single image on Blackwell**, more than the 3 × 36 ms `aws-g6-g7-v090-check`'s standalone loader
  (`modload.c`) timed for the three modules no longer loaded; the rest was not investigated. v0.9.0 against itself
  differed by 0.8 % or less there.
- **A grayscale JPEG with `--yuv 444` loads two:** the encoder's init prepares the 4:4:4 instantiation its
  configuration predicts, and the monochrome frame, with no 4:4:4 frame beside it, runs in the 4:2:0 group, which
  the encode prepares before it starts. It still saves 110 ms. At the default 4:2:0, grayscale JPEGs use the predicted
  one.
- **The L4 gains little** (its k_recon modules are 7–8 MB of sm_89 code): 3–11 ms on the JPEGs, about the noise
  (0.1–2.3 %), and 22 ms on the RGBA PNG.
- **`--batch` is unchanged:** batch mode loads every module eagerly, so a mid-run lazy load cannot stall the other
  encoders' launches (#36). A short list such as an upload's photo and thumbnail could keep lazy loading; not done here.

## Files

- `scripts/run.sh`: the whole run on each instance (inputs, identity, timing); `scripts/wall9.py`: the cold-start
  timer; `scripts/medians.py`: the table above from `run.log`.
- `results.json`: per GPU, the identity counts and every timing run.
- `logs/{l4,bw}/`: `run.log`, `gpu.txt`, `inputs.sha256`, the file lists and each identity run's logs and exit codes.
