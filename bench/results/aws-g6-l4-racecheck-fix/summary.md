# k_recon racecheck fix (#43) on an NVIDIA L4

AWS g6.4xlarge (on demand), 2026-11-02. The fix (78e374c) built alongside main d7d94b3 (v0.6.0), both with
`-DCMAKE_CUDA_ARCHITECTURES=89`.

## The change

After the concurrent U / V chroma finals (#37), every thread read `S.eobUV[0..1]`. nvcc merged those reads with
thread 0's read of `S.nTxb` into one `LDS.128 [0x5af0]`, so every thread also read `nTxb` while thread 0 updated it.
The value was never used, but racecheck reported the pair `k_recon.cuh:1373 <-> 1374`. The two reads are now
volatile single-word loads.

SASS of `k_recon<1,8>` (`logs/sass/`):

| | eob reads (every thread) | thread 0 |
|---|---|---|
| d7d94b3 | `LDS.128 R8, [0x5af0]` (nzFlag[1], eobUV[0], eobUV[1], nTxb) | `@!P0 LDS R0, [0x5b00]` (nCoef); writes `[0x5afc]`, `[0x5b00]` |
| 78e374c | `LDS R8, [0x5af4]`, `LDS R9, [0x5af8]` | `@!P0 LDS` of `[0x5b00]` and `[0x5afc]`; same writes |

No wide shared load covers `nTxb` anywhere in the kernel now. The 4:2:0 10-bit kernel had the same wide load
and is fixed the same way. The kernel is still 160,368 instructions.

## racecheck

`compute-sanitizer --tool racecheck`, `av1gpu --batch` on the 3 photos of the v0.6 release check (`logs/race/`):

| | d7d94b3 default | 78e374c default | d7d94b3 `--yuv 444 --depth 10` | 78e374c `--yuv 444 --depth 10` |
|---|---:|---:|---:|---:|
| `k_recon` 1373 / 1374 | 52,056 | **0** | 0 | 0 |
| `k_entropy_win` 1090 / 992 | 7,046,576 | 7,046,576 | 7,133,332 | 7,133,332 |
| `k_entropy_win` 1090 / 1061 | 1,927,868 | 1,927,868 | 1,968,372 | 1,968,372 |
| reports displayed | 103 | 8 | 8 | 8 |

The remaining reports are the known `k_entropy_win` ones (docs/RESULTS.md, Correctness), equal in both builds apart
from the `vload` / `vstore` spin-wait counts, which vary from run to run.

memcheck: 0 access errors in both (only the 256 KB variance-boost tables not freed at exit). synccheck: 0 errors.

## Byte identity

`cmp` of every output, 78e374c against d7d94b3 (`logs/identity.txt`): 13 sets, 1,267 files each, **0 differences**.
Size range (100 JPEGs) default, `--preset turbo3 --small-preset none`, `--yuv 444`, `--depth 10`,
`--yuv 444 --depth 10`, `--preset slow --small-preset none`, `GP_ENCODERS=1` and `6`; standard set default; Kodak
lossy and `--lossless --ycgco-re`; 39 RGBA PNGs; 280 thumbnails. Every set's digest also matches the v0.6 release
check's.

ctest: 5/5 pass (`logs/ctest.txt`).

## Speed

MP/s, median of 6 (2 rounds alternating, spd.py method; `runs/speed.json`):

| Set | 78e374c | d7d94b3 | change |
|---|---:|---:|---:|
| size range, default | 237.7 | 236.0 | +0.7 % |
| size range, `--yuv 444 --depth 10` | 170.6 | 172.2 | −0.9 % (−1.4 % / +0.3 % by round) |
| standard set, default | 76.9 | 76.7 | +0.3 % |
| main against main (noise) | | | 1.4 % |

Neutral within noise. This instance ran 3–6 % faster than the release check's for both builds.
