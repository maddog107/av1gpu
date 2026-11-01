# CPU threads for batch mode (L4, 2026-11-01)

The table in docs/RESULTS.md comes from `followup/`. This top level is the first round, run on real instances of three
sizes, with the earlier batch-pick order (PNG / WebP frames first).

Three AWS L4 instances: g6.2xlarge (8 vCPUs), g6.4xlarge (16) and g6.8xlarge (32), all AMD EPYC 7R13 with
driver 595.112.07 and CUDA 13.2. The builds are `-DCMAKE_CUDA_ARCHITECTURES=89`. `main` is main before the
change (v0.3.0). `new` is the branch, with thread counts from the usable CPU count. `bf` is PR #23
(contrib/batch-fallback). In `final.jsonl` (16 vCPUs), `main2` and `reb` are main with #21/#22 and the
branch rebased onto it.

Lists (`scripts/mkdata.py`):
- jpeg: the 100 size-range JPEGs, 1,434 MP.
- png: the same photos as RGB PNGs, 956 MP.
- mix: per size class 5 PNG, 3 WebP and 12 JPEG, 956 MP.

Runs (`scripts/tthr.py`): `sweep` is 3 rounds in alternating order. `confirm` is 5 rounds and also tests
`GP_FALLBACK_NICE`, a test-only knob that is not in the final code. `qtest` is 3 rounds of
`GP_FALLBACK_QUEUE_MP`. `hang` and `aff` are the hang and CPU-detection checks. Each line records MP/s, CPU
cores, peak GPU memory (nvidia-smi, 250 ms samples) and `md5_mismatch` against main's first run of the
same list and arguments. That mismatch is 0 for every run.

`python3 scripts/pooled.py runs` prints the table in docs/RESULTS.md:

```
tiles: --tiles 2,2,2
| JPEG | 124.6 → 126.0 (+1 %) [3/3] | 123.2 → 125.1 (+2 %) [3/3] | 124.1 → 121.3 (-2 %) [3/3] |
| JPEG, CPU entropy decode (GP_INGEST_DECODER=1) | 118.4 → 122.3 (+3 %) [3/3] | 115.0 → 120.2 (+5 %) [3/3] | 115.3 → 120.7 (+5 %) [3/3] |
| PNG | 95.7 → 106.4 (+11 %) [8/11] | 97.2 → 109.8 (+13 %) [8/11] | 96.8 → 116.0 (+20 %) [8/11] |
| PNG to 10-bit 4:4:4 | 60.8 → 82.8 (+36 %) [8/11] | 61.2 → 88.1 (+44 %) [8/11] | 61.5 → 92.4 (+50 %) [8/11] |
| mixed JPEG / PNG / WebP | 117.0 → 110.9 (-5 %) [11/11] | 114.0 → 110.4 (-3 %) [11/11] | 118.2 → 115.5 (-2 %) [11/11] |
tiles: 2x2 (main's default)
| JPEG | 110.2 → 115.5 (+5 %) [3/3] | 108.8 → 114.0 (+5 %) [3/3] | 110.8 → 115.2 (+4 %) [3/3] |
| JPEG, CPU entropy decode (GP_INGEST_DECODER=1) | 106.8 → 110.8 (+4 %) [3/3] | 102.7 → 109.7 (+7 %) [3/3] | 105.0 → 113.8 (+8 %) [3/3] |
| PNG | 88.2 → 97.2 (+10 %) [3/3] | 91.0 → 99.5 (+9 %) [3/3] | 91.3 → 107.5 (+18 %) [3/3] |
| PNG to 10-bit 4:4:4 | 61.1 → 78.3 (+28 %) [3/3] | 61.6 → 79.3 (+29 %) [3/3] | 60.2 → 82.4 (+37 %) [3/3] |
| mixed JPEG / PNG / WebP | 108.0 → 105.5 (-2 %) [3/3] | 106.0 → 104.5 (-1 %) [3/3] | 102.1 → 104.4 (+2 %) [3/3] |
```

## followup/

One g6.8xlarge (32 vCPUs). The 8- and 16-vCPU sets ran under `taskset -c 0-3,16-19` and `0-7,16-23` (CPU k and k+16
are SMT siblings). All runs use `--tiles 2,2,2`. `main` is main with #21, #22 and #27, plus `GP_BATCH_LOG`. `new` is
the branch plus `GP_BATCH_LOG` and the test-only `GP_FB_POLICY`. `p1` is the final branch. `p1n` adds the test-only
`GP_FALLBACK_NICE`. The `GP_FB_POLICY` values are 0 (fallback frames first), 1 (ready JPEG batches first, the
final code), 2 (#23's rule), 3 (alternate) and 4 (half a batch of fallback frames at most).

- `runs/t2_instr.jsonl`, `batch-logs/`: runs with `GP_BATCH_LOG`. `scripts/blog.py batch-logs/*.txt` prints
  composition, encoder coverage and the tail. `scripts/instr.py` is the logging patch.
- `runs/t2_speed.jsonl`: the batch-pick policies, 5 rounds.
- `runs/t2_jpeg*.jsonl`: `GP_INGEST_WORKERS` on 32 vCPUs.
- `runs/t3.jsonl`, `runs/t4.jsonl`: main against the final code on the three CPU sets. `t4` also has the nice
  test. `python3 scripts/t3sum.py runs/t3.jsonl runs/t4.jsonl` prints the RESULTS.md table.

Every run's `md5_mismatch` against main's first run at the same settings is 0.
