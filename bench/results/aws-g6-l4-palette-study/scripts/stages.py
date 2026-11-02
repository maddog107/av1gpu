#!/usr/bin/env python3
"""median --stats stage times per (tag, binary) of spd_pal.py rows: where a speed difference comes from"""
import collections
import json
import statistics
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
keys = ["mp_s", "analysis_ms", "dp_ms", "recon_ms", "pack_ms", "postfilter_ms", "gpu_total_ms", "entropy_gpu_ms", "peak_dev_mb"]
by = collections.defaultdict(list)
for r in rows:
    if r["timed"] and r["stats"]:
        by[(r["tag"], r["bin"])].append(r["stats"])
for (tag, b), st in sorted(by.items()):
    print(f"{tag:22s} {b:4s} n={len(st)} " + " ".join(f"{k}={statistics.median(s[k] for s in st):.1f}" for k in keys))
