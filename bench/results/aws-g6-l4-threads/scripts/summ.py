#!/usr/bin/env python3
"""summ.py FILE.jsonl... : median MP/s per config (rounds), cores, GPU memory, md5 mismatches."""
import json, statistics, sys
from collections import defaultdict
g = defaultdict(list)
for f in sys.argv[1:]:
    for l in open(f):
        r = json.loads(l)
        g[(r["n"], r["list"], " ".join(r["args"]), r["bin"], json.dumps(r["env"], sort_keys=True))].append(r)
for k in sorted(g, key=lambda k: (k[0], k[1], k[2], k[3] != "main", k[4])):
    rs = g[k]
    ok = [r for r in rs if r["rc"] == 0 and r.get("mp_s")]
    med = statistics.median(r["mp_s"] for r in ok) if ok else None
    print(f"n={k[0]:2} {k[1]:5} {k[2]:38} {k[3]:4} {k[4]:58} runs={len(rs)} ok={len(ok)} "
          f"mp/s={med and round(med, 1)} [{', '.join(str(round(r['mp_s'], 1)) for r in ok)}] "
          f"cores={ok and round(statistics.median(r['cores'] for r in ok), 2)} gmem={max((r.get('gpu_mem_max_mb') or 0) for r in rs)} "
          f"util={ok and round(statistics.median(r.get('gpu_util_mean') or 0 for r in ok))} "
          f"mism={sum(r.get('md5_mismatch') or 0 for r in rs)} out={set(r.get('n_out') for r in rs)}")
