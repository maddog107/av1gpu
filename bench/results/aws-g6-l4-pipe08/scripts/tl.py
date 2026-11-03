#!/usr/bin/env python3
"""tl.py REP.sqlite [min_ms]: timeline of an nsys report: CUDA API calls (runtime + driver) and OS runtime calls of at
least min_ms (default 2), GPU kernels / memcpy busy spans, relative to the first event; per-thread; then totals."""
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
mn = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
cur = db.cursor()
tabs = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
ev = []
def names():
    return {i: s for i, s in cur.execute("select id, value from StringIds")}
S = names()
if "CUPTI_ACTIVITY_KIND_RUNTIME" in tabs:
    for st, en, nid, tid in cur.execute("select start, end, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
        ev.append((st, en, "api", S.get(nid, "?"), tid & 0xffffff))
if "OSRT_API" in tabs:
    for st, en, nid, tid in cur.execute("select start, end, nameId, globalTid from OSRT_API"):
        ev.append((st, en, "os", S.get(nid, "?"), tid & 0xffffff))
gpu = []
if "CUPTI_ACTIVITY_KIND_KERNEL" in tabs:
    for st, en, nid in cur.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
        gpu.append((st, en, "kernel", S.get(nid, "?")))
if "CUPTI_ACTIVITY_KIND_MEMCPY" in tabs:
    for st, en, k in cur.execute("select start, end, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
        gpu.append((st, en, "memcpy", str(k)))
t0 = None
if "PROCESSES" in tabs:
    pass
if "TARGET_INFO_SESSION_START_TIME" in tabs:
    pass
allst = [e[0] for e in ev] + [g[0] for g in gpu]
t0 = min(allst)
print(f"first event at 0; last API end {(max(e[1] for e in ev) - t0) / 1e6:.1f} ms")
if gpu:
    print(f"GPU: first {(min(g[0] for g in gpu) - t0) / 1e6:.1f} ms, last end {(max(g[1] for g in gpu) - t0) / 1e6:.1f} ms, "
          f"busy {sum(g[1] - g[0] for g in gpu) / 1e6:.1f} ms, {len(gpu)} ops")
for st, en, k, n, tid in sorted(ev):
    if (en - st) / 1e6 >= mn:
        print(f"{(st - t0) / 1e6:9.1f} {(en - st) / 1e6:8.1f} ms  {k:3s} t{tid:<7d} {n}")
