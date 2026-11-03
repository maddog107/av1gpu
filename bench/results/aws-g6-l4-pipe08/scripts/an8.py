#!/usr/bin/env python3
"""an8.py JSONL...: per phase / set / binary: median of the timed runs (MP/s and img/s from --stats, wall MP/s =
MP over the process's wall time), per-round medians, postfilter ms, bytes the same in every run; markdown rows."""
import collections, json, statistics, sys
runs = collections.defaultdict(list)
for fn in sys.argv[1:]:
    for ln in open(fn):
        r = json.loads(ln)
        if r["timed"] and not r["rc"]:
            runs[(r["phase"], r["tag"], r["bin"])].append(r)
print("| phase | set | binary | MP/s | img/s | rounds (MP/s) | wall MP/s | postfilter ms | bytes same | runs |")
print("|---|---|---|---:|---:|---|---:|---:|---|---:|")
for (ph, tag, b), v in sorted(runs.items()):
    mps = [r["stats"]["mp_s"] for r in v]
    ims = [r["stats"]["img_s"] for r in v]
    wall = [r["stats"]["mp"] / r["wall_s"] for r in v]
    pf = [r["stats"].get("postfilter_ms", 0) for r in v]
    rounds = [statistics.median(mps[i:i + 3]) for i in range(0, len(mps), 3)]
    same = len({r["stats"]["bytes"] for r in v}) == 1
    print(f"| {ph} | {tag} | {b} | {statistics.median(mps):.1f} | {statistics.median(ims):.1f} | "
          f"{' / '.join(f'{x:.1f}' for x in rounds)} | {statistics.median(wall):.1f} | {statistics.median(pf):.0f} | {same} | {len(v)} |")
