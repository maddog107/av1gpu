#!/usr/bin/env python3
"""analyze8.py spd_TAG.jsonl...: per set, the median of the 6 timed runs per binary, B/A, and the per-round ratio
(round 1: A then B, round 2: B then A; the order spd8.py runs them in); single-file: median wall of 12 runs each."""
import json, statistics, sys

for fn in sys.argv[1:]:
    rows = [json.loads(l) for l in open(fn)]
    tags = []
    for r in rows:
        if r["tag"] not in tags:
            tags.append(r["tag"])
    for t in tags:
        rs = [r for r in rows if r["tag"] == t]
        if t == "single":
            a = [r["wall_s"] for r in rs if r["bin"] == "A" and r["timed"]]
            b = [r["wall_s"] for r in rs if r["bin"] == "B" and r["timed"]]
            ma, mb = statistics.median(a), statistics.median(b)
            print(f"{fn.split('/')[-1]} {t}: A {ma * 1000:.1f} ms B {mb * 1000:.1f} ms  B/A {mb / ma:.4f}  ({len(a)} + {len(b)} runs; "
                  f"A min/max {min(a) * 1000:.0f}/{max(a) * 1000:.0f}, B {min(b) * 1000:.0f}/{max(b) * 1000:.0f})")
            continue
        key = "img_s" if t.startswith("thumbs") else "mp_s"
        val = lambda r: r["stats"].get(key, 0)
        n = len(rs) // 2   # runs per round (both binaries)
        rounds = [rs[:n], rs[n:]]
        out = []
        for k, rr in enumerate(rounds):
            a = [val(r) for r in rr if r["bin"] == "A" and r["timed"]]
            b = [val(r) for r in rr if r["bin"] == "B" and r["timed"]]
            out.append((statistics.median(a), statistics.median(b)))
        a6 = [val(r) for r in rs if r["bin"] == "A" and r["timed"]]
        b6 = [val(r) for r in rs if r["bin"] == "B" and r["timed"]]
        ma, mb = statistics.median(a6), statistics.median(b6)
        print(f"{fn.split('/')[-1]} {t} {key}: A {ma:.2f} B {mb:.2f} B/A {mb / ma:.4f} | per round B/A "
              f"{out[0][1] / out[0][0]:.4f} (A first), {out[1][1] / out[1][0]:.4f} (B first) | run range A {min(a6):.1f}-{max(a6):.1f}"
              f" B {min(b6):.1f}-{max(b6):.1f}")
