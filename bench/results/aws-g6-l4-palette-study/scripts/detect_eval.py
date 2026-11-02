#!/usr/bin/env python3
"""detection variants per class from detect.sh output (GP_SC_DEBUG lines): images with score > 1"""
import collections
import sys

rows = []
for l in open(sys.argv[1]):
    p = l.split()
    if len(p) < 18:
        continue
    d = dict(zip(p[2::2], p[3::2]))
    rows.append((p[0], p[1], float(p[9]), float(p[11]), float(p[13]), float(p[15]), float(p[17])))
names = ["libaom", "range>=8", "range>=16", "range>=32", "range>=64"]
by = collections.defaultdict(list)
for r in rows:
    by[r[0]].append(r)
print("class      n  " + "  ".join(f"{n:>10s}" for n in names))
for s in ("screen", "flat_png", "flat_jpg", "painted", "comics", "photo", "std100"):
    rs = by[s]
    print(f"{s:9s} {len(rs):3d}  " + "  ".join(f"{sum(1 for r in rs if r[2 + k] > 1):10d}" for k in range(5)))
for s in ("photo", "std100", "flat_png", "flat_jpg", "painted", "screen"):
    for r in by[s]:
        if r[2] > 1:
            print(f"  {s:8s} {r[1][:48]:48s} " + " ".join(f"{v:5.2f}" for v in r[2:]))
