#!/usr/bin/env python3
"""av1gpu's port of libaom's detection (libaom-score > 1, detect.sh output) against libaom's own decision (B, q60, pstat)
  detect_vs_aom.py SCORED_LOSSY_JSONL DETECT_TXT"""
import json
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
b = [r for r in rows if r["enc"] == "aom" and r["cfg"] == "B" and r["fmt"] == "420" and r["q"] == 60]
lib = {(r["set"], r["image"]) for r in b if (r.get("pstat") or {}).get("sct")}
allimg = {(r["set"], r["image"]) for r in b}
mine, seen = set(), set()
for l in open(sys.argv[2]):
    p = l.split()
    if p[0] == "std100":
        continue
    seen.add((p[0], p[1]))
    if float(p[9]) > 1:
        mine.add((p[0], p[1]))
print(len(allimg), "images,", len(seen), "with av1gpu scores; libaom", len(lib), "av1gpu port", len(mine))
print("only libaom:", sorted(lib - mine))
print("only av1gpu:", sorted(mine - lib))
