#!/usr/bin/env python3
"""Progress of the CVVDP rows: unique (cfg, set, image, q) rows per configuration, all and for sources >= 5 MP."""
import collections
import csv
import json
import os

H = os.path.expanduser("~")
big = {r["file"] for r in csv.DictReader(open(f"{H}/bench/openimages_v4_gamut.csv")) if r["size_class"] in ("large", "huge")}
u, ub = collections.Counter(), collections.Counter()
seen = set()
for l in open(f"{H}/lr/rows/cvvdp.jsonl"):
    r = json.loads(l)
    k = (r["cfg"], r["set"], r["image"], r["q"])
    if k in seen:
        continue
    seen.add(k)
    u[r["cfg"]] += 1
    if r["image"] in big:
        ub[r["cfg"]] += 1
print("all:", dict(u))
print(">= 5 MP (of 200 each: 40 photos x 5 q):", dict(ub))
