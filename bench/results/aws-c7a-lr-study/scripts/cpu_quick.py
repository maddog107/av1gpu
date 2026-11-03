#!/usr/bin/env python3
"""CPU seconds per configuration, format and set (and q60 only), from ~/lr/rows/enc.jsonl, with ratios to s6 / s4."""
import collections
import json
import os

rows = [json.loads(l) for l in open(os.path.expanduser("~/lr/rows/enc.jsonl"))]
t = collections.defaultdict(float)
for r in rows:
    for s in (r["set"], "all"):
        t[(r["fmt"], r["cfg"], s, "all")] += r["cpu"]
        if r["q"] == 60:
            t[(r["fmt"], r["cfg"], s, "q60")] += r["cpu"]
for fmt in ("420", "444"):
    for qq in ("all", "q60"):
        print(f"--- {fmt} {qq}")
        for s in ("all", "std", "gamut", "kodak"):
            if (fmt, "s6", s, qq) not in t:
                continue
            line = [f"{s:6s}"]
            for c in ("s6", "s6r1", "p6keep", "p6full", "p6wiener", "p6sgr", "s4", "s4r1", "p4full"):
                if (fmt, c, s, qq) in t:
                    a = "s4" if c.startswith(("s4", "p4")) else "s6"
                    line.append(f"{c} {t[(fmt, c, s, qq)]:.0f} ({t[(fmt, c, s, qq)] / t[(fmt, a, s, qq)]:.3f})")
            print("  ".join(line))
