#!/usr/bin/env python3
"""sass_sum.py SASS.csv: warp-stall samples and executed instructions by the average active threads per instruction
(single-lane code = serial parts), plus the hottest instructions."""
import csv, sys

rows = list(csv.reader(open(sys.argv[1])))
hdr = None
data = []
for r in rows:
    if len(r) > 3 and r[0] == "Address":
        hdr = r
        continue
    if hdr and len(r) == len(hdr):
        data.append(dict(zip(hdr, r)))
def f(x):
    try: return float(x.replace(",", ""))
    except: return 0.0
tot = sum(f(d["Warp Stall Sampling (All Samples)"]) for d in data)
ti = sum(f(d["Instructions Executed"]) for d in data)
print(f"{len(data)} instructions, {tot:.0f} samples, {ti:.0f} warp instructions executed")
bins = [(0, 1.5), (1.5, 4.5), (4.5, 8.5), (8.5, 16.5), (16.5, 31.5), (31.5, 33)]
for lo, hi in bins:
    s = sum(f(d["Warp Stall Sampling (All Samples)"]) for d in data if lo < f(d["Avg. Threads Executed"]) <= hi)
    n = sum(f(d["Instructions Executed"]) for d in data if lo < f(d["Avg. Threads Executed"]) <= hi)
    print(f"avg threads ({lo:4.1f}, {hi:4.1f}]: samples {100 * s / max(tot, 1):5.1f} %  instructions {100 * n / max(ti, 1):5.1f} %")
loc = sum(f(d["Warp Stall Sampling (All Samples)"]) for d in data if d.get("Address Space") == "Local")
print(f"local-memory instructions: samples {100 * loc / max(tot, 1):5.1f} %")
print("== hottest 40")
for d in sorted(data, key=lambda d: -f(d["Warp Stall Sampling (All Samples)"]))[:40]:
    print(f'{f(d["Warp Stall Sampling (All Samples)"]):7.0f} thr {f(d["Avg. Threads Executed"]):5.1f} {d["Address"][-5:]} {d["Source"].strip()[:70]}')
