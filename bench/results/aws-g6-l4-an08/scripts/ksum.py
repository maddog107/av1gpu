#!/usr/bin/env python3
"""ksum.py NCU.csv: total GPU time per kernel (ncu --csv --metrics gpu__time_duration.sum), grouped by the kernel
name with and without template arguments; percent of the total."""
import csv, collections, re, sys

rows = []
with open(sys.argv[1]) as f:
    lines = [l for l in f if l.startswith('"')]
for r in csv.DictReader(lines):
    if r.get("Metric Name") != "gpu__time_duration.sum":
        continue
    v = float(r["Metric Value"].replace(",", ""))
    unit = r.get("Metric Unit", "ns")
    v = v / 1e6 if unit in ("ns", "nsecond") else v / 1e3 if unit in ("us", "usecond") else v
    rows.append((r["Kernel Name"], v))
tot = sum(v for _, v in rows)
full = collections.defaultdict(lambda: [0.0, 0])
base = collections.defaultdict(lambda: [0.0, 0])
for k, v in rows:
    full[k][0] += v; full[k][1] += 1
    b = re.sub(r"<.*", "", k).replace("void ", "")
    base[b][0] += v; base[b][1] += 1
print(f"total {tot:.1f} ms over {len(rows)} kernels")
print("== by kernel")
for k, (v, n) in sorted(base.items(), key=lambda x: -x[1][0]):
    print(f"{v:10.1f} ms {100 * v / tot:5.1f} % {n:7d}  {k}")
print("== by instantiation (top 40)")
for k, (v, n) in sorted(full.items(), key=lambda x: -x[1][0])[:40]:
    print(f"{v:10.1f} ms {100 * v / tot:5.1f} % {n:7d}  {k[:150]}")
