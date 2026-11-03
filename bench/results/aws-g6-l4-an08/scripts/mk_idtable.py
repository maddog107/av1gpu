#!/usr/bin/env python3
"""mk_idtable.py IDENTITY.txt...: markdown identity table (one row per cmpset.sh / single-file line) and a total."""
import re
import sys

rows = []
tot = [0, 0, 0, 0]
pat = re.compile(r"(\w+): inputs (\d+), files (\d+) / (\d+), differ (\d+), (?:skipped (\S+), rc (\S+)|failed runs (\d+)), "
                 r"digest \w+, opts \[([^\]]*)\]\s*(.*)")
for fn in sys.argv[1:]:
    for l in open(fn):
        m = pat.match(l.strip())
        if not m:
            continue
        n, ni, fa, fb, d, sk, rc, fr, opts, env = m.groups()
        o = f"`{opts}`" if opts else "default"
        if env:
            o += f", `{env}`"
        rows.append(f"| {n} | {o} | {ni} | {fa} | {fb} | {d} | {sk if sk is not None else ''} |")
        tot[0] += 1
        tot[1] += int(fa)
        tot[2] += int(fb)
        tot[3] += int(d)
print("| set | options | inputs | main files | branch files | differ | skipped (both) |")
print("|---|---|---:|---:|---:|---:|---:|")
print("\n".join(rows))
print(f"| **all ({tot[0]} runs)** | | | **{tot[1]:,}** | **{tot[2]:,}** | **{tot[3]}** | |")
