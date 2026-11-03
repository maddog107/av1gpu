#!/usr/bin/env python3
"""mktable.py LOGDIR: the identity table (markdown) from identity.txt (first pass, binary = commits 1-4), identity_fix.txt
(sets remade, commits 1-6), ident_extra.txt (palette sets, short lists; commits 1-4) and ident_single.txt (single-file
mode and short lists, commits 1-6).  A set of the first pass that identity_fix.txt reran is listed from the rerun only."""
import re, sys
D = sys.argv[1]
pat = re.compile(r"^(\S+): inputs (\d+), files (\d+) / (\d+), differ (\d+)(?:, skipped ([0-9?]+))?(?:, rc (\S+))?(?:, rc mismatches (\d+))?(?:, failed runs (\d+))?.*opts \[(.*?)\]\s*(\S*)")
rows, seen = [], {}
def load(fn, binary):
    for ln in open(f"{D}/{fn}"):
        m = pat.match(ln.strip())
        if not m:
            continue
        name = m.group(1)
        key = (name, m.group(10), m.group(11))
        r = dict(name=name, inputs=int(m.group(2)), a=int(m.group(3)), b=int(m.group(4)), differ=int(m.group(5)),
                 skipped=m.group(6) or "", opts=m.group(10), env=m.group(11), binary=binary, src=fn)
        if name == "single_file":
            key = (name + fn, r["opts"], "")
        seen[key] = r
load("identity.txt", "1-4")
load("ident_extra.txt", "1-4")
load("identity_fix.txt", "1-6")
load("ident_single.txt", "1-6")
tot_a = tot_b = tot_d = 0
print("| set | options | branch state | inputs | main files | branch files | differ | skipped (both) |")
print("|---|---|---|---:|---:|---:|---:|---:|")
for k, r in seen.items():
    if r["src"] == "identity.txt" and r["name"].startswith("single_file"):
        continue   # superseded by ident_single.txt (the first pass lacked sets/gray/g0.png)
    o = r["opts"] + (f" {r['env']}" if r["env"] else "")
    print(f"| {r['name']} | `{o.strip() or 'default'}` | {r['binary']} | {r['inputs']} | {r['a']} | {r['b']} | {r['differ']} | {r['skipped']} |")
    tot_a += r["a"]; tot_b += r["b"]; tot_d += r["differ"]
print(f"| **all ({len([1 for k, r in seen.items() if not (r['src'] == 'identity.txt' and r['name'].startswith('single_file'))])} runs)** | | | | **{tot_a}** | **{tot_b}** | **{tot_d}** | |")
