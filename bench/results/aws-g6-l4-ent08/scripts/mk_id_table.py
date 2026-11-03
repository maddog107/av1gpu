#!/usr/bin/env python3
"""mk_id_table.py identity.txt identity_rerun.txt identity_single_br1.txt identity_single_br2.txt: the identity table
(markdown).  A rerun line replaces the first run's line of the same set (the sets that were empty in the first run)."""
import re, sys

rows, order = {}, []
pat = re.compile(r"^(\S+): inputs (\d+), files (\d+) / (\d+), differ (\d+), skipped (\S+), rc (\d+)/(\d+), digest (\S*), opts \[(.*)\]")
for fn in sys.argv[1:3]:
    for line in open(fn):
        m = pat.match(line)
        if m:
            if m.group(1) not in rows:
                order.append(m.group(1))
            rows[m.group(1)] = m.groups() + (fn.split("/")[-1],)
single = []
for fn in sys.argv[3:]:
    for line in open(fn):
        m = re.match(r"single_file: inputs (\d+), files (\d+) / (\d+), differ (\d+), failed runs (\d+), digest (\S+), opts \[(.*)\]", line)
        if m:
            single.append((fn.split("identity_single_")[-1].replace(".txt", ""),) + m.groups())
print("| set | options | inputs | main files | branch files | differ | skipped (both) | rc | from |")
print("|---|---|---:|---:|---:|---:|---:|---|---|")
tin = tf = td = 0
for n in order:
    g = rows[n]
    print(f"| {g[0]} | `{g[9] or 'default'}` | {g[1]} | {g[2]} | {g[3]} | {g[4]} | {g[5]} | {g[6]}/{g[7]} | {g[10]} |")
    tf += int(g[2]); td += int(g[4])
for s in single:
    print(f"| single-file mode ({s[0]}) | `{s[7] or 'default'}` | {s[1]} | {s[2]} | {s[3]} | {s[4]} | | {s[5]} failed runs | identity_single_{s[0]}.txt |")
    tf += int(s[2]); td += int(s[4])
print(f"| **all ({len(order)} batch runs, {len(single)} single-file runs)** | | | **{tf}** | | **{td}** | | | |")
