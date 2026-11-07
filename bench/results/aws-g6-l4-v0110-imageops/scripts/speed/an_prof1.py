#!/usr/bin/env python3
import glob, json, os, re, sys
d = sys.argv[1]
for f in sorted(glob.glob(f"{d}/*.an")):
    n = os.path.basename(f)[:-3]; t = open(f).read()
    k = {}
    for m in re.finditer(r"^\s+([\d.]+) ms n=\s*(\d+) avg\s+([\d.]+) us (\S+)", t, re.M):
        k[m.group(4)] = (float(m.group(1)), int(m.group(2)), float(m.group(3)))
    w = re.search(r"host wait for trim / tone_range results ([\d.]+) ms in (\d+) waits", t)
    sp = re.search(r"span ([\d.]+) ms, any-kernel busy ([\d.]+) ms = ([\d.]+) %", t)
    try:
        s = json.loads(open(f[:-3] + ".out").read().strip().splitlines()[-1])
    except Exception:
        s = {}
    extra = " ".join(f"{a}={v[0]:.1f}ms/{v[1]}" for a, v in k.items() if a != "k_entropy_win")
    print(f"{n:16s} wall {s.get('wall_ms', 0):7.0f} MP {s.get('mp', 0):6.1f} busy {sp.group(3)}% wait {w.group(1):>7s}/{w.group(2):>3s} | {extra}")
