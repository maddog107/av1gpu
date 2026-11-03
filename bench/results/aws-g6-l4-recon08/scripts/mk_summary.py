#!/usr/bin/env python3
"""Assemble bench/results/aws-g6-l4-recon08/summary.md (the PR's results page) from the logs: mk_summary.py RESULTS_DIR"""
import os, re, sys

D = sys.argv[1]
L = os.path.join(D, "logs")
idtable = os.popen(f"python3 {D}/scripts/mk_identity_table8.py {L}/identity.txt").read().strip()
speed = open(f"{L}/speed_final.txt").read() if os.path.exists(f"{L}/speed_final.txt") else ""

# speed table from speed_final.txt (spd8.py output)
rows, cur = [], None
for l in speed.splitlines():
    m = re.match(r"== (\S+)\s*(.*?): (.*)", l)
    if m:
        cur = [m.group(1), m.group(3), {}]
        rows.append(cur)
        continue
    m = re.match(r"\s+(\w+): ([0-9.]+) MP/s, ([0-9.]+) img/s\s+rounds \[([^\]]*)\]\s+vs main ([+-][0-9.]+) %", l)
    if m and cur:
        cur[2][m.group(1)] = ("batch", float(m.group(2)), float(m.group(3)), m.group(4), m.group(5))
        continue
    m = re.match(r"\s+(\w+): median ([0-9.]+) ms \(min ([0-9.]+), max ([0-9.]+)\)\s+vs main ([+-][0-9.]+) %", l)
    if m and cur:
        cur[2][m.group(1)] = ("single", float(m.group(2)), float(m.group(3)), float(m.group(4)), m.group(5))
names = {"size_range_default": "size range (100 JPEGs), default", "standard_default": "standard (100 JPEGs), default",
         "thumbs_default": "thumbnails (280 JPEGs), default", "single_jpeg_8mp": "single JPEG, cold process",
         "single_jpeg_2.5mp": "single JPEG, cold process", "single_thumb_512x384": "single thumbnail, cold process"}
bt = ["| set | repeated to | main | perf/recon | change | main again (noise) | noise |", "|---|---|---:|---:|---:|---:|---:|"]
st = ["| file | main | perf/recon | change | main again (noise) | noise |", "|---|---:|---:|---:|---:|---:|"]
for name, info, r in rows:
    a, b, c = r.get("main"), r.get("recon08"), r.get("main_noise")
    if not (a and b and c):
        continue
    if a[0] == "batch":
        unit = "img/s" if name.startswith("thumbs") else "MP/s"
        k = 2 if unit == "img/s" else 1
        bt.append(f"| {names.get(name, name)} | {info} | {a[k]:.2f} {unit} | {b[k]:.2f} | {100 * (b[k] / a[k] - 1):+.1f} % | {c[k]:.2f} | "
                  f"{100 * (c[k] / a[k] - 1):+.1f} % |")
    else:
        st.append(f"| {names.get(name, name)}: {info} | {a[1]:.0f} ms | {b[1]:.0f} ms | {100 * (b[1] / a[1] - 1):+.1f} % | {c[1]:.0f} ms | "
                  f"{100 * (c[1] / a[1] - 1):+.1f} % |")
tmpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary_tmpl.txt")).read()
out = tmpl.replace("@@IDTABLE@@", idtable).replace("@@BATCH@@", "\n".join(bt)).replace("@@SINGLE@@", "\n".join(st))
open(os.path.join(D, "summary.md"), "w").write(out)
print(out[:400])
