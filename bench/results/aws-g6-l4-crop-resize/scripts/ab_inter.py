#!/usr/bin/env python3
"""ab_inter.py: run-by-run interleaved A/B on the size-range speed set (1,434 MP), no new options: main, branch and
a second copy of main's binary (control), in the order main, cr, main2 repeated 8 times after one warm-up each;
--stats MP/s.  ~/results/ab_inter.txt"""
import json, os, shutil, statistics, subprocess

H = os.path.expanduser("~")
shutil.copy(f"{H}/main/build/av1gpu", f"{H}/main/build/av1gpu_copy")
B = {"main": f"{H}/main/build/av1gpu", "cr": f"{H}/cr/build/av1gpu", "main2": f"{H}/main/build/av1gpu_copy"}
src = f"{H}/work/spd/sr/speedset"
res = {k: [] for k in B}


def one(k):
    out = f"{H}/work/spd/out_ab"
    shutil.rmtree(out, ignore_errors=True)
    p = subprocess.run([B[k], "--batch", src, "--outdir", out, "--stats"], capture_output=True, text=True)
    return json.loads(p.stdout.strip().splitlines()[-1])["mp_s"]


for k in B:
    one(k)
for i in range(8):
    for k in ("main", "cr", "main2"):
        res[k].append(one(k))
with open(f"{H}/results/ab_inter.txt", "w") as f:
    for k in res:
        f.write(f"{k}: median {statistics.median(res[k]):.1f} MP/s, runs {', '.join(f'{x:.1f}' for x in res[k])}\n")
    m = statistics.median(res["main"])
    f.write(f"cr / main {statistics.median(res['cr']) / m:.4f}, main2 / main {statistics.median(res['main2']) / m:.4f}\n")
print(open(f"{H}/results/ab_inter.txt").read())
