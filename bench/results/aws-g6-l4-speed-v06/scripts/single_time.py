#!/usr/bin/env python3
"""Single-file mode (no --batch), wall time per process including CUDA start-up: the 10 files of ~/sets/single one
at a time, a warm-up pass, then 3 rounds with the binary order alternating; ~/results/single.jsonl (one line per
process) and a summary on stdout."""
import glob, json, os, statistics, subprocess, time
H = os.path.expanduser("~")
BIN = {"new": f"{H}/new/build/av1gpu", "old": f"{H}/old/build/av1gpu"}
files = sorted(glob.glob(f"{H}/sets/single/*"))
out = f"{H}/work/single_t"
os.makedirs(out, exist_ok=True)
rows = []
for rnd in range(4):
    for b in (("new", "old") if rnd % 2 == 0 else ("old", "new")):
        for f in files:
            t0 = time.time()
            p = subprocess.run([BIN[b], f, f"{out}/{os.path.basename(f)}.avif"], capture_output=True)
            r = dict(bin=b, round=rnd, file=os.path.basename(f), rc=p.returncode, wall_s=round(time.time() - t0, 3),
                     bytes=os.path.getsize(f"{out}/{os.path.basename(f)}.avif"), timed=rnd > 0)
            rows.append(r)
            with open(f"{H}/results/single.jsonl", "a") as fo:
                fo.write(json.dumps(r) + "\n")
for b in ("new", "old"):
    tot = [sum(r["wall_s"] for r in rows if r["bin"] == b and r["round"] == k) for k in range(1, 4)]
    per = {f: statistics.median(r["wall_s"] for r in rows if r["bin"] == b and r["file"] == f and r["timed"]) for f in
           sorted({r["file"] for r in rows})}
    print(b, "10 files, total s per round", [round(x, 2) for x in tot], "median", round(statistics.median(tot), 2),
          "| per file median", {k: v for k, v in per.items()}, "| failures", sum(1 for r in rows if r["bin"] == b and r["rc"]))
