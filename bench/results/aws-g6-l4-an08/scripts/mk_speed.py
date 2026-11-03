#!/usr/bin/env python3
"""mk_speed.py AB3.jsonl NOISE.jsonl SINGLE.jsonl SETS.json: markdown speed tables from spd_an.py's runs (median of the
6 timed runs; rounds = medians of each round's 3)."""
import json
import statistics as S
import sys

med = lambda v: S.median(v) if v else float('nan')  # noqa: E731

ab, noise, single, sets = sys.argv[1:5]


def load(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def table(runs, bins, label):
    out = []
    names = {"size_range/default": "size range (100 JPEGs), default", "standard/default": "standard (100 JPEGs), default",
             "thumbnails/default": "thumbnails (280 JPEGs), default", "thumbs/default": "thumbnails (280 JPEGs), default"}
    tags = []
    for r in runs:
        if r["tag"] not in tags:
            tags.append(r["tag"])
    res = {}
    for t in tags:
        for b in bins:
            rr = [r for r in runs if r["tag"] == t and r["bin"] == b and r["timed"] and r["rc"] == 0]
            mp = [r["stats"]["mp_s"] for r in rr]
            im = [r["stats"]["img_s"] for r in rr]
            wall = [r["stats"]["mp"] / r["wall_s"] for r in rr]
            res[(t, b)] = dict(n=len(rr), mp=med(mp), im=med(im), r1=med(mp[:3]), r2=med(mp[3:]),
                               wall=med(wall), bytes=sorted(set(r["stats"]["bytes"] for r in rr)))
    return tags, res, names


abr = load(ab)
tags, res, names = table(abr, ["new2", "new", "old"], "ab3")
L = ["## Speed (the stored method)", "",
     "`scripts/spd_an.py` (speed-v06's method): each set repeated to >= 1,000 MP (`bench.speed_set`), a warm-up and 3 "
     "timed runs (`--stats` MP/s), 2 rounds with the binary order alternating (new2, new, old / old, new, new2), median "
     "of the 6 timed runs; the default 3 encoders. `new` = change 1, `new2` = changes 1 + 2 (the branch), `old` = main. "
     "Raw runs: `logs/runs/ab3.jsonl`; the noise run (main against main, same method): `logs/runs/noise.jsonl`. "
     "Wall MP/s counts the whole process (CUDA start-up included).", "",
     "| set | main MP/s | change 1 MP/s | branch MP/s | change 1 vs main | change 2 vs change 1 | branch vs main | branch img/s | main img/s | rounds main | rounds branch | branch wall MP/s | main wall MP/s |",
     "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|"]
for t in tags:
    o, n1, n2 = res[(t, "old")], res[(t, "new")], res[(t, "new2")]
    L.append(f"| {names.get(t, t)} | {o['mp']:.2f} | {n1['mp']:.2f} | {n2['mp']:.2f} | {100 * (n1['mp'] / o['mp'] - 1):+.1f} % | "
             f"{100 * (n2['mp'] / n1['mp'] - 1):+.1f} % | **{100 * (n2['mp'] / o['mp'] - 1):+.1f} %** | {n2['im']:.1f} | {o['im']:.1f} | "
             f"{o['r1']:.1f} / {o['r2']:.1f} | {n2['r1']:.1f} / {n2['r2']:.1f} | {n2['wall']:.1f} | {o['wall']:.1f} |")
bsame = all(len(res[(t, b)]["bytes"]) == 1 for t in tags for b in ("new2", "new", "old")) and \
    all(res[(t, "new2")]["bytes"] == res[(t, "old")]["bytes"] == res[(t, "new")]["bytes"] for t in tags)
L += ["", f"Output bytes were the same in every timed run of every binary: {bsame}."]
nr = load(noise)
ntags, nres, _ = table(nr, ["old2", "old"], "noise")
L += ["", "**Noise run** (main against main, the same binary under two labels, same alternation):", "",
      "| set | main (label old2) MP/s | main (label old) MP/s | difference | rounds old2 | rounds old |", "|---|---:|---:|---:|---|---|"]
for t in ntags:
    a, b = nres[(t, "old2")], nres[(t, "old")]
    L.append(f"| {names.get(t, t)} | {a['mp']:.2f} | {b['mp']:.2f} | {100 * (a['mp'] / b['mp'] - 1):+.1f} % | {a['r1']:.1f} / {a['r2']:.1f} | "
             f"{b['r1']:.1f} / {b['r2']:.1f} |")
sr = load(single)
files = []
for r in sr:
    if r["file"] not in files:
        files.append(r["file"])
L += ["", "**Single-JPEG cold process** (`av1gpu in.jpg out.avif`, no `--batch`; wall time per process including CUDA "
      "start-up; a warm-up, then 12 runs per binary, order alternating; median; `logs/runs/single.jsonl`):", "",
      "| file | MP | main ms | change 1 ms | branch ms | branch vs main |", "|---|---:|---:|---:|---:|---:|"]
for f in files:
    m = {b: S.median([r["wall_s"] for r in sr if r["file"] == f and r["bin"] == b and r["rc"] == 0]) * 1000 for b in ("old", "new", "new2")}
    mp = next(r["mp"] for r in sr if r["file"] == f)
    L.append(f"| {f} | {mp:.2f} | {m['old']:.1f} | {m['new']:.1f} | {m['new2']:.1f} | {100 * (m['new2'] / m['old'] - 1):+.1f} % |")
try:
    sj = json.load(open(sets))
    L += ["", "Speed sets (files, MP, repeats): " + "; ".join(f"{k}: {v[0]} files, {v[1]:.0f} MP, x{v[2]}" for k, v in sj.items()) + "."]
except Exception:
    pass
print("\n".join(L))
