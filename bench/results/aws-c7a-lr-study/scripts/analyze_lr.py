#!/usr/bin/env python3
"""Tables of the loop-restoration study from ~/lr/rows/{enc,score}.jsonl (lrstudy.py).

  analyze_lr.py OUTDIR       -> OUTDIR/tables.md, OUTDIR/results.json, OUTDIR/rows/{rows.csv.gz, bd_per_image.csv}

BD-rate: bench/bench.py's bd_rate (PCHIP on log size against SSIMULACRA2, per image over the 5 quality points, curves
overlapping by at least 2 points), then the median and mean over images; negative = smaller files at equal quality.
The same on RGB PSNR (overlap 0.2 dB), the metric family libaom's LR search minimises (squared error), and on
butteraugli's 3-norm (bscore.py; negated, overlap 0.1) and CVVDP JOD (cvscore.py, standard_fhd; overlap 0.05 JOD, as
bench.py) for the configurations and images they were run on.
q60 (av1gpu's default qindex 120): total bytes and mean SSIMULACRA2 of both configurations, and the size at equal
quality there: the test curve's size at the anchor's q60 score, against the anchor's q60 size (median over images).
CPU: summed child user + system seconds of all encodes of a configuration, against its anchor."""
import collections
import csv
import gzip
import json
import os
import statistics
import sys

import numpy as np

H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/bench")
from bench import _curve, _pchip_eval, bd_rate  # noqa: E402

OUT = sys.argv[1]
os.makedirs(f"{OUT}/rows", exist_ok=True)
enc = [json.loads(l) for l in open(f"{H}/lr/rows/enc.jsonl")]
sco = [json.loads(l) for l in open(f"{H}/lr/rows/score.jsonl")]
gamut_cls = {r["file"]: r["size_class"] for r in csv.DictReader(open(f"{H}/bench/openimages_v4_gamut.csv"))}
CLASSES = ["tiny", "small", "typical", "large", "huge"]

E = {}
for r in enc:
    if r["rc"] == 0:
        E[(r["fmt"], r["cfg"], r["set"], r["image"], r["q"])] = r
S = {}
for r in sco:
    S[(r["fmt"], r["cfg"], r["set"], r["image"], r["q"])] = r
bp = f"{H}/lr/rows/butter.jsonl"   # bscore.py: butteraugli 3-norm (lower is better; negated for the BD-rate)
for l in (open(bp) if os.path.exists(bp) else []):
    r = json.loads(l)
    k = (r["fmt"], r["cfg"], r["set"], r["image"], r["q"])
    if k in S and r.get("butteraugli") is not None:
        S[k]["nbutter"] = -r["butteraugli"]
cp = f"{H}/lr/rows/cvvdp.jsonl"   # cvscore.py: CVVDP JOD (standard_fhd), as bench.py --cvvdp
for l in (open(cp) if os.path.exists(cp) else []):
    r = json.loads(l)
    k = (r["fmt"], r["cfg"], r["set"], r["image"], r["q"])
    if k in S and r.get("jod") is not None:
        S[k]["jod"] = r["jod"]


def groups(s, image):
    g = [s]
    if s == "gamut":
        g.append("gamut:" + gamut_cls[image])
    if s in ("std", "gamut", "kodak"):
        g.append("photos")
    return g


def curves(fmt, cfg, metric):
    c = collections.defaultdict(list)
    for (f, k, s, im, q), r in S.items():
        if f == fmt and k == cfg and r.get(metric) is not None:
            c[(s, im)].append((r["bytes"], r[metric]))
    return c


PAIRS = [  # (test, anchor, label)
    ("p6keep", "s6", "speed 6 + LR as libaom searches it at speed 4 (patched)"),
    ("p6full", "s6", "speed 6 + LR, full search (patched)"),
    ("p6wiener", "s6", "speed 6 + Wiener only, full search"),
    ("p6sgr", "s6", "speed 6 + self-guided only, full search"),
    ("s4r1", "s4", "speed 4: LR on (libaom's default) vs off"),
    ("p4full", "s4", "speed 4 + LR, full search, vs speed 4 LR off"),
    ("s4", "s6", "speed 4 LR off vs speed 6 (the rest of speed 4)"),
    ("s4r1", "s6", "speed 4 default vs speed 6 default"),
]
GORDER = ["photos", "std", "kodak", "gamut"] + ["gamut:" + c for c in CLASSES]
GNAME = {"photos": "All photos", "std": "Standard JPEGs", "kodak": "Kodak", "gamut": "Size range"}
for c in CLASSES:
    GNAME["gamut:" + c] = f"Size range: {c}"


def bd_pairs(fmt, metric, overlap):
    res, per_image = {}, []
    for t, a, lab in PAIRS:
        ct, ca = curves(fmt, t, metric), curves(fmt, a, metric)
        if not ct or not ca:
            continue
        vals = collections.defaultdict(list)
        for key in ct:
            if key not in ca:
                continue
            v = bd_rate(ct[key], ca[key], overlap)
            if v is None:
                continue
            per_image.append(dict(fmt=fmt, metric=metric, test=t, anchor=a, set=key[0], image=key[1], bd=round(v, 6)))
            for g in groups(*key):
                vals[g].append(v)
        res[f"{t}_vs_{a}"] = {g: dict(n=len(v), median=statistics.median(v), mean=statistics.fmean(v))
                              for g, v in vals.items()}
    return res, per_image


def q60(fmt, t, a):
    out = collections.defaultdict(lambda: dict(bt=0, ba=0, st=[], sa=[], eq=[]))
    ca = curves(fmt, a, "ssim2")
    ct = curves(fmt, t, "ssim2")
    for (f, k, s, im, q), ra in S.items():
        if f != fmt or k != a or q != 60:
            continue
        rt = S.get((fmt, t, s, im, 60))
        if not rt or ra.get("ssim2") is None or rt.get("ssim2") is None:
            continue
        eq = None
        xs, ys = _curve(ct[(s, im)])
        if len(xs) >= 2 and xs[0] <= ra["ssim2"] <= xs[-1]:
            eq = float(np.exp(_pchip_eval(xs, ys, np.array([ra["ssim2"]]))[0]) / ra["bytes"]) - 1
        for g in groups(s, im):
            o = out[g]
            o["bt"] += rt["bytes"]
            o["ba"] += ra["bytes"]
            o["st"].append(rt["ssim2"])
            o["sa"].append(ra["ssim2"])
            if eq is not None:
                o["eq"].append(eq)
    return {g: dict(n=len(o["st"]), bytes=o["bt"] / o["ba"] - 1, ssim2_test=statistics.fmean(o["st"]),
                    ssim2_anchor=statistics.fmean(o["sa"]), eq_median=statistics.median(o["eq"]) if o["eq"] else None,
                    eq_n=len(o["eq"])) for g, o in out.items()}


def cpu(fmt):
    tot = collections.defaultdict(float)
    for (f, k, s, im, q), r in E.items():
        if f == fmt:
            for g in groups(s, im):
                tot[(k, g)] += r["cpu"]
    return tot


def lr_use(fmt):
    """per config and group: files with LR in the sequence header, files with LR on in any plane / per plane, and the
    restoration units by type per plane (none / Wiener / self-guided) over the frames that use LR in that plane"""
    u = collections.defaultdict(lambda: dict(files=0, seq=0, any=0, plane=[0, 0, 0], units=[[0, 0, 0] for _ in range(3)],
                                             ftype=[collections.Counter() for _ in range(3)], usize=collections.Counter()))
    for (f, k, s, im, q), r in S.items():
        if f != fmt or not r.get("lr"):
            continue
        lr = r["lr"]
        for g in groups(s, im) + [f"q{q}"]:
            o = u[(k, g)]
            o["files"] += 1
            o["seq"] += lr["seq_lr"]
            on = [p["type"] != 0 for p in lr["planes"]]
            o["any"] += any(on)
            for i, p in enumerate(lr["planes"]):
                o["plane"][i] += on[i]
                o["ftype"][i][p["type"]] += 1
                if on[i]:
                    o["units"][i][0] += p["none"]
                    o["units"][i][1] += p["wiener"]
                    o["units"][i][2] += p["sgr"]
                    if i < 2:
                        o["usize"][("Y" if i == 0 else "UV") + str(p["unit"])] += 1
    return u


def pct(x, d=1):
    return "" if x is None else f"{100 * x:+.{d}f} %"


md = []
results = {}
per_image_all = []
for fmt in ("420", "444"):
    bd, pi = bd_pairs(fmt, "ssim2", 2.0)
    bdp, pip = bd_pairs(fmt, "psnr", 0.2)
    bdb, pib = bd_pairs(fmt, "nbutter", 0.1)
    bdc, pic = bd_pairs(fmt, "jod", 0.05)
    per_image_all += pi + pip + pib + pic
    results[fmt] = dict(bd_ssim2=bd, bd_psnr=bdp, bd_butteraugli=bdb, bd_cvvdp=bdc)
    if not bd:
        continue
    for name, tab in (("SSIMULACRA2", bd), ("RGB PSNR", bdp), ("butteraugli 3-norm", bdb), ("CVVDP (JOD)", bdc)):
        if not tab:
            continue
        md.append(f"\n## {fmt}: BD-rate on {name}, median (mean)\n")
        cols = [k for k in tab]
        md.append("| Images | " + " | ".join(cols) + " |")
        md.append("|---|" + "---:|" * len(cols))
        for g in GORDER:
            cells = []
            for k in cols:
                v = tab[k].get(g)
                cells.append(f"{pct(v['median'])} ({100 * v['mean']:+.1f}) n={v['n']}" if v else "")
            if any(cells):
                md.append(f"| {GNAME[g]} | " + " | ".join(cells) + " |")
    # q60
    md.append(f"\n## {fmt}: at q60 (qindex 120): bytes, mean SSIMULACRA2 anchor -> test, size at equal quality (median)\n")
    results[fmt]["q60"] = {}
    for t, a, _ in PAIRS:
        qq = q60(fmt, t, a)
        if not qq:
            continue
        results[fmt]["q60"][f"{t}_vs_{a}"] = qq
        md.append(f"\n{t} vs {a}:\n")
        md.append("| Images | n | bytes | SSIMULACRA2 | equal-quality size (n) |")
        md.append("|---|---:|---:|---:|---:|")
        for g in GORDER:
            v = qq.get(g)
            if v:
                md.append(f"| {GNAME[g]} | {v['n']} | {pct(v['bytes'], 2)} | {v['ssim2_anchor']:.2f} -> {v['ssim2_test']:.2f} | "
                          f"{pct(v['eq_median'], 2)} ({v['eq_n']}) |")
    # cpu
    tot = cpu(fmt)
    cfgs = sorted({k for k, g in tot})
    md.append(f"\n## {fmt}: CPU seconds (all 5 quality points), and against the anchor\n")
    md.append("| Images | " + " | ".join(cfgs) + " |")
    md.append("|---|" + "---:|" * len(cfgs))
    results[fmt]["cpu"] = {f"{k}|{g}": v for (k, g), v in tot.items()}
    for g in GORDER:
        if (cfgs[0], g) in tot:
            md.append(f"| {GNAME[g]} | " + " | ".join(f"{tot.get((k, g), 0):.0f}" for k in cfgs) + " |")
    md.append("\n| Images | " + " | ".join(f"{t}/{a}" for t, a, _ in PAIRS) + " |")
    md.append("|---|" + "---:|" * len(PAIRS))
    for g in GORDER:
        if ("s6", g) in tot:
            md.append(f"| {GNAME[g]} | " + " | ".join(
                f"{tot[(t, g)] / tot[(a, g)]:.3f}" if (t, g) in tot and (a, g) in tot else "" for t, a, _ in PAIRS) + " |")
    # LR use
    u = lr_use(fmt)
    results[fmt]["lr_use"] = {f"{k}|{g}": dict(files=o["files"], seq=o["seq"], any=o["any"], plane=o["plane"],
                                               units=o["units"], ftype=[dict(c) for c in o["ftype"]],
                                               usize=dict(o["usize"])) for (k, g), o in u.items()}
    md.append(f"\n## {fmt}: LR use: files with LR in the sequence header / any plane / Y U V; units none:Wiener:SGR per plane\n")
    md.append("| Config | Images | files | seq | any | Y | U | V | Y units n:W:S | U units | V units | unit sizes (frames) |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|")
    for k in sorted({k for k, g in u}):
        for g in ["photos", "std", "kodak", "gamut"] + ["gamut:" + c for c in CLASSES] + [f"q{q}" for q in (30, 45, 60, 75, 90)]:
            o = u.get((k, g))
            if not o:
                continue
            uu = [":".join(str(x) for x in o["units"][i]) for i in range(3)]
            md.append(f"| {k} | {GNAME.get(g, g)} | {o['files']} | {o['seq']} | {o['any']} | {o['plane'][0]} | {o['plane'][1]} | "
                      f"{o['plane'][2]} | {uu[0]} | {uu[1]} | {uu[2]} | {dict(o['usize'])} |")

open(f"{OUT}/tables.md", "w").write("# Loop-restoration study: tables (analyze_lr.py)\n" + "\n".join(md) + "\n")
json.dump(results, open(f"{OUT}/results.json", "w"), indent=1, sort_keys=True)
with open(f"{OUT}/rows/bd_per_image.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["fmt", "metric", "test", "anchor", "set", "image", "bd"])
    w.writeheader()
    w.writerows(per_image_all)
with gzip.open(f"{OUT}/rows/rows.csv.gz", "wt", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fmt", "cfg", "set", "image", "q", "mp", "bytes", "cpu_s", "wall_s", "ssim2", "psnr", "seq_lr",
                "lr_type_yuv", "lr_unit_y", "units_y_none_w_sgr", "units_u", "units_v"])
    for key in sorted(E):
        r, s = E[key], S.get(key, {})
        lr = s.get("lr") or {}
        pl = lr.get("planes", [])
        w.writerow(list(key) + [round(r["mp"], 4), r["bytes"], round(r["cpu"], 3), round(r["wall"], 3),
                                s.get("ssim2"), None if s.get("psnr") is None else round(s["psnr"], 4), lr.get("seq_lr"),
                                "".join(str(p["type"]) for p in pl), pl[0]["unit"] if pl else None] +
                   [":".join(str(p[x]) for x in ("none", "wiener", "sgr")) for p in pl] + [""] * (3 - len(pl)))
print(open(f"{OUT}/tables.md").read())
