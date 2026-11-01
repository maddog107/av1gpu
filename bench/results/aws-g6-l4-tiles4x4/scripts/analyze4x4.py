#!/usr/bin/env python3
"""Every av1gpu compression number of docs/RESULTS.md and the README, from the stored baselines (bench/results/) and a
set of av1gpu rows: the stored ones (must reproduce the published numbers), or the 4x4 run's (RUN_DIR = the pulled
~/results of the instance).

  analyze4x4.py WORKTREE [RUN_DIR] [--json NUMBERS.json] [--write RESULTS_DIR]"""
import csv, json, math, os, statistics, sys

WT = sys.argv[1]
RUN = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None  # noqa: E731
sys.path.insert(0, os.path.join(WT, "bench"))
import bench  # noqa: E402

RES = os.path.join(WT, "bench", "results")
L = lambda n: json.load(open(os.path.join(RES, n, "results.json")))  # noqa: E731
gam, ns, cvv, fmt, nv, svt, tun, std, stdns, enc4, enc2 = (L(n) for n in (
    "aws-g6-l4-gamut", "aws-g6-l4-gamut-turbo-nosmall", "aws-g6-l4-cvvdp", "aws-g6-l4-formats", "aws-g6-l4-nvenc",
    "aws-c6a-gamut-svt", "aws-c6a-cpu-tuning", "aws-g6-l4", "aws-g6-l4-turbo-nosmall", "aws-g6-l4-gamut-enc4",
    "aws-g6-l4-gamut-enc2"))
man = {m["file"]: m for m in csv.DictReader(open(os.path.join(WT, "bench", "openimages_v4_gamut.csv")))}
SIZES = {k: int(m["width"]) * int(m["height"]) / 1e6 for k, m in man.items()}
MISSING = "9fba6f263f3c4f3e_tiny.jpg"
Q6 = [30, 40, 50, 60, 70, 80]
FMTS = ("420_8", "444_8", "420_10", "444_10")


def webp_fixed(rows):
    fix = {(x["image"], x["q"]): x["ssimulacra2_icc"] for x in cvv["webp_icc_check"]["rows"]}
    return [dict(x, ssimulacra2=fix.get((x["image"], x["q"]), x["ssimulacra2"])) for x in rows]


def per(rows, metric):
    o = {}
    for r in rows:
        o.setdefault(r["image"], []).append((r["bytes"], r.get(metric)))
    return o


def bd(test, anchor, metric="ssimulacra2", min_overlap=2.0, images=None):
    """({n, median, mean}, {image: bd}) of rows `test` against rows `anchor` (fractions)"""
    t, a = per(test, metric), per(anchor, metric)
    vals = {im: bench.bd_rate(t[im], a[im], min_overlap) for im in t if im in a and (images is None or im in images)}
    vals = {k: v for k, v in vals.items() if v is not None}
    v = list(vals.values())
    return (dict(n=len(v), median=statistics.median(v), mean=statistics.fmean(v)) if v else None), vals


def by_size(vals):
    o = {}
    for cls, lo, hi in bench.SIZE_CLASSES:
        v = [x for im, x in vals.items() if lo <= SIZES[im] < hi]
        if v:
            o[cls] = dict(n=len(v), median=statistics.median(v))
    return o


def interp(b, p0, p1):   # log-linear speed between (bd0, MP/s 0) and (bd1, MP/s 1)
    (b0, s0), (b1, s1) = p0, p1
    return math.exp(math.log(s0) + (b - b0) / (b1 - b0) * math.log(s1 / s0))


# ---------------------------------------------------------------- av1gpu rows
def rows_published():
    return dict(default6=gam["quality"]["av1gpu"], nosmall6=ns["quality"]["av1gpu"], default8=cvv["quality"]["av1gpu"],
                fmt={f: fmt["quality"][f"av1gpu_{f}"] for f in FMTS}, turbo3ns=None, turbo3=None,
                luma=[r for r in cvv["cvvdp_luma_only"]["rows"] if r["codec"] == "av1gpu"])


def rows_run(which):
    sc = json.load(open(os.path.join(RUN, "scores.json")))

    def pick(cfg, qs=None):
        return [dict(image=r["image"], q=r["q"], bytes=r["bytes"], ssimulacra2=r["ssimulacra2"],
                     **({"cvvdp": r["cvvdp"]} if "cvvdp" in r else {}))
                for r in sc if r["config"] == cfg and r["bin"] == which and (qs is None or r["q"] in qs)]
    o = dict(default8=pick("gamut_default"), default6=pick("gamut_default", Q6), nosmall6=pick("gamut_nosmall"),
             fmt={"420_8": pick("gamut_default", Q6), **{f: pick(f"fmt_{f}") for f in FMTS[1:]}},
             turbo3ns=pick("gamut_turbo3ns"), turbo3=pick("gamut_turbo3"))
    if which == "new":
        lv = json.load(open(os.path.join(RUN, "luma_variant.json")))["variant"]
        new = {(r["image"], r["q"]): r for r in sc if r["config"] == "gamut_default" and r["bin"] == "new"}
        luma = []
        for r in cvv["cvvdp_luma_only"]["rows"]:
            if r["codec"] != "av1gpu":
                continue
            n = new[(r["image"], r["q"])]
            if n["md5"] and SIZES[r["image"]] >= 2:   # re-encoded: its own luma-only score
                v = n.get(lv) if n.get(lv) is not None else n.get("cvvdp_luma")
                luma.append(dict(image=r["image"], q=r["q"], bytes=n["bytes"], cvvdp_luma=v))
            else:                                    # unchanged file (< 2 MP): the published score
                assert n["bytes"] == r["bytes"], r
                luma.append(r)
        o["luma"] = luma
    return o


def numbers(A):
    N = {}
    aom6, webp = gam["quality"]["aom_s6"], webp_fixed(gam["quality"]["webp"])
    d, dv = bd(A["default6"], aom6)
    n_, nvv = bd(A["nosmall6"], aom6)
    N["default"], N["default_by_size"] = d, by_size(dv)
    N["nosmall"], N["nosmall_by_size"] = n_, by_size(nvv)
    N["default_vs_webp"] = bd(A["default6"], webp)[0]
    N["default_per_image"] = dv
    for k in ("turbo3ns", "turbo3"):
        if A.get(k):
            N[k] = bd(A[k], aom6)[0]
    # one-tile libaom speed 6 anchor and SVT-AV1 presets 4 / 6 against it
    sw = tun["quality"]["sweeps"]
    nt = sw["stock_aom_s6_notiles"]["rows"]
    N["default_notiles"] = bd(A["default6"], nt)[0]
    N["nosmall_notiles"] = bd(A["nosmall6"], nt)[0]
    p4n, p6n = bd(sw["stock_svt_p4_notiles"]["rows"], nt)[0], bd(sw["stock_svt_p6_notiles"]["rows"], nt)[0]
    tm = {}
    for t in tun["primary"]["timings"]:   # the first (main) measurement of each label
        tm.setdefault(t["label"], t["mp_s"])
    t4, t6 = svt["bd_rate"]["svt_p4_vs_aom_s6"]["median"], svt["bd_rate"]["svt_p6_vs_aom_s6"]["median"]
    x = N["default_notiles"]["median"]
    N["svt_notiles"] = dict(p4=p4n, p6=p6n)
    N["svt46_interp"] = dict(
        stock_untiled=interp(x, (p4n["median"], svt_speed(4)), (p6n["median"], svt_speed(6))),
        pgo_untiled=interp(x, (p4n["median"], tm["pgo_svt_p4_j1"]), (p6n["median"], tm["pgo_svt_p6_j1"])),
        pgo_tiled_8x2=interp(d["median"], (t4, tm["pgo_svt_p4_j2"]), (t6, tm["pgo_svt_p6_j2"])),
        stock_tiled=interp(d["median"], (t4, svt_speed(4)), (t6, svt_speed(6))))
    # the README's "2.4-4x faster than the best CPU encoder" (SVT-AV1 interpolated at av1gpu's sizes)
    N["svt_interp_at"] = {k: svt_at(N[k]["median"]) for k in ("default", "nosmall", "turbo3ns") if N.get(k)}
    # NVENC
    nvr = nv["quality"]["nvenc"]
    even = {im for im in SIZES if int(man[im]["height"]) % 2 == 0}
    N["vs_nvenc"] = bd(A["default6"], nvr)[0]
    N["vs_nvenc_even"] = bd(A["default6"], nvr, images=even)[0]
    # totals at each quality point (100 photos, and the 99 NVENC photos at q60)
    N["totals"] = {q: dict(images=len(s), bytes=sum(r["bytes"] for r in s), ssimulacra2=statistics.fmean(r["ssimulacra2"] for r in s))
                   for q in Q6 for s in [[r for r in A["default6"] if r["q"] == q]]}
    q60 = [r for r in A["default6"] if r["q"] == 60 and r["image"] != MISSING]
    N["q60_99"] = dict(bytes=sum(r["bytes"] for r in q60), ssimulacra2=statistics.fmean(r["ssimulacra2"] for r in q60))
    N["q60_files"] = {r["image"]: (r["bytes"], r["ssimulacra2"]) for r in A["default6"] if r["q"] == 60}
    # formats
    F = {}
    for f in FMTS:
        F[f"av1gpu_{f}_vs_aom_s6_{f}"] = bd(A["fmt"][f], fmt["quality"][f"aom_s6_{f}"])[0]
    for a_, b_ in (("444_8", "420_8"), ("444_10", "420_10"), ("420_10", "420_8"), ("444_10", "444_8")):
        F[f"av1gpu_{a_}_vs_av1gpu_{b_}"] = bd(A["fmt"][a_], A["fmt"][b_])[0]
    N["formats"] = F
    # CVVDP (99 photos, 8 points)
    cset = {r["image"] for r in cvv["quality"]["aom_s6"]}   # the CVVDP run's 99 photos
    cq = dict(cvv["quality"], av1gpu=[r for r in A["default8"] if r["image"] in cset])
    cj, cjv = bd(cq["av1gpu"], cq["aom_s6"], "cvvdp", 0.05)
    cs, csv_ = bd(cq["av1gpu"], cq["aom_s6"])
    pub = lambda rows: [r for r in rows if r["q"] in Q6]  # noqa: E731
    tgt = bench.jod_target(cq, dict(SIZES), 9.5)
    N["cvvdp"] = dict(bd=cj, bd_published_points=bd(pub(cq["av1gpu"]), pub(cq["aom_s6"]), "cvvdp", 0.05)[0],
                      by_size=by_size(cjv), ssimulacra2_8pts=cs, target=tgt["codecs"]["av1gpu"],
                      jod_worse_than_ssimulacra2=sum(1 for im in cjv if im in csv_ and cjv[im] > csv_[im]), images=len(cjv))
    an_l = [r for r in cvv["cvvdp_luma_only"]["rows"] if r["codec"] == "aom_s6"]
    N["cvvdp"]["luma_only"] = bd(A["luma"], an_l, "cvvdp_luma", 0.05)[0]
    cvb = {k.split("_vs_")[0]: v["median"] for k, v in cvv["bd_rate_cvvdp"].items() if k.endswith("_vs_aom_s6")}
    N["cvvdp"]["svt_speed"] = interp(cj["median"], (cvb["svt_p6"], svt_speed(6)), (cvb["svt_p9"], svt_speed(9)))
    s6, s9 = aom_speed(6), aom_speed(9)
    N["cvvdp"]["aom_speed"] = interp(cj["median"], (0.0, s6), (cvb["aom_s9"], s9))
    return N


def svt_speed(p):
    return {int(x["codec"].split()[2].rstrip(",")): x["mp_s"] for x in svt["speed"] if x["codec"].startswith("SVT")}[p]


def aom_speed(s):
    return next(x["mp_s"] for x in enc2["speed"] if f"speed {s}" in x["codec"])


def svt_at(b):
    pts = sorted((svt["bd_rate"][f"svt_p{p}_vs_aom_s6"]["median"], svt_speed(p)) for p in (2, 4, 6, 8, 9))
    for p0, p1 in zip(pts, pts[1:]):
        if p0[0] <= b <= p1[0]:
            return interp(b, p0, p1)
    return None


A = rows_run("new") if RUN else rows_published()
N = numbers(A)
if RUN:
    O = rows_published()   # v0.3.0 = the published files (checks4x4.py); turbo3 was first scored on the L4 here
    oo = rows_run("old")
    O["turbo3ns"], O["turbo3"] = oo["turbo3ns"], oo["turbo3"]
    NO = numbers(O)
    N["v030_on_l4"] = {k: NO[k] for k in ("default", "nosmall", "turbo3ns", "turbo3", "default_notiles", "nosmall_notiles",
                                          "default_vs_webp", "vs_nvenc", "formats", "cvvdp", "totals")}

pct = lambda v: round(100 * v, 2)  # noqa: E731


def show(o, ind=""):
    for k, v in o.items():
        if k in ("default_per_image", "q60_files", "per_image_bpp"):
            continue
        if isinstance(v, dict) and set(v) >= {"n", "median", "mean"}:
            print(f"{ind}{k}: {pct(v['median']):+.2f} % ({pct(v['mean']):+.2f} %) n={v['n']}")
        elif isinstance(v, dict) and set(v) >= {"n", "median"}:
            print(f"{ind}{k}: {pct(v['median']):+.2f} % n={v['n']}")
        elif isinstance(v, dict):
            print(f"{ind}{k}:")
            show(v, ind + "  ")
        elif isinstance(v, float):
            print(f"{ind}{k}: {v:.4f}")
        else:
            print(f"{ind}{k}: {v}")


show(N)
if arg("--json"):
    json.dump(N, open(arg("--json"), "w"), indent=1, default=str)
if arg("--write") and RUN:
    exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "write4x4.py")).read())
