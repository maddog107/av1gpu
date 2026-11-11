#!/usr/bin/env python3
"""v0.13.0 docs refresh: bench/results/aws-g6-l4-v0130/{results.json,summary.md} from the instance's ~/results (pulled).

  analyze.py WORKTREE RUN_DIR [--skip PART,...] [--allow-failed-runs] [--out DIR]
  analyze.py WORKTREE --selftest

Speed: the docs speed list (spd13.py) aggregated as aws-g6-l4-speed-v06's analyze_spd.py did ("<phase>/<tag>/<new|old>":
median of the timed runs, power from nvidia-smi dmon), the scheme of aws-g6-g7-v090-check/results.json "speed", which
docs/charts/make_charts.py reads.  Quality: av1gpu's rows against the stored CPU baselines with aws-g6-l4-tiles4x4's
analyze4x4.py numbers() (same keys as tiles4x4's results.json: quality, bd_rate, bd_rate_by_size, bd_rate_cvvdp, ...),
for NEW (v0.13.0) and OLD (v0.12.0), plus NEW vs OLD per config, byte / size checks against the stored rows and the
scorer controls.  Gates, start-up times, environment.

Every part of run.sh must have finished OK, with all its files, unless named in --skip (then results.json lists it in
"not_measured" and summary.md says so at the top).  --selftest feeds the stored tiles4x4 rows and the v0.9.0 check's
speed logs through the same code and requires the published numbers back."""
import calendar, csv, glob, json, math, os, re, statistics, sys, time

WT = sys.argv[1]
SELFTEST = "--selftest" in sys.argv
RUN = None if SELFTEST else sys.argv[2]
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None  # noqa: E731
SKIP = set(filter(None, (arg("--skip") or "").split(",")))
OUTD = arg("--out") or os.path.join(WT, "bench", "results", "aws-g6-l4-v0130")
sys.path.insert(0, os.path.join(WT, "bench"))
import bench  # noqa: E402

RES = os.path.join(WT, "bench", "results")
L = lambda n: json.load(open(os.path.join(RES, n, "results.json")))  # noqa: E731
gam, cvv, fmt, nv, svt, tun, enc2, std0, stdns, t4 = (L(n) for n in (
    "aws-g6-l4-gamut", "aws-g6-l4-cvvdp", "aws-g6-l4-formats", "aws-g6-l4-nvenc", "aws-c6a-gamut-svt", "aws-c6a-cpu-tuning",
    "aws-g6-l4-gamut-enc2", "aws-g6-l4", "aws-g6-l4-turbo-nosmall", "aws-g6-l4-tiles4x4"))
man = {m["file"]: m for m in csv.DictReader(open(os.path.join(WT, "bench", "openimages_v4_gamut.csv")))}
SIZES = {k: int(m["width"]) * int(m["height"]) / 1e6 for k, m in man.items()}
MISSING = "9fba6f263f3c4f3e_tiny.jpg"
Q6, Q8 = [30, 40, 50, 60, 70, 80], [30, 40, 50, 60, 70, 80, 85, 90]
FMTS = ("420_8", "444_8", "420_10", "444_10")
PARTS = ["speed", "quality", "speed_small", "speed_std", "speed_ll", "gates", "startup", "alpha", "cvvdp", "quality_old",
         "speed_extra"]
ERR = []


def die(msg):
    sys.exit(f"analyze.py: {msg}")


# ------------------------------------------------------------------ quality numbers (analyze4x4.py's numbers())
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


def numbers(A):
    N = {}
    aom6, webp = gam["quality"]["aom_s6"], webp_fixed(gam["quality"]["webp"])
    d, dv = bd(A["default6"], aom6)
    n_, nvv = bd(A["nosmall6"], aom6)
    N["default"], N["default_by_size"] = d, by_size(dv)
    N["nosmall"], N["nosmall_by_size"] = n_, by_size(nvv)
    N["default_vs_webp"] = bd(A["default6"], webp)[0]
    for k in ("turbo3ns", "turbo3"):
        if A.get(k):
            N[k] = bd(A[k], aom6)[0]
    sw = tun["quality"]["sweeps"]
    nt = sw["stock_aom_s6_notiles"]["rows"]
    N["default_notiles"] = bd(A["default6"], nt)[0]
    N["nosmall_notiles"] = bd(A["nosmall6"], nt)[0]
    p4n, p6n = bd(sw["stock_svt_p4_notiles"]["rows"], nt)[0], bd(sw["stock_svt_p6_notiles"]["rows"], nt)[0]
    tm = {}
    for t in tun["primary"]["timings"]:
        tm.setdefault(t["label"], t["mp_s"])
    t4_, t6_ = svt["bd_rate"]["svt_p4_vs_aom_s6"]["median"], svt["bd_rate"]["svt_p6_vs_aom_s6"]["median"]
    x = N["default_notiles"]["median"]
    N["svt_notiles"] = dict(p4=p4n, p6=p6n)
    N["svt46_interp"] = dict(
        stock_untiled=interp(x, (p4n["median"], svt_speed(4)), (p6n["median"], svt_speed(6))),
        pgo_untiled=interp(x, (p4n["median"], tm["pgo_svt_p4_j1"]), (p6n["median"], tm["pgo_svt_p6_j1"])),
        pgo_tiled_8x2=interp(d["median"], (t4_, tm["pgo_svt_p4_j2"]), (t6_, tm["pgo_svt_p6_j2"])),
        stock_tiled=interp(d["median"], (t4_, svt_speed(4)), (t6_, svt_speed(6))))
    N["svt_interp_at"] = {k: svt_at(N[k]["median"]) for k in ("default", "nosmall", "turbo3ns") if N.get(k)}
    nvr = nv["quality"]["nvenc"]
    even = {im for im in SIZES if int(man[im]["height"]) % 2 == 0}
    N["vs_nvenc"] = bd(A["default6"], nvr)[0]
    N["vs_nvenc_even"] = bd(A["default6"], nvr, images=even)[0]
    N["totals"] = {q: dict(images=len(s), bytes=sum(r["bytes"] for r in s), ssimulacra2=statistics.fmean(r["ssimulacra2"] for r in s))
                   for q in Q6 for s in [[r for r in A["default6"] if r["q"] == q]]}
    q60 = [r for r in A["default6"] if r["q"] == 60 and r["image"] != MISSING]
    N["q60_99"] = dict(bytes=sum(r["bytes"] for r in q60), ssimulacra2=statistics.fmean(r["ssimulacra2"] for r in q60))
    F = {}
    for f in FMTS:
        if A["fmt"].get(f):
            F[f"av1gpu_{f}_vs_aom_s6_{f}"] = bd(A["fmt"][f], fmt["quality"][f"aom_s6_{f}"])[0]
    for a_, b_ in (("444_8", "420_8"), ("444_10", "420_10"), ("420_10", "420_8"), ("444_10", "444_8")):
        if A["fmt"].get(a_) and A["fmt"].get(b_):
            F[f"av1gpu_{a_}_vs_av1gpu_{b_}"] = bd(A["fmt"][a_], A["fmt"][b_])[0]
    N["formats"] = F
    if A.get("default8") and any(r.get("cvvdp") is not None for r in A["default8"]):
        cset = {r["image"] for r in cvv["quality"]["aom_s6"]}
        cq = dict(cvv["quality"], av1gpu=[r for r in A["default8"] if r["image"] in cset])
        cj, cjv = bd(cq["av1gpu"], cq["aom_s6"], "cvvdp", 0.05)
        cs, csv_ = bd(cq["av1gpu"], cq["aom_s6"])
        pub = lambda rows: [r for r in rows if r["q"] in Q6]  # noqa: E731
        tgt = bench.jod_target(cq, dict(SIZES), 9.5)
        N["cvvdp"] = dict(bd=cj, bd_published_points=bd(pub(cq["av1gpu"]), pub(cq["aom_s6"]), "cvvdp", 0.05)[0],
                          by_size=by_size(cjv), ssimulacra2_8pts=cs, target=tgt["codecs"]["av1gpu"],
                          jod_worse_than_ssimulacra2=sum(1 for im in cjv if im in csv_ and cjv[im] > csv_[im]), images=len(cjv))
        if A.get("luma"):
            an_l = [r for r in cvv["cvvdp_luma_only"]["rows"] if r["codec"] == "aom_s6"]
            N["cvvdp"]["luma_only"] = bd(A["luma"], an_l, "cvvdp_luma", 0.05)[0]
        cvb = {k.split("_vs_")[0]: v["median"] for k, v in cvv["bd_rate_cvvdp"].items() if k.endswith("_vs_aom_s6")}
        N["cvvdp"]["svt_speed"] = interp(cj["median"], (cvb["svt_p6"], svt_speed(6)), (cvb["svt_p9"], svt_speed(9)))
        N["cvvdp"]["aom_speed"] = interp(cj["median"], (0.0, aom_speed(6)), (cvb["aom_s9"], aom_speed(9)))
    return N


def strip(rows):
    return [{k: r[k] for k in ("image", "q", "bytes", "ssimulacra2", "cvvdp") if k in r} for r in rows]


def quality_block(A, N):
    """the tiles4x4 results.json keys"""
    q = {"av1gpu": strip(A["default6"]), "av1gpu_nosmall": strip(A["nosmall6"])}
    if A.get("turbo3ns"):
        q["av1gpu_turbo3_nosmall"] = strip(A["turbo3ns"])
    if A.get("turbo3"):
        q["av1gpu_turbo3"] = strip(A["turbo3"])
    if A.get("default8"):
        q["av1gpu_8pts"] = strip(A["default8"])
    q.update({f"av1gpu_{f}": strip(A["fmt"][f]) for f in FMTS[1:] if A["fmt"].get(f)})
    b = {"av1gpu_vs_aom_s6": N["default"], "av1gpu_vs_webp": N["default_vs_webp"], "av1gpu_nosmall_vs_aom_s6": N["nosmall"]}
    if N.get("turbo3ns"):
        b["av1gpu_turbo3_nosmall_vs_aom_s6"] = N["turbo3ns"]
    if N.get("turbo3"):
        b["av1gpu_turbo3_vs_aom_s6"] = N["turbo3"]
    b.update({"av1gpu_vs_aom_s6_notiles": N["default_notiles"], "av1gpu_nosmall_vs_aom_s6_notiles": N["nosmall_notiles"],
              "av1gpu_vs_nvenc": N["vs_nvenc"], "av1gpu_vs_nvenc_even_height": N["vs_nvenc_even"], **N["formats"]})
    o = dict(quality=q, bd_rate=b,
             bd_rate_by_size={"av1gpu_vs_aom_s6": N["default_by_size"], "av1gpu_nosmall_vs_aom_s6": N["nosmall_by_size"]},
             totals=N["totals"])
    if "cvvdp" in N:
        c = N["cvvdp"]
        o["bd_rate_cvvdp"] = {"av1gpu_vs_aom_s6": c["bd"], "av1gpu_vs_aom_s6_published_points": c["bd_published_points"]}
        if "luma_only" in c:
            o["bd_rate_cvvdp"]["av1gpu_vs_aom_s6_luma_only"] = c["luma_only"]
        o["bd_rate_cvvdp_by_size"] = {"av1gpu_vs_aom_s6": c["by_size"]}
        o["cvvdp_target"] = dict(c["target"])
        o["av1gpu_per_image_agreement"] = dict(images=c["images"], jod_worse_than_ssimulacra2=c["jod_worse_than_ssimulacra2"])
    o["derived"] = dict(svt_interp_at=N["svt_interp_at"], svt46_interp=N["svt46_interp"], svt_notiles=N["svt_notiles"],
                        q60_99=N["q60_99"], **({"cvvdp_svt_speed": N["cvvdp"]["svt_speed"], "cvvdp_aom_speed": N["cvvdp"]["aom_speed"],
                                                "cvvdp_ssimulacra2_8pts": N["cvvdp"]["ssimulacra2_8pts"]} if "cvvdp" in N else {}))
    return o


def std_block(S):
    """standard set (100 photos) against aws-g6-l4's stored libaom speed 6 / WebP rows"""
    a6, wp = std0["quality"]["aom_s6"], std0["quality"]["webp"]
    o = {}
    for k, key in (("default", "av1gpu"), ("nosmall", "av1gpu_nosmall"), ("turbo3ns", "av1gpu_turbo3_nosmall")):
        if S.get(k):
            o[f"{key}_vs_aom_s6"], v = bd(S[k], a6)
            o[f"{key}_vs_aom_s6_by_size"] = by_size_std(v)
    if S.get("default"):
        o["av1gpu_vs_webp"] = bd(S["default"], wp)[0]
    return o


def by_size_std(vals):
    return {}   # the standard set has no size classes in the docs


# ------------------------------------------------------------------ speed (analyze_spd.py)
def load_dmon(path):
    power = []
    if path and os.path.exists(path):
        for l in open(path):
            f = l.split()
            if len(f) < 6 or l.startswith("#"):
                continue
            try:
                t = calendar.timegm(time.strptime(f[0] + f[1], "%Y%m%d%H:%M:%S"))
                power.append((t, float(f[3]), float(f[6]), float(f[-1]) if len(f) >= 14 else float("nan")))
            except (ValueError, IndexError):
                pass
    return power


def aggregate(rows, power):
    def watts(runs):
        s = [(w, u, c) for t, w, u, c in power for r in runs if r["t0"] + 1 <= t <= r["t1"] - 1]
        return (statistics.fmean(w for w, _, _ in s), statistics.fmean(u for _, u, _ in s), statistics.fmean(c for _, _, c in s),
                len(s)) if s else None
    res = {}
    for phase in ("jpeg", "formats", "lossless", "alpha", "thumbs", "kodak"):
        pr = [r for r in rows if r["phase"] == phase]
        for key in sorted({(r["tag"], r["bin"]) for r in pr}):
            rs = [r for r in pr if (r["tag"], r["bin"]) == key and r["timed"] and r["rc"] == 0]
            if not rs:
                continue
            st = lambda k: statistics.median(r["stats"][k] for r in rs)  # noqa: E731
            d = dict(n=len(rs), mp_s=st("mp_s"), img_s=st("img_s"), cpu_cores=st("cpu_cores"), mp=rs[0]["stats"]["mp"],
                     images=rs[0]["stats"]["images"], bytes=rs[0]["stats"]["bytes"], wall_s=statistics.median(r["wall_s"] for r in rs),
                     mp_s_wall=statistics.median(r["stats"]["mp"] / r["wall_s"] for r in rs),
                     img_s_wall=statistics.median(r["stats"]["images"] / r["wall_s"] for r in rs),
                     rounds_img=[statistics.median(r["stats"]["img_s"] for r in rs[i:i + 3]) for i in range(0, len(rs), 3)],
                     rounds_img_wall=[statistics.median(r["stats"]["images"] / r["wall_s"] for r in rs[i:i + 3])
                                      for i in range(0, len(rs), 3)],
                     startup_s=statistics.median(r["wall_s"] - r["stats"]["wall_ms"] / 1000 for r in rs),
                     runs=[round(r["stats"]["mp_s"], 1) for r in rs], peak_dev_mb=max(r["stats"].get("peak_dev_mb", 0) for r in rs),
                     workers=[rs[0]["stats"].get("ingest_workers"), rs[0]["stats"].get("fallback_workers")],
                     bytes_same=len({r["stats"]["bytes"] for r in rs}) == 1,
                     rounds=[statistics.median(r["stats"]["mp_s"] for r in rs[i:i + 3]) for i in range(0, len(rs), 3)],
                     round_cores=[statistics.median(r["stats"]["cpu_cores"] for r in rs[i:i + 3]) for i in range(0, len(rs), 3)],
                     rounds_wall=[statistics.median(r["stats"]["mp"] / r["wall_s"] for r in rs[i:i + 3]) for i in range(0, len(rs), 3)])
            pw = watts(rs)
            if pw:
                d["watts"], d["sm_util"], d["sm_mhz"], d["power_samples"] = round(pw[0], 1), round(pw[1], 1), round(pw[2]), pw[3]
                d["j_per_mp"] = round(pw[0] / d["mp_s"], 3)
            res[f"{phase}/{key[0]}/{key[1]}"] = d
    return res


def load_jsonl(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


# ------------------------------------------------------------------ start-up (wall9.py output) and gates
def parse_wall(path, label):
    """{name: {label(spec): {median, min, max, n}}} from '== name' blocks of wall9.py output (its JSON line)"""
    out, name = {}, None
    for l in open(path):
        l = l.rstrip("\n")
        if l.startswith("== "):
            name = l[3:]
        elif l.startswith("{") and name:
            runs = json.loads(l)["runs"]
            out[name] = {label(s): dict(median=statistics.median(v), min=min(v), max=max(v), n=len(v)) for s, v in runs.items()}
    return out


def parse_gates(d):
    g = {}
    for f in sorted(glob.glob(os.path.join(d, "*.log"))):
        g[os.path.basename(f)[:-4]] = [x for x in open(f).read().splitlines() if x.startswith("[gate]")]
    return g


# ------------------------------------------------------------------ self-test
def close(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        bad = [k for k in set(a) | set(b) if k not in a or k not in b]
        return [f"{path}/{k}: only in {'computed' if k in a else 'stored'}" for k in bad] + \
            [e for k in a if k in b for e in close(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [e for i, (x, y) in enumerate(zip(a, b)) for e in close(x, y, f"{path}[{i}]")]
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or not (abs(a - b) <= 1e-9 * max(1.0, abs(b)) or (a != a and b != b)):
            return [f"{path}: {a} vs {b}"]
        return []
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def selftest():
    q = t4["quality"]
    A = dict(default6=q["av1gpu"], default8=q["av1gpu_8pts"], nosmall6=q["av1gpu_nosmall"], turbo3ns=q["av1gpu_turbo3_nosmall"],
             turbo3=q["av1gpu_turbo3"], fmt={"420_8": q["av1gpu"], **{f: q[f"av1gpu_{f}"] for f in FMTS[1:]}},
             luma=t4["cvvdp_luma_only"])
    blk = json.loads(json.dumps(quality_block(A, numbers(A))))
    fails = []
    for k in ("quality", "bd_rate", "bd_rate_by_size", "bd_rate_cvvdp", "bd_rate_cvvdp_by_size", "cvvdp_target",
              "av1gpu_per_image_agreement", "totals"):
        e = close(blk[k], t4[k], k)
        print(f"selftest quality {k}: {'reproduces aws-g6-l4-tiles4x4' if not e else f'{len(e)} DIFFERENCES'}")
        fails += e
    # the README's 2.4-4x (SVT-AV1 at av1gpu's sizes) from the stored rows and the v0.9.0 speeds
    v9 = L("aws-g6-g7-v090-check")["speed"]
    for k, tag in (("default", "default_enc4"), ("nosmall", "nosmall_enc4"), ("turbo3ns", "turbo3ns_enc4")):
        s = blk["derived"]["svt_interp_at"][k]
        print(f"  derived: av1gpu {k} {v9[f'jpeg/size_range/{tag}/new']['mp_s']:.1f} MP/s / SVT-AV1 at its size {s:.1f} = "
              f"{v9[f'jpeg/size_range/{tag}/new']['mp_s'] / s:.2f}x")
    # speed: the v0.9.0 check's docs list through aggregate()
    D = os.path.join(RES, "aws-g6-g7-v090-check", "logs", "l4")
    sp = json.loads(json.dumps(aggregate(load_jsonl(sorted(glob.glob(os.path.join(D, "docs", "*.jsonl")))),
                                         load_dmon(os.path.join(D, "dmon.txt")))))
    e = close(sp, v9, "speed")
    print(f"selftest speed: {len(sp)} entries, {'reproduces aws-g6-g7-v090-check' if not e else f'{len(e)} DIFFERENCES'}")
    fails += e
    # start-up: the v0.9.0 check's spd9_single.txt
    lab = lambda s: os.path.basename(s.split("@")[0]) + ("+1q" if "@GP_MAX_CONNECTIONS=1" in s else "")  # noqa: E731
    w = json.loads(json.dumps(parse_wall(os.path.join(D, "spd9_single.txt"), lab)))
    e = close(w, L("aws-g6-g7-v090-check")["single"]["l4"], "single")
    print(f"selftest start-up: {'reproduces aws-g6-g7-v090-check single' if not e else f'{len(e)} DIFFERENCES'}")
    fails += e
    g = parse_gates(os.path.join(RES, "aws-g6-l4-tiles4x4", "logs"))
    g = {k: v for k, v in g.items() if v}
    e = close(g, t4["decoder_gate"], "decoder_gate")
    print(f"selftest gates: {'reproduces aws-g6-l4-tiles4x4' if not e else f'{len(e)} DIFFERENCES'}")
    fails += e
    for x in fails[:40]:
        print("  ", x)
    sys.exit(1 if fails else 0)


if SELFTEST:
    selftest()


# ------------------------------------------------------------------ the run
def need(part, rel, loader=json.load):
    p = os.path.join(RUN, rel)
    if part in SKIP:
        return None
    if not os.path.exists(p):
        ERR.append(f"part '{part}': missing {rel}")
        return None
    with open(p) as f:
        return loader(f)


def rd(rel):
    p = os.path.join(RUN, rel)
    return open(p).read() if os.path.exists(p) else ""


env_txt = rd("env.txt")
if not env_txt:
    die(f"no {RUN}/env.txt: is RUN_DIR the instance's ~/results?")
m = re.search(r"^rgba: (\d+) files", env_txt, re.M)
HAVE_RGBA = bool(m and int(m.group(1)) > 0)
if not HAVE_RGBA:
    SKIP.add("alpha")
prog = rd("progress.log")
status = {}
for l in prog.splitlines():
    mm = re.match(r"## (\w+): (\w+)", l)
    if mm:
        status[mm.group(1)] = mm.group(2)   # the last run of each part counts
for p in PARTS:
    if p in SKIP:
        continue
    if status.get(p) != "OK" and not (p == "alpha" and status.get(p) == "SKIPPED"):
        ERR.append(f"part '{p}': {'not run' if p not in status else status[p]} (progress.log)")

# speed
groups = {"speed": "core", "speed_small": "small", "speed_std": "std", "speed_ll": "ll", "speed_extra": "extra", "alpha": "alpha"}
rows = []
for part, g in groups.items():
    r = need(part, f"speed/{g}.jsonl", lambda f: [json.loads(l) for l in f if l.strip()])
    if r is not None:
        bad = [x for x in r if x["rc"] or not x["stats"]]
        if bad:
            msg = f"part '{part}': {len(bad)} failed runs in speed/{g}.jsonl ({sorted({x['tag'] + '/' + x['bin'] for x in bad})})"
            if "--allow-failed-runs" in sys.argv:
                print("WARNING:", msg)
            else:
                ERR.append(msg + "; --allow-failed-runs accepts them")
        rows += r
SPEED = aggregate(rows, load_dmon(os.path.join(RUN, "dmon.txt")))
for k, v in SPEED.items():
    if v["n"] < 6:
        print(f"WARNING: {k}: {v['n']} timed runs (6 expected)")
    if v["n"] < 3:
        ERR.append(f"speed {k}: only {v['n']} timed runs")
    if not v["bytes_same"]:
        ERR.append(f"speed {k}: output bytes vary between runs")
EXPECT = {"speed": ["jpeg/size_range/" + t for t in ("default_enc4", "default_enc3", "nosmall_enc4", "turbo3ns_enc4", "turbo3_enc4")]
          + ["formats/" + f for f in ("444_8", "420_10", "444_10")],
          "speed_small": ["thumbs/default", "thumbs/nosmall", "kodak/default", "kodak/nosmall", "kodak/turbo3_nosmall"],
          "speed_std": ["jpeg/standard/" + t for t in ("default_enc4", "default_enc2", "default_enc3", "nosmall_enc4", "nosmall_enc3")],
          "speed_ll": ["lossless/" + t for t in ("cmp_1pass", "cmp_1pass_ycgco", "cmp_4x", "cmp_4x_ycgco")],
          "speed_extra": ["jpeg/size_range/" + t for t in ("default_enc2", "nosmall_enc3", "tiles211_enc3", "tiles233_enc3")],
          "alpha": ["alpha/" + t for t in ("default", "q100", "lossless", "no_alpha")]}
for part, keys in EXPECT.items():
    if part in SKIP:
        continue
    for k in keys:
        for b in ("new", "old"):
            if f"{k}/{b}" not in SPEED:
                ERR.append(f"part '{part}': no speed entry {k}/{b}")
SPEED_CHANGE = {k[:-4]: v["mp_s"] / SPEED[k[:-4] + "/old"]["mp_s"] - 1 for k, v in SPEED.items()
                if k.endswith("/new") and k[:-4] + "/old" in SPEED}
SETS = {}
for f in sorted(glob.glob(os.path.join(RUN, "speed", "*_set*.json"))):
    SETS[os.path.basename(f)[:-5]] = json.load(open(f))

# quality
manifests = need("quality", "quality/manifests.json")
compare = need("quality", "quality/compare.json")
sc_new = need("quality", "quality/scores_new.json")
sc_old = need("quality_old", "quality/scores_old.json")
sc_cv = need("cvvdp", "quality/scores_cvvdp.json")
cvinfo = need("cvvdp", "quality/cvvdp_info.json")
gates = parse_gates(os.path.join(RUN, "quality", "gates")) if "gates" not in SKIP else None
if gates is not None and len([k for k, v in gates.items() if v]) < 3:
    ERR.append(f"part 'gates': {len(gates)} gate logs with [gate] lines (3 expected)")
startup = need("startup", "startup.txt", lambda f: f.name)
alpha = need("alpha", "quality/alpha.json") if HAVE_RGBA else None
if ERR:
    die("missing or failed inputs (pass --skip PART,... to accept a part as not measured):\n  " + "\n  ".join(ERR))

ALL_SC = (sc_new or []) + (sc_old or [])


def pick(cfg, b, qs=None):
    rs = [dict(image=r["image"], q=r["q"], bytes=r["bytes"], ssimulacra2=r["ssimulacra2"], md5=r["md5"])
          for r in ALL_SC if r["config"] == cfg and r["bin"] == b and (qs is None or r["q"] in qs)]
    seen, out = set(), []
    for r in rs:
        if (r["image"], r["q"]) not in seen:
            seen.add((r["image"], r["q"]))
            out.append(r)
    return out


def identical(cfg):
    c = compare["compare"].get(cfg)
    return c and c["new"] == c["old"] and c["all"]["identical"] == c["all"]["files"] == c["new"]


def rows_for(b):
    def get(cfg, qs=None):
        r = pick(cfg, b, qs)
        if not r and b == "old" and identical(cfg):   # OLD's files are NEW's: the same scores
            r = pick(cfg, "new", qs)
        return r
    cvm = {r["md5"]: r["cvvdp"] for r in (sc_cv or []) if r.get("cvvdp") is not None}
    d8 = [dict(r, cvvdp=cvm[r["md5"]]) if r["md5"] in cvm else dict(r) for r in get("gamut_default")]
    A = dict(default8=d8 if b == "new" else [], default6=get("gamut_default", Q6), nosmall6=get("gamut_nosmall"),
             turbo3ns=get("gamut_turbo3ns"), turbo3=get("gamut_turbo3"),
             fmt={"420_8": get("gamut_default", Q6), **{f: get(f"fmt_{f}") for f in FMTS[1:]}})
    S = dict(default=get("std_default"), nosmall=get("std_nosmall"), turbo3ns=get("std_turbo3ns"))
    return A, S


A_new, S_new = rows_for("new")
for k, n_exp in (("default8", 800), ("default6", 600), ("nosmall6", 600), ("turbo3ns", 600), ("turbo3", 600)):
    if len(A_new[k]) != n_exp:
        die(f"NEW {k}: {len(A_new[k])} scored rows ({n_exp} expected)")
for f in FMTS[1:]:
    if len(A_new["fmt"][f]) != 600:
        die(f"NEW fmt {f}: {len(A_new['fmt'][f])} scored rows (600 expected)")
nnone = [r for r in A_new["default8"] if r["ssimulacra2"] is None]
if nnone:
    die(f"{len(nnone)} NEW default rows without a SSIMULACRA2 score")
if sc_cv is not None and sum(1 for r in A_new["default8"] if r.get("cvvdp") is not None) < 780:
    die("CVVDP: fewer than 780 of the 792 NEW default rows scored")
N_new = numbers(A_new)
QB = quality_block(A_new, N_new)

A_old, S_old = rows_for("old")
old_ok = len(A_old["default6"]) == 600   # the other OLD configs only if quality_old ran (or their files equal NEW's)
N_old = numbers(A_old) if old_ok else None
OLDB = None
if N_old:
    OLDB = {"bd_rate": quality_block(A_old, N_old)["bd_rate"], "totals": N_old["totals"],
            "configs_scored": {k: len(v) for k, v in A_old.items() if isinstance(v, list)}}


# NEW vs OLD per config
def new_vs_old():
    out = {}
    for cfg, c in compare["compare"].items():
        o = {k: c[k] for k in ("new", "old") if k in c}
        o.update({k: c[k] for k in ("lt1", "1to2", "ge2", "all", "unknown_size") if k in c})
        o["bytes_change"] = (c["all"]["bytes_new"] / c["all"]["bytes_old"] - 1) if c["all"]["bytes_old"] else None
        n, ol = pick(cfg, "new"), pick(cfg, "old")
        if not ol and identical(cfg):
            ol = n
        if n and ol and cfg.startswith(("gamut_", "fmt_", "std_")):
            o["bd_rate_new_vs_old"] = bd(n, ol)[0]
            if cfg.startswith(("gamut_", "fmt_")):
                o["bd_rate_new_vs_old_by_size"] = by_size(bd(n, ol)[1])
        out[cfg] = o
    return out


NVO = new_vs_old()

# checks: sizes against the stored rows, and the scorer against the stored scores
STORED = {"gamut_default": t4["quality"]["av1gpu_8pts"], "gamut_nosmall": t4["quality"]["av1gpu_nosmall"],
          "gamut_turbo3ns": t4["quality"]["av1gpu_turbo3_nosmall"], "gamut_turbo3": t4["quality"]["av1gpu_turbo3"],
          "fmt_444_8": t4["quality"]["av1gpu_444_8"], "fmt_420_10": t4["quality"]["av1gpu_420_10"],
          "fmt_444_10": t4["quality"]["av1gpu_444_10"], "std_default": std0["quality"]["av1gpu"],
          "std_nosmall": stdns["quality"]["av1gpu"]}


def checks():
    o = dict(sizes_vs_published={}, scorer={})
    inv = {bench.qindex_for_quality(q): q for q in Q8}
    for cfg, rows_ in STORED.items():
        st = {(r["image"], r["q"]): r for r in rows_}
        for b in ("new", "old"):
            key = f"{b}/{cfg}"
            if key not in manifests:
                continue
            files = manifests[key]["files"]
            run = {}
            for f, mm in files.items():
                img, qq = f.rsplit(".q", 1)
                run[(img, inv[int(qq[:-5])])] = mm["bytes"]
            com = [k for k in st if k in run]
            o["sizes_vs_published"][key] = dict(stored=len(st), common=len(com), same_size=sum(run[k] == st[k]["bytes"] for k in com),
                                                bytes_run=sum(run[k] for k in com), bytes_stored=sum(st[k]["bytes"] for k in com))
    # scorer: our score of a file with the stored row's (image, q, bytes) against the stored score.  A control is a file
    # that is presumably the published one: OLD's (v0.12.0, whose sizes all equal the stored rows'), or NEW's when its
    # md5 equals OLD's.  NEW files that only share a stored row's byte count are other bitstreams: listed apart.
    old_md5 = {}
    for key, v in manifests.items():
        b_, cfg_ = key.split("/", 1)
        if b_ == "old":
            for f, mm in v["files"].items():
                if ".q" in f:
                    img, qq = f.rsplit(".q", 1)
                    old_md5[(cfg_, img, inv[int(qq[:-5])])] = mm["md5"]
    for metric, src_rows in (("ssimulacra2", ALL_SC), ("cvvdp", sc_cv or [])):
        diffs, eq, other = [], 0, []
        for r in src_rows:
            if r.get(metric) is None:
                continue
            st = STORED.get(r["config"])
            if not st or r["config"].startswith("std_"):
                continue
            s = next((x for x in st if x["image"] == r["image"] and x["q"] == r["q"] and x["bytes"] == r["bytes"]
                      and x.get(metric) is not None), None)
            if s:
                d = abs(r[metric] - s[metric])
                if r["bin"] == "old" or old_md5.get((r["config"], r["image"], r["q"])) == r["md5"]:
                    diffs.append((d, r["config"], r["bin"], r["image"], r["q"]))
                    eq += d == 0
                else:
                    other.append(dict(diff=r[metric] - s[metric], config=r["config"], image=r["image"], q=r["q"], bytes=r["bytes"]))
        diffs.sort(reverse=True)
        o["scorer"][metric] = dict(rows=len(diffs), equal=eq, max_abs_diff=diffs[0][0] if diffs else None,
                                   worst=[dict(diff=d, config=c, bin=b, image=i, q=q) for d, c, b, i, q in diffs[:5]],
                                   same_size_other_bitstream=other)
    return o


CHK = checks()

# environment, inputs against the v0.9.0 check's
inputs_cmp = {}
mine = {}
for l in rd("inputs.sha256").splitlines():
    h, p = l.split("  ", 1)
    mine[p] = h
ref = {}
for l in open(os.path.join(RES, "aws-g6-g7-v090-check", "logs", "l4", "inputs.sha256")):
    h, p = l.rstrip("\n").split("  ", 1)
    ref[p] = h
for top in sorted({"/".join(p.split("/")[:2]) if p.startswith(("ll/", "sets/", "data/")) else p.split("/")[0] for p in mine}):
    ks = [p for p in mine if p.startswith(top + "/")]
    inputs_cmp[top] = dict(files=len(ks), in_v090=sum(p in ref for p in ks), identical=sum(ref.get(p) == mine[p] for p in ks))
lab = {"new": "new", "old": "old", "old2": "old_noise"}
SINGLE = parse_wall(os.path.join(RUN, "startup.txt"), lambda s: lab.get(os.path.basename(s), s)) if startup else None
env = [l for l in env_txt.splitlines() if l.strip()]
cpu = dict(nproc=next((l.split()[1] for l in env if l.startswith("nproc ")), None),
           model=next((l.split(":", 1)[1].strip() for l in env if l.startswith("Model name")), None))
bins = [l.replace(" unknown option --version", "") for l in rd("binaries.txt").splitlines() if l.strip()]   # (no --version)

# derived speed claims (the ratios make_charts.py and the README compute)
DER = dict(QB.pop("derived"))
sp = lambda k: SPEED.get(f"jpeg/size_range/{k}/new", {}).get("mp_s")  # noqa: E731
s6, s9 = aom_speed(6), aom_speed(9)
nvbest = max(nv["speed"], key=lambda x: x["mp_s"])["mp_s"]
DER.update(
    default_vs_aom_s6=sp("default_enc4") / s6, default_vs_nvenc=sp("default_enc4") / nvbest,
    turbo3ns_vs_svt_p9=sp("turbo3ns_enc4") / svt_speed(9), turbo3ns_vs_aom_s9=sp("turbo3ns_enc4") / s9,
    vs_svt_at_same_size={k: sp(t) / DER["svt_interp_at"][k] for k, t in
                         (("default", "default_enc4"), ("nosmall", "nosmall_enc4"), ("turbo3ns", "turbo3ns_enc4"))
                         if DER["svt_interp_at"].get(k) and sp(t)},
    baselines=dict(aom_s6_mp_s=s6, aom_s9_mp_s=s9, svt_p9_mp_s=svt_speed(9), nvenc_best_mp_s=nvbest,
                   sources="aws-g6-l4-gamut-enc2 (libaom), aws-c6a-gamut-svt (SVT-AV1), aws-g6-l4-nvenc (NVENC)"))
if "cvvdp" in N_new:
    DER["cvvdp_vs_svt"] = sp("default_enc4") / DER["cvvdp_svt_speed"]
    DER["cvvdp_vs_aom"] = sp("default_enc4") / DER["cvvdp_aom_speed"]

new_line = next((l for l in bins if l.startswith("new ")), "new ?")
old_line = next((l for l in bins if l.startswith("old ")), "old ?")
res = dict(
    description="v0.13.0 docs refresh on an NVIDIA L4 (AWS g6.4xlarge, 16 vCPUs): the published v0.13.0 Linux CUDA 13 "
                "package (new) against v0.12.0's (old). speed: the docs speed list of aws-g6-l4-speed-v06's spd.py "
                "(scripts/spd13.py: each set repeated to >= 1,000 MP, the lossless and RGBA sets as published; a warm-up "
                "then 3 timed runs with --stats, 2 rounds with the binary order alternating; median of the 6 timed runs; "
                "keys as aws-g6-g7-v090-check). quality: av1gpu's side re-encoded and re-scored (libjxl 0.11.1 "
                "ssimulacra2, libavif 1.4.0 avifdec with dav1d, cvvdp 0.5.7) against the stored CPU / NVENC baselines, as "
                "aws-g6-l4-tiles4x4 (same keys); v0120_on_l4 = the same numbers for v0.12.0.",
    date=time.strftime("%Y-%m-%d", time.gmtime(os.path.getmtime(os.path.join(RUN, "env.txt")))),
    code=dict(new=new_line, old=old_line), binaries=bins, environment=env, cpu=cpu,
    not_measured=sorted(SKIP), parts=status,
    sets=SETS, speed=SPEED, speed_change=SPEED_CHANGE, single=SINGLE,
    cvvdp=cvinfo, images=100, megapixels=sum(SIZES.values()),
    baselines=dict(size_range="aws-g6-l4-gamut (libaom speed 6 / 9, WebP with aws-g6-l4-cvvdp's ICC re-scores)",
                   cvvdp="aws-g6-l4-cvvdp (99 photos, 8 quality points)", formats="aws-g6-l4-formats",
                   one_tile_anchor="aws-c6a-cpu-tuning (stock_aom_s6_notiles)", nvenc="aws-g6-l4-nvenc",
                   svt="aws-c6a-gamut-svt", standard="aws-g6-l4 (libaom speed 6, WebP on the standard set)"),
    **QB, derived=DER,
    quality_std={k: strip(v) for k, v in S_new.items() if v}, bd_rate_std=std_block(S_new),
    v0120_on_l4=OLDB, new_vs_old=NVO, checks=CHK, inputs_vs_v090=inputs_cmp,
    decoder_gate=gates, alpha=alpha)
os.makedirs(OUTD, exist_ok=True)
json.dump(res, open(os.path.join(OUTD, "results.json"), "w"), indent=1)

# ------------------------------------------------------------------ summary.md
p = lambda v: "—" if not v else f"{100 * v['median']:+.1f} %".replace("-", "−")  # noqa: E731
pm = lambda v: "—" if not v else f"{100 * v['median']:+.2f} % ({100 * v['mean']:+.2f} %)".replace("-", "−")  # noqa: E731
ob = (OLDB or {}).get("bd_rate", {})
out = ["# v0.13.0 on the NVIDIA L4: speed and file sizes for the docs", ""]
if SKIP:
    out += [f"**NOT MEASURED in this run: {', '.join(sorted(SKIP))}** — the docs rows that depend on these keep their "
            "old values with a \"measured with\" note (REFRESH_MAP.md).", ""]
for mt, sc in CHK["scorer"].items():
    oth = sc["same_size_other_bitstream"]
    if sc["rows"] and sc["max_abs_diff"] > 1e-6:
        out += [f"**WARNING: the scorer does not reproduce the stored {mt} scores**: {sc['rows']} files identical to the "
                f"published ones (v0.12.0's, or v0.13.0's with v0.12.0's md5), {sc['equal']} scored equal, max |diff| "
                f"{sc['max_abs_diff']:.4g}. BD-rates against the stored CPU baselines mix two scorers.", ""]
    elif sc["rows"]:
        out += [f"Scorer check ({mt}): all {sc['rows']} files identical to the published ones (v0.12.0's, whose sizes all "
                f"equal the stored rows', and v0.13.0's with v0.12.0's md5) score exactly as published."
                + (f" {len(oth)} other v0.13.0 files merely have the byte count of a stored row (different bitstreams, "
                   f"md5 unlike v0.12.0's) and score differently (by {min(x['diff'] for x in oth):+.4g} to "
                   f"{max(x['diff'] for x in oth):+.4g}), as different files do: not a scorer difference." if oth else ""), ""]
out += [f"new: `{new_line}`  ", f"old: `{old_line}`  ", f"CPU: {cpu['model']}, {cpu['nproc']} vCPUs", "",
        "## Speed (MP/s, median of 6; av1gpu --stats)", "", "| setting | v0.13.0 | v0.12.0 | change | img/s new | CPU cores new |",
        "|---|---:|---:|---:|---:|---:|"]
for k in sorted(SPEED):
    if not k.endswith("/new"):
        continue
    b = k[:-4]
    n_, o_ = SPEED[k], SPEED.get(b + "/old")
    out.append(f"| {b} | {n_['mp_s']:.1f} | {o_['mp_s']:.1f} | {100 * SPEED_CHANGE[b]:+.1f} % | {n_['img_s']:.1f} | "
               f"{n_['cpu_cores']:.2f} |" if o_ else f"| {b} | {n_['mp_s']:.1f} | — | — | {n_['img_s']:.1f} | {n_['cpu_cores']:.2f} |")
out += ["", "## File size: median BD-rate on SSIMULACRA2 (mean in brackets), 100 size-range photos", "",
        "| av1gpu | v0.13.0 | v0.12.0 | published (tiles4x4) |", "|---|---:|---:|---:|"]
ROWS = [("default vs libaom speed 6", "av1gpu_vs_aom_s6"), ("`--small-preset none`", "av1gpu_nosmall_vs_aom_s6"),
        ("turbo3 nosmall", "av1gpu_turbo3_nosmall_vs_aom_s6"), ("turbo3", "av1gpu_turbo3_vs_aom_s6"),
        ("default vs one-tile libaom speed 6", "av1gpu_vs_aom_s6_notiles"), ("default vs WebP", "av1gpu_vs_webp"),
        ("default vs NVENC AV1", "av1gpu_vs_nvenc")] + \
    [(f"{f.replace('_', ' / ')}-bit vs libaom speed 6, same format", f"av1gpu_{f}_vs_aom_s6_{f}") for f in FMTS]
for lab_, k in ROWS:
    out.append(f"| {lab_} | {pm(QB['bd_rate'].get(k))} | {pm(ob.get(k))} | {pm(t4['bd_rate'].get(k))} |")
if "bd_rate_cvvdp" in QB:
    out.append(f"| CVVDP (JOD), default vs libaom speed 6 (99 photos, 8 points) | {pm(QB['bd_rate_cvvdp']['av1gpu_vs_aom_s6'])} | — | "
               f"{pm(t4['bd_rate_cvvdp']['av1gpu_vs_aom_s6'])} |")
out += ["", "By size, default vs libaom speed 6: " + ", ".join(f"{c} {p(v)} (n {v['n']})" for c, v in QB["bd_rate_by_size"]["av1gpu_vs_aom_s6"].items()), ""]
if res["bd_rate_std"]:
    out += ["Standard set (100 photos) vs aws-g6-l4's libaom speed 6: " +
            ", ".join(f"{k} {p(v)}" for k, v in res["bd_rate_std"].items() if isinstance(v, dict) and "median" in v), ""]
out += ["## Derived", "", f"- default (GP_ENCODERS=4) / libaom speed 6 (16 threads): {DER['default_vs_aom_s6']:.2f}×",
        f"- default / NVENC AV1 best: {DER['default_vs_nvenc']:.2f}×; turbo3 nosmall / SVT-AV1 p9 {DER['turbo3ns_vs_svt_p9']:.2f}×, "
        f"/ libaom speed 9 {DER['turbo3ns_vs_aom_s9']:.2f}×",
        "- vs SVT-AV1 interpolated at av1gpu's file size: " + ", ".join(f"{k} {v:.2f}×" for k, v in DER["vs_svt_at_same_size"].items()),
        f"- q60 totals (99 photos): {DER['q60_99']['bytes']:,} bytes, mean SSIMULACRA2 {DER['q60_99']['ssimulacra2']:.2f}", "",
        "## v0.13.0 against v0.12.0, per config (bytes of the common files; BD-rate NEW vs OLD on SSIMULACRA2)", "",
        "| config | identical | bytes | BD-rate | < 1 MP | 1-2 MP | ≥ 2 MP identical |", "|---|---:|---:|---:|---:|---:|---:|"]
for cfg, o in NVO.items():
    bc = "—" if o["bytes_change"] is None else f"{100 * o['bytes_change']:+.2f} %"
    out.append(f"| {cfg} | {o['all']['identical']}/{o['all']['files']} | {bc} | "
               f"{pm(o.get('bd_rate_new_vs_old'))} | {o['lt1']['identical']}/{o['lt1']['files']} | {o['1to2']['identical']}/{o['1to2']['files']} | "
               f"{o['ge2']['identical']}/{o['ge2']['files']} |")
out += ["", "## Checks", ""]
for k, v in CHK["sizes_vs_published"].items():
    out.append(f"- sizes {k} vs the stored rows: {v['same_size']}/{v['common']} same size")
for k, v in CHK["scorer"].items():
    out.append(f"- scorer {k}: {v['rows']} files identical to the published ones, {v['equal']} equal, max |diff| {v['max_abs_diff']}; "
               f"{len(v['same_size_other_bitstream'])} other bitstreams of a stored row's byte count")
out += [f"- inputs vs the v0.9.0 check: " + "; ".join(f"{k} {v['identical']}/{v['files']}" for k, v in inputs_cmp.items()), ""]
if gates:
    out += ["## Decoder gates (NEW, tests/decoder_gate.py, 100 size-range photos)", ""] + [f"- {k}: {x}" for k, v in gates.items() for x in v] + [""]
if SINGLE:
    out += ["## Cold start, one process per image (ms, median of 12)", "", "| input | v0.13.0 | v0.12.0 | v0.12.0 again |", "|---|---:|---:|---:|"]
    for k, v in SINGLE.items():
        out.append(f"| {k} | {v['new']['median']:.1f} | {v['old']['median']:.1f} | {v.get('old_noise', {}).get('median', float('nan')):.1f} |")
    out.append("")
if alpha:
    out += ["## RGBA set (39 PNGs)", ""] + [f"- {k}: {v['bytes']:,} bytes (alpha {v['alpha']:,}), alpha exact {v['alpha_exact']}/{v['files']}"
                                             for k, v in alpha.items()] + [""]
open(os.path.join(OUTD, "summary.md"), "w").write("\n".join(out) + "\n")
print("wrote", os.path.join(OUTD, "results.json"), "and summary.md;", len(SPEED), "speed entries")
