#!/usr/bin/env python3
"""v0.13.0 docs refresh, quality side (on the instance; aws-g6-l4-tiles4x4's run4x4.py with v0.13.0 = NEW and v0.12.0 = OLD,
the published Linux CUDA 13 packages; setup.sh puts them at ~/bin/new and ~/bin/old).

  qual13.py encode     every lossy config with NEW and OLD (size + md5 manifests), the lossless sets and (if ~/data/rgba
                       exists) the RGBA sets: ~/results/quality/{manifests,compare}.json
  qual13.py score      SSIMULACRA2 of every NEW lossy file the tables need (+ OLD gamut_default at the six published
                       points, for the headline NEW vs OLD comparison): ~/results/quality/scores_new.json
  qual13.py score_old  SSIMULACRA2 of the other OLD configs (NEW vs OLD per config): ~/results/quality/scores_old.json
  qual13.py cvvdp      CVVDP (standard_fhd) of NEW gamut_default on the CVVDP run's 99 photos, 8 points, and of OLD on 8
                       control photos: ~/results/quality/scores_cvvdp.json
  qual13.py alpha      RGBA set: av1gpu rows (bytes, colour/alpha split, alpha error), NEW and OLD (needs ~/data)
  qual13.py gates      tests/decoder_gate.py with NEW on the size-range photos (default, 4:4:4 10-bit, turbo3 nosmall)

Scores are cached by the AVIF's md5 (~/results/quality/score_cache.json), so files identical in NEW and OLD are scored
once and a restarted part only scores what is missing.  Run with ~/venv/bin/python (numpy / Pillow / torch of the
published CVVDP run).  QUALITY_STD=0 leaves out the standard-set configs."""
import concurrent.futures as cf, glob, hashlib, json, math, os, subprocess, sys, tempfile, time
H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/src/bench")
import bench  # noqa: E402

R, E = f"{H}/results/quality", f"{H}/enc"
os.makedirs(R, exist_ok=True)
os.makedirs(E, exist_ok=True)
BIN = {"new": f"{H}/bin/new", "old": f"{H}/bin/old"}
AVIFDEC, SSIM2 = f"{H}/opt/bin/avifdec", f"{H}/opt/bin/ssimulacra2"
IMG = sorted(glob.glob(f"{H}/images/*.jpg"))
STD = sorted(glob.glob(f"{H}/images_std/*.jpg"))
Q6, Q8 = [30, 40, 50, 60, 70, 80], [30, 40, 50, 60, 70, 80, 85, 90]
WITH_STD = os.environ.get("QUALITY_STD", "1") != "0"
# name: (inputs, av1gpu options, quality points, depth for decoding) -- tiles4x4's list
LOSSY = {
    "gamut_default": (IMG, [], Q8, 8),
    "gamut_nosmall": (IMG, ["--small-preset", "none"], Q6, 8),
    "gamut_turbo3ns": (IMG, ["--preset", "turbo3", "--small-preset", "none"], Q6, 8),
    "gamut_turbo3": (IMG, ["--preset", "turbo3"], Q6, 8),
    "fmt_444_8": (IMG, ["--yuv", "444"], Q6, 8),
    "fmt_420_10": (IMG, ["--depth", "10"], Q6, 10),
    "fmt_444_10": (IMG, ["--yuv", "444", "--depth", "10"], Q6, 10),
}
if WITH_STD:
    LOSSY.update({"std_default": (STD, [], Q6, 8), "std_nosmall": (STD, ["--small-preset", "none"], Q6, 8),
                  "std_turbo3ns": (STD, ["--preset", "turbo3", "--small-preset", "none"], Q6, 8)})
LL = {
    "ll_jpeg_gamut": (IMG, ["--lossless"]),
    "ll_cmp": (sorted(glob.glob(f"{H}/ll/cmp/*.png")), ["--lossless"]),
    "ll_cmp_ycgco": (sorted(glob.glob(f"{H}/ll/cmp/*.png")), ["--lossless", "--ycgco-re"]),
    "ll_misc": (sorted(glob.glob(f"{H}/ll/misc/*.png")), ["--lossless"]),
    "ll_jpeg_misc": (sorted(glob.glob(f"{H}/ll/jpeg/*")), ["--lossless"]),
}
RGBA = sorted(glob.glob(f"{H}/data/rgba/set/*.png")) + sorted(glob.glob(f"{H}/data/rgba/extra/*"))
ALPHA = {"alpha_default": [], "alpha_q100": ["--alpha-quality", "100"], "alpha_lossless": ["--lossless"]} if RGBA else {}
INV = {bench.qindex_for_quality(q): q for q in Q8}


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def batch(binname, name, files, opts):
    out = f"{E}/{binname}/{name}"
    subprocess.run(["rm", "-rf", out])
    os.makedirs(out)
    lst = f"{out}/inputs.txt"
    open(lst, "w").write("\n".join(files) + "\n")
    t0 = time.time()
    p = subprocess.run([BIN[binname], "--batch", lst, "--outdir", out] + opts, capture_output=True, text=True)
    open(f"{out}.log", "w").write(p.stderr)
    man = {f: dict(bytes=os.path.getsize(os.path.join(out, f)), md5=md5(os.path.join(out, f)))
           for f in sorted(os.listdir(out)) if f.endswith(".avif")}
    return dict(rc=p.returncode, wall_s=round(time.time() - t0, 2), files=man, inputs=len(files), opts=opts,
                summary=next((l for l in p.stderr.splitlines() if "MP/s" in l), ""))


def encode():
    for n in ("images", "images_std"):
        if len(glob.glob(f"{H}/{n}/*.jpg")) != 100:
            sys.exit(f"~/{n} does not hold 100 photos: run setup.sh")
    res = {}
    for name, (files, opts, qs, _) in LOSSY.items():
        qi = ",".join(str(bench.qindex_for_quality(q)) for q in qs)
        for b in BIN:
            res[f"{b}/{name}"] = batch(b, name, files, ["--qindex", qi, "--name-q"] + opts)
            print(b, name, res[f"{b}/{name}"]["rc"], len(res[f"{b}/{name}"]["files"]), "files", res[f"{b}/{name}"]["wall_s"], "s",
                  flush=True)
    for name, (files, opts) in list(LL.items()) + [(k, (RGBA, v)) for k, v in ALPHA.items()]:
        for b in BIN:
            res[f"{b}/{name}"] = batch(b, name, files, opts)
            print(b, name, res[f"{b}/{name}"]["rc"], len(res[f"{b}/{name}"]["files"]), "files", flush=True)
    json.dump(res, open(f"{R}/manifests.json", "w"), indent=1)
    from PIL import Image
    mp = {}
    for f in IMG + STD + RGBA + [x for v in LL.values() for x in v[0]]:
        try:
            w, h = Image.open(f).size
            mp[os.path.basename(f)] = w * h / 1e6
        except Exception:
            pass
    cmp = {}
    for name in list(LOSSY) + list(LL) + list(ALPHA):
        n, o = res[f"new/{name}"]["files"], res[f"old/{name}"]["files"]
        c = dict(new=len(n), old=len(o), only_new=sorted(set(n) - set(o)), only_old=sorted(set(o) - set(n)))
        for lo, hi, lab in ((0, 1, "lt1"), (1, 2, "1to2"), (2, 1e9, "ge2")):
            ks = [k for k in n if k in o and lo <= mp.get(k.split(".q")[0] if ".q" in k else k[:-5], -1) < hi]
            c[lab] = dict(files=len(ks), identical=sum(n[k]["md5"] == o[k]["md5"] for k in ks),
                          bytes_new=sum(n[k]["bytes"] for k in ks), bytes_old=sum(o[k]["bytes"] for k in ks))
        unk = [k for k in n if k in o and (k.split(".q")[0] if ".q" in k else k[:-5]) not in mp]
        c["unknown_size"] = dict(files=len(unk), identical=sum(n[k]["md5"] == o[k]["md5"] for k in unk))
        c["all"] = dict(files=len([k for k in n if k in o]), identical=sum(n[k]["md5"] == o[k]["md5"] for k in n if k in o),
                        bytes_new=sum(n[k]["bytes"] for k in n if k in o), bytes_old=sum(o[k]["bytes"] for k in n if k in o))
        cmp[name] = c
        print(name, {k: v for k, v in c.items() if k not in ("only_new", "only_old")}, flush=True)
    json.dump(dict(megapixels=mp, compare=cmp), open(f"{R}/compare.json", "w"), indent=1)
    bad = [k for k, v in res.items() if v["rc"] != 0 and not k.split("/")[1].startswith(("ll_", "alpha_"))]
    if bad:
        sys.exit(f"encode failed: {bad}")


def _src(name):
    return {os.path.basename(f): f for f in (STD if name.startswith("std_") else IMG)}


def _score(plan, out_name, want_cv=False):
    """plan: [(config, bin, qs or None, images or None)]; SSIMULACRA2 (+ CVVDP if want_cv) of each file, cached by md5"""
    man = json.load(open(f"{R}/manifests.json"))
    cache_p = f"{R}/score_cache.json"
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    cv = None
    if want_cv:
        cv = bench.Cvvdp("standard_fhd")
        json.dump(cv.info, open(f"{R}/cvvdp_info.json", "w"), indent=1)
    jobs = []
    for name, b, qs, imgs in plan:
        if f"{b}/{name}" not in man:
            sys.exit(f"no {b}/{name} in manifests.json: run the encode step")
        src, depth = _src(name), LOSSY[name][3]
        for f, m in man[f"{b}/{name}"]["files"].items():
            img, q = f.rsplit(".q", 1)
            q = INV[int(q[:-5])]
            if (qs and q not in qs) or (imgs is not None and img not in imgs):
                continue
            jobs.append(dict(config=name, bin=b, image=img, q=q, file=f, depth=depth, md5=m["md5"], bytes=m["bytes"],
                             src=src[img]))
    print(len(jobs), "score jobs", "(+ CVVDP)" if want_cv else "", flush=True)

    def one(j):
        key = f"{j['md5']}:{j['depth']}"
        c = dict(cache.get(key, {}))
        path = f"{E}/{j['bin']}/{j['config']}/{j['file']}"
        if "ssimulacra2" not in c or (want_cv and "cvvdp" not in c):
            with tempfile.TemporaryDirectory(dir=E) as td:
                png = os.path.join(td, "d.png")
                bench.run([AVIFDEC, "-c", "dav1d"] + (["-d", "16"] if j["depth"] > 8 else []) + [path, png])
                if "ssimulacra2" not in c:
                    p = bench.run([SSIM2, j["src"], png])
                    try:
                        c["ssimulacra2"] = float(p.stdout.strip().split()[-1])
                    except (ValueError, IndexError):
                        c["ssimulacra2"] = None
                if want_cv and "cvvdp" not in c:
                    c["cvvdp"] = cv(j["src"], png, path)
        return key, c, j

    rows, done, t0 = [], 0, time.time()
    with cf.ThreadPoolExecutor(16) as ex:
        for key, c, j in ex.map(one, jobs):
            cache[key] = c
            r = {k: j[k] for k in ("config", "bin", "image", "q", "bytes", "md5")}
            r["ssimulacra2"] = c.get("ssimulacra2")
            if want_cv:
                r["cvvdp"] = c.get("cvvdp")
            rows.append(r)
            done += 1
            if done % 200 == 0:
                print(done, "scored", round(time.time() - t0), "s", flush=True)
                json.dump(cache, open(cache_p, "w"))
    json.dump(cache, open(cache_p, "w"))
    json.dump(rows, open(f"{R}/{out_name}", "w"), indent=0)
    nnone = sum(1 for r in rows if r["ssimulacra2"] is None or (want_cv and r.get("cvvdp") is None))
    print(len(rows), "rows,", nnone, "with a failed score", flush=True)
    if nnone > len(rows) // 100:
        sys.exit(f"{nnone} of {len(rows)} scores failed")


def score():
    plan = [(n, "new", None, None) for n in LOSSY] + [("gamut_default", "old", Q6, None)]
    _score(plan, "scores_new.json")


def score_old():
    _score([(n, "old", None, None) for n in LOSSY if n != "gamut_default"], "scores_old.json")


def cvvdp():
    # the CVVDP run's 99 photos (aws-g6-l4-cvvdp has no row for 9fba6f263f3c4f3e_tiny.jpg); OLD: 8 control photos (4 with
    # non-sRGB ICC profiles, 4 others of >= 2 MP), to check the scorer against the stored tiles4x4 / CVVDP-run scores
    cset = {os.path.basename(f) for f in IMG} - {"9fba6f263f3c4f3e_tiny.jpg"}
    ctrl = {"370f7c330a75892f_large.jpg", "bd4ff3b451c3ba6b_large.jpg", "ca12d94f3a1dca2c_large.jpg",
            "fe00a8d898a01166_large.jpg", "0cf2daca3f1c50e1_large.jpg", "16aaa6421eaf9905_huge.jpg",
            "4bbc8bd3085f9e02_large.jpg", "8c67f6e6c2ca8a6d_typical.jpg"}
    _score([("gamut_default", "new", None, cset), ("gamut_default", "old", None, ctrl)], "scores_cvvdp.json", want_cv=True)


def gates():
    os.makedirs(f"{R}/gates", exist_ok=True)
    runs = [("default_q60_120_180", ["--q", "60,120,180"], []),
            ("yuv444_10_q120", ["--q", "120", "--expect-format", "444", "--expect-depth", "10"], ["--yuv", "444", "--depth", "10"]),
            ("turbo3_nosmall_q120", ["--q", "120"], ["--preset", "turbo3", "--small-preset", "none"])]
    bad = []
    for name, gopts, aopts in runs:
        t0 = time.time()
        p = subprocess.run([sys.executable, f"{H}/src/tests/decoder_gate.py", "--bin", BIN["new"], "--images", f"{H}/images",
                            "--jobs", "16"] + gopts + (["--"] + aopts if aopts else []), capture_output=True, text=True)
        open(f"{R}/gates/{name}.log", "w").write(p.stdout + p.stderr + f"\nrc={p.returncode}\n")
        print(name, "rc", p.returncode, round(time.time() - t0), "s", (p.stdout.strip().splitlines() or [""])[-1], flush=True)
        if p.returncode:
            bad.append(name)
    if bad:
        sys.exit(f"gate FAILED: {bad}")


def alpha():
    if not RGBA or not os.path.exists(f"{H}/data/tools/rgba.py"):
        print("no ~/data/rgba set or ~/data/tools/rgba.py: alpha rows not measured")
        return
    import numpy as np
    import importlib.util
    sys.path.insert(0, f"{H}/data/tools")
    import rgba
    spec = importlib.util.spec_from_file_location("gate", f"{H}/src/tests/decoder_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    man = json.load(open(f"{H}/data/rgba/manifest.json"))
    files = sorted(glob.glob(f"{H}/data/rgba/set/*.png"))
    src = {}
    for f in files:
        px, bits = rgba.read_png(f)
        if bits == 16:
            px = rgba.avif_depth(px, 16, 8)
        src[os.path.basename(f)] = px
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for b in BIN:
            for name in ALPHA:
                d = f"{E}/{b}/{name}"
                tot = col = alp = 0
                se = n = mx = 0
                sa = 0.0
                exact = nf = 0
                per = {}
                for base, px in src.items():
                    o = os.path.join(d, base + ".avif")
                    if not os.path.exists(o):
                        continue
                    data = open(o, "rb").read()
                    c, a_, _ = gate.avif_items(data)
                    tot += len(data); col += len(c) if c else 0; alp += len(a_) if a_ else 0
                    png = os.path.join(td, "d.png")
                    subprocess.run([AVIFDEC, "-c", "dav1d", "-d", "8", "--ignore-icc", o, png], capture_output=True, check=True)
                    dec, _ = rgba.read_png(png)
                    if dec.shape[2] < 4:
                        dec = np.concatenate([dec, np.full(dec.shape[:2] + (1,), 255, dec.dtype)], axis=2)
                    err = np.abs(dec[:, :, 3].astype(np.int64) - px[:, :, 3].astype(np.int64))
                    fse = float((err.astype(np.float64) ** 2).sum())
                    se += fse; n += err.size; mx = max(mx, int(err.max())); sa += float(err.sum())
                    exact += int(err.max() == 0); nf += 1
                    per[base] = dict(bytes=len(data), colour=len(c) if c else 0, alpha=len(a_) if a_ else 0,
                                     alpha_psnr=99.0 if fse == 0 else 10 * math.log10(255 * 255 * err.size / fse),
                                     transparent=bool((px[:, :, 3] < 255).any()))
                if not nf:
                    sys.exit(f"no RGBA outputs in {d}: the quality part ran before ~/data was there; run it again")
                mse = se / n
                out[f"{b}/{name}"] = dict(files=nf, bytes=tot, colour=col, alpha=alp, alpha_exact=exact, alpha_max_err=mx,
                                          alpha_mean_abs=sa / n, alpha_psnr=99.0 if mse == 0 else 10 * math.log10(255 * 255 / mse),
                                          per_file=per, set_mp=sum(man["set/" + k]["w"] * man["set/" + k]["h"] for k in src) / 1e6)
                print(b, name, {k: v for k, v in out[f"{b}/{name}"].items() if k != "per_file"}, flush=True)
    json.dump(out, open(f"{R}/alpha.json", "w"), indent=1)


if __name__ == "__main__":
    for step in sys.argv[1:]:
        {"encode": encode, "score": score, "score_old": score_old, "cvvdp": cvvdp, "gates": gates, "alpha": alpha}[step]()
