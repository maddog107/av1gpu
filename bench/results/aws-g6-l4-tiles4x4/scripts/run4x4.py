#!/usr/bin/env python3
"""4x4-tile default: re-encode and re-score av1gpu's side of the published L4 compression results (on the instance).

  run4x4.py encode     every config with NEW (perf/tiles-4x4) and OLD (v0.3.0); size + md5 manifests
  run4x4.py score      SSIMULACRA2 (+ CVVDP, + luma-only CVVDP) of the files the tables need; cache by md5
  run4x4.py gates      tests/decoder_gate.py with NEW on the size-range photos
  run4x4.py alpha      RGBA set: av1gpu rows (bytes, colour/alpha split, alpha error), NEW and OLD

Run with ~/venv/bin/python (Pillow/numpy of the published CVVDP run).  Results in ~/results/."""
import concurrent.futures as cf, glob, hashlib, json, math, os, subprocess, sys, tempfile, time, types
H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/new/bench")
import bench  # noqa: E402
import numpy as np  # noqa: E402

R, E = f"{H}/results", f"{H}/enc"
os.makedirs(R, exist_ok=True)
BIN = {"new": f"{H}/new/build/av1gpu", "old": f"{H}/old/build/av1gpu"}
AVIFDEC, SSIM2 = f"{H}/opt/bin/avifdec", f"{H}/opt/bin/ssimulacra2"
IMG = sorted(glob.glob(f"{H}/images/*.jpg"))
STD = sorted(glob.glob(f"{H}/images_std/*.jpg"))
Q6, Q8 = [30, 40, 50, 60, 70, 80], [30, 40, 50, 60, 70, 80, 85, 90]
# name: (inputs, av1gpu options, quality points or None (lossless: one pass), depth for decoding)
LOSSY = {
    "gamut_default": (IMG, [], Q8, 8),
    "gamut_nosmall": (IMG, ["--small-preset", "none"], Q6, 8),
    "gamut_turbo3ns": (IMG, ["--preset", "turbo3", "--small-preset", "none"], Q6, 8),
    "gamut_turbo3": (IMG, ["--preset", "turbo3"], Q6, 8),
    "fmt_444_8": (IMG, ["--yuv", "444"], Q6, 8),
    "fmt_420_10": (IMG, ["--depth", "10"], Q6, 10),
    "fmt_444_10": (IMG, ["--yuv", "444", "--depth", "10"], Q6, 10),
    "std_default": (STD, [], Q6, 8),
    "std_nosmall": (STD, ["--small-preset", "none"], Q6, 8),
    "std_turbo3ns": (STD, ["--preset", "turbo3", "--small-preset", "none"], Q6, 8),
}
LL = {
    "ll_jpeg_gamut": (IMG, ["--lossless"]),
    "ll_cmp": (sorted(glob.glob(f"{H}/ll/cmp/*.png")), ["--lossless"]),
    "ll_cmp_ycgco": (sorted(glob.glob(f"{H}/ll/cmp/*.png")), ["--lossless", "--ycgco-re"]),
    "ll_misc": (sorted(glob.glob(f"{H}/ll/misc/*.png")), ["--lossless"]),
    "ll_jpeg_misc": (sorted(glob.glob(f"{H}/ll/jpeg/*")), ["--lossless"]),
}
RGBA = sorted(glob.glob(f"{H}/data/rgba/set/*.png")) + sorted(glob.glob(f"{H}/data/rgba/extra/*"))
ALPHA = {"alpha_default": [], "alpha_q100": ["--alpha-quality", "100"], "alpha_lossless": ["--lossless"]}


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
    return dict(rc=p.returncode, wall_s=round(time.time() - t0, 2), files=man, inputs=len(files),
                summary=next((l for l in p.stderr.splitlines() if "MP/s" in l), ""))


def encode():
    res = {}
    for name, (files, opts, qs, _) in LOSSY.items():
        qi = ",".join(str(bench.qindex_for_quality(q)) for q in qs)
        for b in BIN:
            res[f"{b}/{name}"] = batch(b, name, files, ["--qindex", qi, "--name-q"] + opts)
            print(b, name, res[f"{b}/{name}"]["rc"], len(res[f"{b}/{name}"]["files"]), "files", flush=True)
    for name, (files, opts) in list(LL.items()) + [(k, (RGBA, v)) for k, v in ALPHA.items()]:
        for b in BIN:
            res[f"{b}/{name}"] = batch(b, name, files, opts)
            print(b, name, res[f"{b}/{name}"]["rc"], len(res[f"{b}/{name}"]["files"]), "files", flush=True)
    json.dump(res, open(f"{R}/manifests.json", "w"), indent=1)
    # NEW vs OLD per config
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
        for lo, hi, lab in ((0, 2, "lt2"), (2, 1e9, "ge2")):
            ks = [k for k in n if k in o and lo <= mp.get(k.split(".q")[0] if ".q" in k else k[:-5], -1) < hi]
            c[lab] = dict(files=len(ks), identical=sum(n[k]["md5"] == o[k]["md5"] for k in ks),
                          bytes_new=sum(n[k]["bytes"] for k in ks), bytes_old=sum(o[k]["bytes"] for k in ks))
        unk = [k for k in n if k in o and (k.split(".q")[0] if ".q" in k else k[:-5]) not in mp]
        c["unknown_size"] = dict(files=len(unk), identical=sum(n[k]["md5"] == o[k]["md5"] for k in unk))
        cmp[name] = c
        print(name, {k: v for k, v in c.items() if k not in ("only_new", "only_old")}, flush=True)
    json.dump(dict(megapixels=mp, compare=cmp), open(f"{R}/compare.json", "w"), indent=1)


def score():
    man = json.load(open(f"{R}/manifests.json"))
    src = {os.path.basename(f): f for f in IMG}
    inv = {bench.qindex_for_quality(q): q for q in Q8}
    big = json.load(open(f"{R}/compare.json"))["megapixels"]
    icc4 = ["370f7c330a75892f_large.jpg", "bd4ff3b451c3ba6b_large.jpg", "ca12d94f3a1dca2c_large.jpg", "fe00a8d898a01166_large.jpg"]
    others = sorted(k for k in src if big.get(k, 0) >= 2 and k not in icc4)[::12][:4]
    ctrl = set(icc4 + others)
    # (config, bin) to score, and which metrics
    plan = [("gamut_default", "new", True, True), ("gamut_default", "old", "ctrl", "ctrl"),
            ("gamut_nosmall", "new", False, False), ("gamut_turbo3ns", "new", False, False),
            ("gamut_turbo3ns", "old", False, False), ("gamut_turbo3", "new", False, False),
            ("gamut_turbo3", "old", False, False), ("fmt_444_8", "new", False, False),
            ("fmt_420_10", "new", False, False), ("fmt_444_10", "new", False, False)]
    cache_p = f"{R}/score_cache.json"
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    cv = bench.Cvvdp("standard_fhd")
    json.dump(cv.info, open(f"{R}/cvvdp_info.json", "w"), indent=1)
    jobs = []
    for name, b, want_cv, want_luma in plan:
        depth = LOSSY[name][3]
        for f, m in man[f"{b}/{name}"]["files"].items():
            img, q = f.rsplit(".q", 1)
            q = inv[int(q[:-5])]
            if want_cv == "ctrl" and img not in ctrl:
                continue
            jobs.append(dict(config=name, bin=b, image=img, q=q, file=f, depth=depth, md5=m["md5"], bytes=m["bytes"],
                             cv=bool(want_cv), luma=bool(want_luma) and (b == "old" or big.get(img, 0) >= 2),
                             luma_noicc=bool(want_luma) and img in icc4))
    print(len(jobs), "score jobs", flush=True)

    def one(j):
        key = f"{j['md5']}:{j['depth']}"
        c = cache.get(key, {})
        path = f"{E}/{j['bin']}/{j['config']}/{j['file']}"
        need = ("ssimulacra2" not in c) or (j["cv"] and "cvvdp" not in c) or (j["luma"] and "luma" not in c) or \
               (j["luma_noicc"] and "luma_noicc" not in c)
        if need:
            with tempfile.TemporaryDirectory(dir=E) as td:
                png = os.path.join(td, "d.png")
                bench.run([AVIFDEC, "-c", "dav1d"] + (["-d", "16"] if j["depth"] > 8 else []) + [path, png])
                if "ssimulacra2" not in c:
                    p = bench.run([SSIM2, src[j["image"]], png])
                    try:
                        c["ssimulacra2"] = float(p.stdout.strip().split()[-1])
                    except (ValueError, IndexError):
                        c["ssimulacra2"] = None
                if j["cv"] and "cvvdp" not in c:
                    c["cvvdp"] = cv(src[j["image"]], png, path)
                if (j["luma"] and "luma" not in c) or (j["luma_noicc"] and "luma_noicc" not in c):
                    from PIL import Image

                    def grey(a):
                        return np.asarray(Image.fromarray(a).convert("L"))
                    test = grey(bench.srgb8(png))
                    if j["luma"] and "luma" not in c:
                        ref = grey(bench.srgb8(src[j["image"]]))
                        with cv.lock:
                            c["luma"] = float(cv.metric.predict(test, ref, dim_order="HW")[0])
                    if j["luma_noicc"] and "luma_noicc" not in c:
                        keep = bench._jpeg_icc
                        bench._jpeg_icc = lambda im: None
                        try:
                            ref = grey(bench.srgb8(src[j["image"]]))
                        finally:
                            bench._jpeg_icc = keep
                        with cv.lock:
                            c["luma_noicc"] = float(cv.metric.predict(test, ref, dim_order="HW")[0])
        return key, c, j

    rows = []
    done = 0
    with cf.ThreadPoolExecutor(16) as ex:
        for key, c, j in ex.map(one, jobs):
            cache[key] = c
            r = {k: j[k] for k in ("config", "bin", "image", "q", "bytes", "md5")}
            r.update(ssimulacra2=c.get("ssimulacra2"))
            if j["cv"]:
                r["cvvdp"] = c.get("cvvdp")
            if j["luma"]:
                r["cvvdp_luma"] = c.get("luma")
            if j["luma_noicc"]:
                r["cvvdp_luma_noicc"] = c.get("luma_noicc")
            rows.append(r)
            done += 1
            if done % 200 == 0:
                print(done, "scored", flush=True)
                json.dump(cache, open(cache_p, "w"))
    json.dump(cache, open(cache_p, "w"))
    json.dump(rows, open(f"{R}/scores.json", "w"), indent=0)
    print(len(rows), "rows", flush=True)


def gates():
    os.makedirs(f"{R}/gates", exist_ok=True)
    runs = [("default_q60_120_180", ["--q", "60,120,180"], []),
            ("yuv444_10_q120", ["--q", "120", "--expect-format", "444", "--expect-depth", "10"], ["--yuv", "444", "--depth", "10"]),
            ("turbo3_nosmall_q120", ["--q", "120"], ["--preset", "turbo3", "--small-preset", "none"])]
    for name, gopts, aopts in runs:
        t0 = time.time()
        p = subprocess.run([sys.executable, f"{H}/new/tests/decoder_gate.py", "--bin", BIN["new"], "--images", f"{H}/images",
                            "--jobs", "16"] + gopts + (["--"] + aopts if aopts else []), capture_output=True, text=True)
        open(f"{R}/gates/{name}.log", "w").write(p.stdout + p.stderr)
        print(name, "rc", p.returncode, round(time.time() - t0), "s", (p.stdout.strip().splitlines() or [""])[-1], flush=True)


def alpha():
    import importlib.util
    sys.path.insert(0, f"{H}/data/tools")
    import rgba
    spec = importlib.util.spec_from_file_location("gate", f"{H}/new/tests/decoder_gate.py")
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
                mse = se / n
                out[f"{b}/{name}"] = dict(files=nf, bytes=tot, colour=col, alpha=alp, alpha_exact=exact, alpha_max_err=mx,
                                          alpha_mean_abs=sa / n, alpha_psnr=99.0 if mse == 0 else 10 * math.log10(255 * 255 / mse),
                                          per_file=per, set_mp=sum(man["set/" + k]["w"] * man["set/" + k]["h"] for k in src) / 1e6)
                print(b, name, {k: v for k, v in out[f"{b}/{name}"].items() if k != "per_file"}, flush=True)
    json.dump(out, open(f"{R}/alpha.json", "w"), indent=1)


if __name__ == "__main__":
    {"encode": encode, "score": score, "gates": gates, "alpha": alpha}[sys.argv[1]]()
