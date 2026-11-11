#!/usr/bin/env python3
"""v0.13.0 docs refresh, speed side (on the instance): the docs speed list of aws-g6-l4-speed-v06/scripts/spd.py (as rerun
for v0.9.0 in aws-g6-g7-v090-check), NEW = v0.13.0 (~/bin/new) against OLD = v0.12.0 (~/bin/old), same method:
each set repeated to >= 1,000 MP (bench.speed_set; the lossless and RGBA sets as published), per setting and binary a
warm-up then 3 timed runs with --stats, 2 rounds with the binary order alternating; median of the 6 timed runs.

The list is split into groups, so that the most quoted rows finish first and the rest can be skipped by name:

  spd13.py core     size range: default with GP_ENCODERS=4 and 3 (the default), --small-preset none (4), turbo3 and
                    turbo3 nosmall (4); --yuv 444 / --depth 10 / both (GP_ENCODERS=4, as bench/formats.py ran)
  spd13.py small    the thumbnails (~/sets/thumbs) and the 24 Kodak PNGs: default, nosmall (+ turbo3 nosmall on Kodak)
  spd13.py std      the standard set (100 photos): default with 4 / 2 / 3 encoders, nosmall with 4 / 3
  spd13.py ll       the lossless comparison set (~/ll/cmp): one pass (incl. start-up) and 4x, with and without --ycgco-re
  spd13.py extra    size range: default with 2 encoders, nosmall with 3, --tiles 2,1,1 and 2,3,3 (3 encoders)
  spd13.py alpha    the 39-PNG RGBA set repeated 20x, GP_ENCODERS=4: default, --alpha-quality 100, --lossless,
                    --no-alpha (only with ~/data/rgba)

Each run appends a JSON line to ~/results/speed/<group>.jsonl (phase / tag as spd.py's, so the keys are
"<phase>/<tag>/<new|old>" as in the published results.json); a restarted group starts its file afresh."""
import glob, json, os, shutil, statistics, subprocess, sys, time

H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/src/bench")
import bench  # noqa: E402
from PIL import Image  # noqa: E402

R = f"{H}/results/speed"
W = f"{H}/work"
os.makedirs(R, exist_ok=True)
os.makedirs(W, exist_ok=True)
BIN = {"new": f"{H}/bin/new", "old": f"{H}/bin/old"}
OUT = None   # the group's jsonl


def sizes_of(files):
    o = {}
    for f in files:
        w, h = Image.open(f).size
        o[os.path.basename(f)] = w * h / 1e6
    return o


def speedset(name, files):
    if not files:
        sys.exit(f"empty input set for {name}: run setup.sh")
    d, sfiles, mp, rep = bench.speed_set(files, sizes_of(files), 1000, f"{W}/{name}")
    return d, len(sfiles), mp, rep


def one(phase, b, src, opts, env, tag, timed):
    out = f"{W}/out"
    shutil.rmtree(out, ignore_errors=True)
    e = dict(os.environ, **env)
    t0 = time.time()
    p = subprocess.run([BIN[b], "--batch", src, "--outdir", out, "--stats"] + opts, capture_output=True, text=True, env=e)
    t1 = time.time()
    st = {}
    try:
        st = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        pass
    line = next((l for l in p.stderr.splitlines() if "MP/s" in l), "")
    r = dict(phase=phase, tag=tag, bin=b, opts=opts, env=env, timed=timed, rc=p.returncode, t0=t0, t1=t1,
             wall_s=round(t1 - t0, 3), stats=st, line=line, files=len(os.listdir(out)) if os.path.isdir(out) else 0)
    with open(OUT, "a") as f:
        f.write(json.dumps(r) + "\n")
    if p.returncode or not st:
        print("FAILED", phase, tag, b, p.returncode, p.stderr[-500:], flush=True)
    return r


def speed_call(phase, b, src, opts, env, tag, repeats=3):
    """bench.gpu_speed: a warm-up, then `repeats` timed runs"""
    runs = [r for i in range(repeats + 1) for r in [one(phase, b, src, opts, env, tag, i > 0)] if i]
    mps = [r["stats"].get("mp_s", 0) for r in runs]
    print(f"{phase} {tag} {b}: {statistics.median(mps):.1f} MP/s (runs {', '.join(f'{x:.1f}' for x in mps)}), "
          f"cores {statistics.median(r['stats'].get('cpu_cores', 0) for r in runs):.2f}", flush=True)
    return runs


E = lambda n: {"GP_ENCODERS": str(n)} if n else {}  # noqa: E731


def jpeg(cfg):
    names = sorted({s for s, *_ in cfg})
    pats = {"size_range": "images/*.jpg", "standard": "images_std/*.jpg"}
    sets = {s: speedset({"size_range": "sr", "standard": "std"}[s], sorted(glob.glob(f"{H}/{pats[s]}"))) for s in names}
    json.dump({k: v[1:] for k, v in sets.items()}, open(f"{R}/jpeg_sets_{'_'.join(names)}.json", "w"))
    for rnd in range(2):
        for s, tag, n, opts in cfg:
            for b in (("new", "old") if rnd % 2 == 0 else ("old", "new")):
                speed_call("jpeg", b, sets[s][0], opts, E(n), f"{s}/{tag}")


def formats():
    d = speedset("sr", sorted(glob.glob(f"{H}/images/*.jpg")))[0]
    F = [("444_8", ["--yuv", "444"]), ("420_10", ["--depth", "10"]), ("444_10", ["--yuv", "444", "--depth", "10"])]
    for rnd in range(2):
        for k, opts in (F if rnd % 2 == 0 else list(reversed(F))):
            for b in (("new", "old") if rnd % 2 == 0 else ("old", "new")):
                speed_call("formats", b, d, opts, {"GP_ENCODERS": "4"}, k)


def rep_dir(name, files, n):
    d = f"{W}/{name}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for k in range(n):
        for f in files:
            os.symlink(f, f"{d}/r{k:02d}_{os.path.basename(f)}")
    return d


def small(phase, files, turbo3=False):
    d, n, mp, rep = speedset(phase, files)
    json.dump(dict(files=len(files), repeated_files=n, mp=mp, repeats=rep), open(f"{R}/{phase}_set.json", "w"))
    X = [("default", []), ("nosmall", ["--small-preset", "none"])] + \
        ([("turbo3_nosmall", ["--preset", "turbo3", "--small-preset", "none"])] if turbo3 else [])
    for rnd in range(2):
        for tag, opts in (X if rnd % 2 == 0 else list(reversed(X))):
            for b in (("new", "old") if rnd % 2 == 0 else ("old", "new")):
                speed_call(phase, b, d, opts, {}, tag)


def g_core():
    jpeg([("size_range", "default_enc4", 4, []),
          ("size_range", "default_enc3", None, []),
          ("size_range", "nosmall_enc4", 4, ["--small-preset", "none"]),
          ("size_range", "turbo3ns_enc4", 4, ["--preset", "turbo3", "--small-preset", "none"]),
          ("size_range", "turbo3_enc4", 4, ["--preset", "turbo3"])])
    formats()


def g_small():
    small("thumbs", sorted(glob.glob(f"{H}/sets/thumbs/*.jpg")))
    small("kodak", sorted(glob.glob(f"{H}/ll/kodak/*.png")), turbo3=True)


def g_std():
    jpeg([("standard", "default_enc4", 4, []),
          ("standard", "default_enc2", 2, []),
          ("standard", "default_enc3", None, []),
          ("standard", "nosmall_enc4", 4, ["--small-preset", "none"]),
          ("standard", "nosmall_enc3", None, ["--small-preset", "none"])])


def g_extra():
    jpeg([("size_range", "default_enc2", 2, []),
          ("size_range", "nosmall_enc3", None, ["--small-preset", "none"]),
          ("size_range", "tiles211_enc3", None, ["--tiles", "2,1,1"]),   # 2 x 2 tiles (the default up to v0.3.0)
          ("size_range", "tiles233_enc3", None, ["--tiles", "2,3,3"])])  # 8 x 8


def g_ll():
    cmp = sorted(glob.glob(f"{H}/ll/cmp/*.png"))
    if not cmp:
        sys.exit("no ~/ll/cmp: run setup.sh")
    one_pass = rep_dir("ll1", cmp, 1)
    four = rep_dir("ll4", cmp, 4)
    mp = sum(sizes_of(cmp).values())
    json.dump(dict(files=len(cmp), mp=mp, bytes=sum(os.path.getsize(f) for f in cmp)), open(f"{R}/lossless_set.json", "w"))
    for rnd in range(2):
        for b in (("new", "old") if rnd % 2 == 0 else ("old", "new")):
            for tag, src, opts in (("cmp_1pass", one_pass, ["--lossless"]), ("cmp_1pass_ycgco", one_pass, ["--lossless", "--ycgco-re"]),
                                   ("cmp_4x", four, ["--lossless"]), ("cmp_4x_ycgco", four, ["--lossless", "--ycgco-re"])):
                speed_call("lossless", b, src, opts, {}, tag)


def g_alpha():
    files = sorted(glob.glob(f"{H}/data/rgba/set/*.png"))
    if not files:
        print("no ~/data/rgba/set: the alpha rows are not measured")
        return
    d = rep_dir("rgba20", files, 20)
    man = json.load(open(f"{H}/data/rgba/manifest.json"))
    mp = 20 * sum(v["w"] * v["h"] for k, v in man.items() if k.startswith("set/")) / 1e6
    json.dump(dict(files=20 * len(files), mp=mp), open(f"{R}/alpha_set.json", "w"))
    X = [("default", []), ("q100", ["--alpha-quality", "100"]), ("lossless", ["--lossless"]), ("no_alpha", ["--no-alpha"])]
    for rnd in range(2):
        for b in (("new", "old") if rnd % 2 == 0 else ("old", "new")):
            for tag, opts in X:
                speed_call("alpha", b, d, opts, {"GP_ENCODERS": "4"}, tag)


GROUPS = {"core": g_core, "small": g_small, "std": g_std, "ll": g_ll, "extra": g_extra, "alpha": g_alpha}

if __name__ == "__main__":
    for g in sys.argv[1:]:
        OUT = f"{R}/{g}.jsonl"
        if os.path.exists(OUT):
            os.rename(OUT, f"{OUT}.{int(time.time())}.restarted")
        GROUPS[g]()
        bad = [json.loads(l) for l in open(OUT)] if os.path.exists(OUT) else []
        bad = [r for r in bad if r["rc"] or not r["stats"]]
        if bad:
            sys.exit(f"{g}: {len(bad)} failed runs")
