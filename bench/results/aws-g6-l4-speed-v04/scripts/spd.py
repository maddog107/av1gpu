#!/usr/bin/env python3
"""Speed rerun for v0.4 (perf/cpu-threads fdb0628, 4x4 tiles + one decode thread per CPU) against v0.3.0, on an L4
g6.4xlarge.  Every av1gpu speed in README.md / docs/RESULTS.md, measured the way it was published:

  spd.py jpeg      bench.speed_set (each set repeated to >= 1,000 MP) + bench.gpu_speed's method (a warm-up, then
                   3 timed runs, --stats); 2 interleaved rounds, binary order alternating; median of the 6 runs
  spd.py formats   the same for --yuv 444 / --depth 10 / both (GP_ENCODERS=4, as bench/formats.py ran)
  spd.py lossless  the 47-PNG comparison set: one pass incl. start-up (wall), the set 4x (av1gpu's MP/s), and the
                   size-range JPEGs (av1gpu's MP/s); a warm-up, then 3 runs
  spd.py alpha     the 39-PNG RGBA set repeated 20x (202 MP), GP_ENCODERS=4, wall time incl. start-up; a warm-up, 3 runs
  spd.py demo      the README terminal recording (rec.py session in ~/av1gpu-readme)

Every run appends a JSON line (start/end time, settings, --stats) to ~/results/<phase>.jsonl."""
import glob, json, os, shutil, statistics, subprocess, sys, time

H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/new/bench")
import bench  # noqa: E402

R = f"{H}/results"
W = f"{H}/work"
os.makedirs(R, exist_ok=True)
os.makedirs(W, exist_ok=True)
BIN = {"new": f"{H}/new/build/av1gpu", "old": f"{H}/old/build/av1gpu"}
from PIL import Image  # noqa: E402


def sizes_of(files):
    return {os.path.basename(f): float(Image.open(f).size[0] * Image.open(f).size[1]) / 1e6 for f in files}


def speedset(name, files):
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
    with open(f"{R}/{phase}.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
    if p.returncode:
        print("FAILED", tag, b, p.stderr[-500:], flush=True)
    return r


def speed_call(phase, b, src, opts, env, tag, repeats=3):
    """bench.gpu_speed: a warm-up, then `repeats` timed runs"""
    runs = []
    for i in range(repeats + 1):
        r = one(phase, b, src, opts, env, tag, i > 0)
        if i:
            runs.append(r)
    mps = [r["stats"].get("mp_s", 0) for r in runs]
    print(f"{phase} {tag} {b}: {statistics.median(mps):.1f} MP/s (runs {', '.join(f'{x:.1f}' for x in mps)}), "
          f"cores {statistics.median(r['stats'].get('cpu_cores', 0) for r in runs):.2f}", flush=True)
    return runs


def jpeg():
    gam = sorted(glob.glob(f"{H}/images/*.jpg"))
    std = sorted(glob.glob(f"{H}/images_std/*.jpg"))
    sets = {"size_range": speedset("sr", gam), "standard": speedset("std", std)}
    json.dump({k: v[1:] for k, v in sets.items()}, open(f"{R}/jpeg_sets.json", "w"))
    E = lambda n: {"GP_ENCODERS": str(n)} if n else {}  # noqa: E731
    cfg = [  # set, tag, encoders (None: the default), options, binaries
        ("size_range", "default_enc4", 4, [], ("new", "old")),
        ("size_range", "default_enc2", 2, [], ("new", "old")),
        ("size_range", "default_enc3", None, [], ("new", "old")),
        ("size_range", "nosmall_enc4", 4, ["--small-preset", "none"], ("new", "old")),
        ("size_range", "nosmall_enc3", None, ["--small-preset", "none"], ("new", "old")),
        ("size_range", "turbo3ns_enc4", 4, ["--preset", "turbo3", "--small-preset", "none"], ("new", "old")),
        ("size_range", "turbo3_enc4", 4, ["--preset", "turbo3"], ("new", "old")),
        ("standard", "default_enc4", 4, [], ("new", "old")),
        ("standard", "default_enc2", 2, [], ("new",)),
        ("standard", "default_enc3", None, [], ("new",)),
        ("standard", "nosmall_enc4", 4, ["--small-preset", "none"], ("new", "old")),
        ("standard", "nosmall_enc3", None, ["--small-preset", "none"], ("new", "old")),
    ]
    for rnd in range(2):
        for s, tag, n, opts, bins in cfg:
            for b in (bins if rnd % 2 == 0 else tuple(reversed(bins))):
                speed_call("jpeg", b, sets[s][0], opts, E(n), f"{s}/{tag}")


def formats():
    gam = sorted(glob.glob(f"{H}/images/*.jpg"))
    d = speedset("sr", gam)[0]
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


def lossless():
    cmp = sorted(glob.glob(f"{H}/ll/cmp/*.png"))
    one_pass = rep_dir("ll1", cmp, 1)
    four = rep_dir("ll4", cmp, 4)
    jpg = sorted(glob.glob(f"{H}/images/*.jpg"))
    mp = sum(sizes_of(cmp).values())
    json.dump(dict(files=len(cmp), mp=mp, bytes=sum(os.path.getsize(f) for f in cmp)), open(f"{R}/lossless_set.json", "w"))
    for rnd in range(2):
        for b in (("new", "old") if rnd % 2 == 0 else ("old", "new")):
            for tag, src, opts in (("cmp_1pass", one_pass, ["--lossless"]), ("cmp_1pass_ycgco", one_pass, ["--lossless", "--ycgco-re"]),
                                   ("cmp_4x", four, ["--lossless"]), ("cmp_4x_ycgco", four, ["--lossless", "--ycgco-re"])):
                speed_call("lossless", b, src, opts, {}, tag)


def alpha():
    files = sorted(glob.glob(f"{H}/data/rgba/set/*.png"))
    d = rep_dir("rgba20", files, 20)
    man = json.load(open(f"{H}/data/rgba/manifest.json"))
    mp = 20 * sum(v["w"] * v["h"] for k, v in man.items() if k.startswith("set/")) / 1e6
    json.dump(dict(files=20 * len(files), mp=mp), open(f"{R}/alpha_set.json", "w"))
    X = [("default", []), ("q100", ["--alpha-quality", "100"]), ("lossless", ["--lossless"]), ("no_alpha", ["--no-alpha"])]
    for rnd in range(2):
        for b in (("new", "old") if rnd % 2 == 0 else ("old", "new")):
            for tag, opts in X:
                speed_call("alpha", b, d, opts, {"GP_ENCODERS": "4"}, tag)


def demo():
    d = f"{H}/av1gpu-readme"
    shutil.rmtree(f"{d}/avif", ignore_errors=True)
    subprocess.run(f"{d}/av1gpu/build/av1gpu --batch {d}/photos --outdir /tmp/av1gpu-warm > /dev/null 2>&1; rm -rf /tmp/av1gpu-warm",
                   shell=True)
    cmds = ["ls photos | wc -l; du -sh photos", "bash -c 'time av1gpu --batch photos/ --outdir avif/'", "du -sh avif/"]
    p = subprocess.run([sys.executable, f"{H}/rec.py", json.dumps(cmds)], capture_output=True, text=True, cwd=d)
    open(f"{R}/rec.json", "w").write(p.stdout)
    print(p.stdout[:2000], p.stderr[-500:], flush=True)


if __name__ == "__main__":
    for ph in sys.argv[1:]:
        {"jpeg": jpeg, "formats": formats, "lossless": lossless, "alpha": alpha, "demo": demo}[ph]()
