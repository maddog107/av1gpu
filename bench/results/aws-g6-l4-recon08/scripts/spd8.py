#!/usr/bin/env python3
"""recon08 speed A/B (the method of bench/results/aws-g6-l4-speed-v04 / -v06): every set repeated to >= 1,000 MP
(bench.speed_set), a warm-up and 3 timed runs per binary, 2 rounds with the binary order alternating, median of the
6 timed runs (av1gpu's --stats MP/s and img/s).  Binaries: spd8.py A=path B=path ... (labels free; a noise run is
two labels of the same binary).  Single-file mode: one JPEG, cold process (`av1gpu in.jpg out.avif`), a warm-up,
then 12 runs per binary alternating, median wall time.  Every run is a JSON line in ~/results/spd8/<tag>.jsonl."""
import glob, json, os, shutil, statistics, subprocess, sys, time

H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/new/bench")
import bench  # noqa: E402
from PIL import Image  # noqa: E402

TAG = os.environ.get("TAG", "ab")
R = f"{H}/results/spd8"
W = f"{H}/work/spd8"
os.makedirs(R, exist_ok=True)
os.makedirs(W, exist_ok=True)
BIN = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
SETS = os.environ.get("SETS", "sr,std,thumbs,single").split(",")
LOG = f"{R}/{TAG}.jsonl"


def sizes_of(files):
    return {os.path.basename(f): Image.open(f).size[0] * Image.open(f).size[1] / 1e6 for f in files}


def one(label, src, opts, timed, rnd, name):
    out = f"{W}/out"
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.time()
    p = subprocess.run([BIN[label], "--batch", src, "--outdir", out, "--stats"] + opts, capture_output=True, text=True)
    t1 = time.time()
    st = {}
    try:
        st = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        pass
    line = next((l for l in p.stderr.splitlines() if "MP/s" in l), "")
    stage = next((l for l in p.stderr.splitlines() if "stage ms" in l), "")
    r = dict(set=name, label=label, opts=opts, timed=timed, round=rnd, rc=p.returncode, wall_s=round(t1 - t0, 3), stats=st,
             line=line, stage=stage)
    with open(LOG, "a") as f:
        f.write(json.dumps(r) + "\n")
    if p.returncode:
        print("FAILED", name, label, p.stderr[-400:], flush=True)
    return r


def mps(r):
    s = r["stats"]
    for k in ("mp_per_s", "mps", "MPps"):
        if k in s:
            return s[k]
    import re
    m = re.search(r"([0-9.]+) MP/s", r["line"])
    return float(m.group(1)) if m else float("nan")


def ips(r):
    import re
    m = re.search(r"([0-9.]+) img/s", r["line"])
    return float(m.group(1)) if m else float("nan")


def batch_set(name, files, opts=()):
    d, sf, mp, rep = bench.speed_set(files, sizes_of(files), float(os.environ.get("MINMP", "1000")), f"{W}/{name}")
    labels = list(BIN)
    res = {l: [] for l in labels}
    for rnd in range(2):
        order = labels if rnd % 2 == 0 else labels[::-1]
        for l in order:
            one(l, d, list(opts), False, rnd, name)
            for k in range(3):
                res[l].append(one(l, d, list(opts), True, rnd, name))
    print(f"== {name} {' '.join(opts)}: {len(sf)} files, {mp:.0f} MP ({rep}x)", flush=True)
    base = None
    for l in labels:
        m = statistics.median(mps(r) for r in res[l])
        i = statistics.median(ips(r) for r in res[l])
        rounds = [round(statistics.median(mps(r) for r in res[l] if r["round"] == k), 2) for k in range(2)]
        base = base or m
        print(f"  {l}: {m:.2f} MP/s, {i:.2f} img/s  rounds {rounds}  vs {labels[0]} {100 * (m / base - 1):+.2f} %", flush=True)


def single(name, f):
    labels = list(BIN)
    out = f"{W}/single.avif"
    for l in labels:
        subprocess.run([BIN[l], f, out], capture_output=True)   # warm-up
    t = {l: [] for l in labels}
    for k in range(12):
        for l in (labels if k % 2 == 0 else labels[::-1]):
            t0 = time.time()
            p = subprocess.run([BIN[l], f, out], capture_output=True)
            w = time.time() - t0
            t[l].append(w)
            with open(LOG, "a") as fo:
                fo.write(json.dumps(dict(set=name, label=l, round=k, rc=p.returncode, wall_s=round(w, 4), file=f)) + "\n")
    print(f"== {name}: {os.path.basename(f)} {Image.open(f).size}", flush=True)
    base = None
    for l in labels:
        m = statistics.median(t[l])
        base = base or m
        print(f"  {l}: median {1000 * m:.1f} ms (min {1000 * min(t[l]):.1f}, max {1000 * max(t[l]):.1f})  vs {labels[0]} "
              f"{100 * (m / base - 1):+.2f} %", flush=True)


photos = sorted(glob.glob(f"{H}/images/*.jpg"))
print("start", time.strftime("%H:%M:%S", time.gmtime()), BIN, flush=True)
if "sr" in SETS:
    batch_set("size_range_default", photos)
if "std" in SETS:
    batch_set("standard_default", sorted(glob.glob(f"{H}/images_std/*.jpg")))
if "thumbs" in SETS:
    batch_set("thumbs_default", sorted(glob.glob(f"{H}/sets/thumbs/*.jpg")))
if "single" in SETS:
    # one JPEG of 2-12 MP: the size-range photo closest to 8 MP
    cand = [(abs(Image.open(f).size[0] * Image.open(f).size[1] / 1e6 - 8), f) for f in photos]
    single("single_jpeg_8mp", min(cand)[1])
if "single2" in SETS:   # also a 2-3 MP photo and a 512x384 thumbnail
    cand = [(abs(Image.open(f).size[0] * Image.open(f).size[1] / 1e6 - 2.5), f) for f in photos]
    single("single_jpeg_2.5mp", min(cand)[1])
    th = sorted(glob.glob(f"{H}/sets/thumbs/d512_*.jpg"))
    single("single_thumb_512x384", th[len(th) // 2])
print("end", time.strftime("%H:%M:%S", time.gmtime()), flush=True)
print("SPD8_DONE", flush=True)
