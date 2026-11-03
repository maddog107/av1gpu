#!/usr/bin/env python3
"""Speed A/B for ent08 (the stored method of bench/results/aws-g6-l4-speed-v06/scripts/spd.py):

  spd8.py TAG BIN_A BIN_B [SET...]

SETs: sr (size range, default), std (standard 100, default), thumbs (~/sets/thumbs, default), single (one 2-12 MP
JPEG, `av1gpu in.jpg out.avif`, cold process, median of 12 runs per binary, order alternating).  Each batch set is
repeated to >= 1,000 MP (bench.speed_set); per binary a warm-up + 3 timed runs (--stats); 2 rounds with the binary
order alternating; the median of the 6 timed runs.  Every run appends a JSON line to ~/results/spd_TAG.jsonl and a
summary line to stdout.  BIN_A == a copy of BIN_B's build gives the noise run."""
import glob, json, os, shutil, statistics, subprocess, sys, time

H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/main/bench")
import bench  # noqa: E402
from PIL import Image  # noqa: E402

TAG, A, B = sys.argv[1:4]
SETS = sys.argv[4:] or ["sr", "std", "thumbs", "single"]
R = f"{H}/results"
W = f"{H}/work"
os.makedirs(R, exist_ok=True)
os.makedirs(W, exist_ok=True)
BIN = {"A": A, "B": B}
LOG = f"{R}/spd_{TAG}.jsonl"


def speedset(name, files):
    sizes = {os.path.basename(f): Image.open(f).size[0] * Image.open(f).size[1] / 1e6 for f in files}
    d, sfiles, mp, rep = bench.speed_set(files, sizes, 1000, f"{W}/{name}")
    return d


def one(b, src, opts, tag, timed):
    out = f"{W}/out"
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.time()
    p = subprocess.run([BIN[b], "--batch", src, "--outdir", out, "--stats"] + opts, capture_output=True, text=True)
    t1 = time.time()
    st = {}
    try:
        st = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        pass
    stage = next((l for l in p.stderr.splitlines() if "stage ms" in l), "")
    r = dict(tag=tag, bin=b, path=BIN[b], timed=timed, rc=p.returncode, wall_s=round(t1 - t0, 3), stats=st, stage=stage)
    with open(LOG, "a") as f:
        f.write(json.dumps(r) + "\n")
    if p.returncode:
        print("FAILED", tag, b, p.stderr[-500:], flush=True)
    return r


def batch_set(name, src, key):
    res = {"A": [], "B": []}
    for rnd in range(2):
        for b in (("A", "B") if rnd == 0 else ("B", "A")):
            for i in range(4):
                r = one(b, src, [], name, i > 0)
                if i:
                    res[b].append(r["stats"].get(key, 0))
    ma, mb = statistics.median(res["A"]), statistics.median(res["B"])
    print(f"{TAG} {name} {key}: A {ma:.2f} B {mb:.2f} B/A {mb / ma:.4f} | A {res['A']} | B {res['B']}", flush=True)


def single():
    photos = sorted(glob.glob(f"{H}/images/*.jpg"))
    f = next(p for p in photos if 7e6 <= Image.open(p).size[0] * Image.open(p).size[1] <= 9e6)
    res = {"A": [], "B": []}
    for k in range(13):
        for b in (("A", "B") if k % 2 == 0 else ("B", "A")):
            t0 = time.time()
            p = subprocess.run([BIN[b], f, f"{W}/single_{b}.avif"], capture_output=True)
            w = time.time() - t0
            with open(LOG, "a") as fo:
                fo.write(json.dumps(dict(tag="single", bin=b, file=f, rc=p.returncode, wall_s=round(w, 4), timed=k > 0)) + "\n")
            if k:
                res[b].append(w)
    ma, mb = statistics.median(res["A"]), statistics.median(res["B"])
    print(f"{TAG} single {os.path.basename(f)} wall s: A {ma:.4f} B {mb:.4f} B/A {mb / ma:.4f}", flush=True)


for s in SETS:
    if s == "sr":
        batch_set("sr_default", speedset("sr", sorted(glob.glob(f"{H}/images/*.jpg"))), "mp_s")
    elif s == "std":
        batch_set("std_default", speedset("std", sorted(glob.glob(f"{H}/images_std/*.jpg"))), "mp_s")
    elif s == "thumbs":
        batch_set("thumbs_default", speedset("thumbs", sorted(glob.glob(f"{H}/sets/thumbs/*.jpg"))), "img_s")
    elif s == "single":
        single()
print("SPD_DONE", TAG, flush=True)
