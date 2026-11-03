#!/usr/bin/env python3
"""an08 speed A/B, the stored method (speed-v06 spd.py): each set repeated to >= 1,000 MP (bench.speed_set), a warm-up
and 3 timed runs (--stats MP/s), 2 rounds with the binary order alternating, median of the 6 timed runs.

  spd_an.py ab      ~/new (this branch) against ~/old (main cde1f74): size range default, standard default,
                    thumbnails default (3 encoders, the default)
  spd_an.py noise   the same with main against main (~/old twice, labelled old / old2)
  spd_an.py single  single-JPEG cold process: `av1gpu in.jpg out.avif` (no --batch), 2.16 MP and 12 MP photos of
                    the size range, a warm-up then 12 runs per binary alternating; wall time per process

Every run appends a JSON line to ~/results/<phase>.jsonl; a summary goes to stdout."""
import glob, json, os, shutil, statistics, subprocess, sys, time

H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/old/bench")
import bench  # noqa: E402
from PIL import Image  # noqa: E402

R = f"{H}/results"
W = f"{H}/work"
os.makedirs(R, exist_ok=True)
os.makedirs(W, exist_ok=True)
BIN = {"new": f"{H}/new/build/av1gpu", "new2": f"{H}/new2/build/av1gpu", "old": f"{H}/old/build/av1gpu",
       "old2": f"{H}/old/build/av1gpu"}


def sizes_of(files):
    return {os.path.basename(f): float(Image.open(f).size[0] * Image.open(f).size[1]) / 1e6 for f in files}


def speedset(name, files):
    d, sfiles, mp, rep = bench.speed_set(files, sizes_of(files), 1000, f"{W}/{name}")
    return d, len(sfiles), mp, rep


def one(phase, b, src, opts, tag, timed):
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
    line = next((l for l in p.stderr.splitlines() if "MP/s" in l), "")
    stage = next((l for l in p.stderr.splitlines() if "stage ms" in l), "")
    r = dict(phase=phase, tag=tag, bin=b, opts=opts, timed=timed, rc=p.returncode, t0=t0, t1=t1, wall_s=round(t1 - t0, 3),
             stats=st, line=line, stage=stage, files=len(os.listdir(out)) if os.path.isdir(out) else 0)
    with open(f"{R}/{phase}.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
    if p.returncode:
        print("FAILED", tag, b, p.stderr[-500:], flush=True)
    return r


def speed_call(phase, b, src, opts, tag, repeats=3):
    runs = []
    for i in range(repeats + 1):
        r = one(phase, b, src, opts, tag, i > 0)
        if i:
            runs.append(r)
    mps = [r["stats"].get("mp_s", 0) for r in runs]
    ims = [r["stats"].get("img_s", 0) for r in runs]
    print(f"{phase} {tag} {b}: {statistics.median(mps):.2f} MP/s {statistics.median(ims):.2f} img/s (runs "
          f"{', '.join(f'{x:.1f}' for x in mps)})", flush=True)
    return runs


def ab(phase, pair):
    gam = sorted(glob.glob(f"{H}/images/*.jpg"))
    std = sorted(glob.glob(f"{H}/images_std/*.jpg"))
    th = sorted(glob.glob(f"{H}/sets/thumbs/*.jpg"))
    sets = {"size_range": speedset("sr", gam), "standard": speedset("std", std), "thumbs": speedset("thumbs", th)}
    json.dump({k: v[1:] for k, v in sets.items()}, open(f"{R}/{phase}_sets.json", "w"))
    res = {}
    for rnd in range(2):
        for s in sets:
            for b in (pair if rnd % 2 == 0 else pair[::-1]):
                res.setdefault((s, b), []).extend(speed_call(phase, b, sets[s][0], [], f"{s}/default"))
    print(f"== {phase} summary (median of 6; rounds = medians of each round's 3 runs)")
    for s in sets:
        med = {}
        for b in pair:
            runs = res[(s, b)]
            v = [r["stats"].get("mp_s", 0) for r in runs]
            iv = [r["stats"].get("img_s", 0) for r in runs]
            med[b] = statistics.median(v)
            print(f"{s} {b}: {statistics.median(v):.2f} MP/s, {statistics.median(iv):.2f} img/s, rounds "
                  f"{statistics.median(v[:3]):.2f} / {statistics.median(v[3:]):.2f}")
        for b in pair[:-1]:
            print(f"{s} {b} vs {pair[-1]}: {100 * (med[b] / med[pair[-1]] - 1):+.2f} %", flush=True)


SB = tuple(os.environ.get("SB", "new old").split())


def single():
    gam = sorted(glob.glob(f"{H}/images/*.jpg"))
    sz = sizes_of(gam)
    pick = []
    for target in (2.16, 12.0):
        pick.append(min(gam, key=lambda f: abs(sz[os.path.basename(f)] - target)))
    res = {}
    out = f"{W}/single_out.avif"
    for f in pick:
        for b in SB:   # warm-up
            subprocess.run([BIN[b], f, out], capture_output=True)
        for i in range(12):
            for b in (SB if i % 2 == 0 else SB[::-1]):
                t0 = time.time()
                p = subprocess.run([BIN[b], f, out], capture_output=True, text=True)
                t = time.time() - t0
                res.setdefault((f, b), []).append(t)
                with open(f"{R}/single.jsonl", "a") as fo:
                    fo.write(json.dumps(dict(file=os.path.basename(f), mp=sz[os.path.basename(f)], bin=b, wall_s=round(t, 4),
                                             rc=p.returncode)) + "\n")
    for f in pick:
        m = {b: statistics.median(res[(f, b)]) for b in SB}
        print(f"single {os.path.basename(f)} ({sz[os.path.basename(f)]:.2f} MP): " +
              ", ".join(f"{b} {m[b] * 1000:.1f} ms ({100 * (m[b] / m['old'] - 1):+.2f} %)" for b in SB) + ", 12 runs each", flush=True)


if __name__ == "__main__":
    for ph in sys.argv[1:]:
        if ph == "ab":
            ab("ab", ("new", "old"))
        elif ph == "ab3":
            ab("ab3", ("new2", "new", "old"))
        elif ph == "noise":
            ab("noise", ("old2", "old"))
        elif ph == "single":
            single()
