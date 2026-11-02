#!/usr/bin/env python3
"""Speed A/B for the k_recon racecheck fix (new = 78e374c) against main v0.6.0 (old = d7d94b3), on an L4 g6.4xlarge.
The method of bench/results/aws-g6-l4-speed-v06/scripts/spd.py: bench.speed_set (each set repeated to >= 1,000 MP),
then bench.gpu_speed's method (a warm-up, then 3 timed runs, --stats); 2 interleaved rounds, binary order
alternating; median of the 6 runs.

  spd.py race    size range default, size range --yuv 444 --depth 10, standard set default (encoder count: the
                 default, no GP_ENCODERS)
  spd.py noise   size range default, main against main (the same binary twice), for the noise

Every run appends a JSON line (round, start/end time, settings, --stats, wall time incl. start-up) to
~/results/<phase>.jsonl."""
import glob, json, os, shutil, statistics, subprocess, sys, time

H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/new/bench")
import bench  # noqa: E402

R = f"{H}/results"
W = f"{H}/work"
os.makedirs(R, exist_ok=True)
os.makedirs(W, exist_ok=True)
BIN = {"new": f"{H}/new/build/av1gpu", "old": f"{H}/old/build/av1gpu", "old2": f"{H}/old/build/av1gpu"}
from PIL import Image  # noqa: E402


def sizes_of(files):
    return {os.path.basename(f): float(Image.open(f).size[0] * Image.open(f).size[1]) / 1e6 for f in files}


def speedset(name, files):
    d, sfiles, mp, rep = bench.speed_set(files, sizes_of(files), 1000, f"{W}/{name}")
    return d, len(sfiles), mp, rep


def one(phase, rnd, b, src, opts, env, tag, timed):
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
    r = dict(phase=phase, rnd=rnd, tag=tag, bin=b, opts=opts, env=env, timed=timed, rc=p.returncode, t0=t0, t1=t1,
             wall_s=round(t1 - t0, 3), stats=st, line=line, files=len(os.listdir(out)) if os.path.isdir(out) else 0)
    with open(f"{R}/{phase}.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
    if p.returncode:
        print("FAILED", tag, b, p.stderr[-500:], flush=True)
    return r


def speed_call(phase, rnd, b, src, opts, env, tag, repeats=3):
    """bench.gpu_speed: a warm-up, then `repeats` timed runs"""
    runs = []
    for i in range(repeats + 1):
        r = one(phase, rnd, b, src, opts, env, tag, i > 0)
        if i:
            runs.append(r)
    mps = [r["stats"].get("mp_s", 0) for r in runs]
    wall = [r["stats"].get("mp", 0) / r["wall_s"] for r in runs]
    print(f"{phase} r{rnd} {tag} {b}: {statistics.median(mps):.1f} MP/s (runs {', '.join(f'{x:.1f}' for x in mps)}; wall "
          f"{statistics.median(wall):.1f})", flush=True)
    return runs


def sets():
    gam = sorted(glob.glob(f"{H}/images/*.jpg"))
    std = sorted(glob.glob(f"{H}/images_std/*.jpg"))
    s = {"size_range": speedset("sr", gam), "standard": speedset("std", std)}
    json.dump({k: v[1:] for k, v in s.items()}, open(f"{R}/speed_sets.json", "w"))
    return s


def race():
    s = sets()
    cfg = [("size_range", "default", []), ("size_range", "444_10", ["--yuv", "444", "--depth", "10"]),
           ("standard", "default", [])]
    for rnd in range(2):
        for k, tag, opts in cfg:
            for b in (("new", "old") if rnd % 2 == 0 else ("old", "new")):
                speed_call("race", rnd, b, s[k][0], opts, {}, f"{k}/{tag}")


def noise():
    s = sets()
    for rnd in range(2):
        for b in (("old", "old2") if rnd % 2 == 0 else ("old2", "old")):
            speed_call("noise", rnd, b, s["size_range"][0], [], {}, "size_range/default")


if __name__ == "__main__":
    for ph in sys.argv[1:]:
        {"race": race, "noise": noise}[ph]()
