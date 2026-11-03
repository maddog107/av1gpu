#!/usr/bin/env python3
"""Speed A/B of the palette branch (new = ~/br) against main 60f94ff (old = ~/av1gpu) on an L4 g6.4xlarge.
The method of bench/results/aws-g6-l4-speed-v06/scripts/spd.py: bench.speed_set (each set repeated to >= 1,000 MP),
then bench.gpu_speed's method (a warm-up, then 3 timed runs, --stats); 2 interleaved rounds, binary order
alternating; median of the 6 runs.

  spd_pal.py race     size range default, size range --yuv 444 --depth 10, standard set default (photos: detection
                      runs, no image passes it)
  spd_pal.py noise    size range default, main against main (the same binary twice)
  spd_pal.py screen   the screen set and the flat PNGs (palette search on the images that pass detection): default; the
                      branch also with --palette off and --palette on
  spd_pal.py screenll the same images, --lossless and --lossless --ycgco-re (GP_ENCODE_MP=12)
  spd_pal.py off      the size range, the branch with --palette off against main
  spd_pal.py graphics the screen set and the flat PNGs, default only

Every run appends a JSON line (round, settings, --stats, wall time incl. start-up) to ~/pal/spd/<phase>.jsonl."""
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/br/bench")
import bench  # noqa: E402
from PIL import Image  # noqa: E402

R = f"{H}/pal/spd"
W = f"{H}/pal/spdwork"
os.makedirs(R, exist_ok=True)
os.makedirs(W, exist_ok=True)
BIN = {"new": f"{H}/br/build/av1gpu", "old": f"{H}/av1gpu/build/av1gpu", "old2": f"{H}/av1gpu/build/av1gpu"}


def sizes_of(files):
    out = {}
    for f in files:
        w, h = Image.open(f).size
        out[os.path.basename(f)] = w * h / 1e6
    return out


def speedset(name, files):
    d, sfiles, mp, rep = bench.speed_set(files, sizes_of(files), 1000, f"{W}/{name}")
    return d, len(sfiles), mp, rep


def one(phase, rnd, b, src, opts, tag, timed):
    out = f"{W}/out"
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.time()
    env = dict(os.environ, GP_ENCODE_MP="12") if "--lossless" in opts else None   # lossless needs ~1 GB per batch MP
    p = subprocess.run([BIN[b], "--batch", src, "--outdir", out, "--stats"] + opts, capture_output=True, text=True, env=env)
    t1 = time.time()
    st = {}
    try:
        st = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        pass
    r = dict(phase=phase, rnd=rnd, tag=tag, bin=b, opts=opts, timed=timed, rc=p.returncode, t0=t0, t1=t1,
             wall_s=round(t1 - t0, 3), stats=st, files=len(os.listdir(out)) if os.path.isdir(out) else 0)
    with open(f"{R}/{phase}.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
    if p.returncode:
        print("FAILED", tag, b, p.stderr[-500:], flush=True)
    return r


def speed_call(phase, rnd, b, src, opts, tag, repeats=3):
    runs = []
    for i in range(repeats + 1):
        r = one(phase, rnd, b, src, opts, tag, i > 0)
        if i:
            runs.append(r)
    mps = [r["stats"].get("mp_s", 0) for r in runs]
    print(f"{phase} r{rnd} {tag} {b}: {statistics.median(mps):.1f} MP/s (runs {', '.join(f'{x:.1f}' for x in mps)})", flush=True)
    return runs


def ab(phase, cfg, pair=("new", "old")):
    res = {}
    for rnd in range(2):
        for k, tag, opts_new, opts_old in cfg:
            for b in (pair if rnd % 2 == 0 else pair[::-1]):
                runs = speed_call(phase, rnd, b, k, opts_new if b == pair[0] else opts_old, tag)
                res.setdefault((tag, b), []).extend(r["stats"].get("mp_s", 0) for r in runs)
    out = {}
    for tag in dict.fromkeys(t for t, _ in res):
        a, o = statistics.median(res[(tag, pair[0])]), statistics.median(res[(tag, pair[1])])
        out[tag] = dict(new=a, old=o, change=a / o - 1)
        print(f"SUMMARY {phase} {tag}: {pair[0]} {a:.1f} MP/s, {pair[1]} {o:.1f} MP/s, {100 * (a / o - 1):+.1f} %", flush=True)
    json.dump(out, open(f"{R}/{phase}_summary.json", "w"), indent=1)


def photo_sets():
    sr = speedset("sr", sorted(glob.glob(f"{H}/pal/sizerange/*.jpg")))
    std = speedset("std", sorted(glob.glob(f"{H}/pal/std100/*.jpg")))
    return sr, std


def race():
    sr, std = photo_sets()
    ab("race", [(sr[0], "size_range/default", [], []), (sr[0], "size_range/444_10", ["--yuv", "444", "--depth", "10"],
                                                         ["--yuv", "444", "--depth", "10"]),
                (std[0], "standard/default", [], [])])


def noise():
    sr, _ = photo_sets()
    ab("noise", [(sr[0], "size_range/default", [], [])], pair=("old", "old2"))


def off():
    """diagnosis: the branch with --palette off (no detection, no palette slots) against main"""
    sr, _ = photo_sets()
    ab("off", [(sr[0], "size_range/default palette off", ["--palette", "off"], [])])


def screenll():
    """lossless on the screen set and the flat PNGs (GP_ENCODE_MP=12, as the lossless comparison runs it)"""
    files = sorted(glob.glob(f"{H}/pal/sets/screen/*.png") + glob.glob(f"{H}/pal/sets/flat_png/*.png"))
    sc = speedset("scr", files)
    ab("screenll", [(sc[0], "graphics/lossless", ["--lossless"], ["--lossless"]),
                    (sc[0], "graphics/lossless ycgco-re", ["--lossless", "--ycgco-re"], ["--lossless", "--ycgco-re"])])


def graphics():
    """the screen set and the flat PNGs, default settings only (e.g. with GP_PAL_FAST=1 in the environment)"""
    files = sorted(glob.glob(f"{H}/pal/sets/screen/*.png") + glob.glob(f"{H}/pal/sets/flat_png/*.png"))
    sc = speedset("scr", files)
    ab("graphics", [(sc[0], "graphics/default", [], [])])


def screen():
    files = sorted(glob.glob(f"{H}/pal/sets/screen/*.png") + glob.glob(f"{H}/pal/sets/flat_png/*.png"))
    sc = speedset("scr", files)
    ab("screen", [(sc[0], "graphics/default", [], []),
                  (sc[0], "graphics/default palette off", ["--palette", "off"], []),
                  (sc[0], "graphics/default palette on", ["--palette", "on"], [])])


if __name__ == "__main__":
    for ph in sys.argv[1:]:
        {"race": race, "noise": noise, "screen": screen, "off": off, "screenll": screenll, "graphics": graphics}[ph]()
