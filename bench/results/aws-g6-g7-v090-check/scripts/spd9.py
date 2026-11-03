#!/usr/bin/env python3
"""rc09 speed A/B (speed-v06's spd.py method, as pipe08's spd8.py): v0.9.0 candidate B13 (main c553ffb, CUDA 13 package)
against v0.8.0 A13 (CUDA 13 release binary), plus the noise run A13c (the same v0.8.0 binary from a second path).

  spd9.py batch [SETS]  size range (default, 3 encoders), standard 100, the 280 thumbnails and the size range at
                        --yuv 444 --depth 10, each repeated to >= 1,000 MP (bench.speed_set); a warm-up and 3 timed runs
                        (--stats) per binary; 2 rounds with the binary order alternating (B13 A13 A13c, then A13c A13
                        B13); median of the 6 timed runs.
  spd9.py single        cold-process wall time (wall9.py), 12 runs per binary alternating run by run after a warm-up:
                        B13, A13, A13c and A13 with GP_MAX_CONNECTIONS=1; `av1gpu IN OUT` for the 2-3 MP photo, the
                        10-12 MP photo, the 512 x 384 thumbnail and the 320 x 320 thumbnail, and the production pair
                        (photo + 512 x 384 thumbnail) as a 2-entry --batch.
Every run appends a JSON line to ~/results/spd9_<phase>.jsonl; wall9 output to ~/results/spd9_single.txt."""
import glob, json, os, shutil, statistics, subprocess, sys, time

H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/src/bench")
import bench  # noqa: E402
from PIL import Image  # noqa: E402

R = f"{H}/results"; W = f"{H}/work"
os.makedirs(R, exist_ok=True); os.makedirs(W, exist_ok=True)
BIN = {"B13": f"{H}/bin/B13", "A13": f"{H}/bin/A13", "A13c": f"{H}/bin/A13c"}
SETS = {"size_range": ("images/*.jpg", []), "standard": ("images_std/*.jpg", []), "thumbs": ("sets/thumbs/*.jpg", []),
        "size_range_444_10": ("images/*.jpg", ["--yuv", "444", "--depth", "10"])}


def sizes_of(files):
    out = {}
    for f in files:
        w, h = Image.open(f).size
        out[os.path.basename(f)] = w * h / 1e6
    return out


def one(phase, b, src, opts, tag, timed):
    out = f"{W}/spd_out"
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
    r = dict(phase=phase, tag=tag, bin=b, opts=opts, timed=timed, rc=p.returncode, t0=t0, t1=t1, wall_s=round(t1 - t0, 3),
             stats=st, stage=stage, files=len(os.listdir(out)) if os.path.isdir(out) else 0)
    with open(f"{R}/spd9_{phase}.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
    if p.returncode:
        print("FAILED", tag, b, p.stderr[-500:], flush=True)
    return r


def speed_call(phase, b, src, opts, tag, repeats=3):
    runs = [r for i in range(repeats + 1) for r in [one(phase, b, src, opts, tag, i > 0)] if i]
    mps = [r["stats"].get("mp_s", 0) for r in runs]
    ips = [r["stats"].get("img_s", 0) for r in runs]
    print(f"{phase} {tag} {b}: {statistics.median(mps):.1f} MP/s {statistics.median(ips):.1f} img/s "
          f"(runs {', '.join(f'{x:.1f}' for x in mps)}), wall {statistics.median(r['wall_s'] for r in runs):.2f} s", flush=True)
    return runs


def batch(names):
    sets = {}
    for name in names:
        pat, opts = SETS[name]
        files = sorted(glob.glob(f"{H}/{pat}"))
        d, sfiles, mp, rep = bench.speed_set(files, sizes_of(files), 1000, f"{W}/spd9_{name}")
        sets[name] = (d, opts)
        json.dump(dict(files=len(files), repeated_files=len(sfiles), mp=mp, repeats=rep, opts=opts),
                  open(f"{R}/spd9_{name}_set.json", "w"))
    for rnd in range(2):
        for s in names:
            order = ("B13", "A13", "A13c") if rnd % 2 == 0 else ("A13c", "A13", "B13")
            for b in order:
                speed_call("batch", b, sets[s][0], sets[s][1], f"{s}/r{rnd}")


def single():
    bins = [BIN["B13"], BIN["A13"], BIN["A13c"], BIN["A13"] + "@GP_MAX_CONNECTIONS=1"]
    out = open(f"{R}/spd9_single.txt", "a")
    jobs = [(f, [f"{H}/st/{f}", "/tmp/s{bin}.avif"]) for f in ("photo.jpg", "big.jpg", "thumb512.jpg", "t320.jpg")]
    jobs.append(("pair batch (photo.jpg + thumb512.jpg)", ["--batch", f"{H}/st/pair.txt", "--outdir", "/tmp/pb{bin}"]))
    for name, args in jobs:
        p = subprocess.run([sys.executable, f"{H}/wall9.py", "12"] + bins + ["--"] + args, capture_output=True, text=True)
        out.write(f"== {name}\n{p.stdout}{p.stderr[-2000:]}\n"); out.flush()
        print(f"== {name}\n" + "\n".join(p.stdout.splitlines()[:-1]) + p.stderr[-500:], flush=True)


if __name__ == "__main__":
    ph = sys.argv[1]
    if ph == "batch":
        batch(sys.argv[2:] or list(SETS))
    elif ph == "single":
        single()
