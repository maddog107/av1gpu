#!/usr/bin/env python3
"""pipe08 speed A/B (speed-v06's spd.py method): this branch (new = ~/new/build/av1gpu) against main cde1f74
(old = ~/old/build/av1gpu), plus the noise run: main against a copy of itself (old2 = ~/old2/av1gpu).

  spd8.py batch    size range (default, 3 encoders), standard 100 (default) and the 280 thumbnails (default), each
                   repeated to >= 1,000 MP (bench.speed_set); a warm-up and 3 timed runs (--stats) per binary; 2 rounds
                   with the binary order alternating; median of the 6 timed runs.  Pairs: new/old and old2/old.
  spd8.py single   cold-process wall time of `av1gpu IN OUT` (no --batch), one process per run: main, the states after
                   each commit (~/c1 .. ~/c6, c6 = this branch) and old2; 4 JPEGs (320 x 213 and 512 x 384 thumbnails,
                   2.4 and 10 MP photos); one warm-up each, then 10 runs per binary, binaries alternating run by run
                   (wall.py); and the upload pair (photo + its 512 x 384 thumbnail) as a 2-image --batch.

  spd8.py split    the size range with ~/c1, ~/c2, ~/c3 (2 rounds, order alternating): commits 2 and 3 separately.

Every run appends a JSON line to ~/results/spd8_<phase>.jsonl."""
import glob, json, os, shutil, statistics, subprocess, sys, time

H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/old/bench")
import bench  # noqa: E402
from PIL import Image  # noqa: E402

R = f"{H}/results"; W = f"{H}/work"
os.makedirs(R, exist_ok=True); os.makedirs(W, exist_ok=True)
BIN = {"new": f"{H}/new/build/av1gpu", "old": f"{H}/old/build/av1gpu", "old2": f"{H}/old2/av1gpu"}
BIN.update({f"c{n}": f"{H}/c{n}/build/av1gpu" for n in range(1, 7)})


def sizes_of(files):
    return {os.path.basename(f): float(Image.open(f).size[0] * Image.open(f).size[1]) / 1e6 for f in files}


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
    r = dict(phase=phase, tag=tag, bin=b, opts=opts, timed=timed, rc=p.returncode, t0=t0, t1=t1, wall_s=round(t1 - t0, 3),
             stats=st, line=line, files=len(os.listdir(out)) if os.path.isdir(out) else 0)
    with open(f"{R}/spd8_{phase}.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
    if p.returncode:
        print("FAILED", tag, b, p.stderr[-500:], flush=True)
    return r


def speed_call(phase, b, src, opts, tag, repeats=3):
    runs = [r for i in range(repeats + 1) for r in [one(phase, b, src, opts, tag, i > 0)] if i]
    mps = [r["stats"].get("mp_s", 0) for r in runs]
    print(f"{phase} {tag} {b}: {statistics.median(mps):.1f} MP/s (runs {', '.join(f'{x:.1f}' for x in mps)}), "
          f"wall {statistics.median(r['wall_s'] for r in runs):.2f} s", flush=True)
    return runs


def batch():
    sets = {}
    for name, files in (("size_range", sorted(glob.glob(f"{H}/images/*.jpg"))), ("standard", sorted(glob.glob(f"{H}/images_std/*.jpg"))),
                        ("thumbs", sorted(glob.glob(f"{H}/sets/thumbs/*.jpg")))):
        d, sfiles, mp, rep = bench.speed_set(files, sizes_of(files), 1000, f"{W}/spd8_{name}")
        sets[name] = d
        json.dump(dict(files=len(files), repeated_files=len(sfiles), mp=mp, repeats=rep), open(f"{R}/spd8_{name}_set.json", "w"))
    for rnd in range(2):
        for s in ("size_range", "standard", "thumbs"):
            order = ("new", "old", "old2") if rnd % 2 == 0 else ("old2", "old", "new")
            for b in order:
                speed_call("batch", b, sets[s], [], f"{s}/default")


def single():
    bins = [f"{H}/old/build/av1gpu"] + [f"{H}/c{n}/build/av1gpu" for n in range(1, 7)] + [BIN["old2"]]
    out = open(f"{R}/spd8_single.txt", "a")
    for f in ("t320.jpg", "thumb512.jpg", "photo.jpg", "big10.jpg"):
        p = subprocess.run([sys.executable, f"{H}/wall.py", "10"] + bins + ["--", f"{H}/st/{f}", "/tmp/s{bin}.avif"],
                           capture_output=True, text=True)
        out.write(f"== {f}\n{p.stdout}{p.stderr[-2000:]}\n"); out.flush()
        print(f"== {f}\n" + "\n".join(p.stdout.splitlines()[:-1]), flush=True)
    p = subprocess.run([sys.executable, f"{H}/wall.py", "10"] + bins + ["--", "--batch", f"{H}/st/pair.txt", "--outdir", "/tmp/pb{bin}"],
                       capture_output=True, text=True)
    out.write(f"== pair batch (photo.jpg + thumb512.jpg)\n{p.stdout}{p.stderr[-2000:]}\n")
    print("== pair batch\n" + "\n".join(p.stdout.splitlines()[:-1]), flush=True)


def split():
    """the size range (as in batch) with the trees after commits 1, 2 and 3: which commit the batch gain comes from"""
    d = f"{W}/spd8_size_range/speedset"
    for rnd in range(2):
        for b in (("c1", "c2", "c3") if rnd % 2 == 0 else ("c3", "c2", "c1")):
            speed_call("split", b, d, [], "size_range/default")


if __name__ == "__main__":
    for ph in sys.argv[1:]:
        {"batch": batch, "single": single, "split": split}[ph]()
