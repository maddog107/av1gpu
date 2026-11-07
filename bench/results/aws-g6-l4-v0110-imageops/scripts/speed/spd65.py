#!/usr/bin/env python3
"""spd65.py TAG ROUNDS SPECS -- SET...   : interleaved batch speed runs.

SPECS: comma-separated NAME=PATH[|extra args] (a binary, optionally with options; e.g. A=/x/av1gpu,C=/y/av1gpu,B=/z/av1gpu).
SET: sr std thumbs comics pngwebp c444 (see SETS).  Every set is repeated to >= 1,000 MP (bench.speed_set), outputs go to a
tmpfs (/dev/shm), so disk does not time anything.  One warm-up of each spec, then ROUNDS rounds; each round runs every spec
once, the order rotating and reversing round by round.  Per run: --stats JSON, the process wall time, its peak RSS (wait4), a
line in ~/results/spd_TAG.jsonl.  Prints for each set per spec: median mp_s and img_s, median wall ms, median RSS, peak dev,
and per-round paired ratios against the first spec (median, min, max)."""
import glob, itertools, json, os, shutil, statistics as st, subprocess, sys, time
H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/main/bench")
import bench  # noqa: E402
from PIL import Image  # noqa: E402
TAG, ROUNDS = sys.argv[1], int(sys.argv[2])
k = sys.argv.index("--")
SPECS = []
for s in sys.argv[3].split(";" if ";" in sys.argv[3] else ","):
    n, _, v = s.partition("=")
    p, _, ex = v.partition("|")
    SPECS.append((n, p, ex.split()))
NAMES = [s[0] for s in SPECS]
SETLIST = sys.argv[k + 1:]
R = f"{H}/results"; W = f"{H}/work"; O = "/dev/shm/out"
os.makedirs(R, exist_ok=True); os.makedirs(W, exist_ok=True)
LOG = f"{R}/spd_{TAG}.jsonl"
C444 = "--yuv 444 --depth 10 --tiles 2,2,2 --max-mp 100 --max-input-mp 100".split()
SETS = {"sr": (f"{H}/images/*.jpg", []), "std": (f"{H}/images_std/*.jpg", []), "thumbs": (f"{H}/sets/thumbs/*.jpg", []),
        "comics": (f"{H}/sets/comics/*.jpg", []), "border": (f"{H}/sets/border/*.jpg", []), "b444": (f"{H}/sets/border/*.jpg", C444), "pngwebp": (f"{H}/sets/pngwebp/*", []), "c444": (f"{H}/sets/comics/*.jpg", C444)}
def mkset(name, pat):
    files = sorted(glob.glob(pat))
    sizes = {os.path.basename(f): Image.open(f).size[0] * Image.open(f).size[1] / 1e6 for f in files}
    d, _, mp, rep = bench.speed_set(files, sizes, 1000, f"{W}/set_{name}")
    return d, mp
def one(spec, src, opts, tag, rnd, timed):
    n, path, ex = spec
    shutil.rmtree(O, ignore_errors=True)
    so, se = open(f"{W}/o.txt", "wb"), open(f"{W}/e.txt", "wb")
    t0 = time.perf_counter()
    p = subprocess.Popen([path, "--batch", src, "--outdir", O, "--stats"] + opts + ex, stdout=so, stderr=se)
    _, status, ru = os.wait4(p.pid, 0)
    wall = time.perf_counter() - t0
    so.close(); se.close()
    rc = os.waitstatus_to_exitcode(status)
    out = open(f"{W}/o.txt").read(); err = open(f"{W}/e.txt").read()
    try:
        s = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        s = {}
    stage = next((l for l in err.splitlines() if "stage ms" in l), "")
    trim = next((l for l in err.splitlines() if "trim:" in l), "")
    r = dict(tag=tag, spec=n, rnd=rnd, timed=timed, rc=rc, wall_s=round(wall, 3), rss_mb=round(ru.ru_maxrss / 1024, 1), utime=ru.ru_utime, stime=ru.ru_stime,
             stats=s, stage=stage, trim=trim)
    with open(LOG, "a") as f:
        f.write(json.dumps(r) + "\n")
    if rc:
        print("FAILED", tag, n, rc, err[-400:], flush=True)
    return r
def bset(name):
    pat, opts = SETS[name]
    src, mp = mkset(name, pat)
    res = {n: [] for n in NAMES}
    for s in SPECS:
        one(s, src, opts, name, -1, False)
    order = list(SPECS)
    for r in range(ROUNDS):
        rot = order[r % len(order):] + order[:r % len(order)]
        if r % 2:
            rot = rot[::-1]
        for s in rot:
            res[s[0]].append(one(s, src, opts, name, r, True))
    base = NAMES[0]
    key = "img_s" if name == "thumbs" else "mp_s"
    print(f"== {TAG} {name}: {mp:.0f} MP, {ROUNDS} rounds, {key}", flush=True)
    for n in NAMES:
        v = [x["stats"].get(key, 0) for x in res[n]]
        w = [x["wall_s"] for x in res[n]]
        rs = [x["rss_mb"] for x in res[n]]
        pk = [x["stats"].get("peak_dev_mb", 0) for x in res[n]]
        pr = [a / b for a, b in zip(v, [x["stats"].get(key, 0) for x in res[base]]) if b]
        print(f"  {n:12s} {key} med {st.median(v):9.2f} min {min(v):9.2f} max {max(v):9.2f} | wall {st.median(w):7.3f} s | rss {st.median(rs):7.0f} MB | peak dev {st.median(pk):6.0f} MB"
              f" | {n}/{base} med ratio of rounds {st.median(pr):.4f} [{min(pr):.4f}, {max(pr):.4f}] ratio of medians {st.median(v) / st.median([x['stats'].get(key, 0) for x in res[base]]):.4f}", flush=True)
for s in SETLIST:
    bset(s)
print("SPD_DONE", TAG, flush=True)
